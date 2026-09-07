"""`just braintrust-cockpit` -- create/update the Braintrust cockpit objects
(views, dashboard, Topics, one Pattern) idempotently via REST, replay the
refined judge-span trace for the hero case, and write
`data/reports/demo_manifest.json` (spec `m7b.md`).

M7b is additive: nothing M1-M7a computed is rerun or retuned. This module
only replays already-computed local artifacts (`data/eval/judge_scores.jsonl`,
`data/eval/calibration/`, `data/results/`, `data/reports/`) and issues
idempotent-by-name REST calls. No model is ever called here.

Two client seams, both optional:
- `sdk_client` -- a `braintrust`-module-shaped object (or fake) used only to
  replay the hero-case judge-span trace as a dedicated experiment
  (`m7b-hero-case`), the same `start_span`/`log` shape `braintrust_sync.py`
  uses.
- `rest_client` -- a thin REST seam (`get`/`post`/`patch`) for `/v1/view`
  (saved views + the dashboard), `/v1/function` (Topics facet), and reads
  against `/v1/experiment`, `/v1/dataset`, `/v1/organization` for the
  manifest's `experiments`/`datasets`/`permalinks` keys. Patterns have no
  REST creation path at all (see `PATTERN_REST_LIMITATION`). A fake test
  double drives the offline gate; `RestClient` is the real (but never
  network-called in tests) HTTP wrapper.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dealpoint.config import (
    DEMO_MANIFEST_PATH,
    JUDGE_SCORES_PATH,
    JUDGED_SUBSET_HASH,
    JUDGED_SUBSET_PATH,
    M7B_MAX_NEW_SCORES,
)
from dealpoint.eval.braintrust_adapter import PROJECT
from dealpoint.eval.braintrust_sync import _load_jsonl, _packet_id_for

JUDGE_DIMENSIONS: tuple[str, ...] = ("reasoning", "evidence", "trajectory", "professional")

HERO_VARIANTS: tuple[str, str] = ("A@haiku", "D@haiku")

HERO_RULE_TEXT = (
    "From the judged subset (subset_hash "
    f"{JUDGED_SUBSET_HASH}): candidates are every case_id where arm A@haiku and "
    "arm D@haiku disagree on deterministic obj/grounded_accuracy. Among "
    "candidates, the hero case is the one with the largest max pairwise "
    "spread (max(dim_scores) - min(dim_scores), across the three judges) on "
    "any of the four rubric dimensions, for either variant. Ties (including "
    "no candidates found) are broken by ascending case_id string sort."
)


# --- human-scoring path probe (spec section 3) ------------------------------

HUMAN_SCORING_PROBE = {
    "probed_via": (
        "mcp_braintrust_get_project_settings(project_id=dealpoint-eval) live on "
        "2026-09-06, corroborated with mcp_braintrust_search_docs('configure human "
        "review scores Pro Enterprise plan') and data/reports/braintrust_sync.json"
    ),
    "probed_at": "2026-09-06T23:00:00+00:00",
    "finding": (
        "get_project_settings on the live dealpoint-eval project returned "
        "settings={} -- zero configured review scores exist on this workspace "
        "today, so there is nothing to extend rather than something already at "
        "a hard cap. The docs' 'Configure review scores' page states human "
        "review scorers are 'only available on Pro and Enterprise plans', and "
        "separately (docs 'Upgrade your plan') that Starter is 'limited to 1 "
        "per project' even where available -- 1 slot, not the 4 rubric "
        "dimensions (reasoning, evidence, trajectory, professional) this "
        "milestone needs. That this workspace is Starter-tier is independently "
        "evidenced by the M7a live sync's real, observed block: "
        "data/reports/braintrust_sync.json's live_run_status recorded "
        "blocked_by='Braintrust workspace plan limit: num_scores_calendar_months' "
        "with usage 11016 against a limit of 11000 -- a Starter-plan monthly "
        "score quota actually hit live, not inferred from a doc alone."
    ),
    "decision": "local_form",
    "decision_reason": (
        "Configured Review scores cannot carry all four rubric dimensions on "
        "this workspace's plan (1 slot available, 4 needed). Branch B of spec "
        "section 3 applies: data/eval/calibration/form.md is the scoring "
        "surface (as M5 already built), `just calibration` computes agreement "
        "locally, and `just braintrust-cockpit` pushes the rows as "
        "human/<dimension> scores on the matching experiment/trace rows. The "
        "walkthrough shows human scores in the experiment table and the trace, "
        "not in Review mode."
    ),
}


# --- hero case selection -----------------------------------------------------


def _judged_subset() -> dict:
    return json.loads(Path(JUDGED_SUBSET_PATH).read_text(encoding="utf-8"))


def _variant_results(subset: dict, variant_id: str) -> dict[str, dict]:
    for variant in subset.get("variants", []):
        if variant["variant_id"] == variant_id:
            return {row["case_id"]: row for row in _load_jsonl(variant["results_path"])}
    return {}


def _judge_rows_for_packet(judge_rows: list[dict], packet_id: str) -> list[dict]:
    return [r for r in judge_rows if r.get("packet_id") == packet_id and r.get("ok", True)]


def _max_pairwise_spread(judge_rows: list[dict], packet_id: str) -> float:
    rows = _judge_rows_for_packet(judge_rows, packet_id)
    best = 0.0
    for dim in JUDGE_DIMENSIONS:
        vals = [r[dim] for r in rows if r.get(dim) is not None]
        if len(vals) >= 2:
            best = max(best, max(vals) - min(vals))
    return best


def hero_case() -> dict:
    """The one case the demo follows, chosen by `HERO_RULE_TEXT` -- pure
    function over local files, deterministic, no model calls.
    """
    subset = _judged_subset()
    assert subset.get("subset_hash") == JUDGED_SUBSET_HASH, (
        f"judged_subset.json subset_hash changed ({subset.get('subset_hash')!r}); "
        "the hero rule is pinned to a specific subset"
    )
    a_rows = _variant_results(subset, HERO_VARIANTS[0])
    d_rows = _variant_results(subset, HERO_VARIANTS[1])
    judge_rows = _load_jsonl(JUDGE_SCORES_PATH)

    candidates: list[tuple[str, float, bool | None, bool | None]] = []
    for case_id in subset.get("case_ids", []):
        ga_a = (a_rows.get(case_id, {}).get("scores") or {}).get("grounded_accuracy")
        ga_d = (d_rows.get(case_id, {}).get("scores") or {}).get("grounded_accuracy")
        if ga_a is None or ga_d is None or ga_a == ga_d:
            continue
        spread = max(
            _max_pairwise_spread(judge_rows, _packet_id_for(case_id, HERO_VARIANTS[0])),
            _max_pairwise_spread(judge_rows, _packet_id_for(case_id, HERO_VARIANTS[1])),
        )
        candidates.append((case_id, spread, ga_a, ga_d))

    if not candidates:
        return {
            "rule": HERO_RULE_TEXT,
            "case_id": None,
            "max_pairwise_spread": None,
            "grounded_accuracy_a": None,
            "grounded_accuracy_d": None,
            "note": "no case_id in the judged subset has arm A/D disagreement on grounded_accuracy",
        }

    candidates.sort(key=lambda c: (-c[1], c[0]))
    case_id, spread, ga_a, ga_d = candidates[0]
    return {
        "rule": HERO_RULE_TEXT,
        "case_id": case_id,
        "max_pairwise_spread": spread,
        "grounded_accuracy_a": ga_a,
        "grounded_accuracy_d": ga_d,
        "n_candidates": len(candidates),
    }


# --- judge spans (spec section 2) -------------------------------------------


def _judge_family_map() -> dict[str, str]:
    """`judge_model -> lowercase family` (`judge/mistral`, `judge/nvidia`,
    `judge/bytedance`), read from the frozen `judge_slate.json` -- never
    hardcoded, so a judge-trio change is picked up automatically.
    """
    from dealpoint.config import JUDGE_SLATE_PATH

    payload = json.loads(Path(JUDGE_SLATE_PATH).read_text(encoding="utf-8"))
    return {j["model"]: j["family"].lower() for j in payload.get("judges", []) if j.get("ok")}


def _packet_text_for(case_id: str, variant_id: str) -> str | None:
    from dealpoint.eval.judge_run import build_all_packets

    for item in build_all_packets():
        if item["case_id"] == case_id and item["variant_id"] == variant_id:
            return item["packet_text"]
    return None


def _human_row_for_packet(packet_id: str) -> dict | None:
    from dealpoint.config import CALIBRATION_DIR

    path = CALIBRATION_DIR / "human_scores.jsonl"
    for row in _load_jsonl(path):
        if row.get("packet_id") == packet_id:
            return row
    return None


AGGREGATE_ROUNDING_RULE = (
    "M5 decision D5: the mean-of-judges is rounded to the nearest integer "
    "(Python round-half-to-even) only when computing weighted Cohen's kappa "
    "against a human score, which is defined on integer ratings. Spearman "
    "rho and this aggregate span's own scores use the raw (unrounded) mean."
)


def judge_spans(case_id: str, variant_id: str) -> dict:
    """Build the `scoring` span's children per spec section 2: one
    `judge/<family>` span per judge carrying the blinded packet as input and
    the judge's raw JSON as output, one `judge/aggregate` span with the
    mean-of-judges per dimension and per-judge scores in metadata (never as
    scores -- the score budget stays six plus four), and the `human/<dimension>`
    scores when a human score exists for this packet.

    Replayed entirely from `data/eval/judge_scores.jsonl`,
    `data/eval/calibration/human_scores.jsonl` and the calibration packets.
    No model call. If a stored judge result lacks the packet or the raw
    JSON, the span says so in metadata rather than inventing content.
    """
    packet_id = _packet_id_for(case_id, variant_id)
    packet_text = _packet_text_for(case_id, variant_id)
    judge_rows = _judge_rows_for_packet(_load_jsonl(JUDGE_SCORES_PATH), packet_id)
    family_map = _judge_family_map()

    spans: list[dict] = []
    per_judge_scores: dict[str, dict] = {}
    for row in judge_rows:
        family: str = family_map.get(row.get("judge_model") or "") or str(row.get("judge_family", "unknown")).lower()
        output = {dim: row.get(dim) for dim in JUDGE_DIMENSIONS}
        output["notes"] = row.get("notes")
        span = {
            "name": f"judge/{family}",
            "input": packet_text,
            "output": output,
            "metadata": {
                "judge_model": row.get("judge_model"),
                "rubric_version": row.get("rubric_version"),
                "judge_price_usd": row.get("usd"),
                "call_cost_usd": row.get("usd"),
                "subset_hash": JUDGED_SUBSET_HASH,
            },
        }
        if packet_text is None:
            span["metadata"]["missing_packet"] = "no stored blinded packet found for this case/variant"
        spans.append(span)
        per_judge_scores[family] = output

    if not judge_rows:
        spans.append(
            {
                "name": "judge/unavailable",
                "metadata": {
                    "reason": "no stored judge result for this packet_id",
                    "packet_id": packet_id,
                    "case_id": case_id,
                    "variant_id": variant_id,
                },
            }
        )

    aggregate: dict[str, float] = {}
    for dim in JUDGE_DIMENSIONS:
        vals = [r[dim] for r in judge_rows if r.get(dim) is not None]
        if vals:
            aggregate[dim] = sum(vals) / len(vals)

    human_row = _human_row_for_packet(packet_id)
    human_scores = {dim: human_row[dim] for dim in JUDGE_DIMENSIONS if human_row and human_row.get(dim) is not None}

    spans.append(
        {
            "name": "judge/aggregate",
            "output": aggregate,
            "metadata": {
                "rounding_rule": AGGREGATE_ROUNDING_RULE,
                "per_judge_scores": per_judge_scores,
                "subset_hash": JUDGED_SUBSET_HASH,
                "human_scores": human_scores or None,
            },
        }
    )

    return {
        "packet_id": packet_id,
        "case_id": case_id,
        "variant_id": variant_id,
        "spans": spans,
        "aggregate": aggregate,
        "human_scores": human_scores,
    }


# --- saved views + dashboard (spec sections 4-5) ----------------------------

BRAINTRUST_QUERIES_MD_PATH = Path(__file__).resolve().parents[2] / "docs" / "braintrust-queries.md"

# query title -> its 1-based position in docs/braintrust-queries.md, so the
# BTQL text used by a saved view is READ from the doc, never duplicated --
# a test asserts the two cannot drift apart.
QUERY_NUMBERS = {
    "retrieval_rescue": 1,
    "failure_attribution": 2,
    "deepeval_disagreement": 4,
    "trajectory_inefficiency": 6,
}


def documented_btql_query(query_number: int) -> str:
    """The exact BTQL query text (the first fenced, non-bash code block)
    under `## {query_number}.` in `docs/braintrust-queries.md`.
    """
    text = BRAINTRUST_QUERIES_MD_PATH.read_text(encoding="utf-8")
    marker = f"## {query_number}."
    start = text.index(marker)
    rest = text[start:]
    fence_start = rest.index("```\n") + 4
    fence_end = rest.index("```", fence_start)
    return rest[fence_start:fence_end].strip()


def _review_set_case_ids() -> list[str]:
    from dealpoint.eval.braintrust_sync import review_set

    return sorted({p["case_id"] for p in review_set()})


def view_definitions() -> list[dict]:
    """The seven named table views (spec section 4). Names are stable;
    `sync_cockpit` resolves by name so a re-run updates rather than
    duplicates. BTQL filters are the documented queries verbatim; every
    `definition` carries a `btql` key so `_view_data_for` maps it onto the
    real `view_data.search.filter` REST shape without a second mechanism.
    """
    review_case_ids = _review_set_case_ids()
    review_btql = " or ".join(f"metadata.case_id = '{cid}'" for cid in review_case_ids) or "false"
    return [
        {
            "name": "Judged traces by variant",
            "view_type": "experiments",
            "caption": "Every judge- experiment, tagged stage=evaluation: variant, judge/* aggregates, obj/grounded_accuracy.",
            "definition": {"btql": "tags.stage = 'evaluation' and name like 'judge-%'"},
        },
        {
            "name": "Judge disagreement",
            "view_type": "experiment",
            "bind_to": "hero_experiment",
            "caption": "Rows on the hero variant where max pairwise judge spread >= 2 on any dimension, sorted by spread.",
            "definition": {
                "btql": "metadata.judge_spread_max >= 2",
                "sort": "metadata.judge_spread_max desc",
            },
        },
        {
            "name": "Retrieval rescue",
            "view_type": "logs",
            "caption": "BTQL query 1 (docs/braintrust-queries.md): which cases did a later search_agreement call surface gold evidence that the first search missed.",
            "definition": {"btql": documented_btql_query(QUERY_NUMBERS["retrieval_rescue"])},
        },
        {
            "name": "Failure attribution",
            "view_type": "logs",
            "caption": "BTQL query 2: EXECUTION_FAILED / CAP_HIT counts grouped by model and arm.",
            "definition": {"btql": documented_btql_query(QUERY_NUMBERS["failure_attribution"])},
        },
        {
            "name": "DeepEval vs judge disagreement",
            "view_type": "logs",
            "caption": "BTQL query 4: traces where deepeval/task_completion and obj/grounded_accuracy disagree.",
            "definition": {"btql": documented_btql_query(QUERY_NUMBERS["deepeval_disagreement"])},
        },
        {
            "name": "Trajectory inefficiency",
            "view_type": "logs",
            "caption": "BTQL query 6: tool calls vs outcome -- does more searching correlate with a worse result.",
            "definition": {"btql": documented_btql_query(QUERY_NUMBERS["trajectory_inefficiency"])},
        },
        {
            "name": "Review set (12)",
            "view_type": "for_review_experiments",
            "caption": "The 12 blinded traces flagged for human calibration (dealpoint.eval.braintrust_sync.review_set), matched by case_id.",
            "definition": {"btql": review_btql},
        },
    ]


def dashboard_charts() -> list[dict]:
    """The five `DealPoint eval overview` dashboard charts, in spec order,
    as data (never a string template) so a fake-client test can assert on
    the structure directly.
    """
    return [
        {
            "title": "obj/grounded_accuracy by arm, grouped by model",
            "measure": "avg(scores.\"obj/grounded_accuracy\")",
            "group_by": ["metadata.arm", "metadata.model"],
        },
        {
            "title": "judge/<dimension> mean by variant",
            "measure": [f'avg(scores."judge/{dim}")' for dim in JUDGE_DIMENSIONS],
            "group_by": ["metadata.variant_id"],
        },
        {
            "title": "judge vs human per dimension (review set)",
            "measure": [f'avg(scores."judge/{dim}")' for dim in JUDGE_DIMENSIONS]
            + [f'avg(scores."human/{dim}")' for dim in JUDGE_DIMENSIONS],
            "group_by": ["metadata.variant_id"],
            "caption_if_empty": "pending human calibration",
        },
        {
            "title": "$/case by model",
            "measure": "avg(metadata.secondary_diagnostics.\"obj/usd\")",
            "group_by": ["metadata.model"],
        },
        {
            "title": "DeepEval vs obj/ agreement rate",
            "measure": "avg(if(sign(scores.\"deepeval/task_completion\" - 0.5) = sign(scores.\"obj/grounded_accuracy\" - 0.5), 1, 0))",
            "group_by": [],
        },
    ]


def dashboard_definition() -> dict:
    return {"name": "DealPoint eval overview", "view_type": "monitor", "charts": dashboard_charts()}


# --- Topics + one Pattern (spec section 7) ----------------------------------
#
# Confirmed live, 2026-09-06 (direct REST probes against api.braintrust.dev,
# this repo's real key): `/v1/facet` and `/v1/pattern` do NOT exist -- both
# return the Next.js app-router's catch-all 404 page (not an API 400/404),
# proving those paths aren't API routes at all. A Topics facet is really a
# saved `Function` (`function_data.type == "facet"`, confirmed via
# `braintrust/_generated_types.py`'s `FacetData`/`FunctionData` and a live
# `POST /v1/function` that created and then idempotently updated one by
# `(project_id, slug)`). Patterns have no public REST/API surface in this
# SDK version at all -- every plausible path (`/v1/pattern(s)`,
# `/v1/project-pattern`) 404s the same way. This is a genuine platform gap,
# not a code defect: per spec section 4, "If the REST surface ... genuinely
# does not exist for this plan, do not fake it ... That is a legitimate
# outcome." `sync_topics_and_pattern` therefore creates Topics for real via
# `/v1/function`, and records (never invents) that Patterns have no REST
# creation path -- see PATTERN_REST_LIMITATION.

PATTERN_REST_LIMITATION = (
    "No public REST endpoint exists to create a Braintrust Pattern in this "
    "API surface (probed live: GET/POST /v1/pattern, /v1/patterns and "
    "/v1/project-pattern all return the web app's catch-all 404 page, not an "
    "API error -- there is no such route). The only creation path observed is "
    "the product's internal Loop/pattern-analysis feature, exposed to an "
    "MCP-enabled agent session as `new_pattern` but not as public REST. This "
    "module cannot create a Pattern from pure Python; `pattern_definition()` "
    "still defines the payload the milestone specifies (name, description, "
    ">= 3 supporting trace ids), and the id recorded in the manifest, when "
    "present, was created once via that MCP tool and is looked up from "
    "data/reports/pattern_record.json (committed, analogous to "
    "braintrust_runs.json) so a re-sync never tries to recreate it."
)


def topics_preprocessor_code() -> str:
    """Renders one `case > agent` span as text (question, tool sequence,
    final answer, obj/grounded_accuracy) for Topics clustering. A plain
    string of JS, matching `test_preprocessor_on_trace`'s inline-code shape
    -- never a new dependency.
    """
    return (
        "function handler(span) {\n"
        "  const meta = span.metadata || {};\n"
        "  const trajectory = (span.metadata && span.metadata.trajectory) || [];\n"
        "  const tools = trajectory.map((s) => s.tool).filter(Boolean).join(' -> ');\n"
        "  const finding = meta.finding || {};\n"
        "  const question = span.input || meta.question || '';\n"
        "  const answer = finding.answer || meta.answer || '';\n"
        "  const grounded = meta['obj/grounded_accuracy'];\n"
        "  return [\n"
        "    `question: ${question}`,\n"
        "    `tool sequence: ${tools || '(none)'}`,\n"
        "    `final answer: ${answer}`,\n"
        "    `obj/grounded_accuracy: ${grounded}`,\n"
        "  ].join('\\n');\n"
        "}\n"
    )


TOPICS_FACET_PROMPT = (
    "In one sentence, describe what went wrong, or what made it succeed, in "
    "this agent trace. Ground your sentence in the tool sequence and the "
    "final answer's relationship to obj/grounded_accuracy -- never invent a "
    "cause the trace doesn't show."
)


def topics_config() -> dict:
    return {
        "name": "DealPoint agent traces",
        "preprocessor_code": topics_preprocessor_code(),
        "facet_name": "trace-outcome-summary",
        "facet_prompt": TOPICS_FACET_PROMPT,
        "clustering_enabled": True,
    }


def _inefficient_trajectory_trace_ids(limit: int = 5) -> list[str]:
    """>= 3 case ids satisfying the same `inefficient_trajectory` predicate
    `braintrust_sync._category_for_row` uses for the representative-case
    rule (spec section 7: "the recurring behaviour from query 6 ... with
    >= 3 supporting trace ids"), sorted for a stable pick.
    """
    from dealpoint.config import RESULTS_DIR
    from dealpoint.eval.braintrust_sync import _category_for_row
    from dealpoint.eval.cases import find_case, resolve_question

    found: set[str] = set()
    if RESULTS_DIR.exists():
        for path in sorted(RESULTS_DIR.glob("*.jsonl")):
            if path.name == "spend_ledger.jsonl":
                continue
            for row in _load_jsonl(path):
                case_id = row.get("case_id")
                if not case_id or case_id in found:
                    continue
                try:
                    case = find_case(case_id)
                except KeyError:
                    continue
                reasoning_type = resolve_question(case).reasoning_type
                if _category_for_row(row, reasoning_type) == "inefficient_trajectory":
                    found.add(case_id)
    return sorted(found)[:limit]


def pattern_definition() -> dict:
    """One Pattern: the recurring trajectory-inefficiency behaviour from
    BTQL query 6, with its supporting trace ids and a one-paragraph
    description.
    """
    trace_ids = _inefficient_trajectory_trace_ids()
    return {
        "name": "Trajectory inefficiency: cap-hit or >=6 tool calls without a correct answer",
        "description": (
            "A recurring failure shape across the agent-arm results: the agent "
            "hits its tool-call cap (`cap_hit`) or makes six or more tool calls "
            "without landing a grounded-correct answer (`obj/grounded_accuracy` "
            "is not True). BTQL query 6 (docs/braintrust-queries.md) surfaces the "
            "same tool_calls-vs-grounded_accuracy relationship at the trace "
            "level; this Pattern names the specific traces where more searching "
            "did not translate into a better result, so a reviewer investigating "
            "cost/latency outliers has a concrete starting set rather than the "
            "raw BTQL table."
        ),
        "supporting_trace_ids": trace_ids,
        "source_query": "trajectory_inefficiency (BTQL query 6)",
    }


# --- human/<dimension> score push (spec section 3, branch B) ---------------


class HumanScoreBudgetError(RuntimeError):
    pass


def human_score_rows() -> list[dict]:
    """One row per human-scored packet that maps to a known judged variant:
    `{packet_id, case_id, variant_id, experiment_name, scores}` where
    `scores` is `{human/<dimension>: value in [0,1]}` (rescaled from the
    1-5 rubric scale the same way `_normalize_score_value` rescales
    `judge/`). Only entries present in `human_scores.jsonl` are returned --
    absent entries never appear (never a logged zero).
    """
    from dealpoint.config import CALIBRATION_DIR

    subset = _judged_subset()
    variant_by_case_packet: dict[str, tuple[str, str]] = {}
    for variant in subset.get("variants", []):
        variant_id = variant["variant_id"]
        for case_id in subset.get("case_ids", []):
            variant_by_case_packet[_packet_id_for(case_id, variant_id)] = (case_id, variant_id)

    rows = []
    for human_row in _load_jsonl(CALIBRATION_DIR / "human_scores.jsonl"):
        packet_id = human_row.get("packet_id") or ""
        located = variant_by_case_packet.get(packet_id)
        if located is None:
            continue
        case_id, variant_id = located
        scores = {}
        for dim in JUDGE_DIMENSIONS:
            value = human_row.get(dim)
            if value is None:
                continue
            scores[f"human/{dim}"] = (value - 1) / 4  # same 1-5 -> [0,1] rescale as judge/
        if not scores:
            continue
        rows.append(
            {
                "packet_id": packet_id,
                "case_id": case_id,
                "variant_id": variant_id,
                "experiment_name": f"judge-{variant_id}",
                "scores": scores,
            }
        )
    return rows


def assert_human_score_budget(rows: list[dict]) -> int:
    """M7b adds at most `human/<dimension>` (4) per human-scored trace on the
    review set (12 to 54 traces) and nothing else (spec "Budget and disk").
    Returns the total number of scores this run is about to write, so the
    caller can record it.
    """
    total = sum(len(r["scores"]) for r in rows)
    n_traces = len(rows)
    if total > M7B_MAX_NEW_SCORES:
        raise HumanScoreBudgetError(
            f"human-score push would write {total} scores, exceeding the "
            f"{M7B_MAX_NEW_SCORES}-score-per-run guard"
        )
    if n_traces and total > 4 * n_traces:
        raise HumanScoreBudgetError(
            f"human-score push logs {total} scores across {n_traces} traces -- "
            "more than 4 per trace (reasoning, evidence, trajectory, professional)"
        )
    return total


# --- REST seam (idempotent by name, spec sections 4-5) ----------------------


class RestClient:
    """Real HTTP wrapper for the subset of Braintrust's public REST API this
    module uses (`GET/POST/PATCH /v1/view`, `GET/POST /v1/function`, `GET
    /v1/project`, `/v1/experiment`, `/v1/dataset`, `/v1/organization`).
    Never imported/instantiated by the offline test suite; a fake test
    double with the same `get`/`post`/`patch` shape drives every gate test.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.braintrust.dev") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def get(self, path: str, params: dict | None = None) -> dict:
        import requests

        response = requests.get(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            params=params or {},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def post(self, path: str, json_body: dict) -> dict:
        import requests

        response = requests.post(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=json_body,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def patch(self, path: str, json_body: dict) -> dict:
        import requests

        response = requests.patch(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=json_body,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()


def _resolve_project_id(rest_client) -> str:
    payload = rest_client.get("/v1/project", {"project_name": PROJECT})
    objects = payload.get("objects", payload) if isinstance(payload, dict) else payload
    if not objects:
        raise RuntimeError(f"project {PROJECT!r} not found via GET /v1/project")
    return objects[0]["id"]


def _resolve_experiment_id(rest_client, project_id: str, name: str) -> str | None:
    """`GET /v1/experiment?project_id=...&experiment_name=...` -- best
    effort: unresolvable (unknown route on a fake/dry-run client, or the
    experiment not created yet) falls back to `None` rather than raising,
    so a view that wants to bind to it can fall back to the project.
    """
    try:
        payload = rest_client.get("/v1/experiment", {"project_id": project_id, "experiment_name": name})
    except Exception:  # noqa: BLE001 - best-effort fallback to the project on any unresolvable/fake route
        return None
    objects = payload.get("objects", payload) if isinstance(payload, dict) else payload
    if not objects:
        return None
    return objects[0]["id"]


_SCORE_AVG_RE = re.compile(r'^avg\(scores\."([^"]+)"\)$')

# Namespace for deterministic per-chart uuids (spec section 5: "keep the
# payload as data"; a stable id per chart title keeps a re-run's custom_charts
# payload byte-identical rather than growing new chart ids every sync).
_CHART_ID_NAMESPACE = uuid.UUID("6b3f9e0a-8f3a-4b7f-9b3e-b7f3a4b7f9b3")


def _measure_rest(measure_btql: str) -> dict:
    """One `dashboard_charts()` measure string -> the real `custom_charts`
    measure shape, confirmed live 2026-09-06 via `create_monitoring_view` +
    `GET /v1/view`: a bare `avg(scores."X")` is `{type: aggregateScore,
    scoreName, aggregator}`; anything else (a full BTQL expression, e.g. the
    `$/case` or DeepEval-agreement measures) is `{type: expression, btql}`.
    """
    match = _SCORE_AVG_RE.match(measure_btql)
    if match:
        score_name = match.group(1)
        return {"type": "aggregateScore", "scoreName": score_name, "aggregator": {"type": "avg"}, "displayName": score_name}
    return {"type": "expression", "btql": measure_btql, "displayName": measure_btql}


def _chart_rest_definition(chart: dict) -> dict:
    measures = chart["measure"] if isinstance(chart["measure"], list) else [chart["measure"]]
    group_bys = [{"btql": g, "displayName": g.rsplit(".", 1)[-1]} for g in chart.get("group_by", [])]
    unit_type = "cost" if "usd" in str(chart["measure"]) else "percent"
    return {
        "type": "monitorTimeseries",
        "measures": [_measure_rest(m) for m in measures],
        "groupBys": group_bys,
        "viz": {"type": "timeseries", "timeseriesVizType": "bars", "unitType": unit_type},
    }


def _view_data_for(definition: dict) -> dict:
    """Definition (`{btql, sort}` or `{custom_charts}`) -> the real
    `View.view_data` REST shape, confirmed live 2026-09-06: table/logs/
    experiment views take `{"search": {"filter": [{"btql": ...}]}}`
    (+ optional `"sort"`); the monitor dashboard takes
    `{"custom_charts": {"charts": {<chart_id>: {title, definition}, ...},
    "layout": {"type": "linear", "order": [<chart_id>, ...]}, "version":
    "0.0.0"}}` -- confirmed live via `create_monitoring_view` + `GET
    /v1/view` (a bare `"charts": [...]` array, as an earlier version of this
    module posted, 400s with "Expected object, received array").
    """
    if "custom_charts" in definition:
        charts: dict = {}
        order: list[str] = []
        for chart in definition["custom_charts"]:
            chart_id = str(uuid.uuid5(_CHART_ID_NAMESPACE, chart["title"]))
            entry = {"title": chart["title"], "definition": _chart_rest_definition(chart)}
            if "caption_if_empty" in chart:
                entry["caption_if_empty"] = chart["caption_if_empty"]
            charts[chart_id] = entry
            order.append(chart_id)
        return {"custom_charts": {"charts": charts, "layout": {"type": "linear", "order": order}, "version": "0.0.0"}}
    search: dict = {}
    if "btql" in definition:
        search["filter"] = [{"btql": definition["btql"]}]
    if "sort" in definition:
        search["sort"] = [{"btql": definition["sort"]}]
    return {"search": search} if search else {}


def _upsert_view(rest_client, object_type: str, object_id: str, view_type: str, name: str, view_data: dict) -> dict:
    """`GET /v1/view?object_type=...&object_id=...` then `POST /v1/view`
    (create) or `PATCH /v1/view/{id}` (update) -- confirmed live 2026-09-06:
    `GET /v1/view` takes `object_type`/`object_id`, not `project_id`; `POST
    /v1/view` rejects an `id` key outright ("Extraneous key"), and the
    insert-style upsert this module assumed does not exist -- the real
    update path is `PATCH /v1/view/{id}` with the same body, no `id` field.
    """
    existing_payload = rest_client.get("/v1/view", {"object_type": object_type, "object_id": object_id})
    existing = existing_payload.get("objects", existing_payload) if isinstance(existing_payload, dict) else existing_payload
    match = next((v for v in existing if v.get("name") == name and v.get("view_type") == view_type), None)

    body = {
        "object_type": object_type,
        "object_id": object_id,
        "view_type": view_type,
        "name": name,
        "view_data": view_data,
    }
    if view_type == "monitor":
        # confirmed live 2026-09-06: a monitor view needs `options.viewType ==
        # "monitor"` alongside `view_data.custom_charts`, or the dashboard
        # renders with no chart layout even though the POST succeeds.
        body["options"] = {"viewType": "monitor", "options": {"projectId": object_id, "type": object_type}}
    if match is None:
        result = rest_client.post("/v1/view", body)
        return {"id": result.get("id"), "created": True}
    result = rest_client.patch(f"/v1/view/{match['id']}", body)
    return {"id": result.get("id", match["id"]), "created": False}


def sync_views_and_dashboard(rest_client, hero_experiment_id: str | None = None) -> dict:
    project_id = _resolve_project_id(rest_client)
    views_result = []
    for v in view_definitions():
        object_type, object_id = "project", project_id
        if v.get("bind_to") == "hero_experiment" and hero_experiment_id:
            object_type, object_id = "experiment", hero_experiment_id
        upserted = _upsert_view(rest_client, object_type, object_id, v["view_type"], v["name"], _view_data_for(v["definition"]))
        views_result.append(
            {"name": v["name"], "view_type": v["view_type"], "caption": v["caption"], "object_type": object_type, "object_id": object_id, **upserted}
        )

    dash = dashboard_definition()
    dash_upserted = _upsert_view(rest_client, "project", project_id, dash["view_type"], dash["name"], _view_data_for({"custom_charts": dash["charts"]}))
    dashboard_result = {"name": dash["name"], **dash_upserted}

    return {"project_id": project_id, "views": views_result, "dashboard": dashboard_result}


def sync_topics_and_pattern(rest_client, project_id: str) -> dict:
    """Topics: a real `POST /v1/function` (`function_data.type == "facet"`),
    idempotent by `(project_id, slug)` (confirmed live). Pattern: no REST
    creation path exists (`PATTERN_REST_LIMITATION`) -- `pattern.id` comes
    from the local, committed `data/reports/pattern_record.json` when
    present (written once via the `new_pattern` MCP tool, the only creation
    path this platform offers; never invented here).
    """
    cfg = topics_config()
    function_body = {
        "project_id": project_id,
        "name": cfg["facet_name"],
        "slug": cfg["facet_name"],
        "function_data": {"type": "facet", "prompt": cfg["facet_prompt"]},
    }
    try:
        facet_result = rest_client.post("/v1/function", function_body)
        topics_result: dict = {"id": facet_result.get("id"), "config": cfg}
    except Exception as exc:  # noqa: BLE001 - a live failure must be visible in the manifest, never silently faked
        topics_result = {"id": None, "config": cfg, "error": str(exc)}
    # Cluster ASSIGNMENT is asynchronous, product-UI-driven work on Braintrust's
    # side (a "topic map" job over already-synced traces) with no read-back
    # path in this SDK/REST surface within a single script run -- never
    # invented here. `clusters` stays empty with the reason recorded rather
    # than faked; a later `just braintrust-cockpit` run can populate it once
    # a topic map exists to read (spec section 7's honest-gap precedent).
    topics_result["clusters"] = []
    topics_result["clusters_limitation"] = (
        "Topic clustering runs asynchronously in Braintrust's product UI over "
        "already-synced traces; this SDK/REST surface exposes no read-back of "
        "materialized cluster names within a single script invocation. The "
        "facet function above is created for real (confirmed live); cluster "
        "names are read back and appended here once a topic map exists, not "
        "invented in the meantime."
    )

    pattern_record_path = Path("data/reports/pattern_record.json")
    pattern_id = None
    if pattern_record_path.exists():
        pattern_id = json.loads(pattern_record_path.read_text(encoding="utf-8")).get("pattern_id")
    pattern_result = {"id": pattern_id, "definition": pattern_definition(), "limitation": PATTERN_REST_LIMITATION}

    return {"topics": topics_result, "pattern": pattern_result}


# --- hero-case replay (spec section 1-2) ------------------------------------


JUDGE_SCORE_ALLOWED_NAMES: set[str] = {f"judge/{dim}" for dim in JUDGE_DIMENSIONS}


def _judge_spread_by_dim(judge_rows: list[dict], packet_id: str) -> dict[str, float]:
    rows = _judge_rows_for_packet(judge_rows, packet_id)
    spreads: dict[str, float] = {}
    for dim in JUDGE_DIMENSIONS:
        vals = [r[dim] for r in rows if r.get(dim) is not None]
        if len(vals) >= 2:
            spreads[dim] = max(vals) - min(vals)
    return spreads


def _hero_hierarchy(case_id: str, variant_id: str) -> tuple[dict, dict]:
    from dealpoint.corpus.document import load_document
    from dealpoint.eval.braintrust_sync import _mean_judge_dims_for_packet, log_hierarchy
    from dealpoint.eval.cases import find_case, resolve_document_id

    subset = _judged_subset()
    variant = next(v for v in subset["variants"] if v["variant_id"] == variant_id)
    rows_by_case = {r["case_id"]: r for r in _load_jsonl(variant["results_path"])}
    row = rows_by_case.get(case_id) or {"case_id": case_id, "scores": {}, "record": {"trajectory": []}}
    try:
        case = find_case(case_id)
        doc = load_document(resolve_document_id(case))
    except (KeyError, FileNotFoundError):
        case, doc = {}, None

    packet_id = _packet_id_for(case_id, variant_id)
    all_judge_rows = _load_jsonl(JUDGE_SCORES_PATH)
    judge_dims = _mean_judge_dims_for_packet(all_judge_rows, packet_id)
    spread_by_dim = _judge_spread_by_dim(all_judge_rows, packet_id)

    hierarchy = log_hierarchy(row, case, doc, judge_dims=judge_dims)
    tree = judge_spans(case_id, variant_id)
    agent_node = hierarchy["children"][0]
    for child in agent_node["children"]:
        if child.get("name") == "scoring":
            child["children"] = tree["spans"]
            # spec section 4: the 'Judge disagreement' view filters on this
            # metadata key -- write it on every replayed scoring span so the
            # view has something to match, never a filter over an unwritten field.
            child.setdefault("metadata", {})
            child["metadata"]["judge_spread_by_dim"] = spread_by_dim
            child["metadata"]["judge_spread_max"] = max(spread_by_dim.values()) if spread_by_dim else 0.0
    return hierarchy, tree


def replay_hero_case(sdk_client, case_id: str) -> dict:
    """`m7b-hero-case`: `case -> agent -> ... -> scoring -> {judge/<family>
    x3, judge/aggregate}` for each of `HERO_VARIANTS`, entirely replayed from
    stored results/judge scores. No model call.
    """
    from dealpoint.eval.braintrust_sync import _emit_span_tree, _init_experiment

    experiment = _init_experiment(sdk_client, "m7b-hero-case")
    variant_trees: dict[str, dict] = {}
    seen = _ledger_load()
    for variant_id in HERO_VARIANTS:
        hierarchy, tree = _hero_hierarchy(case_id, variant_id)
        variant_trees[variant_id] = tree
        if ("m7b-hero-case", f"{case_id}:{variant_id}") in seen:
            continue                      # already logged; never re-write scores
        root_span = experiment.start_span(name=f"case:{variant_id}")
        if type(root_span).__name__ == "_NoopSpan":
            raise RuntimeError("hero-case replay span landed on _NoopSpan")
        if hasattr(root_span, "log"):
            root_span.log(metadata={"case_id": case_id, "variant_id": variant_id})
        n_scores_emitted = 0
        for child in hierarchy.get("children", []):
            n_scores_emitted += _emit_span_tree(
                experiment, child, parent_span=root_span, allowed_score_names=JUDGE_SCORE_ALLOWED_NAMES
            )
        if hasattr(root_span, "end"):
            root_span.end()
        # Record what was ACTUALLY emitted, not the constant JUDGE_AGGREGATE_DIMENSIONS --
        # a ledger that misreports counts cannot support the "never re-log" audit (spec
        # spend guard). Falls back to 0 for fake test doubles whose spans don't track scores.
        _ledger_record("m7b-hero-case", f"{case_id}:{variant_id}", n_scores_emitted)
    if hasattr(experiment, "flush"):
        experiment.flush()
    return {"experiment_name": "m7b-hero-case", "variant_trees": variant_trees}



# --- live spend guard (operator instruction, 2026-09-06 evening) ----------------
# The org's Braintrust plan is at its monthly score cap with pay-as-you-go
# overage. Every live write is therefore (1) opt-in via `--live`, (2) bounded by
# LIVE_SCORE_CAP, computed BEFORE the first write, and (3) recorded in a local
# ledger so a re-run never re-logs a score that already exists.

LIVE_SCORE_CAP = 600
SCORE_LEDGER_PATH = Path("data/reports/braintrust_score_ledger.jsonl")
JUDGE_AGGREGATE_DIMENSIONS = 4          # judge/<dimension> scores per replayed tree


class ScoreBudgetError(RuntimeError):
    """Raised before any live write when the planned score count exceeds LIVE_SCORE_CAP."""


def planned_replay_trees() -> int:
    """How many trees `replay_hero_case` will emit; extend this when the replay grows."""
    return len(HERO_VARIANTS)


def planned_live_scores(n_human_scores: int) -> int:
    return n_human_scores + JUDGE_AGGREGATE_DIMENSIONS * planned_replay_trees()


def assert_live_score_budget(planned: int, cap: int = LIVE_SCORE_CAP) -> int:
    if planned > cap:
        raise ScoreBudgetError(
            f"live run would write {planned} scores, above the cap of {cap}; "
            "nothing was written. Reduce the plan or raise LIVE_SCORE_CAP deliberately."
        )
    return planned


# The ledger is a property of LIVE runs only: `main(["--live"])` arms it. Fake
# and dry-run clients never read or write it, so offline tests stay hermetic.
_LEDGER_ACTIVE = False


def _ledger_load() -> set[tuple[str, str]]:
    if not _LEDGER_ACTIVE or not SCORE_LEDGER_PATH.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    with open(SCORE_LEDGER_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                seen.add((row["experiment"], row["key"]))
    return seen


def _ledger_record(experiment: str, key: str, n_scores: int) -> None:
    if not _LEDGER_ACTIVE:
        return
    SCORE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORE_LEDGER_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"experiment": experiment, "key": key, "n_scores": n_scores,
                             "ts": datetime.now(UTC).isoformat()}) + "\n")


def push_human_scores(sdk_client, rows: list[dict]) -> int:
    """Push `human/<dimension>` scores onto the matching `judge-<variant_id>`
    experiment rows, keyed by the SAME stable row id `braintrust_sync.sync`
    used for that row (spec: "human/<dimension> scores are logged on the
    same row"), so this updates the existing row rather than duplicating it.
    """
    from dealpoint.eval.braintrust_sync import _init_experiment, _stable_dataset_row_id

    pushed = 0
    seen = _ledger_load()
    by_experiment: dict[str, list[dict]] = {}
    for row in rows:
        by_experiment.setdefault(row["experiment_name"], []).append(row)
    for experiment_name, exp_rows in by_experiment.items():
        experiment = _init_experiment(sdk_client, experiment_name)
        for row in exp_rows:
            row_id = _stable_dataset_row_id(experiment_name, row["case_id"])
            if (experiment_name, row_id) in seen:
                continue                  # already logged; never re-write scores
            experiment.log(
                input=row["case_id"],
                output=row["variant_id"],
                scores=row["scores"],
                metadata={"packet_id": row["packet_id"]},
                id=row_id,
            )
            pushed += len(row["scores"])
            _ledger_record(experiment_name, row_id, len(row["scores"]))
        if hasattr(experiment, "flush"):
            experiment.flush()
    return pushed


# --- driver + manifest -------------------------------------------------------


def sync_cockpit(rest_client, sdk_client=None) -> dict:
    """Create/update every persistent cockpit object, idempotently, and
    return the raw result dict `build_demo_manifest` turns into
    `data/reports/demo_manifest.json`. `sdk_client=None` skips the hero-case
    replay and human-score push (used by the REST-only offline gate tests).
    """
    hero = hero_case()
    human_rows = human_score_rows()
    n_human_scores = assert_human_score_budget(human_rows)

    planned_scores = planned_live_scores(n_human_scores)
    replay_result = None
    pushed_scores = 0
    hero_experiment_id = None
    if sdk_client is not None and hero.get("case_id"):
        assert_live_score_budget(planned_scores)      # before the first write
        replay_result = replay_hero_case(sdk_client, hero["case_id"])
        pushed_scores = push_human_scores(sdk_client, human_rows)
        # 'Judge disagreement' (spec section 4) binds to the hero experiment, not
        # the project -- resolvable now that the replay has created it.
        hero_experiment_id = _resolve_experiment_id(rest_client, _resolve_project_id(rest_client), "m7b-hero-case")

    views_dashboard = sync_views_and_dashboard(rest_client, hero_experiment_id=hero_experiment_id)
    topics_pattern = sync_topics_and_pattern(rest_client, views_dashboard["project_id"])

    return {
        "views_dashboard": views_dashboard,
        "topics_pattern": topics_pattern,
        "hero_case": hero,
        "human_score_rows": human_rows,
        "n_human_scores_planned": n_human_scores,
        "n_human_scores_pushed": pushed_scores,
        "n_live_scores_planned": planned_scores,
        "live_score_cap": LIVE_SCORE_CAP,
        "replay": replay_result,
        "human_scoring_probe": HUMAN_SCORING_PROBE,
        "synced_at": datetime.now(UTC).isoformat(),
    }


EXPERIMENT_NAME_PREFIXES: tuple[str, ...] = ("A-", "B-", "C-", "D-", "judge-", "judged-", "m7-", "m7b-")

# The frozen M7a draft (docs/templates/demo-walkthrough.draft.md) cites these
# 14 experiment names literally (RAG lab families section 2, one representative
# agent run section 3, the five Pareto-sweep runs section 10) -- fixed,
# unsuffixed names from a specific milestone run, not the "newest wins"
# per-prefix families EXPERIMENT_NAME_PREFIXES resolves. Confirmed live: each
# name below also exists suffixed (e.g. "rag-m3-dense-7f166caa", an M7a re-sync
# artifact); only the EXACT unsuffixed name is the one the doc quotes.
EXACT_EXPERIMENT_NAMES: tuple[str, ...] = (
    "rag-m3-dense",
    "rag-m3-bm25",
    "rag-m3-hybrid",
    "rag-m3-hybrid-rerank",
    "rag-m3-fusion",
    "rag-m3-fusion-rerank",
    "rag-m7-li-crosscheck",
    "rag-m7-synthetic",
    "A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc",
    "pareto-deepseek_deepseek-v4-flash",
    "pareto-qwen_qwen3.7-flash",
    "pareto-google_gemini-3.1-flash-lite",
    "pareto-anthropic_claude-haiku-4.5",
    "pareto-z-ai_glm-5.3-flash",
)
DATASET_NAMES: tuple[str, ...] = (
    "maud-dealpoint-dev-v1",
    "maud-dealpoint-test-v1",
    "maud-dealpoint-counterfactual-v1",
    "maud-dealpoint-review-set",
    "maud-dealpoint-judged_calibration",
)


def _resolve_experiments(rest_client, project_id: str) -> list[dict]:
    """Every experiment name/id the walkthrough cites, newest-by-prefix
    (spec section 6: "resolved by name prefix, newest wins, so a re-sync
    never strands the doc"). Falls back to an empty list on a fake/dry-run
    client that doesn't implement `/v1/experiment` (never raises).
    """
    try:
        payload = rest_client.get("/v1/experiment", {"project_id": project_id, "limit": 200})
    except Exception:  # noqa: BLE001 - offline/fake clients don't implement this route; empty list, never raise
        return []
    objects = payload.get("objects", payload) if isinstance(payload, dict) else payload
    objects = objects or []
    newest_by_prefix: dict[str, dict] = {}
    for exp in objects:
        name = exp.get("name") or ""
        for prefix in EXPERIMENT_NAME_PREFIXES:
            if not name.startswith(prefix):
                continue
            current = newest_by_prefix.get(prefix)
            if current is None or (exp.get("created") or "") > (current.get("created") or ""):
                newest_by_prefix[prefix] = exp
    resolved = [
        {"prefix": prefix, "name": exp["name"], "id": exp["id"], "created": exp.get("created")}
        for prefix, exp in sorted(newest_by_prefix.items())
    ]
    by_exact_name = {exp["name"]: exp for exp in objects if exp.get("name") in EXACT_EXPERIMENT_NAMES}
    resolved.extend(
        {"prefix": None, "name": name, "id": by_exact_name[name]["id"], "created": by_exact_name[name].get("created")}
        for name in EXACT_EXPERIMENT_NAMES
        if name in by_exact_name
    )
    return resolved


def _resolve_datasets(rest_client, project_id: str) -> list[dict]:
    try:
        payload = rest_client.get("/v1/dataset", {"project_id": project_id, "limit": 200})
    except Exception:  # noqa: BLE001 - offline/fake clients don't implement this route; empty list, never raise
        return []
    objects = payload.get("objects", payload) if isinstance(payload, dict) else payload
    by_name = {ds["name"]: ds for ds in (objects or []) if ds.get("name") in DATASET_NAMES}
    return [{"name": name, "id": by_name[name]["id"]} for name in DATASET_NAMES if name in by_name]


def _org_name(rest_client) -> str | None:
    try:
        payload = rest_client.get("/v1/organization", {})
    except Exception:  # noqa: BLE001 - offline/fake clients don't implement this route; None, never raise
        return None
    objects = payload.get("objects", payload) if isinstance(payload, dict) else payload
    return objects[0]["name"] if objects else None


# Table-view URLs pin the saved view's id via a `?v=` query param on the
# page that renders that view_type -- the same "pin a view by id" mechanism
# Braintrust documents for traces/datasets (`&tv=`/`&dv=`), generalized to
# every view_type this project creates. The monitor dashboard's confirmed-live
# shape (`/dashboards/<id>`, matched against a real `monitor_url`) is handled
# separately below since a dashboard is not a `?v=`-pinned table view.
_VIEW_TYPE_PAGE: dict[str, str] = {
    "logs": "logs",
    "experiments": "experiments",
    "for_review_experiments": "experiments",
    "experiment": "experiments",
}


def permalinks(
    rest_client,
    project_id: str,
    experiments: list[dict],
    datasets: list[dict],
    views: list[dict] | None = None,
    dashboard: dict | None = None,
    hero_experiment_name: str | None = None,
    hero_variants: list[str] | None = None,
) -> dict:
    """One URL per walkthrough stop (spec section 6): every resolved
    experiment/dataset, the seven saved views, the dashboard, and one entry
    per hero-case variant trace -- built from the SAME
    `https://www.braintrust.dev/app/<org>/p/<project>/...` shape
    `data/reports/braintrust_runs.json` already records and
    `mcp_braintrust_generate_permalink` confirms for experiments. Empty when
    the org can't be resolved (fake/dry-run client) rather than a guess.
    """
    org = _org_name(rest_client)
    if not org:
        return {}
    import urllib.parse

    base = f"https://www.braintrust.dev/app/{urllib.parse.quote(org)}/p/{PROJECT}"
    links: dict[str, str] = {f"experiment:{e['name']}": f"{base}/experiments/{e['name']}" for e in experiments}
    links.update({f"dataset:{d['name']}": f"{base}/datasets/{d['name']}" for d in datasets})
    for v in views or []:
        if v.get("object_type") == "experiment" and hero_experiment_name:
            page = f"experiments/{hero_experiment_name}"
        else:
            page = _VIEW_TYPE_PAGE.get(v.get("view_type") or "", "logs")
        links[f"view:{v['name']}"] = f"{base}/{page}?v={v['id']}"
    if dashboard and dashboard.get("id"):
        links[f"dashboard:{dashboard['name']}"] = f"{base}/dashboards/{dashboard['id']}"
    if hero_experiment_name:
        for variant_id in hero_variants or []:
            links[f"hero_trace:{variant_id}"] = f"{base}/experiments/{hero_experiment_name}"
    return links


def build_demo_manifest(result: dict, rest_client=None) -> dict:
    """`data/reports/demo_manifest.json`'s full shape (spec section 6).
    `rest_client`, when given, resolves the live experiments/datasets/
    permalinks the manifest needs; omitted (or a client without those
    routes) yields empty lists rather than a crash -- offline callers still
    get every OTHER key.
    """
    from dealpoint.eval.cases import git_sha7
    from dealpoint.eval.rubric import rubric_version

    views_dashboard = result["views_dashboard"]
    project_id = views_dashboard["project_id"]
    experiments = _resolve_experiments(rest_client, project_id) if rest_client is not None else []
    datasets = _resolve_datasets(rest_client, project_id) if rest_client is not None else []
    review_case_ids = _review_set_case_ids()
    replay_raw = result.get("replay") or {}
    hero_experiment_name = replay_raw.get("experiment_name")
    hero_variants = sorted((replay_raw.get("variant_trees") or {}).keys())
    return {
        "project_id": project_id,
        "views": views_dashboard["views"],
        "dashboard": views_dashboard["dashboard"],
        "topics": result["topics_pattern"]["topics"],
        "pattern": result["topics_pattern"]["pattern"],
        "hero_case": result["hero_case"],
        "human_scoring_probe": result["human_scoring_probe"],
        "human_score_rows": [
            {k: v for k, v in r.items() if k != "scores"} | {"dimensions": sorted(r["scores"])}
            for r in result["human_score_rows"]
        ],
        "n_human_scores_planned": result["n_human_scores_planned"],
        "n_human_scores_pushed": result["n_human_scores_pushed"],
        "n_live_scores_planned": result["n_live_scores_planned"],
        "live_score_cap": result["live_score_cap"],
        "replay": {
            "experiment_name": (result.get("replay") or {}).get("experiment_name"),
            "variants": sorted((result.get("replay") or {}).get("variant_trees", {}).keys()),
        },
        "experiments": experiments,
        "datasets": datasets,
        "review_set": review_case_ids,
        "permalinks": permalinks(
            rest_client,
            project_id,
            experiments,
            datasets,
            views=views_dashboard["views"],
            dashboard=views_dashboard["dashboard"],
            hero_experiment_name=hero_experiment_name,
            hero_variants=hero_variants,
        )
        if rest_client is not None
        else {},
        "git_sha7": git_sha7(),
        "rubric_version": rubric_version(),
        "subset_hash": JUDGED_SUBSET_HASH,
        "synced_at": result["synced_at"],
    }


class _DryRunRestClient:
    """A same-shape REST double that never touches the network -- mirrors
    `braintrust_sync._DryRunClient`'s purpose. `get('/v1/project', ...)`
    returns a stable fake project id; `get('/v1/view', ...)` returns
    whatever this instance has already "created" via `post`, so a second
    `sync_cockpit` call against the SAME instance is genuinely idempotent.
    """

    def __init__(self) -> None:
        self._project_id = "dry-run-project-id"
        self._views: list[dict] = []
        self._functions: list[dict] = []
        self._next_id = 1

    def get(self, path: str, params: dict | None = None) -> dict:
        if path == "/v1/project":
            return {"objects": [{"id": self._project_id, "name": (params or {}).get("project_name", PROJECT)}]}
        if path == "/v1/view":
            return {"objects": list(self._views)}
        if path == "/v1/function":
            return {"objects": list(self._functions)}
        # /v1/experiment, /v1/dataset, /v1/organization -- deliberately
        # unhandled: a dry run never resolves real experiments/datasets/org,
        # so `_resolve_experiment_id`/`_resolve_experiments`/`permalinks` fall
        # back to their offline defaults (None / [] / {}) via the caught
        # exception, exactly like a fake test double without those routes.
        raise ValueError(f"unhandled dry-run GET {path}")

    def post(self, path: str, json_body: dict) -> dict:
        if path == "/v1/view":
            new_view = dict(json_body)
            new_view["id"] = f"view-{self._next_id}"
            self._next_id += 1
            self._views.append(new_view)
            return new_view
        if path == "/v1/function":
            for fn in self._functions:
                if fn.get("slug") == json_body.get("slug"):
                    fn.update(json_body)
                    return fn
            new_fn = dict(json_body)
            new_fn["id"] = f"function-{self._next_id}"
            self._next_id += 1
            self._functions.append(new_fn)
            return new_fn
        raise ValueError(f"unhandled dry-run POST {path}")

    def patch(self, path: str, json_body: dict) -> dict:
        view_id = path.rsplit("/", 1)[-1]
        for v in self._views:
            if v["id"] == view_id:
                v.update(json_body)
                return v
        raise ValueError(f"unhandled dry-run PATCH {path}")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    # Dry run is the default. A live write needs an explicit `--live`; `--dry-run`
    # is accepted for symmetry with braintrust_sync and always wins.
    live = "--live" in argv and "--dry-run" not in argv

    rest_client_for_manifest = None
    if not live:
        rest_client_for_manifest = _DryRunRestClient()
        result = sync_cockpit(rest_client_for_manifest, sdk_client=None)
        print(f"dry run (pass --live to write); planned live scores: {result['n_live_scores_planned']} "
              f"(cap {LIVE_SCORE_CAP})")
    else:
        import os

        # T1 fix: resolve the key via the SAME resolver `braintrust_available()`
        # uses (env -> .env.braintrust -> .braintrust.json), not raw
        # `os.environ` -- the key lives in `.env.braintrust` (`just` only
        # dotenv-loads `.env`), so the old `os.environ.get(...)` read "" and
        # built `RestClient("")`: every REST call would 401 mid-run. Also
        # exports the resolved key to the process environment so the
        # `braintrust` SDK (which reads `BRAINTRUST_API_KEY` from `os.environ`
        # internally) authenticates as the SAME identity as the REST seam.
        from dealpoint.eval.braintrust_adapter import load_braintrust_key

        api_key = load_braintrust_key()
        if not api_key:
            print(
                "braintrust --live requested but no API key resolved (checked env, "
                ".env.braintrust, .braintrust.json) -- aborting, nothing written",
                file=sys.stderr,
            )
            return 1
        try:
            import braintrust
        except ImportError:
            print(
                "braintrust --live requested but the braintrust package is not installed "
                "-- aborting, nothing written",
                file=sys.stderr,
            )
            return 1

        os.environ["BRAINTRUST_API_KEY"] = api_key
        print("live run: key resolved, writing to Braintrust")
        global _LEDGER_ACTIVE
        _LEDGER_ACTIVE = True
        rest_client_for_manifest = RestClient(api_key)
        result = sync_cockpit(rest_client_for_manifest, sdk_client=braintrust)

    manifest = build_demo_manifest(result, rest_client=rest_client_for_manifest)
    DEMO_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEMO_MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        fh.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
