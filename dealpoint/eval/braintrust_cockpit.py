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
            "name": "RAG tournament",
            "view_type": "experiments",
            "caption": "The six frozen M3 retriever configs plus the M7 LlamaIndex cross-check and synthetic-query study: li/hit_rate, li/mrr, obj/hit@5, obj/hit@10 side by side.",
            "definition": {"btql": "name like 'rag-%'"},
        },
        {
            "name": "Review set (12)",
            "view_type": "for_review_experiments",
            "caption": "The 12 blinded traces flagged for human calibration (dealpoint.eval.braintrust_sync.review_set), matched by case_id.",
            "definition": {"btql": review_btql},
        },
    ]


# Every ranking chart compares like with like: canonical traces only (`comparable = 1`: the judged and
# sweep traces of the five slate models; no representative picks, live replays or partial runs) and one
# case pool at a time. The five arm-D models share the 18 judged cases; the four GLM systems share the
# 32-case subset. `correct_all` credits a correct abstention on a counterfactual as a correct outcome.
GLM32 = "metadata.comparable = 1 and metadata.model_label = 'glm' and (metadata.category = 'agent' or metadata.category = 'judged')"
D18 = "metadata.comparable = 1 and metadata.arm = 'D' and metadata.category = 'judged'"
JUDGED18 = "metadata.comparable = 1 and metadata.category = 'judged'"
AGENT_LOGS = "(metadata.category = 'agent' or metadata.category = 'judged')"
DASHBOARD_RANGE = "30d"      # pinned on every dashboard: the logs were batch-ingested in one afternoon, and a
                             # "recent window" default blanks every chart as soon as that window slides past them
SYS_POOL = "the four systems on the same 32 cases, GLM 5.3 flash"
MOD_POOL = "arm D on the same 18 cases, five models"
JUDGE_DIMS_ = ("reasoning", "evidence", "trajectory", "professional")
FAMILIES = (("mistral", "Mistral Small"), ("nvidia", "NVIDIA Nemotron"), ("bytedance", "ByteDance Seed"))


def chart_catalogue() -> dict[str, dict]:
    """Every chart, keyed, with a short title (the dashboard's name carries the question). Each is a
    toplist over the metadata mirror: see `dashboard_definitions` for why nothing here is a time series."""
    return {
        "big_best_config": {"kind": "bignumber", "title": "BEST MEASURED CONFIG: A@haiku (single shot on Claude Haiku 4.5), correct outcome on the same 18 cases; 0 fabrication, 0 cap-hits, 3.7 s, $0.0034",
                            "measure": [{"btql": "avg(metadata.correct_all)", "name": "A@haiku"}], "group_by": [], "filters": [f"{JUDGED18} and metadata.variant_label = 'A@haiku'"]},
        "big_best_agent": {"kind": "bignumber", "title": "BEST AGENT: D@gemini (agent + hybrid + skill on Gemini 3.1 flash lite), correct outcome on the same 18 cases; the only agent that never fabricates",
                           "measure": [{"btql": "avg(metadata.correct_all)", "name": "D@gemini"}], "group_by": [], "filters": [f"{JUDGED18} and metadata.variant_label = 'D@gemini-3.1-flash-lite'"]},
        "big_best_retriever": {"kind": "bignumber", "title": "BEST RETRIEVER: hybrid_rrf (dense + BM25, reciprocal-rank fusion, no reranker), gold-span hit@5 on the 58 dev queries",
                               "measure": [{"btql": "avg(metadata.hit_at_5)", "name": "hybrid_rrf"}], "group_by": [], "filters": ["metadata.category = 'retrieval' and metadata.retriever = 'hybrid_rrf'"]},
        "big_judges": {"kind": "bignumber", "title": "BEST JUDGE PANEL: Mistral Small for reasoning, evidence and professional quality, NVIDIA Nemotron for trajectory. Shown: Mistral's closeness to the lawyer on evidence",
                       "measure": [{"btql": "avg(metadata.judge_mistral_evidence_closeness)", "name": "Mistral on evidence"}], "group_by": [], "filters": [f"metadata.has_human = 1 and {JUDGED18}"]},
        "big_best_prompt": {"kind": "bignumber", "title": "BEST BASELINE PROMPT: cite-first, Judge: evidence over the 18 arm-A packets",
                            "measure": [{"btql": "avg(metadata.judge_evidence)", "name": "cite-first"}], "group_by": [], "filters": ["metadata.category = 'prompt-variant' and metadata.prompt_variant = 'cite-first'"]},
        "sys_correct": {"title": f"Correct outcome over every case: A pipeline+dense, B agent+dense, C agent+hybrid, D agent+hybrid+skill ({SYS_POOL}; a correct abstention counts)",
                        "measure": "avg(metadata.correct_all)", "group_by": ["metadata.system_label"], "filters": [GLM32]},
        "sys_cap": {"title": f"Cap-hit rate, the loop never stops ({SYS_POOL})", "measure": "avg(metadata.cap_hit)", "group_by": ["metadata.system_label"], "filters": [GLM32]},
        "sys_fab": {"title": f"Fabrication rate, invented clauses ({SYS_POOL})", "measure": "avg(metadata.fabrication)", "group_by": ["metadata.system_label"], "filters": [GLM32]},
        "sys_abstain": {"title": "Correct abstention when the definition is not there (counterfactual cases, all comparable traces)",
                        "measure": "avg(metadata.abstain_correct)", "group_by": ["metadata.system_label"], "filters": ["metadata.comparable = 1 and metadata.case_set = 'counterfactual'"]},
        "loop_x_model": {"title": "Does the loop pay off more on the stronger model? A vs D on GLM and Haiku, the same 18 cases, correct outcome",
                         "measure": "avg(metadata.correct_all)", "group_by": ["metadata.variant_label"],
                         "filters": [f"{JUDGED18} and (metadata.arm = 'A' or metadata.arm = 'D') and (metadata.model_label = 'glm' or metadata.model_label = 'haiku')"]},
        "sys_usd": {"title": f"Dollars per case, realized from the local ledger ({SYS_POOL})", "measure": "avg(metadata.usd)", "group_by": ["metadata.system_label"], "filters": [GLM32], "unit": "cost"},
        "sys_sec": {"title": f"Seconds per case ({SYS_POOL})", "measure": "avg(metadata.wall_s)", "group_by": ["metadata.system_label"], "filters": [GLM32], "unit": "duration"},
        "sys_calls": {"title": f"Tool calls per case ({SYS_POOL})", "measure": "avg(metadata.tool_calls)", "group_by": ["metadata.system_label"], "filters": [GLM32], "unit": "count"},
        "sys_per_dollar": {"title": f"Correct outcomes per dollar ({SYS_POOL})", "measure": "sum(metadata.correct_all) / sum(metadata.usd)", "group_by": ["metadata.system_label"], "filters": [GLM32], "unit": "count"},
        "ret_hit5": {"title": "Gold-span hit@5 on the 58 dev queries (our deterministic scorer; the tournament's ranking rule)", "measure": "avg(metadata.hit_at_5)", "group_by": ["metadata.retriever"], "filters": ["metadata.category = 'retrieval'"]},
        "ret_mrr": {"title": "Mean reciprocal rank of the gold span, 58 dev queries", "measure": "avg(metadata.mrr)", "group_by": ["metadata.retriever"], "filters": ["metadata.category = 'retrieval'"]},
        "ret_hit10": {"title": "Gold-span hit@10 on the 58 dev queries", "measure": "avg(metadata.hit_at_10)", "group_by": ["metadata.retriever"], "filters": ["metadata.category = 'retrieval'"]},
        "mod_correct": {"title": f"Correct outcome over every case ({MOD_POOL}; a correct abstention counts)", "measure": "avg(metadata.correct_all)", "group_by": ["metadata.model_label"], "filters": [D18]},
        "mod_usd": {"title": f"Dollars per case, realized from the local ledger ({MOD_POOL})", "measure": "avg(metadata.usd)", "group_by": ["metadata.model_label"], "filters": [D18], "unit": "cost"},
        "mod_sec": {"title": f"Seconds per case ({MOD_POOL})", "measure": "avg(metadata.wall_s)", "group_by": ["metadata.model_label"], "filters": [D18], "unit": "duration"},
        "mod_p90": {"title": f"Tail latency, p90 seconds per case ({MOD_POOL})", "measure": "percentile(metadata.wall_s, 0.9)", "group_by": ["metadata.model_label"], "filters": [D18], "unit": "duration"},
        "mod_cap": {"title": f"Cap-hit rate ({MOD_POOL})", "measure": "avg(metadata.cap_hit)", "group_by": ["metadata.model_label"], "filters": [D18]},
        "mod_fail": {"title": f"Execution-failure rate, no finding produced ({MOD_POOL})", "measure": "avg(metadata.execution_failed)", "group_by": ["metadata.model_label"], "filters": [D18]},
        "mod_per_dollar": {"title": f"Correct outcomes per dollar ({MOD_POOL})", "measure": "sum(metadata.correct_all) / sum(metadata.usd)", "group_by": ["metadata.model_label"], "filters": [D18], "unit": "count"},
        "mod_usd_per_correct": {"title": f"Dollars per correct outcome ({MOD_POOL})", "measure": "sum(metadata.usd) / sum(metadata.correct_all)", "group_by": ["metadata.model_label"], "filters": [D18], "unit": "cost"},
        "pareto": {"title": "Correct outcomes per dollar across every system@model on the same 18 cases (the Pareto question in one list)",
                   "measure": "sum(metadata.correct_all) / sum(metadata.usd)", "group_by": ["metadata.variant_label"], "filters": [JUDGED18], "unit": "count"},
        "judge_vs_lawyer": {"title": "Can the panel be trusted? Closeness of the three-judge mean to the lawyer per dimension, 24 blinded packets (100% = identical score)",
                            "measure": [{"btql": f"avg(metadata.panel_{d}_closeness)", "name": d} for d in JUDGE_DIMS_],
                            "group_by": [], "filters": [f"metadata.has_human = 1 and {JUDGED18}"]},
        **{f"judge_bias_{d}": {"title": f"Which judge for {d}? Closeness of each judge to the lawyer (100% = identical score; the tallest bar is the pick, 24 packets)",
                               "measure": [{"btql": f"avg(metadata.judge_{f}_{d}_closeness)", "name": label} for f, label in FAMILIES], "group_by": [], "filters": [f"metadata.has_human = 1 and {JUDGED18}"]}
           for d in JUDGE_DIMS_},
        "judge_family_usd": {"title": "What does a judge call cost? Realized dollars per judged trace per family (108 traces; all three within a hundredth of a cent)",
                             "measure": [{"btql": f"avg(metadata.judge_{f}_usd)", "name": label} for f, label in FAMILIES], "group_by": [], "filters": [JUDGED18], "unit": "cost"},
        "panel_professional": {"title": "Judge panel: professional quality by system@model, 108 judged traces (the judges' view of the model race)", "measure": "avg(metadata.judge_professional)", "group_by": ["metadata.variant_label"], "filters": [JUDGED18]},
        "deepeval": {"kind": "bignumber", "title": "DeepEval agrees with deterministic truth (task completion >= 0.5 vs the expert span), 108 judged traces",
                     "measure": [{"btql": "avg(metadata.deepeval_agrees_with_truth)", "name": "agreement with truth"}], "group_by": [], "filters": [JUDGED18]},
        "deepeval_vs_judges": {"kind": "bignumber", "title": "DeepEval agrees with our judge panel (task completion vs judges' reasoning), 108 judged traces; its evaluator is one of the three judges, so read this as partly self-agreement",
                               "measure": [{"btql": "avg(metadata.deepeval_agrees_with_judges)", "name": "agreement with the panel"}], "group_by": [], "filters": [JUDGED18]},
        "deepeval_vs_lawyer": {"kind": "bignumber", "title": "DeepEval agrees with the lawyer (task completion vs the lawyer's reasoning), 24 packets",
                               "measure": [{"btql": "avg(metadata.deepeval_agrees_with_lawyer)", "name": "agreement with the lawyer"}], "group_by": [], "filters": [f"metadata.has_human = 1 and {JUDGED18}"]},
        "deepeval_task": {"title": "DeepEval task completion by system@model (its own metric, 0-1)", "measure": "avg(metadata.deepeval_task_completion)", "group_by": ["metadata.variant_label"], "filters": [JUDGED18]},
        "deepeval_tools": {"title": "DeepEval tool correctness by system@model (agrees with our required-evidence check 89% of the time)", "measure": "avg(metadata.deepeval_tool_correctness)", "group_by": ["metadata.variant_label"], "filters": [JUDGED18]},
        "deepeval_steps": {"title": "DeepEval step efficiency by system@model (a written-down GEval; the trajectory dimension from outside)", "measure": "avg(metadata.deepeval_step_efficiency)", "group_by": ["metadata.variant_label"], "filters": [JUDGED18]},
        "li_hit_rate": {"title": "LlamaIndex's own hit rate by retriever, 58 dev queries (its RetrieverEvaluator, including its native bm25)", "measure": "avg(metadata.li_hit_rate)", "group_by": ["metadata.retriever"], "filters": ["metadata.category = 'retrieval'"]},
        "li_mrr": {"title": "LlamaIndex's own MRR by retriever, 58 dev queries", "measure": "avg(metadata.li_mrr)", "group_by": ["metadata.retriever"], "filters": ["metadata.category = 'retrieval'"]},
        "li_vs_ours": {"kind": "bignumber", "title": "LlamaIndex vs our gold-span scorer on the same hybrid_rrf hits: LlamaIndex's hit rate (ours is 91.4%; the two rank the six retrievers identically, Spearman 0.99)",
                       "measure": [{"btql": "avg(metadata.li_hit_rate)", "name": "LlamaIndex hit rate, hybrid_rrf"}], "group_by": [], "filters": ["metadata.category = 'retrieval' and metadata.retriever = 'hybrid_rrf'"]},
        "prompt_professional": {"title": "Judge: professional by arm-A prompt variant, 18 packets, GLM", "measure": "avg(metadata.judge_professional)", "group_by": ["metadata.prompt_variant"], "filters": ["metadata.category = 'prompt-variant'"]},
        "prompt_evidence": {"title": "Judge: evidence by arm-A prompt variant, 18 packets, GLM", "measure": "avg(metadata.judge_evidence)", "group_by": ["metadata.prompt_variant"], "filters": ["metadata.category = 'prompt-variant'"]},
        "prompt_reasoning": {"title": "Judge: reasoning by arm-A prompt variant, 18 packets, GLM", "measure": "avg(metadata.judge_reasoning)", "group_by": ["metadata.prompt_variant"], "filters": ["metadata.category = 'prompt-variant'"]},
    }


DASHBOARDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("DealPoint eval overview",
     ("VERDICTS (from the question dashboards, 2026-09-07; small samples, one run each). "
      "SEARCH: hybrid (dense + BM25, reciprocal-rank fusion), no reranker: 91% gold-span hit@5 vs 81% dense. "
      "BEST MEASURED CONFIG (the Pareto list, every system@model on the same 18 cases): A@haiku, the single-shot baseline on Claude Haiku 4.5: 56% "
      "correct outcomes, zero fabrication, zero cap-hits, zero failures, 3.7 s and $0.0034 per case; it abstains on 39% of cases, right on the "
      "counterfactuals, costly on a few answerable ones. Best agentic config: D@gemini (50%, the only agent that never fabricates, 12.7 s). "
      "SYSTEM (four systems on GLM, same 32 cases): the loop only pays off once a stopping rule ends the search-in-circles cap-hits; C beats D "
      "(the skill trades cap-hits for fabrication and loses); until the rule lands, A wins on correct outcomes, fabrication and cost. "
      "MODEL (arm D, same 18 cases): Gemini 3.1 flash lite for quality (most correct outcomes, zero execution failures); Qwen 3.7 flash for the cost "
      "floor (most correct outcomes per dollar, but two runs in three cap out or fail); Haiku out of the agent at sixteen times the cost. "
      "JUDGES: Mistral Small for reasoning, evidence and professional quality, NVIDIA Nemotron for trajectory; two judges, not three; no judge "
      "is trustworthy on trajectory, keep the lawyer there. "
      "PROMPT: cite-first for the baseline (best evidence score), terse loses everywhere. "
      "How to read any chart here: a ranked list over the traces in Logs (metadata mirrored from the stored results; scores are metered, metadata is free), "
      "one case pool of comparable traces at a time; 'correct outcome' counts an unanswered case as wrong and a correct abstention as right."),
     ("big_best_config", "big_best_agent", "big_best_retriever", "big_judges", "big_best_prompt",
      "pareto", "sys_correct", "mod_correct", "ret_hit5", "judge_vs_lawyer", "prompt_evidence")),
    ("Which system?",
     ("VERDICT: A wins today (most correct outcomes, zero fabrication, a quarter of the cost); C beats D and earns its place once a stopping rule ends the cap-hits. Four systems, one config key changed per step (A pipeline+dense, B agent+dense, C agent+hybrid, D agent+hybrid+skill), all on GLM 5.3 flash "
     "and the same 32 test cases. Read top to bottom: does agency help (correct outcome), what it costs in failure modes (cap-hits, fabrication, "
     "abstention), whether the loop pays off more on a stronger model, then dollars, seconds, tool calls and correct outcomes per dollar."),
     ("sys_correct", "sys_cap", "sys_fab", "sys_abstain", "loop_x_model", "sys_usd", "sys_sec", "sys_calls", "sys_per_dollar")),
    ("Which model?",
     ("VERDICT: Gemini 3.1 flash lite for quality, Qwen 3.7 flash for the cost floor, Haiku out of the loop; and the Pareto list says the best config overall is the single-shot baseline on Haiku. Arm D held fixed, five models on the same 18 judged cases. Quality first (correct outcome), then price and speed (dollars, seconds, p90), "
     "then reliability (cap-hits, execution failures), then the two numbers a deployment decision turns on: correct outcomes per dollar and "
     "dollars per correct outcome. The Pareto list ranks every system@model by correct outcomes per dollar; the judge panel's chart is the "
     "judges' view of the same race."),
     ("mod_correct", "mod_usd", "mod_sec", "mod_p90", "mod_cap", "mod_fail", "mod_per_dollar", "mod_usd_per_correct", "pareto", "panel_professional")),
    ("Which retriever?",
     ("VERDICT: hybrid_rrf, no reranker. Six retrievers on the same 58 dev queries, scored against the expert's gold span: hit@5, hit@10 and mean reciprocal rank. "
     "Hybrid (dense + BM25 with reciprocal-rank fusion) wins; reranking adds nothing."),
     ("ret_hit5", "ret_hit10", "ret_mrr", "li_hit_rate")),
    ("Judges and the lawyer",
     ("VERDICT: Mistral Small for reasoning, evidence and professional quality, NVIDIA Nemotron for trajectory, two judges not three, and the lawyer keeps trajectory. Three cheap LLM judges (Mistral Small, NVIDIA Nemotron, ByteDance Seed) scored 108 blinded traces on four dimensions; a lawyer scored 24 of "
     "the same packets. First chart: can the panel be trusted (closeness of the three-judge mean to the lawyer, per dimension; 100% is identical). "
     "Next four: which judge for which dimension, each judge's closeness to the lawyer; the tallest bar is the pick. Direction, for the record: "
     "every judge scores higher than the lawyer on every dimension, most of all on trajectory, so a gap always means over-credit. Then cost per "
     "call (a wash). Last: DeepEval as an independent second opinion against deterministic truth."),
     ("judge_vs_lawyer", "judge_bias_reasoning", "judge_bias_evidence", "judge_bias_trajectory", "judge_bias_professional", "judge_family_usd")),
    ("Second opinions",
     ("VERDICT: keep both frameworks as cross-checks, never as gates. DeepEval (an off-the-shelf agent-eval framework: task completion, tool "
      "correctness, argument correctness, step efficiency) read the same 108 traces our judges scored. It agrees with the judge panel far more than "
      "with deterministic truth, and its evaluator model is one of our three judges, so that agreement is partly self-agreement; its tool-correctness "
      "metric is the one that tracks our required-evidence check. LlamaIndex's RetrieverEvaluator scored the same six retrievers and ranks them "
      "exactly as our gold-span scorer does (Spearman 0.99); its genuinely native bm25 config disagrees with ours on three queries, all explained by "
      "duplicate relevant chunks. Read: the tiles, then DeepEval's own metrics by system, then LlamaIndex's numbers next to ours."),
     ("deepeval", "deepeval_vs_judges", "deepeval_vs_lawyer", "li_vs_ours", "deepeval_task", "deepeval_tools", "deepeval_steps", "li_hit_rate", "li_mrr")),
    ("Which prompt?",
     ("VERDICT: cite-first (best evidence score); terse loses everywhere; abstain-first buys nothing. Four system prompts for the single-shot baseline (base, terse, cite-first, abstain-first) over the same 18 packets and the same model, "
     "judged on professional quality, evidence and reasoning by the LLM scorers. Only the prompt varies."),
     ("prompt_professional", "prompt_evidence", "prompt_reasoning")),
)


def dashboard_charts() -> list[dict]:
    """The overview dashboard's charts (the first entry of DASHBOARDS)."""
    return dashboard_definitions()[0]["charts"]


def dashboard_definitions() -> list[dict]:
    """One dashboard per engineering question, plus a short overview, as data (never a string template).

    The dashboard is Braintrust's *monitor* surface over PROJECT LOGS. Two facts shape it: (1) the logs
    carry zero obj/*, judge/*, human/* or li/* scores (those live in experiments, where the Starter plan
    meters them), so a chart over those names is blank by construction, which is what the first version
    of this dashboard did; (2) a monitor chart is either a time series (x = time, right for production
    traffic) or a `scalars` toplist (groups ranked by an aggregate, x = the group). Experiment comparisons
    are not time series. So every chart is a toplist over the METADATA MIRROR the showroom merges onto
    each log (`braintrust_showroom.metadata_mirror`), the dashboard's name carries the question, and the
    time range is pinned (DASHBOARD_RANGE) so a sliding default window cannot blank batch-ingested logs.
    """
    cat = chart_catalogue()
    return [{"name": name, "description": desc, "view_type": "monitor", "charts": [{"kind": "toplist", **cat[k]} for k in keys]}
            for name, desc, keys in DASHBOARDS]


def dashboard_definition() -> dict:
    return dashboard_definitions()[0]


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


PREPROCESSOR_SLUG = "dealpoint-trace-preprocessor"


def topics_preprocessor_code() -> str:
    """Renders one DealPoint log as text for Topics: agent traces (system, model, question, status, tool
    calls, answer, rationale, objective grounding, from the showroom's metadata mirror on the root span),
    retrieval logs and prompt-variant runs. A plain string of JS (the inline-code preprocessor shape);
    installed as the project's default preprocessor by `sync_topics_and_pattern`.
    """
    return """function handler(span) {
  const m = span.metadata || {};
  const out = span.output || {};
  if (m.category === "retrieval") {
    return [
      "retrieval log",
      `retriever: ${m.retriever}`,
      `question: ${span.input || ""}`,
      `gold span first hit at rank: ${m.first_hit_rank == null ? "none in top 20" : m.first_hit_rank}`,
    ].join("\\n");
  }
  if (m.category === "prompt-variant") {
    return [
      "arm-A prompt variant run",
      `prompt variant: ${m.prompt_variant}`,
      `answer: ${typeof span.output === "string" ? span.output : (out.answer || JSON.stringify(out)).slice(0, 300)}`,
      `judge evidence: ${m.judge_evidence}, judge professional: ${m.judge_professional}`,
    ].join("\\n");
  }
  const answer = typeof out === "string" ? out : (out.answer || "(no finding)");
  const rationale = typeof out === "object" && out.rationale ? String(out.rationale).slice(0, 400) : "";
  return [
    "agent trace",
    `system: ${m.system_label || m.arm || ""}`,
    `model: ${m.model_label || m.model || ""}`,
    `question: ${span.input || ""}`,
    `status: ${m.status || ""} (tool calls: ${m.tool_calls == null ? "n/a" : m.tool_calls}; cap hit: ${m.cap_hit ? "yes" : "no"})`,
    `final answer: ${answer}`,
    rationale ? `rationale: ${rationale}` : "",
    `objectively grounded (matches the expert span): ${m.ga_scored == null ? "no finding to score" : (m.ga_scored ? "yes" : "no")}`,
    m.fabrication ? "fabrication: the citation does not support the answer" : "",
  ].filter(Boolean).join("\\n");
}
"""


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


GROUP_DISPLAY = {"metadata.system_label": "system", "metadata.model_label": "model", "metadata.variant_label": "system@model",
                 "metadata.retriever": "retriever", "metadata.prompt_variant": "prompt variant"}


def _measure_rest(measure) -> dict:
    """One `chart_catalogue()` measure -> the real `custom_charts` measure shape, confirmed live 2026-09-06
    via `create_monitoring_view` + `GET /v1/view`: a bare `avg(scores."X")` is `{type: aggregateScore,
    scoreName, aggregator}`; anything else (a full BTQL expression) is `{type: expression, btql}`.
    A measure may be a string or `{"btql": ..., "name": ...}`; `name` is what the chart shows as the row
    label, so a multi-measure chart reads "Mistral / NVIDIA / ByteDance", not three BTQL expressions.
    """
    btql, name = (measure["btql"], measure.get("name")) if isinstance(measure, dict) else (measure, None)
    match = _SCORE_AVG_RE.match(btql)
    if match:
        score_name = match.group(1)
        return {"type": "aggregateScore", "scoreName": score_name, "aggregator": {"type": "avg"}, "displayName": name or score_name}
    return {"type": "expression", "btql": btql, "displayName": name or btql}


def _chart_rest_definition(chart: dict) -> dict:
    measures = chart["measure"] if isinstance(chart["measure"], list) else [chart["measure"]]
    group_bys = [{"btql": g, "displayName": GROUP_DISPLAY.get(g, g.rsplit(".", 1)[-1])} for g in chart.get("group_by", [])]
    unit_type = chart.get("unit") or ("cost" if "usd" in str(chart["measure"]) else "percent")
    filters = [{"btql": f} for f in chart.get("filters", [])]
    if chart.get("kind") == "bignumber":
        return {"type": "scalars", "measures": [_measure_rest(m) for m in measures], "groupBys": [], "filters": filters,
                "viz": {"type": "singleValue", "unitType": unit_type}}
    if chart.get("kind") == "toplist":
        return {"type": "scalars", "measures": [_measure_rest(m) for m in measures], "groupBys": group_bys, "filters": filters,
                "sortByOptions": {"type": "value", "direction": "desc"}, "viz": {"type": "toplist", "unitType": unit_type}}
    return {
        "type": "monitorTimeseries",
        "measures": [_measure_rest(m) for m in measures],
        "groupBys": group_bys,
        "filters": filters,
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


def _upsert_view(rest_client, object_type: str, object_id: str, view_type: str, name: str, view_data: dict, description: str | None = None) -> dict:
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
    if description:
        body["description"] = description
    if view_type == "monitor":
        # confirmed live 2026-09-06: a monitor view needs `options.viewType ==
        # "monitor"` alongside `view_data.custom_charts`, or the dashboard
        # renders with no chart layout even though the POST succeeds.
        body["options"] = {"viewType": "monitor", "options": {"projectId": object_id, "type": object_type,
                                                              "spanType": "range", "rangeValue": DASHBOARD_RANGE}}
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

    dashboards_result = []
    for dash in dashboard_definitions():
        dash_upserted = _upsert_view(rest_client, "project", project_id, dash["view_type"], dash["name"], _view_data_for({"custom_charts": dash["charts"]}),
                                     description=dash.get("description"))
        dashboards_result.append({"name": dash["name"], "n_charts": len(dash["charts"]), **dash_upserted})
    dashboard_result = dashboards_result[0]

    return {"project_id": project_id, "views": views_result, "dashboard": dashboard_result, "dashboards": dashboards_result}


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
    # The preprocessor is a real function (`function_type == "preprocessor"`, inline node code) and the
    # project's default preprocessor points at it (PATCH /v1/project settings.default_preprocessor),
    # so the facet above reads the metadata mirror rather than the built-in thread rendering.
    pre_body = {
        "project_id": project_id, "name": PREPROCESSOR_SLUG, "slug": PREPROCESSOR_SLUG, "function_type": "preprocessor",
        "description": "Renders each DealPoint log for Topics: agent traces (system, model, question, status, answer, grounded), retrieval logs and prompt-variant runs.",
        "function_data": {"type": "code", "data": {"type": "inline", "runtime_context": {"runtime": "node", "version": "22"}, "code": cfg["preprocessor_code"]}},
    }
    try:
        pre = rest_client.post("/v1/function", pre_body)
        topics_result["preprocessor"] = {"id": pre.get("id"), "slug": PREPROCESSOR_SLUG}
        rest_client.patch(f"/v1/project/{project_id}", {"settings": {"default_preprocessor": {"type": "function", "id": pre.get("id")}}})
        topics_result["preprocessor"]["set_as_default"] = True
    except Exception as exc:  # noqa: BLE001 - recorded, never faked
        topics_result.setdefault("preprocessor", {})["error"] = str(exc)
        topics_result["preprocessor"]["set_as_default"] = False
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
# One ledger per org, resolved lazily from the active key into data/reports/orgs/<org-slug>/
# (shared with the showroom). A set value here (tests) or BRAINTRUST_LEDGER_FILE wins.
SCORE_LEDGER_PATH: Path | None = None
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


def _score_ledger_path() -> Path:
    from dealpoint.eval.braintrust_adapter import org_report_path
    return SCORE_LEDGER_PATH or org_report_path("braintrust_score_ledger.jsonl", override_env="BRAINTRUST_LEDGER_FILE")


def _ledger_load() -> set[tuple[str, str]]:
    if not _LEDGER_ACTIVE or not _score_ledger_path().exists():
        return set()
    seen: set[tuple[str, str]] = set()
    with open(_score_ledger_path(), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                seen.add((row["experiment"], row["key"]))
    return seen


def _ledger_record(experiment: str, key: str, n_scores: int) -> None:
    if not _LEDGER_ACTIVE:
        return
    path = _score_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"experiment": experiment, "key": key, "n_scores": n_scores,
                             "ts": datetime.now(UTC).isoformat()}) + "\n")


REPRESENTATIVE_EXPERIMENT = "m7-representative-traces"


def replay_representative_cases(sdk_client) -> dict:
    """Replay the six rule-chosen representative traces (representative_cases())
    into `m7-representative-traces` as full span trees with NO scores
    (`allowed_score_names=set()`): the walkthrough's "Logs trace" stop needs
    the trees, and trees without scores cost nothing on the plan. Ledger
    entries carry n_scores=0; re-runs skip trees already recorded.
    """
    from dealpoint.corpus.document import load_document
    from dealpoint.eval.braintrust_sync import (
        _emit_span_tree,
        _init_experiment,
        log_hierarchy,
        representative_cases,
    )
    from dealpoint.eval.cases import find_case, resolve_document_id

    selections = representative_cases().get("selections", [])
    seen = _ledger_load()
    experiment = _init_experiment(sdk_client, REPRESENTATIVE_EXPERIMENT)
    replayed: list[dict] = []
    for sel in selections:
        case_id, variant_id = sel.get("case_id"), sel.get("variant_id")
        key = f"{case_id}:{variant_id}"
        if not case_id or (REPRESENTATIVE_EXPERIMENT, key) in seen:
            continue
        rows = {r.get("case_id"): r for r in _load_jsonl(sel["results_path"])} if sel.get("results_path") else {}
        row = rows.get(case_id) or {"case_id": case_id, "scores": {}, "record": {"trajectory": []}}
        try:
            case = find_case(case_id)
            doc = load_document(resolve_document_id(case))
        except (KeyError, FileNotFoundError):
            case, doc = {}, None
        hierarchy = log_hierarchy(row, case, doc, judge_dims=None)
        root_span = experiment.start_span(name=f"{sel.get('category')}:{variant_id}")
        if hasattr(root_span, "log"):
            root_span.log(metadata={"case_id": case_id, "variant_id": variant_id, "category": sel.get("category")})
        n_scores = 0
        for child in hierarchy.get("children", []):
            n_scores += _emit_span_tree(experiment, child, parent_span=root_span, allowed_score_names=set())
        if hasattr(root_span, "end"):
            root_span.end()
        _ledger_record(REPRESENTATIVE_EXPERIMENT, key, n_scores)
        replayed.append({"category": sel.get("category"), "case_id": case_id, "variant_id": variant_id, "n_scores": n_scores})
    if hasattr(experiment, "flush"):
        experiment.flush()
    return {"experiment_name": REPRESENTATIVE_EXPERIMENT, "trees": replayed, "n_trees": len(replayed)}


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


def sync_cockpit(rest_client, sdk_client=None, replay_representative: bool = False) -> dict:
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
    replay_representative_result = None
    if sdk_client is not None and replay_representative:
        replay_representative_result = replay_representative_cases(sdk_client)   # score-free by construction
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
        "replay_representative": replay_representative_result,
        "replay": replay_result,
        "human_scoring_probe": HUMAN_SCORING_PROBE,
        "synced_at": datetime.now(UTC).isoformat(),
    }


EXPERIMENT_NAME_PREFIXES: tuple[str, ...] = ("A-", "B-", "C-", "D-", "judge-", "judged-", "m7-", "m7b-")
_SYNC_COPY_SUFFIX = re.compile(r"-[0-9a-f]{8}$")

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
    # A trailing "-<8 hex>" marks a re-sync copy (M7a's syncs minted one per run;
    # the copies are 1-row stubs once the score quota was hit). Within a prefix
    # family prefer an unsuffixed experiment (the scored original), then newest.
    def _rank(exp: dict) -> tuple[int, str]:
        name = exp.get("name") or ""
        return (0 if _SYNC_COPY_SUFFIX.search(name) else 1, exp.get("created") or "")

    newest_by_prefix: dict[str, dict] = {}
    for exp in objects:
        name = exp.get("name") or ""
        for prefix in EXPERIMENT_NAME_PREFIXES:
            if not name.startswith(prefix):
                continue
            current = newest_by_prefix.get(prefix)
            if current is None or _rank(exp) > _rank(current):
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


def ledger_totals() -> dict:
    """Live scores written so far, from the score ledger (read regardless of
    _LEDGER_ACTIVE: reporting is not writing)."""
    totals = {"n_live_scores": 0, "n_human_scores": 0, "n_entries": 0, "last_ts": None}
    path = _score_ledger_path()
    if not path.exists():
        return totals
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            totals["n_entries"] += 1
            n = int(row.get("n_scores", 0))
            totals["n_live_scores"] += n
            if str(row.get("experiment", "")).startswith("judge-"):
                totals["n_human_scores"] += n
            totals["last_ts"] = row.get("ts")
    return totals


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
        "ledger": ledger_totals(),
        "n_live_scores_planned": result["n_live_scores_planned"],
        "live_score_cap": result["live_score_cap"],
        "replay_representative": result.get("replay_representative"),
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
        result = sync_cockpit(rest_client_for_manifest, sdk_client=braintrust, replay_representative=True)

    manifest = build_demo_manifest(result, rest_client=rest_client_for_manifest)
    DEMO_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEMO_MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        fh.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
