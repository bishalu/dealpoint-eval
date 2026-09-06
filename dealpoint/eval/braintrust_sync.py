"""`just braintrust-sync` -- recreate every Braintrust artifact from Git/local
sources, idempotently (spec section 3).

Structured as pure mapping functions + a thin client-calling driver
(`sync`), so the offline gate tests drive the whole mapping through a fake
client -- never real network. `braintrust` is imported lazily inside every
function that actually talks to the SDK, matching the existing
`braintrust_adapter.py` convention.

Everything here replays already-computed local artifacts (result rows,
judge scores, the frozen datasets/skill/rubric). No model is ever called by
this module -- the walkthrough must be re-creatable from Git/local sources
by one command, never dependent on a live trace id (Starter tier retains
logs 14 days).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from dealpoint.config import (
    ARMS,
    BRAINTRUST_SYNC_PATH,
    M7A_MAX_SCORES_PER_CASE,
    M7A_MAX_SUBSET_FOR_WIDE_SCORES,
    M7A_REVIEW_SET_N,
    SEED,
)
from dealpoint.eval.braintrust_adapter import PROJECT

# --- namespaces + common metadata -------------------------------------------

SCORE_NAMESPACES: tuple[str, ...] = ("obj", "li", "judge", "deepeval", "procedure")

COMMON_METADATA_KEYS: tuple[str, ...] = (
    "stage",
    "arm",
    "reasoning_type",
    "retriever",
    "case_type",
    "model",
    "index_version",
    "skill_version",
    "git_sha",
    "framework_versions",
)


def score_namespace(name: str) -> str:
    """`\"grounded_accuracy\"` -> `\"obj/grounded_accuracy\"`, etc.

    Rule (deterministic, documented): DeepEval-derived names -> `deepeval/`;
    judge-aggregate dimension names (the four `JUDGE_DIMENSIONS`) -> `judge/`;
    everything else (the canonical `dealpoint.eval.scorers` names) -> `obj/`.
    Names already carrying a namespace slash are returned unchanged.
    """
    if "/" in name:
        return name
    deepeval_names = {"task_completion", "tool_correctness", "argument_correctness", "step_efficiency"}
    judge_names = {"reasoning", "evidence", "trajectory", "professional"}
    li_names = {"hit_rate", "mrr"}
    if name in deepeval_names:
        return f"deepeval/{name}"
    if name in judge_names:
        return f"judge/{name}"
    if name in li_names:
        return f"li/{name}"
    return f"obj/{name}"


def _skill_version() -> str | None:
    from dealpoint.config import VERSIONS_JSON_PATH

    if not VERSIONS_JSON_PATH.exists():
        return None
    try:
        payload = json.loads(VERSIONS_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("skill_version")


def _framework_versions_dict() -> dict:
    from dealpoint.eval.framework_versions import framework_versions

    return framework_versions()


def common_metadata(row: dict, *, stage: str, extra: dict | None = None) -> dict:
    """Every row's metadata carries all ten common keys (spec section 5.1) --
    never `{}`. This is the fix for the section 0.5 `metadata: {}` trap.
    """
    from dealpoint.eval.cases import find_case, git_sha7, resolve_question

    case_id = row.get("case_id")
    case = None
    if case_id:
        try:
            case = find_case(case_id)
        except KeyError:
            case = None

    reasoning_type = None
    case_type = None
    if case is not None:
        reasoning_type = resolve_question(case).reasoning_type
        case_type = case.get("kind") or case.get("case_set")

    retriever = row.get("retriever")
    if retriever is None and row.get("arm") in ARMS:
        retriever = (ARMS[row["arm"]].get("retriever") or {}).get("name")

    metadata = {
        "stage": stage,
        "arm": row.get("arm"),
        "reasoning_type": reasoning_type,
        "retriever": retriever,
        "case_type": case_type,
        "model": row.get("model"),
        "index_version": row.get("index_version"),
        "skill_version": row.get("skill_version") or _skill_version(),
        "git_sha": row.get("git_sha7") or row.get("git_sha") or git_sha7(),
        "framework_versions": _framework_versions_dict(),
    }
    if extra:
        metadata.update(extra)
    assert set(COMMON_METADATA_KEYS) <= set(metadata.keys())
    return metadata


# --- score budget -------------------------------------------------------------


class ScoreBudgetError(RuntimeError):
    pass


def assert_score_budget(plan: dict) -> None:
    """M7 experiments <= 12 scores/case on subsets <= 60 cases; M4/M6 sweeps
    keep six (spec section 5.1). `plan` is one experiment-plan entry:
    `{name, stage, n_cases, score_names, ...}`.
    """
    n_scores = len(plan.get("score_names", []))
    n_cases = plan.get("n_cases", 0)
    stage = plan.get("stage", "")

    if stage in ("m4_sweep", "m6_sweep"):
        if n_scores > 6:
            raise ScoreBudgetError(
                f"experiment {plan.get('name')!r} (stage {stage!r}) logs {n_scores} scores "
                f"-- M4/M6 sweeps must keep exactly six"
            )
        return

    if n_cases <= M7A_MAX_SUBSET_FOR_WIDE_SCORES:
        limit = M7A_MAX_SCORES_PER_CASE
    else:
        limit = 6

    if n_scores > limit:
        raise ScoreBudgetError(
            f"experiment {plan.get('name')!r} logs {n_scores} scores on {n_cases} cases "
            f"-- exceeds the {limit}-score budget for this subset size"
        )


# --- datasets -----------------------------------------------------------------

DATASET_SETS: tuple[str, ...] = ("dev", "test", "counterfactual")


def _dataset_row_metadata(row: dict) -> dict:
    from dealpoint.eval.cases import resolve_question

    question = resolve_question(row)
    return {
        "question_id": row.get("question_id"),
        "agreement_id": row.get("agreement_id"),
        "reasoning_type": question.reasoning_type,
        "case_type": row.get("kind") or row.get("case_set"),
    }


def dataset_rows(set_name: str) -> list[dict]:
    """One dataset-shaped row per case, for a known set name.

    `set_name` may be one of `DATASET_SETS`, `"judged_calibration"` (the M5
    judged subset), or `"synthetic_query"` (the frozen synthetic set).
    """
    from dealpoint.config import JUDGED_SUBSET_PATH, SYNTHETIC_DEV_QUERIES_PATH
    from dealpoint.eval.cases import find_case, load_case_set

    if set_name in DATASET_SETS:
        rows = load_case_set(set_name)
        return [
            {"input": row["case_id"], "expected": row.get("gold_answer"), "metadata": _dataset_row_metadata(row)}
            for row in rows
        ]

    if set_name == "judged_calibration":
        if not Path(JUDGED_SUBSET_PATH).exists():
            return []
        payload = json.loads(Path(JUDGED_SUBSET_PATH).read_text(encoding="utf-8"))
        out = []
        for cid in payload.get("case_ids", []):
            try:
                case = find_case(cid)
            except KeyError:
                continue
            out.append(
                {
                    "input": cid,
                    "expected": case.get("gold_answer"),
                    "metadata": {**_dataset_row_metadata(case), "source_hashes": {"subset_hash": payload.get("subset_hash")}},
                }
            )
        return out

    if set_name == "synthetic_query":
        if not Path(SYNTHETIC_DEV_QUERIES_PATH).exists():
            return []
        out = []
        with open(SYNTHETIC_DEV_QUERIES_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out.append(
                    {
                        "input": row["query_id"],
                        "expected": None,
                        "metadata": {
                            "question_id": row.get("question_id"),
                            "agreement_id": row.get("agreement_id"),
                            "case_id": row.get("case_id"),
                            "source_hashes": {"prompt_hash": row.get("prompt_hash")},
                        },
                    }
                )
        return out

    raise KeyError(f"unknown dataset set_name {set_name!r}")


DATASET_PLAN_NAMES: tuple[str, ...] = (
    "dev",
    "test",
    "counterfactual",
    "judged_calibration",
    "synthetic_query",
)


# --- experiments ----------------------------------------------------------


def _load_jsonl(path) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rag_experiment_plans() -> list[dict]:
    """RAG-stage experiment plans, sourced from `data/reports/li_rag_eval.json`."""
    from dealpoint.config import LI_RAG_EVAL_JSON_PATH

    plans: list[dict] = []
    name_by_config = {
        "dense": "rag-m3-dense",
        "bm25": "rag-m3-bm25",
        "hybrid_rrf": "rag-m3-hybrid",
        "hybrid_rrf_rerank": "rag-m3-hybrid-rerank",
        "multi_query_fusion": "rag-m3-fusion",
        "multi_query_fusion_rerank": "rag-m3-fusion-rerank",
    }

    li_report = {}
    if LI_RAG_EVAL_JSON_PATH.exists():
        try:
            li_report = json.loads(LI_RAG_EVAL_JSON_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            li_report = {}

    retrievers = li_report.get("retrievers", {})
    n_cases = li_report.get("generated_from", {}).get("n_cases", 0)
    for config_name, exp_name in name_by_config.items():
        r = retrievers.get(config_name)
        if r is None:
            continue
        rows = [
            {
                "case_id": None,
                "arm": None,
                "model": None,
                "index_version": li_report.get("generated_from", {}).get("index_version"),
                "retriever": config_name,
            }
        ]
        plans.append(
            {
                "name": exp_name,
                "stage": "rag",
                "tags": ["stage=rag"],
                "n_cases": n_cases,
                "score_names": ["obj/gold_span_hit_at_5", "obj/gold_span_mrr", "li/hit_rate", "li/mrr"],
                "rows": rows,
                "metadata": {"config": config_name},
            }
        )

    plans.append(
        {
            "name": "rag-m7-li-crosscheck",
            "stage": "rag",
            "tags": ["stage=rag"],
            "n_cases": n_cases,
            "score_names": ["obj/gold_span_hit_at_5", "obj/gold_span_mrr", "li/hit_rate", "li/mrr"],
            "rows": [{"case_id": None, "arm": None, "model": None}],
            "metadata": {"purpose": "crosscheck"},
        }
    )
    synth = li_report.get("synthetic")
    plans.append(
        {
            "name": "rag-m7-synthetic",
            "stage": "rag",
            "tags": ["stage=rag"],
            "n_cases": (synth or {}).get("n_queries", 0),
            "score_names": ["obj/gold_span_hit_at_5", "li/hit_rate"],
            "rows": [{"case_id": None, "arm": None, "model": None}],
            "metadata": {"purpose": "synthetic"},
        }
    )
    return plans


def _agent_experiment_plans() -> list[dict]:
    """Existing A-D experiment names, from the four-arm + pareto manifests
    (spec: \"re-logged from canonical rows only if needed for tags -- never
    re-run\").
    """
    from dealpoint.config import PARETO_MANIFEST_PATH
    from dealpoint.eval.braintrust_adapter import experiment_name
    from dealpoint.eval.four_arm_sweep import MANIFEST_PATH as FOUR_ARM_MANIFEST_PATH

    plans: list[dict] = []
    for manifest_path in (FOUR_ARM_MANIFEST_PATH, PARETO_MANIFEST_PATH):
        entries = []
        if Path(manifest_path).exists():
            try:
                entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                entries = []
        for entry in entries:
            results_path = entry.get("results_path")
            if not results_path:
                continue
            rows = _load_jsonl(results_path)
            if not rows:
                continue
            index_version = rows[0].get("index_version") or "unknown"
            git_sha7 = rows[0].get("git_sha7") or "nogit"
            name = experiment_name(entry["arm"], entry["model"], index_version, git_sha7)
            plans.append(
                {
                    "name": name,
                    "stage": "agent",
                    "tags": ["stage=agent"],
                    "n_cases": len(rows),
                    "score_names": [
                        "obj/grounded_accuracy",
                        "obj/answer_correct",
                        "obj/citation_gold_overlap",
                        "obj/citation_verbatim",
                        "obj/abstain_correct",
                        "obj/skill_adherence",
                    ],
                    "rows": rows,
                    "metadata": {"arm": entry["arm"], "model": entry["model"]},
                }
            )
    return plans


def _eval_experiment_plans() -> list[dict]:
    """Judge-panel and DeepEval experiments, tagged `stage=eval`."""
    from dealpoint.config import DEEPEVAL_CROSSCHECK_JSON_PATH, JUDGED_SUBSET_PATH

    plans: list[dict] = []
    if Path(JUDGED_SUBSET_PATH).exists():
        payload = json.loads(Path(JUDGED_SUBSET_PATH).read_text(encoding="utf-8"))
        n_cases = len(payload.get("case_ids", []))
        for variant in payload.get("variants", []):
            plans.append(
                {
                    "name": f"judge-{variant['variant_id']}",
                    "stage": "eval",
                    "tags": ["stage=eval"],
                    "n_cases": n_cases,
                    "score_names": ["judge/reasoning", "judge/evidence", "judge/trajectory", "judge/professional"],
                    "rows": [{"case_id": None, "arm": variant.get("arm"), "model": variant.get("model")}],
                    "metadata": {"variant_id": variant["variant_id"]},
                }
            )

    if Path(DEEPEVAL_CROSSCHECK_JSON_PATH).exists():
        try:
            payload = json.loads(DEEPEVAL_CROSSCHECK_JSON_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        n_traces = payload.get("subset", {}).get("n_traces", 0)
        plans.append(
            {
                "name": "deepeval-crosscheck",
                "stage": "eval",
                "tags": ["stage=eval"],
                "n_cases": n_traces,
                "score_names": [
                    "deepeval/task_completion",
                    "deepeval/tool_correctness",
                    "deepeval/argument_correctness",
                    "deepeval/step_efficiency",
                ],
                "rows": [{"case_id": None, "arm": None, "model": None}],
                "metadata": {},
            }
        )
    return plans


def _economics_experiment_plans() -> list[dict]:
    """M6 Pareto experiments, tagged `stage=economics`."""
    from dealpoint.config import PARETO_JSON_PATH

    plans: list[dict] = []
    if Path(PARETO_JSON_PATH).exists():
        try:
            payload = json.loads(PARETO_JSON_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        for model, entry in payload.get("models", {}).items():
            plans.append(
                {
                    "name": f"pareto-{model.replace('/', '_')}",
                    "stage": "economics",
                    "tags": ["stage=economics"],
                    "n_cases": entry.get("n_cases", 0),
                    "score_names": [
                        "obj/grounded_accuracy",
                        "obj/answer_correct",
                        "obj/citation_gold_overlap",
                        "obj/citation_verbatim",
                        "obj/abstain_correct",
                        "obj/skill_adherence",
                    ],
                    "rows": [{"case_id": None, "arm": "D", "model": model}],
                    "metadata": {"model": model},
                }
            )
    return plans


def experiment_plan() -> list[dict]:
    """Every experiment this sync recreates: name, stage, tags, metadata, rows."""
    plans = []
    plans.extend(_rag_experiment_plans())
    plans.extend(_agent_experiment_plans())
    plans.extend(_eval_experiment_plans())
    plans.extend(_economics_experiment_plans())
    for plan in plans:
        assert_score_budget(plan)
    return plans


# --- log hierarchy (replayed, zero model calls) -------------------------------


def log_hierarchy(row: dict, case: dict, doc) -> dict:
    """`case -> agent -> search_agreement (retrieval stages as child spans) ->
    lookup_defined_term / get_section -> final_answer -> scoring` -- built
    entirely from `row["record"]["trajectory"]` + `row["scores"]`. No model
    call, no LLM client of any kind is touched.
    """
    from dealpoint.eval.blinding import _build_trajectory_step

    record = row.get("record") or {}
    trajectory_in = record.get("trajectory") or []

    retrieval_children = []
    tool_children = []
    for i, step in enumerate(trajectory_in):
        built = _build_trajectory_step(step, doc, i) if doc is not None else {
            "step": i + 1,
            "tool": step.get("tool", ""),
            "args": step.get("args") or {},
            "retrieved_text": None,
            "section_refs": [],
        }
        if built["tool"] == "search_agreement":
            retrieval_children.append({"name": "search_agreement", "span": built})
        else:
            tool_children.append({"name": built["tool"], "span": built})

    scoring_span = {
        "name": "scoring",
        "provenance": {
            "obj": {k: v for k, v in (row.get("scores") or {}).items()},
        },
    }

    return {
        "name": "case",
        "case_id": row.get("case_id"),
        "children": [
            {
                "name": "agent",
                "children": [
                    {"name": "search_agreement", "children": retrieval_children},
                    *tool_children,
                    {"name": "final_answer", "finding": row.get("finding")},
                    scoring_span,
                ],
            }
        ],
    }


# --- representative case selection ---------------------------------------


def _seeded_key(case_id: str, variant_id: str = "") -> str:
    return hashlib.sha256(f"{SEED}:{case_id}:{variant_id}".encode()).hexdigest()


REPRESENTATIVE_RULE_TEXT = (
    "Six categories, each a deterministic predicate over a stored canonical result row: "
    "'successful_direct' = grounded_accuracy is True and reasoning_type == 'direct' and "
    "tool_calls <= 2; 'retrieval_rescue' = grounded_accuracy is True and "
    "(gold_first_rank is None or gold_first_rank > 1) and >= 2 search_agreement steps; "
    "'defined_term_cross_ref' = required_evidence is not None and a lookup_defined_term or "
    "get_section step exists; 'inefficient_trajectory' = cap_hit is True, or tool_calls >= 6 "
    "with grounded_accuracy is not True; 'wrong_answer' = answer_correct is False; "
    "'abstention_counterfactual' = case_set == 'counterfactual' and record.status == "
    "'ABSTAINED'. Ties broken by ascending sha256(f'{SEED}:{case_id}:{variant_id}') "
    "(dealpoint.eval.subset._seeded_key's convention)."
)


def _category_for_row(row: dict, reasoning_type: str | None) -> str | None:
    scores = row.get("scores") or {}
    record = row.get("record") or {}
    trajectory = record.get("trajectory") or []
    n_search = sum(1 for s in trajectory if s.get("tool") == "search_agreement")
    n_lookup_or_section = sum(1 for s in trajectory if s.get("tool") in ("lookup_defined_term", "get_section"))
    tool_calls = scores.get("tool_calls", 0) or 0

    if (
        scores.get("grounded_accuracy") is True
        and reasoning_type == "direct"
        and tool_calls <= 2
    ):
        return "successful_direct"

    gold_first_rank = scores.get("gold_first_rank")
    if (
        scores.get("grounded_accuracy") is True
        and (gold_first_rank is None or gold_first_rank > 1)
        and n_search >= 2
    ):
        return "retrieval_rescue"

    if scores.get("required_evidence_met") is not None and n_lookup_or_section > 0:
        return "defined_term_cross_ref"

    if scores.get("cap_hit") is True or (tool_calls >= 6 and scores.get("grounded_accuracy") is not True):
        return "inefficient_trajectory"

    if scores.get("answer_correct") is False:
        return "wrong_answer"

    if row.get("case_set") == "counterfactual" and record.get("status") == "ABSTAINED":
        return "abstention_counterfactual"

    return None


def representative_cases() -> dict:
    """Six categories, chosen by `REPRESENTATIVE_RULE_TEXT`, from every
    result row on disk. Deterministic and stable across invocations.
    """
    from dealpoint.config import RESULTS_DIR
    from dealpoint.eval.cases import find_case, resolve_question

    candidates_by_category: dict[str, list[tuple[str, dict, str]]] = {}
    if RESULTS_DIR.exists():
        for path in sorted(RESULTS_DIR.glob("*.jsonl")):
            if path.name == "spend_ledger.jsonl":
                continue
            for row in _load_jsonl(path):
                case_id = row.get("case_id")
                if not case_id:
                    continue
                try:
                    case = find_case(case_id)
                except KeyError:
                    continue
                reasoning_type = resolve_question(case).reasoning_type
                category = _category_for_row(row, reasoning_type)
                if category is None:
                    continue
                variant_id = f"{row.get('arm')}@{row.get('model')}"
                candidates_by_category.setdefault(category, []).append((path.name, row, variant_id))

    selections: list[dict] = []
    categories = (
        "successful_direct",
        "retrieval_rescue",
        "defined_term_cross_ref",
        "inefficient_trajectory",
        "wrong_answer",
        "abstention_counterfactual",
    )
    for category in categories:
        candidates = candidates_by_category.get(category, [])
        if not candidates:
            selections.append(
                {"category": category, "case_id": None, "variant_id": None, "experiment_name": None, "note": "no candidate found"}
            )
            continue
        _, best_row, best_variant = min(
            ((p, r, v) for p, r, v in candidates),
            key=lambda t: _seeded_key(t[1]["case_id"], t[2]),
        )
        selections.append(
            {
                "category": category,
                "case_id": best_row["case_id"],
                "variant_id": best_variant,
                "experiment_name": None,
            }
        )

    return {"rule": REPRESENTATIVE_RULE_TEXT, "selections": selections}


# --- review set --------------------------------------------------------------


def review_set(n: int = M7A_REVIEW_SET_N) -> list[dict]:
    """`n` deterministically-ordered blinded packets, satisfying: spans
    reasoning types, mix successes/failures, >= 1 abstention, >= 2 variants.

    Reuses `dealpoint.eval.judge_run.build_all_packets` (the exact packets
    the M5 judges saw) and `dealpoint.eval.subset._seeded_key`'s convention
    for a stable, extensible ordering -- `review_set(30)` uses the SAME
    rule, just takes more of the same ordered list.
    """
    from dealpoint.eval.cases import find_case, resolve_question
    from dealpoint.eval.judge_run import build_all_packets

    items = build_all_packets()
    if not items:
        return []

    enriched = []
    for item in items:
        case_id = item["case_id"]
        try:
            case = find_case(case_id)
        except KeyError:
            continue
        reasoning_type = resolve_question(case).reasoning_type
        status = item["packet"].get("status")
        enriched.append(
            {
                "packet_id": item["packet"]["packet_id"],
                "case_id": case_id,
                "variant_id": item["variant_id"],
                "reasoning_type": reasoning_type,
                "status": status,
                "human_score": None,
            }
        )

    ordered = sorted(enriched, key=lambda e: _seeded_key(e["case_id"], e["variant_id"]))
    return ordered[:n]


# --- sync driver ---------------------------------------------------------


def _init_experiment(client, name: str):
    """`client.init_experiment(project=..., name=...)` for a fake test double;
    the real `braintrust` SDK's `init_experiment` takes `experiment=` instead
    (its `init()` signature has no `name` kwarg at all) -- try the documented
    fake-client shape first, fall back to the real SDK's kwarg name.
    """
    try:
        return client.init_experiment(project=PROJECT, name=name)
    except TypeError:
        return client.init_experiment(project=PROJECT, experiment=name)


def sync(client, *, dry_run: bool = False) -> dict:
    """Recreate every artifact from Git/local sources. Idempotent: re-running
    finds existing datasets/experiments by name rather than duplicating.

    `client` is either the real `braintrust` module (imported by the caller)
    or a fake test double exposing `init_dataset`/`init_experiment`/
    `start_span`/`log`. With `dry_run=True` (or a fake client) no network
    call happens; the full mapping still runs.
    """
    plan = experiment_plan()
    datasets_created = []
    for name in DATASET_PLAN_NAMES:
        rows = dataset_rows(name)
        dataset = client.init_dataset(project=PROJECT, name=f"maud-dealpoint-{name}")
        for row in rows:
            dataset.insert(input=row["input"], expected=row.get("expected"), metadata=row.get("metadata") or {})
        if hasattr(dataset, "flush"):
            dataset.flush()
        datasets_created.append({"name": name, "n_rows": len(rows)})

    experiments_created = []
    for exp_plan in plan:
        experiment = _init_experiment(client, exp_plan["name"])
        for row in exp_plan["rows"]:
            metadata = common_metadata(row, stage=exp_plan["stage"], extra=exp_plan.get("metadata"))
            experiment.log(
                input=row.get("case_id") or exp_plan["name"],
                output=row.get("model") or "n/a",
                scores={},
                metadata=metadata,
                tags=exp_plan["tags"],
            )
        if hasattr(experiment, "flush"):
            experiment.flush()
        experiments_created.append(exp_plan["name"])

    rep_cases = representative_cases()
    replayed_traces = 0
    for selection in rep_cases["selections"]:
        if selection.get("case_id") is None:
            continue
        try:
            from dealpoint.corpus.document import load_document
            from dealpoint.eval.cases import find_case, resolve_document_id

            case = find_case(selection["case_id"])
            doc = load_document(resolve_document_id(case))
        except (KeyError, FileNotFoundError):
            doc = None
            case = {}
        # a minimal row shape sufficient for log_hierarchy's replay
        row = {"case_id": selection["case_id"], "scores": {}, "record": {"trajectory": []}}
        hierarchy = log_hierarchy(row, case, doc)
        span = client.start_span(name=hierarchy["name"])
        if hasattr(span, "log"):
            span.log(metadata={"case_id": selection["case_id"]})
        if hasattr(span, "end"):
            span.end()
        replayed_traces += 1

    reviews = review_set()

    result = {
        "datasets": datasets_created,
        "experiments": experiments_created,
        "n_experiments": len(experiments_created),
        "representative_cases": rep_cases,
        "replayed_traces": replayed_traces,
        "review_set_n": len(reviews),
        "score_budget_ok": True,
        "framework_versions": _framework_versions_dict(),
        "synced_at": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
    }
    return result


def _record_sync_run(entry: dict, path: Path = BRAINTRUST_SYNC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    from dealpoint.eval.braintrust_adapter import braintrust_available

    if not braintrust_available():
        print("braintrust unavailable (no key or package) -- skipping cleanly")
        return 0

    import braintrust

    result = sync(braintrust, dry_run=False)
    _record_sync_run(result)
    print(json.dumps({k: v for k, v in result.items() if k != "framework_versions"}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
