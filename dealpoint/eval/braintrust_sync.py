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
import sys
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

# "procedure" was declared in an earlier version of this constant but never
# had any scores mapped to it -- there is no procedural score to log, and
# inventing one to fill a declared-but-empty namespace would be exactly the
# kind of placeholder the no-bloat rule forbids. Dropped rather than kept
# as dead vocabulary.
SCORE_NAMESPACES: tuple[str, ...] = ("obj", "li", "judge", "deepeval")

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


JUDGE_DIMENSION_SCALE_MIN = 1
JUDGE_DIMENSION_SCALE_MAX = 5


def _normalize_score_value(namespaced_name: str, value: float) -> float:
    """Braintrust requires every logged score in [0, 1]. Judge dimensions
    (`judge/reasoning`, `judge/evidence`, `judge/trajectory`,
    `judge/professional`) are a 1-5 Likert scale, not already 0-1 -- logging
    them raw fails the real SDK's `_validate_and_sanitize_experiment_log_partial_args`
    with "score values must be between 0 and 1" (hit live during this
    repair). Rescaled to `(value - 1) / 4`; the raw 1-5 value is preserved
    separately in metadata by callers, never silently discarded. Every
    other namespace (`obj/`, `li/`, `deepeval/`) is already 0-1 and passed
    through unchanged.
    """
    if namespaced_name.startswith("judge/"):
        return (value - JUDGE_DIMENSION_SCALE_MIN) / (JUDGE_DIMENSION_SCALE_MAX - JUDGE_DIMENSION_SCALE_MIN)
    return value


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


def _score_budget_limit(plan: dict) -> int:
    stage = plan.get("stage", "")
    n_cases = plan.get("n_cases", 0)
    if stage in ("m4_sweep", "m6_sweep"):
        return 6
    if n_cases <= M7A_MAX_SUBSET_FOR_WIDE_SCORES:
        return M7A_MAX_SCORES_PER_CASE
    return 6


def assert_score_budget(plan: dict) -> None:
    """M7 experiments <= 12 scores/case on subsets <= 60 cases; M4/M6 sweeps
    keep six (spec section 5.1). `plan` is one experiment-plan entry:
    `{name, stage, n_cases, score_names, ...}`.

    This checks the plan's DECLARED `score_names` list -- a cheap sanity
    check at plan-construction time. It is not sufficient on its own:
    `assert_actual_score_budget` below checks what `sync()` actually logs,
    since `_log_scores_for_row` previously namespaced every numeric/bool key
    in a stored row's `scores` dict regardless of what was declared.
    """
    n_scores = len(plan.get("score_names", []))
    n_cases = plan.get("n_cases", 0)
    stage = plan.get("stage", "")
    limit = _score_budget_limit(plan)

    if n_scores > limit:
        label = "M4/M6 sweeps must keep exactly six" if stage in ("m4_sweep", "m6_sweep") else (
            f"exceeds the {limit}-score budget for this subset size"
        )
        raise ScoreBudgetError(
            f"experiment {plan.get('name')!r} (stage {stage!r}) logs {n_scores} scores "
            f"on {n_cases} cases -- {label}"
        )


def assert_actual_score_budget(plan: dict, actual_score_names: set[str]) -> None:
    """Same budget, checked against the score names ACTUALLY logged for one
    experiment (the union across every row `sync()` logged) -- not the
    plan's declaration. This is what closes the gap the pre-repair
    `assert_score_budget` left open: `_log_scores_for_row` could log more
    names than `plan["score_names"]` declared, and this was never checked.
    """
    limit = _score_budget_limit(plan)
    n_actual = len(actual_score_names)
    if n_actual > limit:
        raise ScoreBudgetError(
            f"experiment {plan.get('name')!r} ACTUALLY logged {n_actual} distinct score "
            f"names ({sorted(actual_score_names)}) on {plan.get('n_cases', 0)} cases -- "
            f"exceeds the {limit}-score budget. Only plan['score_names'] may be logged "
            f"as scores; everything else must go to metadata."
        )


# --- datasets -----------------------------------------------------------------

DATASET_SETS: tuple[str, ...] = ("dev", "test", "counterfactual")


def _file_sha256(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _case_set_file_hash(set_name: str) -> str | None:
    from dealpoint.eval.cases import CASE_SET_PATHS

    path = CASE_SET_PATHS.get(set_name)
    return _file_sha256(path) if path is not None else None


def _dataset_row_metadata(row: dict, *, source_hashes: dict | None = None) -> dict:
    from dealpoint.eval.cases import resolve_question

    question = resolve_question(row)
    return {
        "question_id": row.get("question_id"),
        "agreement_id": row.get("agreement_id"),
        "reasoning_type": question.reasoning_type,
        "case_type": row.get("kind") or row.get("case_set"),
        "source_hashes": source_hashes or {},
    }


def _stable_dataset_row_id(dataset_name: str, input_value) -> str:
    """A deterministic row id, so a second live sync's `dataset.insert(...)`
    updates the SAME row instead of minting a fresh uuid every run (spec:
    datasets are "re-creatable ... by one command", which requires stable
    identity, not accumulation).
    """
    return hashlib.sha256(f"{dataset_name}:{input_value}".encode()).hexdigest()[:32]


def dataset_rows(set_name: str) -> list[dict]:
    """One dataset-shaped row per case, for a known set name.

    `set_name` may be one of `DATASET_SETS`, `"judged_calibration"` (the M5
    judged subset), or `"synthetic_query"` (the frozen synthetic set).
    Every row's metadata carries `source_hashes` -- the sha256 of the local
    file the row was derived from (spec section 3: dataset metadata
    "source hashes").
    """
    from dealpoint.config import JUDGED_SUBSET_PATH, SYNTHETIC_DEV_QUERIES_PATH
    from dealpoint.eval.cases import find_case, load_case_set, resolve_question

    if set_name in DATASET_SETS:
        rows = load_case_set(set_name)
        file_hash = _case_set_file_hash(set_name)
        return [
            {
                "input": row["case_id"],
                "expected": row.get("gold_answer"),
                "metadata": _dataset_row_metadata(row, source_hashes={"case_set_file_sha256": file_hash}),
            }
            for row in rows
        ]

    if set_name == "judged_calibration":
        if not Path(JUDGED_SUBSET_PATH).exists():
            return []
        payload = json.loads(Path(JUDGED_SUBSET_PATH).read_text(encoding="utf-8"))
        subset_file_hash = _file_sha256(JUDGED_SUBSET_PATH)
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
                    "metadata": _dataset_row_metadata(
                        case,
                        source_hashes={
                            "subset_hash": payload.get("subset_hash"),
                            "judged_subset_file_sha256": subset_file_hash,
                        },
                    ),
                }
            )
        return out

    if set_name == "synthetic_query":
        if not Path(SYNTHETIC_DEV_QUERIES_PATH).exists():
            return []
        file_hash = _file_sha256(SYNTHETIC_DEV_QUERIES_PATH)
        out = []
        with open(SYNTHETIC_DEV_QUERIES_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                case_id = row.get("case_id")
                reasoning_type = None
                case_type = None
                if case_id:
                    try:
                        case = find_case(case_id)
                        reasoning_type = resolve_question(case).reasoning_type
                        case_type = case.get("kind") or case.get("case_set")
                    except KeyError:
                        pass
                out.append(
                    {
                        "input": row["query_id"],
                        "expected": None,
                        "metadata": {
                            "question_id": row.get("question_id"),
                            "agreement_id": row.get("agreement_id"),
                            "case_id": case_id,
                            "reasoning_type": reasoning_type,
                            "case_type": case_type,
                            "source_hashes": {
                                "prompt_hash": row.get("prompt_hash"),
                                "synthetic_file_sha256": file_hash,
                            },
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


def _rag_per_case_rows(r: dict, *, config_name: str, index_version: str | None) -> list[dict]:
    """Real per-case rows for one retriever config, built from the LI/obj
    `per_case` arrays `dealpoint.rag_lab.report.build_report` persists
    (spec section 3: "RAG experiments get one row per dev case ... from
    canonical local rows -- never re-run"). `obj/gold_span_hit_at_5` and
    `_at_10` are derived from `first_hit_rank`, the same rule
    `dealpoint.eval.tournament` uses to compute the aggregate hit@k.
    """
    li_by_case = {row["case_id"]: row for row in (r.get("li", {}).get("per_case") or [])}
    obj_by_case = {row["case_id"]: row for row in (r.get("obj", {}).get("per_case") or [])}
    case_ids = sorted(set(li_by_case) | set(obj_by_case))
    rows = []
    for case_id in case_ids:
        li_row = li_by_case.get(case_id, {})
        obj_row = obj_by_case.get(case_id, {})
        rank = obj_row.get("first_hit_rank")
        scores = {}
        if "hit_rate" in li_row:
            scores["hit_rate"] = li_row["hit_rate"]
        if "mrr" in li_row:
            scores["mrr"] = li_row["mrr"]
        if case_id in obj_by_case:
            scores["gold_span_hit_at_5"] = rank is not None and rank <= 5
            scores["gold_span_hit_at_10"] = rank is not None and rank <= 10
        rows.append(
            {
                "case_id": case_id,
                "arm": None,
                "model": None,
                "index_version": index_version,
                "retriever": config_name,
                "scores": scores,
            }
        )
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
    rag_score_names = [
        "obj/gold_span_hit_at_5",
        "obj/gold_span_hit_at_10",
        "li/hit_rate",
        "li/mrr",
    ]

    li_report = {}
    if LI_RAG_EVAL_JSON_PATH.exists():
        try:
            li_report = json.loads(LI_RAG_EVAL_JSON_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            li_report = {}

    retrievers = li_report.get("retrievers", {})
    index_version = li_report.get("generated_from", {}).get("index_version")
    n_cases = li_report.get("generated_from", {}).get("n_cases", 0)
    for config_name, exp_name in name_by_config.items():
        r = retrievers.get(config_name)
        if r is None:
            continue
        rows = _rag_per_case_rows(r, config_name=config_name, index_version=index_version)
        if not rows:
            rows = [{"case_id": None, "arm": None, "model": None, "index_version": index_version, "retriever": config_name}]
        plans.append(
            {
                "name": exp_name,
                "stage": "rag",
                "tags": ["stage=rag"],
                "n_cases": len(rows) or n_cases,
                "score_names": rag_score_names,
                "rows": rows,
                "metadata": {"config": config_name},
            }
        )

    native = retrievers.get("li_native_bm25")
    if native is not None:
        crosscheck_rows = _rag_per_case_rows(native, config_name="li_native_bm25", index_version=index_version)
    else:
        crosscheck_rows = []
    if not crosscheck_rows:
        crosscheck_rows = [{"case_id": None, "arm": None, "model": None}]
    plans.append(
        {
            "name": "rag-m7-li-crosscheck",
            "stage": "rag",
            "tags": ["stage=rag"],
            "n_cases": len(crosscheck_rows),
            "score_names": rag_score_names,
            "rows": crosscheck_rows,
            "metadata": {"purpose": "crosscheck"},
        }
    )

    synth = li_report.get("synthetic")
    synth_results = (synth or {}).get("results") or {}
    synth_rows = [
        {
            "case_id": f"synthetic:{config_name}",
            "arm": None,
            "model": None,
            "index_version": index_version,
            "retriever": config_name,
            "scores": {"hit_rate": r2.get("li_hit_rate"), "gold_span_hit_at_5": r2.get("obj_hit_at_5")},
        }
        for config_name, r2 in synth_results.items()
    ]
    if not synth_rows:
        synth_rows = [{"case_id": None, "arm": None, "model": None}]
    plans.append(
        {
            "name": "rag-m7-synthetic",
            "stage": "rag",
            "tags": ["stage=rag"],
            "n_cases": (synth or {}).get("n_queries", 0),
            "score_names": ["obj/gold_span_hit_at_5", "li/hit_rate"],
            "rows": synth_rows,
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


def _judge_score_row(case_id: str, arm: str | None, model: str | None, dims: dict) -> dict:
    """One judge-experiment row, its `scores` dict already in the shape
    `common_metadata`/`_log_scores_for_row` expect (bare dimension names --
    `score_namespace` maps them to `judge/...`).
    """
    return {"case_id": case_id, "arm": arm, "model": model, "scores": dims}


def _eval_experiment_plans() -> list[dict]:
    """Judge-panel and DeepEval experiments, tagged `stage=eval`, carrying
    REAL per-trace scores (not placeholder rows) so BTQL investigations #3/#4
    have something to query.
    """
    from dealpoint.config import DEEPEVAL_CROSSCHECK_JSON_PATH, JUDGED_SUBSET_PATH

    plans: list[dict] = []
    if Path(JUDGED_SUBSET_PATH).exists():
        payload = json.loads(Path(JUDGED_SUBSET_PATH).read_text(encoding="utf-8"))
        case_ids = payload.get("case_ids", [])
        n_cases = len(case_ids)
        judge_rows = _load_jsonl(_judge_scores_path())
        for variant in payload.get("variants", []):
            variant_id = variant["variant_id"]
            rows = []
            for cid in case_ids:
                packet_id = _packet_id_for(cid, variant_id)
                dims = _mean_judge_dims_for_packet(judge_rows, packet_id)
                rows.append(
                    _judge_score_row(cid, variant.get("arm"), variant.get("model"), dims)
                )
            plans.append(
                {
                    "name": f"judge-{variant_id}",
                    "stage": "eval",
                    "tags": ["stage=eval"],
                    "n_cases": n_cases,
                    "score_names": ["judge/reasoning", "judge/evidence", "judge/trajectory", "judge/professional"],
                    "rows": rows,
                    "metadata": {"variant_id": variant_id},
                }
            )

    if Path(DEEPEVAL_CROSSCHECK_JSON_PATH).exists():
        try:
            payload = json.loads(DEEPEVAL_CROSSCHECK_JSON_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        per_trace_scores = payload.get("per_trace_scores", [])
        n_traces = payload.get("subset", {}).get("n_traces", len(per_trace_scores))

        det_by_case = _deterministic_scores_by_case_variant()
        rows = []
        for pts in per_trace_scores:
            case_id = pts.get("case_id")
            variant_id = pts.get("variant_id")
            scores: dict = {}
            for metric_name in ("task_completion", "tool_correctness", "argument_correctness", "step_efficiency"):
                v = (pts.get(metric_name) or {}).get("score")
                if v is not None:
                    scores[metric_name] = v
            det = det_by_case.get((case_id, variant_id)) or {}
            if det.get("grounded_accuracy") is not None:
                scores["obj/grounded_accuracy"] = det["grounded_accuracy"]
            rows.append({"case_id": case_id, "arm": None, "model": None, "scores": scores})

        if not rows:
            rows = [{"case_id": None, "arm": None, "model": None}]

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
                    "obj/grounded_accuracy",
                ],
                "rows": rows,
                "metadata": {},
            }
        )
    return plans


def _judge_scores_path():
    from dealpoint.config import JUDGE_SCORES_PATH

    return JUDGE_SCORES_PATH


def _packet_id_for(case_id: str, variant_id: str) -> str:
    from dealpoint.eval.blinding import _packet_id

    return _packet_id(case_id, variant_id)


def _mean_judge_dims_for_packet(judge_rows: list[dict], packet_id: str) -> dict:
    dims = ("reasoning", "evidence", "trajectory", "professional")
    out: dict = {}
    for dim in dims:
        vals = [
            r[dim]
            for r in judge_rows
            if r.get("packet_id") == packet_id and r.get("ok", True) and r.get(dim) is not None
        ]
        if vals:
            out[dim] = sum(vals) / len(vals)
    return out


def _judged_variant_id_for(arm: str | None, model: str | None) -> str | None:
    """Map a representative-case selection's `(arm, model)` (full model id,
    e.g. `A@z-ai/glm-5.3-flash`) to the SHORT `variant_id` the judged subset
    and DeepEval crosscheck use (e.g. `A@haiku`) -- the two conventions
    differ, so a direct string match on `variant_id` would silently miss
    every representative case's judge/deepeval provenance.
    """
    from dealpoint.config import JUDGED_SUBSET_PATH

    if not Path(JUDGED_SUBSET_PATH).exists() or arm is None or model is None:
        return None
    payload = json.loads(Path(JUDGED_SUBSET_PATH).read_text(encoding="utf-8"))
    for variant in payload.get("variants", []):
        if variant.get("arm") == arm and variant.get("model") == model:
            return variant.get("variant_id")
    return None


def _deepeval_scores_by_trace() -> dict[tuple[str, str], dict]:
    """`(case_id, variant_id) -> {metric_name: {score, reason}}` from the
    already-written `deepeval_crosscheck.json`'s `per_trace_scores` -- used
    to attach `deepeval/` provenance to replayed representative-trace
    scoring spans (spec section 3: "scoring spans carrying provenance
    (obj/, judge/, deepeval/)"). Empty if the crosscheck has not run yet.
    """
    from dealpoint.config import DEEPEVAL_CROSSCHECK_JSON_PATH

    if not Path(DEEPEVAL_CROSSCHECK_JSON_PATH).exists():
        return {}
    try:
        payload = json.loads(DEEPEVAL_CROSSCHECK_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[tuple[str, str], dict] = {}
    for row in payload.get("per_trace_scores", []):
        case_id = row.get("case_id")
        variant_id = row.get("variant_id")
        if case_id is None or variant_id is None:
            continue
        out[(case_id, variant_id)] = {
            k: v
            for k, v in row.items()
            if k in ("task_completion", "tool_correctness", "argument_correctness", "step_efficiency")
        }
    return out


def _deterministic_scores_by_case_variant() -> dict[tuple[str, str], dict]:
    """`(case_id, variant_id) -> {score_name: value}`, read from the M5
    judged subset's result files -- the same rows DeepEval scored.
    """
    from dealpoint.config import JUDGED_SUBSET_PATH

    if not Path(JUDGED_SUBSET_PATH).exists():
        return {}
    payload = json.loads(Path(JUDGED_SUBSET_PATH).read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict] = {}
    for variant in payload.get("variants", []):
        variant_id = variant["variant_id"]
        for row in _load_jsonl(variant["results_path"]):
            case_id = row.get("case_id")
            if case_id:
                out[(case_id, variant_id)] = row.get("scores") or {}
    return out


def _economics_experiment_plans() -> list[dict]:
    """M6 Pareto experiments, tagged `stage=economics` -- one row per case,
    sourced from the stored M6 result JSONLs named in `PARETO_MANIFEST_PATH`
    (spec section 3: "pareto experiments get one row per case from the
    stored M6 result JSONLs ... no re-runs, no model calls").
    """
    from dealpoint.config import PARETO_JSON_PATH, PARETO_MANIFEST_PATH

    score_names = [
        "obj/grounded_accuracy",
        "obj/answer_correct",
        "obj/citation_gold_overlap",
        "obj/citation_verbatim",
        "obj/abstain_correct",
        "obj/skill_adherence",
    ]

    manifest_by_model: dict[str, dict] = {}
    if Path(PARETO_MANIFEST_PATH).exists():
        try:
            entries = json.loads(Path(PARETO_MANIFEST_PATH).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = []
        for entry in entries:
            manifest_by_model[entry["model"]] = entry

    plans: list[dict] = []
    if Path(PARETO_JSON_PATH).exists():
        try:
            payload = json.loads(PARETO_JSON_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        for model, entry in payload.get("models", {}).items():
            manifest_entry = manifest_by_model.get(model)
            rows: list[dict] = []
            if manifest_entry and manifest_entry.get("results_path"):
                for r in _load_jsonl(manifest_entry["results_path"]):
                    rows.append(
                        {
                            "case_id": r.get("case_id"),
                            "arm": r.get("arm") or manifest_entry.get("arm"),
                            "model": model,
                            "index_version": r.get("index_version") or manifest_entry.get("index_version"),
                            "scores": r.get("scores") or {},
                        }
                    )
            if not rows:
                rows = [{"case_id": None, "arm": "D", "model": model}]
            plans.append(
                {
                    "name": f"pareto-{model.replace('/', '_')}",
                    "stage": "economics",
                    "tags": ["stage=economics"],
                    "n_cases": len(rows),
                    "score_names": score_names,
                    "rows": rows,
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


def _retrieval_stage_children(arm: str | None) -> list[dict]:
    """Deterministic retrieval-stage breakdown for one `search_agreement`
    call, derived from the arm's static retriever config (spec section 3:
    "search_agreement (retrieval stages as child spans)"). This is a
    structural breakdown of which stages a call of this retriever kind
    runs -- dense/bm25/rrf/rerank/multi-query-fusion -- not a re-run of the
    retrieval itself: no index is touched, no scores are attached, only the
    static pipeline shape from `dealpoint.config.ARMS`.
    """
    from dealpoint.config import ARM_C_RETRIEVER, ARMS

    retriever_cfg = (ARMS.get(arm or "", {}) or {}).get("retriever") or (
        ARM_C_RETRIEVER if arm == "C" else None
    )
    if not retriever_cfg:
        return []
    kind = retriever_cfg.get("kind")
    stages: list[str] = []
    if retriever_cfg.get("multi_query"):
        stages.append("multi_query_fusion")
    if kind == "dense":
        stages.append("dense")
    elif kind == "bm25":
        stages.append("bm25")
    elif kind == "hybrid_rrf":
        stages.extend(["dense", "bm25", "rrf"])
    if retriever_cfg.get("rerank_model"):
        stages.append("rerank")
    return [{"name": stage, "children": []} for stage in stages]


def log_hierarchy(
    row: dict,
    case: dict,
    doc,
    *,
    judge_dims: dict | None = None,
    deepeval_scores: dict | None = None,
) -> dict:
    """`case -> agent -> search_agreement (retrieval stages as child spans) ->
    lookup_defined_term / get_section -> final_answer -> scoring` -- built
    entirely from `row["record"]["trajectory"]` + `row["scores"]` (+ the
    arm's static retriever config for the retrieval-stage breakdown, and
    the caller-supplied `judge_dims`/`deepeval_scores` for the scoring
    span's cross-framework provenance). No model call, no LLM client of any
    kind is touched.
    """
    from dealpoint.eval.blinding import _build_trajectory_step

    record = row.get("record") or {}
    trajectory_in = record.get("trajectory") or []
    stage_children = _retrieval_stage_children(row.get("arm"))

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
            retrieval_children.append({"name": "search_agreement", "span": built, "children": stage_children})
        else:
            tool_children.append({"name": built["tool"], "span": built})

    provenance: dict = {"obj": {k: v for k, v in (row.get("scores") or {}).items()}}
    if judge_dims:
        provenance["judge"] = {k: v for k, v in judge_dims.items() if v is not None}
    if deepeval_scores:
        provenance["deepeval"] = {
            k: v.get("score") if isinstance(v, dict) else v
            for k, v in deepeval_scores.items()
            if (v.get("score") if isinstance(v, dict) else v) is not None
        }
    scoring_span = {"name": "scoring", "provenance": provenance}

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
                candidates_by_category.setdefault(category, []).append((str(path), row, variant_id))

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
        best_path, best_row, best_variant = min(
            ((p, r, v) for p, r, v in candidates),
            key=lambda t: _seeded_key(t[1]["case_id"], t[2]),
        )
        selections.append(
            {
                "category": category,
                "case_id": best_row["case_id"],
                "variant_id": best_variant,
                "experiment_name": None,
                "results_path": best_path,
            }
        )

    return {"rule": REPRESENTATIVE_RULE_TEXT, "selections": selections}


# --- review set --------------------------------------------------------------


def review_set(n: int = M7A_REVIEW_SET_N) -> list[dict]:
    """`n` deterministically-ordered blinded packets, ENFORCING BY CONSTRUCTION
    (not by luck of the current data) the diversity rule: spans reasoning
    types, mixes successes/failures, >= 1 abstention, >= 2 variants.

    Reuses `dealpoint.eval.judge_run.build_all_packets` (the exact packets
    the M5 judges saw) and `dealpoint.eval.subset._seeded_key`'s convention
    for a stable, extensible ordering -- `review_set(30)` uses the SAME
    rule, just takes more of the same ordered list (this function's own
    prefix-stability test asserts `review_set(12) == review_set(30)[:12]`).

    Construction: seed with one packet per reasoning type (in seeded
    order), then one abstention if none is present yet, then fill the rest
    in seeded order. If fewer than 2 distinct variants would result (only
    possible on a degenerate/tiny fixture), one more seeded-order packet
    from an unseen variant is added. This guarantees the rule holds
    whenever the underlying data has enough diversity to satisfy it at
    all, rather than holding only incidentally for today's case mix.

    `packet_text` (the same blinded text the M5 judges and the rubric
    calibration reader see -- arm/model identity never appears in it) is
    included so the engineer has something to actually read in Braintrust
    Review; `variant_id` is kept ONLY in the LOCAL result (never pushed to
    the synced dataset -- see `sync()`), so the review set stays blinded
    once it lands in Braintrust.
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
                "packet_text": item["packet_text"],
                "case_id": case_id,
                "variant_id": item["variant_id"],
                "reasoning_type": reasoning_type,
                "status": status,
                "human_score": None,
            }
        )

    ordered = sorted(enriched, key=lambda e: _seeded_key(e["case_id"], e["variant_id"]))
    if len(ordered) <= n:
        return ordered

    candidate = ordered[:n]
    if _satisfies_review_set_rule(candidate):
        return candidate
    return _construct_diverse_review_subset(ordered, n)


def _satisfies_review_set_rule(candidate: list[dict]) -> bool:
    """Check the diversity rule (spec section 3: "spans reasoning types,
    successes/failures, >= 1 abstention, >= 2 judged variants") against an
    already-chosen candidate list -- the verification `review_set` runs
    before falling back to explicit construction.
    """
    if len({e["reasoning_type"] for e in candidate}) < 2:
        return False
    if len({e["variant_id"] for e in candidate}) < 2:
        return False
    return any(e["status"] == "ABSTAINED" for e in candidate)


def _construct_diverse_review_subset(ordered: list[dict], n: int) -> list[dict]:
    """Explicit fallback construction, used only when the plain seeded-order
    prefix does not already satisfy `_satisfies_review_set_rule` (spec:
    enforce the rule "by construction", not incidentally). Seeds one packet
    per reasoning type, then an abstention if still missing, then fills the
    rest in seeded order; a final pass swaps in a second variant if the
    result still has only one. Re-sorted back into seeded order at the end
    so the result is still a well-defined, order-stable subset.
    """
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def _take(entry: dict) -> None:
        if entry["packet_id"] not in selected_ids:
            selected.append(entry)
            selected_ids.add(entry["packet_id"])

    seen_reasoning_types: set[str] = set()
    for entry in ordered:
        if len(selected) >= n:
            break
        if entry["reasoning_type"] not in seen_reasoning_types:
            seen_reasoning_types.add(entry["reasoning_type"])
            _take(entry)

    if not any(e["status"] == "ABSTAINED" for e in selected):
        abstention = next((e for e in ordered if e["status"] == "ABSTAINED"), None)
        if abstention is not None and len(selected) < n:
            _take(abstention)

    for entry in ordered:
        if len(selected) >= n:
            break
        _take(entry)

    if len({e["variant_id"] for e in selected}) < 2:
        seen_variants = {e["variant_id"] for e in selected}
        extra = next((e for e in ordered if e["variant_id"] not in seen_variants), None)
        if extra is not None:
            selected[-1] = extra

    selected_by_id = {e["packet_id"]: e for e in selected}
    return [e for e in ordered if e["packet_id"] in selected_by_id][:n]


# --- scorers / prompts / parameters -------------------------------------

SCORER_PLAN: tuple[dict, ...] = (
    {"name": "grounded_accuracy", "threshold": 1.0},
    {"name": "answer_correct", "threshold": 1.0},
    {"name": "citation_verbatim", "threshold": 1.0},
    {"name": "citation_gold_overlap", "threshold": 1.0},
    {"name": "required_evidence_met", "threshold": 1.0},
    # skill_adherence is a fraction, not a boolean -- no invented threshold
    # (spec section 3(c): "thresholds only on the booleans").
    {"name": "skill_adherence", "threshold": None},
)


def scorer_plan() -> list[dict]:
    """Braintrust scorer registrations: name, canonical import path, threshold.

    Each entry names the exact `dealpoint.eval.scorers` function the real
    scorer handler imports and calls -- never a re-implementation.
    """
    return [
        {
            "name": s["name"],
            "slug": s["name"].replace("_", "-"),
            "import_path": f"dealpoint.eval.scorers.{s['name']}",
            "threshold": s["threshold"],
        }
        for s in SCORER_PLAN
    ]


def _make_scorer_handler(fn_name: str):
    """A Braintrust-shaped scorer handler that imports and calls the named
    canonical `dealpoint.eval.scorers` function -- never a re-implementation.

    Braintrust scorer handlers see `(input, output, expected, metadata)`;
    the canonical scorers take `(case, finding, record, canonical_text)`.
    The bridge reconstructs those from `output` (the replayed row: the same
    shape `dealpoint.eval.braintrust_adapter._row_to_eval_case` already
    produces) and `metadata` (which carries `case_id`).
    """

    def handler(input=None, output=None, expected=None, metadata=None):
        import dealpoint.eval.scorers as scorers_mod
        from dealpoint.agent.schema import ExecutionRecord, Finding
        from dealpoint.corpus.document import load_document
        from dealpoint.eval.cases import find_case, resolve_document_id

        fn = getattr(scorers_mod, fn_name)
        payload = output or {}
        finding_payload = payload.get("finding")
        record_payload = payload.get("record") or {"status": "EXECUTION_FAILED"}
        case_id = (metadata or {}).get("case_id") or payload.get("case_id")
        if not case_id:
            return None
        case = find_case(case_id)
        doc = load_document(resolve_document_id(case))
        finding = Finding.model_validate(finding_payload) if finding_payload else None
        record = ExecutionRecord.model_validate(record_payload)
        if fn_name in ("required_evidence_met", "skill_adherence"):
            return fn(case, finding, record, doc.text, doc=doc)
        return fn(case, finding, record, doc.text)

    handler.__name__ = fn_name
    return handler


def prompt_plan() -> list[dict]:
    """Base agent instructions, arm-D skill injection, judge rubric -- each
    with its source hash. Canonical ownership stays in Git; this is a mirror.
    """
    from dealpoint.agent.prompts import system_prompt
    from dealpoint.agent.skill import load_skill, skill_version
    from dealpoint.eval.rubric import rubric_text, rubric_version

    return [
        {
            "name": "agent-base-system-prompt",
            "slug": "agent-base-system-prompt",
            "content": system_prompt(),
            "source_hash": hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()[:12],
            "version_label": None,
        },
        {
            "name": "agent-arm-d-skill-injection",
            "slug": "agent-arm-d-skill-injection",
            "content": load_skill(),
            "source_hash": skill_version(),
            "version_label": f"skill_version {skill_version()}",
        },
        {
            "name": "judge-calibrated-rubric",
            "slug": "judge-calibrated-rubric",
            "content": rubric_text(),
            "source_hash": rubric_version(),
            "version_label": f"rubric_version {rubric_version()}",
        },
    ]


def parameters_schema() -> dict:
    """One compact versioned schema of non-secret runtime parameters, built
    by EXPOSING existing config (never re-declaring it).

    Compact: arm names and their retriever KIND only (not the full nested
    `ARMS`/`ARM_C_RETRIEVER` config dicts, which duplicate detail already
    canonical in `dealpoint/config.py` and bloat this schema well past
    "compact"). Includes `skill_version` and `model_alias` -- the two
    parameters the spec names that the pre-repair schema omitted.
    """
    from dealpoint.config import ARMS, MAX_TOOL_CALLS, PARETO_JSON_PATH, RETRIEVER_DEFAULT_K

    arm_retrievers = {
        arm_name: (arm_cfg.get("retriever") or {}).get("name")
        for arm_name, arm_cfg in ARMS.items()
    }

    model_alias: list[str] = []
    if Path(PARETO_JSON_PATH).exists():
        try:
            model_alias = sorted(json.loads(PARETO_JSON_PATH.read_text(encoding="utf-8")).get("models", {}))
        except (OSError, json.JSONDecodeError):
            model_alias = []

    return {
        "version": 2,
        "arm": sorted(ARMS.keys()),
        "arm_retrievers": arm_retrievers,
        "top_k": RETRIEVER_DEFAULT_K,
        "max_tool_calls": MAX_TOOL_CALLS,
        "skill_version": _skill_version(),
        "model_alias": model_alias,
        "dataset_split": ["dev", "test", "counterfactual"],
        "mode": ["fake", "live"],
    }


# --- tools decision (documented, no stubs) -----------------------------

TOOLS_DECISION = {
    "exposed": False,
    "reason": (
        "search_agreement/get_section/lookup_defined_term each need the 68 MB "
        "Qdrant index (data/index/), the 64 MB derived canonical corpus "
        "(data/derived/) and fastembed's ONNX embedding weights loaded "
        "in-process. None of this can run inside Braintrust's function "
        "execution environment without duplicating this project's entire "
        "index/data layer -- exactly what the no-bloat rule and the brief's "
        "'no second tracing/data architecture' guidance forbid. Documented "
        "decision, not an omission; no placeholder stubs are shipped."
    ),
}


# --- sync driver ---------------------------------------------------------


# --- production dry-run client: exercises every code path with zero network ---


class _DryRunDataset:
    def __init__(self, name: str) -> None:
        self.name = name
        self.rows: list[dict] = []

    def insert(self, input=None, expected=None, metadata=None, id=None):
        self.rows.append({"input": input, "expected": expected, "metadata": metadata, "id": id})
        return id or f"row-{len(self.rows)}"

    def flush(self) -> None:
        pass


class _DryRunSpan:
    def __init__(self, name: str | None) -> None:
        self.name = name
        self.logged: list[dict] = []
        self.children: list[_DryRunSpan] = []

    def log(self, **kwargs) -> None:
        self.logged.append(kwargs)

    def start_span(self, name: str | None = None):
        child = _DryRunSpan(name)
        self.children.append(child)
        return child

    def end(self) -> None:
        pass


class _DryRunExperiment:
    def __init__(self, name: str) -> None:
        self.name = name
        self.logged: list[dict] = []
        self.spans: list[_DryRunSpan] = []

    def log(self, **kwargs) -> None:
        self.logged.append(kwargs)

    def start_span(self, name: str | None = None) -> _DryRunSpan:
        span = _DryRunSpan(name)
        self.spans.append(span)
        return span

    def flush(self) -> None:
        pass


class _DryRunScorers:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def create(self, **kwargs) -> None:
        self.created.append(kwargs)


class _DryRunPrompts:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def create(self, **kwargs) -> None:
        self.created.append(kwargs)


class _DryRunParameters:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def create(self, **kwargs) -> None:
        self.created.append(kwargs)


class _DryRunProject:
    def __init__(self, name: str) -> None:
        self.name = name
        self.scorers = _DryRunScorers()
        self.prompts = _DryRunPrompts()
        self.parameters = _DryRunParameters()

    def publish(self) -> None:
        pass


class _DryRunProjects:
    def __init__(self) -> None:
        self._projects: dict[str, _DryRunProject] = {}

    def create(self, name: str) -> _DryRunProject:
        if name not in self._projects:
            self._projects[name] = _DryRunProject(name)
        return self._projects[name]


class _DryRunClient:
    """A same-shape stand-in for the real `braintrust` module that never
    calls the network -- used by `main(["--dry-run"])` so
    `data/reports/braintrust_sync.json` can be regenerated from the CURRENT
    code (including the `scorers`/`prompts`/`parameters`/`tools`/
    `review_set` keys and the real-SDK-shaped branches in
    `_sync_scorers`/`_sync_prompts`/`_sync_parameters`, since it exposes
    `.projects` just like the real module does) without spending any of the
    Braintrust workspace's quota. This is a PRODUCTION dry-run path, not a
    test double -- `sync(client, dry_run=True)`'s own `dry_run` flag is
    otherwise just a label on the result dict and does not, by itself, gate
    any network call; using this client is what actually prevents one.
    """

    def __init__(self) -> None:
        self.datasets: dict[str, _DryRunDataset] = {}
        self.experiments: dict[str, _DryRunExperiment] = {}
        self.projects = _DryRunProjects()

    def init_dataset(self, project: str, name: str) -> _DryRunDataset:
        if name not in self.datasets:
            self.datasets[name] = _DryRunDataset(name)
        return self.datasets[name]

    def init_experiment(self, project: str, experiment: str, update: bool = True) -> _DryRunExperiment:
        if experiment not in self.experiments:
            self.experiments[experiment] = _DryRunExperiment(experiment)
        return self.experiments[experiment]


def _init_experiment(client, name: str):
    """`client.init_experiment(project=..., name=...)` for a fake test double;
    the real `braintrust` SDK's `init_experiment` takes `experiment=` instead
    (its `init()` signature has no `name` kwarg at all) -- try the documented
    fake-client shape first, fall back to the real SDK's kwarg name.

    Passes `update=True` on the real-SDK path so a name collision continues
    logging to the existing experiment (idempotent re-sync) rather than the
    SDK's default of minting a new one on every call -- `braintrust.init`
    documents `update`: "If the experiment already exists, continue logging
    to it." `set_current=True` (the SDK default) is what lets a later
    module-level `braintrust.start_span()` land under this experiment
    instead of `_NoopSpan` (see `sync()`'s replay section).
    """
    try:
        return client.init_experiment(project=PROJECT, name=name)
    except TypeError:
        return client.init_experiment(project=PROJECT, experiment=name, update=True)


def _log_scores_for_row(row: dict, allowed_score_names: set[str] | None = None) -> tuple[dict, dict]:
    """Namespace a stored row's `scores` dict through `score_namespace`.

    Returns `(scores_to_log, secondary_diagnostics)`. When `allowed_score_names`
    is given (the experiment plan's declared, budget-checked list), only
    those namespaced names are returned as scores -- every other
    numeric/bool key is a "secondary diagnostic" (spec section 3: "Secondary
    diagnostics go in metadata") and is returned separately rather than
    logged as a score. `allowed_score_names=None` keeps the old
    keep-everything-as-a-score behaviour, for callers (tests) that do not
    care about the budget split. None-valued scores are always dropped,
    never coerced to 0/False.
    """
    scores = row.get("scores") or {}
    scores_to_log: dict = {}
    secondary: dict = {}
    for name, value in scores.items():
        if value is None:
            continue
        if isinstance(value, bool):
            numeric = 1.0 if value else 0.0
        elif isinstance(value, int | float):
            numeric = float(value)
        else:
            continue
        namespaced = score_namespace(name)
        normalized = _normalize_score_value(namespaced, numeric)
        if allowed_score_names is None or namespaced in allowed_score_names:
            scores_to_log[namespaced] = normalized
        else:
            secondary[namespaced] = normalized
    return scores_to_log, secondary


def sync(client, *, dry_run: bool = False) -> dict:
    """Recreate every artifact from Git/local sources. Idempotent: re-running
    finds existing datasets/experiments/scorers/prompts/parameters by name
    rather than duplicating.

    `client` is either the real `braintrust` module (imported by the caller)
    or a fake test double exposing `init_dataset`/`init_experiment`/
    `start_span`/`log`/`projects`. With `dry_run=True` (or a fake client) no
    network call happens; the full mapping still runs.
    """
    plan = experiment_plan()
    datasets_created = []
    for name in DATASET_PLAN_NAMES:
        dataset_name = f"maud-dealpoint-{name}"
        rows = dataset_rows(name)
        dataset = client.init_dataset(project=PROJECT, name=dataset_name)
        for row in rows:
            row_id = _stable_dataset_row_id(dataset_name, row["input"])
            try:
                dataset.insert(
                    input=row["input"], expected=row.get("expected"), metadata=row.get("metadata") or {}, id=row_id
                )
            except TypeError:
                # fake test double's `insert` may not accept `id=`
                dataset.insert(input=row["input"], expected=row.get("expected"), metadata=row.get("metadata") or {})
        if hasattr(dataset, "flush"):
            dataset.flush()
        datasets_created.append({"name": name, "n_rows": len(rows)})

    experiments_created = []
    for exp_plan in plan:
        experiment = _init_experiment(client, exp_plan["name"])
        allowed = set(exp_plan.get("score_names", []))
        actual_score_names: set[str] = set()
        for i, row in enumerate(exp_plan["rows"]):
            metadata = common_metadata(row, stage=exp_plan["stage"], extra=exp_plan.get("metadata"))
            scores_to_log, secondary = _log_scores_for_row(row, allowed_score_names=allowed)
            if secondary:
                metadata["secondary_diagnostics"] = secondary
            actual_score_names.update(scores_to_log.keys())
            # A stable id (mirroring the dataset rows' `_stable_dataset_row_id`)
            # so a second sync's `experiment.log(...)` UPDATES this row rather
            # than appending a duplicate -- the real SDK's `Experiment.log`
            # accepts `id=` for exactly this purpose.
            row_id = _stable_dataset_row_id(
                exp_plan["name"], row.get("case_id") or f"{exp_plan['name']}:{i}"
            )
            experiment.log(
                input=row.get("case_id") or exp_plan["name"],
                output=row.get("model") or "n/a",
                scores=scores_to_log,
                metadata=metadata,
                tags=exp_plan["tags"],
                id=row_id,
            )
        assert_actual_score_budget(exp_plan, actual_score_names)
        if hasattr(experiment, "flush"):
            experiment.flush()
        experiments_created.append(exp_plan["name"])

    # --- replay representative traces from their ACTUAL stored rows ------
    # Spans are rooted on a DEDICATED experiment (never on the bare `client`/
    # `braintrust` module) so a real sync's `braintrust.start_span()` calls
    # attach to a current experiment instead of silently landing on
    # `_NoopSpan` -- `_init_experiment`'s `set_current=True` default is what
    # makes this experiment "current" for the real SDK.
    replay_experiment = _init_experiment(client, "m7-representative-traces")

    from dealpoint.eval.deepeval_adapter import load_judge_dimension_means

    judge_means_by_trace = load_judge_dimension_means()
    deepeval_scores_by_trace = _deepeval_scores_by_trace()

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

        results_path = selection.get("results_path")
        row = None
        if results_path:
            for r in _load_jsonl(results_path):
                if r.get("case_id") == selection["case_id"]:
                    row = r
                    break
        if row is None:
            # No stored row found (e.g. offline/fixture data) -- fall back
            # to an empty-trajectory row rather than crashing; this never
            # invents a model call, only degrades what gets replayed.
            row = {"case_id": selection["case_id"], "scores": {}, "record": {"trajectory": []}}

        judged_variant_id = _judged_variant_id_for(row.get("arm"), row.get("model"))
        judge_dims = None
        deepeval_scores = None
        if judged_variant_id is not None:
            trace_key = (selection["case_id"], judged_variant_id)
            judge_dims = judge_means_by_trace.get(trace_key)
            deepeval_scores = deepeval_scores_by_trace.get(trace_key)
        hierarchy = log_hierarchy(
            row,
            case,
            doc,
            judge_dims=judge_dims,
            deepeval_scores=deepeval_scores,
        )
        root_span = replay_experiment.start_span(name=hierarchy.get("name", "span"))
        if type(root_span).__name__ == "_NoopSpan":
            raise RuntimeError(
                "replay span landed on _NoopSpan -- the replay experiment was not made "
                "current; a real sync would silently discard every replayed trace"
            )
        if hasattr(root_span, "log"):
            root_span.log(metadata={"case_id": hierarchy.get("case_id")})
        for child in hierarchy.get("children", []):
            _emit_span_tree(replay_experiment, child, parent_span=root_span)
        if hasattr(root_span, "end"):
            root_span.end()
        replayed_traces += 1
    if hasattr(replay_experiment, "flush"):
        replay_experiment.flush()

    # --- scorers / prompts / parameters -----------------------------------
    scorers_result = _sync_scorers(client)
    prompts_result = _sync_prompts(client)
    parameters_result = _sync_parameters(client)

    # --- 12-trace blinded review set: pushed as a scoreable dataset -------
    # `variant_id` (which arm/model produced this trace) is DELIBERATELY
    # never pushed to Braintrust here -- it would un-blind the set the
    # engineer is about to hand-score. It stays in the local `reviews` list
    # returned to `main()`/tests only. `packet_text` (the same blinded body
    # judges/humans see) IS pushed as `expected` so there is something to
    # actually read in Braintrust Review.
    reviews = review_set()
    review_dataset_name = "maud-dealpoint-review-set"
    review_dataset = client.init_dataset(project=PROJECT, name=review_dataset_name)
    for packet in reviews:
        row_id = _stable_dataset_row_id(review_dataset_name, packet["packet_id"])
        metadata = {
            "case_id": packet["case_id"],
            "reasoning_type": packet["reasoning_type"],
            "status": packet["status"],
            "human_score": packet["human_score"],
        }
        try:
            review_dataset.insert(
                input=packet["packet_id"], expected=packet.get("packet_text"), metadata=metadata, id=row_id
            )
        except TypeError:
            review_dataset.insert(input=packet["packet_id"], expected=packet.get("packet_text"), metadata=metadata)
    if hasattr(review_dataset, "flush"):
        review_dataset.flush()

    result = {
        "datasets": datasets_created,
        "experiments": experiments_created,
        "n_experiments": len(experiments_created),
        "representative_cases": rep_cases,
        "replayed_traces": replayed_traces,
        "review_set": {"n": len(reviews), "dataset": "maud-dealpoint-review-set"},
        "review_set_n": len(reviews),
        "scorers": scorers_result,
        "prompts": prompts_result,
        "parameters": parameters_result,
        "tools": TOOLS_DECISION,
        "score_budget_ok": True,
        "framework_versions": _framework_versions_dict(),
        "synced_at": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
    }
    return result


def _emit_span_tree(root_span_source, node: dict, parent_span=None) -> None:
    """Recursively emit `node` (from `log_hierarchy`) as nested spans on
    `root_span_source`/`parent_span`. Zero model calls -- every field comes
    from the already-built hierarchy dict.

    `root_span_source` provides the top-level `start_span` call only (used
    when `parent_span is None`); every recursive call already has a
    `parent_span` and starts a CHILD span on it instead. For the real SDK,
    `root_span_source` must be an experiment/logger that was made "current"
    (`_init_experiment`'s `set_current=True` default) -- calling
    `braintrust.start_span()` with no current experiment/logger returns a
    `_NoopSpan` that is silently discarded, which is why the module-level
    `braintrust` object must never be passed here as the root source for a
    real sync (see `sync()`'s replay section, which asserts against this).
    """
    name = node.get("name", "span")
    log_kwargs: dict = {}
    if "case_id" in node:
        log_kwargs["metadata"] = {"case_id": node["case_id"]}
    if "span" in node:
        log_kwargs.setdefault("metadata", {})
        log_kwargs["metadata"]["span"] = node["span"]
    if "finding" in node:
        log_kwargs.setdefault("metadata", {})
        log_kwargs["metadata"]["finding"] = node["finding"]
    if "provenance" in node:
        scores: dict = {}
        excluded: dict = {}
        for namespace_key in ("obj", "judge", "deepeval"):
            for k, v in (node["provenance"].get(namespace_key) or {}).items():
                if v is None or not isinstance(v, bool | int | float):
                    continue
                namespaced = score_namespace(k)
                normalized = _normalize_score_value(namespaced, float(v))
                # Braintrust requires every logged score in [0, 1]. Most
                # `obj/` provenance values are booleans/fractions and are
                # already in range; a few (tool_calls, wall_ms, token
                # counts, ...) are raw counts -- those are not scores at
                # all and go to span metadata instead of being dropped.
                if 0.0 <= normalized <= 1.0:
                    scores[namespaced] = normalized
                else:
                    excluded[namespaced] = normalized
        log_kwargs["scores"] = scores
        if excluded:
            log_kwargs.setdefault("metadata", {})
            log_kwargs["metadata"]["provenance_excluded_from_scores"] = excluded

    if parent_span is not None and hasattr(parent_span, "start_span"):
        span = parent_span.start_span(name=name)
    else:
        span = root_span_source.start_span(name=name)

    if hasattr(span, "log") and log_kwargs:
        span.log(**log_kwargs)

    for child in node.get("children", []):
        _emit_span_tree(root_span_source, child, parent_span=span)

    if hasattr(span, "end"):
        span.end()


SCORER_PUBLISH_LIMITATION = (
    "braintrust 0.37.0's own `Project.publish()` refuses code functions outright "
    "(\"Code functions cannot be published directly. Use `braintrust push` instead.\" -- "
    "verified by reading Project.publish's source in this venv) and posts only prompts/"
    "parameters via the insert-functions API. `project.scorers.create(...)` therefore "
    "produces a local CodeFunction DECLARATION, not a registered Braintrust scorer -- "
    "actually registering it requires the separate `braintrust push` CLI, which loads "
    "and executes this module server-side and is out of scope for a same-process sync "
    "call. Tested and rejected per the no-bloat rule's \"tested-and-rejected features are "
    "documented, not hidden\": this sync declares the six scorer handlers (proving the "
    "canonical-import bridge works, see test_make_scorer_handler_*) but does not claim "
    "they are live in Braintrust. Prompts and parameters ARE actually published below via "
    "Project.publish(), which the SDK does support for those two function types."
)


def _sync_scorers(client) -> dict:
    """Declare the canonical scorers as Braintrust CodeFunctions.

    Uses `client.projects.create(name=...).scorers.create(...)` when the
    real SDK shape is available (a `braintrust` module); a fake test double
    exposes the simpler `register_scorer(name, ...)` shape instead. See
    `SCORER_PUBLISH_LIMITATION`: this declares but does not publish/register
    the scorers against the live Braintrust project.
    """
    plan = scorer_plan()
    if hasattr(client, "projects"):
        project = client.projects.create(name=PROJECT)
        created = []
        for s in plan:
            handler = _make_scorer_handler(s["name"])
            kwargs: dict = {
                "name": s["name"],
                "slug": s["slug"],
                "handler": handler,
                "parameters": {"input": object, "output": object, "expected": object, "metadata": object},
                "if_exists": "replace",
            }
            project.scorers.create(**kwargs)
            created.append(s["slug"])
        return {
            "registered": created,
            "n": len(created),
            "plan": plan,
            "published": False,
            "limitation": SCORER_PUBLISH_LIMITATION,
        }
    # fake test double
    registered = []
    for s in plan:
        client.register_scorer(name=s["slug"], import_path=s["import_path"], threshold=s["threshold"])
        registered.append(s["slug"])
    return {"registered": registered, "n": len(registered), "plan": plan, "published": False, "limitation": SCORER_PUBLISH_LIMITATION}


def _sync_prompts(client) -> dict:
    """Mirror the base agent instructions, arm-D skill, and judge rubric as
    Braintrust prompts, each carrying its source hash. Unlike scorers,
    `Project.publish()` DOES support prompts (posts them via the
    `insert-functions` API), so this actually registers them.
    """
    plan = prompt_plan()
    if hasattr(client, "projects"):
        project = client.projects.create(name=PROJECT)
        created = []
        for p in plan:
            project.prompts.create(
                name=p["name"],
                slug=p["slug"],
                prompt=p["content"],
                model="z-ai/glm-5.3-flash",
                if_exists="replace",
                metadata={"source_hash": p["source_hash"], "version_label": p["version_label"]},
            )
            created.append(p["slug"])
        project.publish()
        return {
            "registered": created,
            "n": len(created),
            "plan": [{k: v for k, v in p.items() if k != "content"} for p in plan],
            "published": True,
        }
    registered = []
    for p in plan:
        client.register_prompt(name=p["slug"], source_hash=p["source_hash"])
        registered.append(p["slug"])
    return {
        "registered": registered,
        "n": len(registered),
        "plan": [{k: v for k, v in p.items() if k != "content"} for p in plan],
        "published": True,
    }


def _sync_parameters(client) -> dict:
    """Register the one compact versioned parameter schema."""
    schema = parameters_schema()
    if hasattr(client, "projects"):
        from pydantic import create_model

        project = client.projects.create(name=PROJECT)
        ParamsModel = create_model(
            "DealpointRuntimeParameters",
            top_k=(int, schema["top_k"]),
            max_tool_calls=(int, schema["max_tool_calls"]),
        )
        project.parameters.create(
            name="dealpoint-runtime-parameters",
            slug="dealpoint-runtime-parameters",
            schema={"parameters": ParamsModel},
            if_exists="replace",
            metadata=schema,
        )
        project.publish()
        return {"registered": ["dealpoint-runtime-parameters"], "schema": schema, "published": True}
    client.register_parameters(name="dealpoint-runtime-parameters", schema=schema)
    return {"registered": ["dealpoint-runtime-parameters"], "schema": schema, "published": True}


def _record_sync_run(entry: dict, path: Path = BRAINTRUST_SYNC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        fh.write("\n")


def _record_representative_cases(rep_cases: dict) -> None:
    """`data/reports/representative_cases.json` -- previously a stale orphan
    nothing wrote (spec: "re-creatable from Git/local sources by one
    command"). `sync()` writes this every run, from the same
    `representative_cases()` result it replays.
    """
    from dealpoint.config import REPRESENTATIVE_CASES_PATH

    REPRESENTATIVE_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPRESENTATIVE_CASES_PATH, "w", encoding="utf-8") as fh:
        json.dump(rep_cases, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        fh.write("\n")


# Machine-readable record of the one fact `docs/demo-walkthrough.md` was
# previously stating only in prose (spec: "so the limitation is not
# prose-only"): the last attempted full live sync (this repair's D7
# verification pass) was blocked by the Braintrust workspace's own plan
# limit, not by a defect in this code. `blocked_at` is the real timestamp
# from the API's own error payload
# (`[timestamp=1788723714.458]`, `data/results/`-adjacent session logs),
# not a guess. Update this constant (or replace it with `None`) the next
# time a full live sync is actually attempted, whatever the outcome.
LIVE_RUN_STATUS: dict = {
    "last_full_live_sync_blocked": True,
    "blocked_at": "2026-09-06T19:41:54.458000+00:00",
    "blocked_by": "Braintrust workspace plan limit: num_scores_calendar_months",
    "blocked_detail": (
        "Violations of resource constraint num_scores_calendar_months: "
        "(8d085458-23cd-4dae-a88f-383718549e80, 11016, 11000) -- current usage 11016 "
        "against a plan limit of 11000, an account-level quota unrelated to this code."
    ),
    "blocked_step": "_sync_prompts (Project.publish() -> insert-functions API call)",
    "code_verified_before_block": (
        "Experiment logging (score-budget split, per-case RAG/pareto rows, judge-dimension "
        "0-1 rescaling) ran live against the real API before the block and was confirmed "
        "correct via direct BTQL queries against the partially-synced data."
    ),
    "resolution": (
        "Re-run `just braintrust-sync` (or `just braintrust-sync-dry-run` for an offline-safe "
        "check that still exercises every code path except the network calls) once the "
        "workspace's monthly score quota resets or the plan is upgraded."
    ),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    dry_run_flag = "--dry-run" in argv

    if dry_run_flag:
        # A genuinely offline path: `_DryRunClient` never touches the network
        # (see its docstring), so this works even when `braintrust_available()`
        # is false and never risks the workspace's plan quota.
        result = sync(_DryRunClient(), dry_run=True)
        result["live_run_status"] = LIVE_RUN_STATUS
        _record_sync_run(result)
        _record_representative_cases(result["representative_cases"])
        print(
            json.dumps({k: v for k, v in result.items() if k != "framework_versions"}, indent=2, sort_keys=True, default=str)
        )
        return 0

    from dealpoint.eval.braintrust_adapter import braintrust_available

    if not braintrust_available():
        print("braintrust unavailable (no key or package) -- skipping cleanly")
        return 0

    import braintrust

    result = sync(braintrust, dry_run=False)
    result["live_run_status"] = LIVE_RUN_STATUS
    _record_sync_run(result)
    _record_representative_cases(result["representative_cases"])
    print(json.dumps({k: v for k, v in result.items() if k != "framework_versions"}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
