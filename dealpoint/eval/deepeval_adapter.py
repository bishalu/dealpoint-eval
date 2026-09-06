"""DeepEval — independent agent-eval cross-check (spec section 2).

A thin adapter from canonical result rows (finding + execution record +
trajectory) to DeepEval test cases. No second trace store: Braintrust
remains primary observability, and DeepEval results land in local JSON plus
``deepeval/``-namespaced Braintrust scores only.

Every DeepEval-facing function is a factory that imports ``deepeval`` lazily
-- ``import dealpoint.eval.deepeval_adapter`` must succeed with the package
absent, matching the ``openai``/``qdrant``/``braintrust`` convention already
used across this repo.

The single most important design decision here (per the milestone plan):
``row_to_test_case`` is a pure, framework-free function returning a plain
dict. A tiny shim (``dict_to_llm_test_case``) converts that dict into a
real ``LLMTestCase`` only when DeepEval is actually being invoked. This is
what lets the offline gate test drive the mapping with DeepEval absent.
"""

from __future__ import annotations

import os

# DeepEval ships posthog and phones home unless told not to; set this
# BEFORE any `import deepeval` anywhere in the process (module import order
# is not guaranteed, so every entry point into this module sets it too).
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("ERROR_REPORTING", "NO")

import hashlib
import json
import sys
from pathlib import Path

from dealpoint.config import (
    DEEPEVAL_CROSSCHECK_JSON_PATH,
    DEEPEVAL_CROSSCHECK_MD_PATH,
    DEEPEVAL_MODEL_ENV,
    JUDGED_SUBSET_PATH,
    MAX_TOOL_CALLS,
)
from dealpoint.eval.scorers import parse_required_evidence

# No generic RAG metrics weaker than MAUD truth (spec section 2): these were
# considered and deliberately omitted. Gold-span overlap already answers the
# same question with expert labels, at a higher evidentiary bar than any of
# these framework-native heuristics could.
OMITTED_RAG_METRICS: tuple[dict, ...] = (
    {
        "name": "FaithfulnessMetric",
        "reason": (
            "Measures whether the answer is supported by retrieved context using an "
            "LLM judge over paraphrased claims -- weaker than exact-quote gold-span "
            "overlap, which is expert-labelled and character-exact."
        ),
    },
    {
        "name": "AnswerRelevancyMetric",
        "reason": (
            "Measures topical relevance of the answer to the question -- already "
            "implied by answer_correct against the fixed MAUD option list, and less "
            "precise for a closed-option-set task."
        ),
    },
    {
        "name": "ContextualPrecisionMetric",
        "reason": (
            "Measures whether relevant context is ranked above irrelevant context -- "
            "the M7a LlamaIndex RAG lab (li_rag_eval.json) already answers this "
            "directly against the same gold-span truth, at retriever granularity."
        ),
    },
    {
        "name": "ContextualRecallMetric",
        "reason": (
            "Measures whether all needed context was retrieved -- same duplication "
            "concern as ContextualPrecisionMetric; the RAG lab's hit_rate/mrr already "
            "cover this against gold spans."
        ),
    },
)


# --- 4.1: evaluator model resolution ----------------------------------------


def resolve_evaluator_model(env_value: str | None = None) -> dict:
    """`DEEPEVAL_MODEL` if set (still policy-checked); else the first verified
    judge-slate model whose family is not in `judge_slate.CANDIDATE_FAMILIES`.

    Returns `{model, family, source, price_basis}`. Raises `RuntimeError` if
    nothing qualifies. Never hardcodes a model id.
    """
    from dealpoint.config import JUDGE_SLATE_PATH
    from dealpoint.eval.judge_slate import CANDIDATE_FAMILIES, _family_for_model_id

    env_value = env_value if env_value is not None else os.environ.get(DEEPEVAL_MODEL_ENV)

    slate_path = Path(JUDGE_SLATE_PATH)
    slate: dict = {}
    if slate_path.exists():
        try:
            slate = json.loads(slate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            slate = {}
    price_basis = slate.get("price_basis", "unknown")
    verified_judges = [j for j in slate.get("judges", []) if j.get("ok")]

    deepeval_version = _deepeval_library_version()
    trio_models = {j.get("model") for j in slate.get("judges", [])}

    if env_value:
        family = _family_for_model_id(env_value)
        if family in CANDIDATE_FAMILIES:
            raise RuntimeError(
                f"{DEEPEVAL_MODEL_ENV}={env_value!r} is in a candidate-agent family "
                f"({family!r}) -- refusing per the M5 non-candidate-family policy"
            )
        return {
            "model": env_value,
            "provider": env_value.split("/", 1)[0] if "/" in env_value else "unknown",
            "family": family,
            "source": f"env:{DEEPEVAL_MODEL_ENV}",
            "price_basis": "n/a (explicit override)",
            "policy_checked": True,
            "deepeval_version": deepeval_version,
            "shares_model_with_judge_trio": env_value in trio_models,
        }

    for judge in verified_judges:
        model = judge.get("model")
        family = judge.get("family") or _family_for_model_id(model)
        if family in CANDIDATE_FAMILIES:
            continue
        return {
            "model": model,
            "provider": model.split("/", 1)[0] if model and "/" in model else "unknown",
            "family": family,
            "source": "judge_slate.json (verified judges, non-candidate-family)",
            "price_basis": price_basis,
            "policy_checked": True,
            "deepeval_version": deepeval_version,
            "shares_model_with_judge_trio": model in trio_models,
        }

    raise RuntimeError(
        "no evaluator model available: every verified judge-slate entry is in a "
        "candidate-agent family, and DEEPEVAL_MODEL is not set"
    )


def _deepeval_library_version() -> str:
    try:
        from importlib.metadata import version

        return version("deepeval")
    except Exception:  # noqa: BLE001 - version lookup is best-effort, never fatal
        return "unknown"


# --- 4.2: subset + mapping ---------------------------------------------------


def judged_subset_variants() -> dict:
    """The M5 judged subset's case ids + all already-judged variants."""
    payload = json.loads(Path(JUDGED_SUBSET_PATH).read_text(encoding="utf-8"))
    return payload


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


def derive_expected_tools(case: dict) -> list[dict]:
    """Expected tools for one case, derived purely from `required_evidence`
    (spec section 2: "a pure function of the case, never of the output").

    Rule 1 of the skill (always applicable): every case expects at least one
    `search_agreement` call. A case whose `required_evidence` names a
    defined term additionally expects `lookup_defined_term`.
    """
    tools = [{"name": "search_agreement"}]
    candidates = parse_required_evidence(case.get("required_evidence"))
    if candidates:
        tools.append({"name": "lookup_defined_term"})
    return tools


def _tools_called_from_trajectory(trajectory: list[dict]) -> list[dict]:
    return [
        {"name": step.get("tool", ""), "input_parameters": step.get("args") or {}}
        for step in trajectory
    ]


def _reconstruct_retrieval_context(row: dict, doc) -> list[str]:
    """Retrieved text per trajectory step, reusing
    `dealpoint.eval.blinding._build_trajectory_step`'s reconstruction (never
    a second implementation of char_ranges -> text).
    """
    from dealpoint.eval.blinding import _build_trajectory_step

    trajectory = (row.get("record") or {}).get("trajectory") or []
    texts: list[str] = []
    for i, step in enumerate(trajectory):
        built = _build_trajectory_step(step, doc, i)
        retrieved = built.get("retrieved_text")
        if retrieved:
            texts.extend(retrieved)
    return texts


def row_to_test_case(row: dict, case: dict, doc) -> dict:
    """One canonical result row -> a plain, framework-free dict.

    `doc` is a `dealpoint.corpus.document.Document | None`. Pure aside from
    reusing `_build_trajectory_step`'s reconstruction, which itself only
    reads `doc.text`/`doc.sections` -- no I/O, no model calls.

    Keys mirror `deepeval.test_case.LLMTestCase`'s fields exactly, so
    `dict_to_llm_test_case` below is a one-line pass-through.
    """
    from dealpoint.eval.cases import resolve_question

    question = resolve_question(case)
    question_text = f"{question.gloss}\nOptions: {', '.join(question.options) or '(free-form)'}"

    finding = row.get("finding")
    if finding is None:
        actual_output = "The system produced no finding for this case."
    else:
        actual_output = f"answer: {finding.get('answer')}\nrationale: {finding.get('rationale', '')}"

    trajectory = (row.get("record") or {}).get("trajectory") or []
    retrieval_context = _reconstruct_retrieval_context(row, doc) if doc is not None else []

    return {
        "input": question_text,
        "actual_output": actual_output,
        "retrieval_context": retrieval_context,
        "tools_called": _tools_called_from_trajectory(trajectory),
        "expected_tools": derive_expected_tools(case),
        "case_id": case["case_id"],
        "question_id": case["question_id"],
    }


def dict_to_llm_test_case(payload: dict):
    """Thin shim: the pure dict -> a real `deepeval.test_case.LLMTestCase`.

    The only function in this module that imports `deepeval.test_case` --
    kept separate from `row_to_test_case` so the mapping logic itself is
    testable with DeepEval absent.
    """
    from deepeval.test_case import LLMTestCase, ToolCall

    return LLMTestCase(
        input=payload["input"],
        actual_output=payload["actual_output"],
        retrieval_context=payload["retrieval_context"] or None,
        tools_called=[
            ToolCall(name=t["name"], input_parameters=t.get("input_parameters") or {})
            for t in payload["tools_called"]
        ],
        expected_tools=[ToolCall(name=t["name"], input_parameters={}) for t in payload["expected_tools"]],
    )


# --- evaluator model wrapper --------------------------------------------------


def _make_deepeval_model(client, model_id: str):
    """A `DeepEvalBaseLLM` delegating to `OpenRouterClient` -- so every call
    lands in the spend ledger with `milestone_tag="m7a", purpose="deepeval"`.
    A native DeepEval model client would bypass the ledger; this never does.
    """
    from deepeval.models.base_model import DeepEvalBaseLLM

    class _OpenRouterDeepEvalModel(DeepEvalBaseLLM):
        def load_model(self):
            return self

        def generate(self, prompt: str, schema=None, **kwargs) -> str:
            client.context = {
                "milestone_tag": "m7a",
                "purpose": "deepeval",
            }
            result = client.chat(
                messages=[{"role": "user", "content": str(prompt)}],
                model=model_id,
                max_tokens=700,
                temperature=0,
            )
            return result.content or ""

        async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:
            return self.generate(prompt, schema=schema, **kwargs)

        def get_model_name(self) -> str:
            return model_id

    return _OpenRouterDeepEvalModel(model=model_id)


# --- 4.3: metrics -------------------------------------------------------------

STEP_EFFICIENCY_GEVAL_CRITERIA = (
    "Given the number and kind of tool calls in the trajectory (input) and the final "
    "outcome (actual_output), judge whether the agent reached its answer efficiently: "
    "penalise redundant or repeated searches, reward reaching a correct or honest "
    f"abstain answer in few steps, and penalise hitting the {MAX_TOOL_CALLS}-call cap "
    "without a resolved answer."
)


def build_metrics(evaluator_model) -> dict:
    """Instantiate every DeepEval metric this cross-check uses, keyed by name.

    `step_efficiency` always uses the documented `GEval` definition
    (`STEP_EFFICIENCY_GEVAL_CRITERIA`, recorded verbatim and hashed in the
    report), never DeepEval-native `StepEfficiencyMetric`. That metric sets
    `requires_trace = True` and reads `test_case._trace_dict`
    (`deepeval/metrics/step_efficiency/step_efficiency.py`), populated only
    by DeepEval's `@observe` tracing decorator -- `dict_to_llm_test_case`
    never sets a trace, and constructing one by hand would mean adopting
    DeepEval's own tracer as a second tracing path, which the brief section
    4 excludes and the spec's no-bloat rule forbids. Measured on the
    pre-repair run: 93 of 108 traces scored 0.0 with reason "No task or
    trace provided for evaluation" -- real spend for empty input. The spec
    explicitly allows this fallback ("DeepEval-native if present, else a
    documented GEval definition").
    """
    from deepeval.metrics import (
        ArgumentCorrectnessMetric,
        GEval,
        TaskCompletionMetric,
        ToolCorrectnessMetric,
    )
    from deepeval.test_case import ToolCall
    from deepeval.test_case.llm_test_case import SingleTurnParams

    available_tools = [
        ToolCall(name="search_agreement", input_parameters={}),
        ToolCall(name="get_section", input_parameters={}),
        ToolCall(name="lookup_defined_term", input_parameters={}),
    ]
    metrics = {
        "task_completion": TaskCompletionMetric(model=evaluator_model, async_mode=False, include_reason=True),
        "tool_correctness": ToolCorrectnessMetric(
            available_tools=available_tools, model=evaluator_model, async_mode=False, include_reason=True
        ),
        "argument_correctness": ArgumentCorrectnessMetric(
            model=evaluator_model, async_mode=False, include_reason=True
        ),
        "step_efficiency": GEval(
            name="StepEfficiency",
            criteria=STEP_EFFICIENCY_GEVAL_CRITERIA,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=evaluator_model,
        ),
        "step_efficiency_basis": "documented GEval",
    }
    return metrics


def _measure_with_bounded_retry(metric, test_case, *, max_retries: int = 2, backoff_seconds: float = 3.0):
    """Run one metric, retrying up to `max_retries` times (with a short
    backoff) on a rate-limit error before giving up. A transient 429 would
    otherwise silently become a missing datum -- see the `coverage` block
    in the report for how often this still happens after retrying.
    """
    import time

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return metric.measure(test_case), None
        except Exception as exc:  # noqa: BLE001 - classified by name below, never swallowed silently
            last_exc = exc
            is_rate_limit = "RateLimitError" in type(exc).__name__ or "429" in str(exc)
            if is_rate_limit and attempt < max_retries:
                time.sleep(backoff_seconds)
                continue
            break
    return None, last_exc


def measure_test_case(metrics: dict, test_case) -> dict:
    """Run every metric on one test case, returning `{name: {score, reason}}`.

    `argument_correctness` only applies where the case has a checkable
    argument (a defined-term case) -- DeepEval itself scores 1.0/"no tool
    calls" when nothing applicable is present, which this records rather
    than hides. Rate-limit errors get a bounded retry
    (`_measure_with_bounded_retry`); any other exception, or a rate-limit
    that survives the retries, is recorded as a null score with the
    exception type name preserved in `reason` so the report's `coverage`
    block can group nulls by cause.
    """
    out: dict = {}
    for name in ("task_completion", "tool_correctness", "argument_correctness", "step_efficiency"):
        metric = metrics[name]
        score, exc = _measure_with_bounded_retry(metric, test_case)
        if exc is not None:
            out[name] = {
                "score": None,
                "reason": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
            }
        else:
            out[name] = {"score": score, "reason": getattr(metric, "reason", None)}
    return out


def compute_coverage(per_trace_scores: list[dict]) -> dict:
    """Per-metric `{n_scored, n_null, null_reasons}` -- an honest count of how
    many of the `n_traces` traces actually produced a non-null score for
    each metric, and why the rest did not (grouped by exception type name).
    """
    metric_names = ("task_completion", "tool_correctness", "argument_correctness", "step_efficiency")
    coverage: dict = {}
    for name in metric_names:
        n_scored = 0
        n_null = 0
        null_reasons: dict[str, int] = {}
        for row in per_trace_scores:
            cell = row.get(name) or {}
            if cell.get("score") is not None:
                n_scored += 1
            else:
                n_null += 1
                error_type = cell.get("error_type") or "unknown"
                null_reasons[error_type] = null_reasons.get(error_type, 0) + 1
        coverage[name] = {"n_scored": n_scored, "n_null": n_null, "null_reasons": null_reasons}
    return coverage


# --- 4.4: comparisons and conclusion -----------------------------------------


def compare_to_deterministic(deepeval_scores: list[dict], det_rows: list[dict]) -> dict:
    """DeepEval task_completion vs `grounded_accuracy`/`required_evidence_met`/tool cap."""
    from dealpoint.eval.agreement import spearman

    tc = [r.get("task_completion", {}).get("score") for r in deepeval_scores]
    grounded = [
        (1.0 if v is True else 0.0 if v is False else None)
        for v in (r.get("grounded_accuracy") for r in det_rows)
    ]
    tool_correct = [r.get("tool_correctness", {}).get("score") for r in deepeval_scores]
    req_met = [
        (1.0 if v is True else 0.0 if v is False else None)
        for v in (r.get("required_evidence_met") for r in det_rows)
    ]
    tool_calls = [r.get("tool_calls") for r in det_rows]
    cap_hits = sum(1 for tc_ in tool_calls if tc_ is not None and tc_ >= MAX_TOOL_CALLS)

    paired_tool = [
        (a, b) for a, b in zip(tool_correct, req_met, strict=True) if a is not None and b is not None
    ]
    # "Agreement": both sides at/above 0.5 (pass) or both below (fail) -- the
    # same >=0.5-as-pass convention used elsewhere in this module.
    agree = sum(1 for a, b in paired_tool if (a >= 0.5) == (b >= 0.5))

    return {
        "task_completion_vs_grounded_accuracy": {
            "spearman": spearman(tc, grounded),
            "n": sum(1 for a, b in zip(tc, grounded, strict=True) if a is not None and b is not None),
        },
        "tool_correctness_vs_required_evidence_met": {
            "n": len(paired_tool),
            "agreement_rate": (agree / len(paired_tool)) if paired_tool else None,
            "spearman": spearman([a for a, _ in paired_tool], [b for _, b in paired_tool]),
            "note": (
                "available_tools (search_agreement, get_section, lookup_defined_term) "
                "is now passed to ToolCorrectnessMetric so both the tool-CALL and "
                "tool-SELECTION halves of the metric are exercised."
            ),
        },
        "n_at_or_over_tool_cap": cap_hits,
        "max_tool_calls": MAX_TOOL_CALLS,
    }


def load_judge_dimension_means(path=None) -> dict[tuple[str, str], dict[str, float | None]]:
    """`(case_id, variant_id) -> {dimension: mean-of-judges}`, read from
    `data/eval/judge_scores.jsonl` (one row per (packet_id, judge_model)).

    `packet_id` alone does not carry `variant_id`; `judged_subset.json`'s
    packet-id derivation is `sha256(f"{case_id}|{variant_id}")[:12]`
    (`dealpoint.eval.blinding._packet_id`), so this reconstructs the key by
    recomputing that same hash for every (case_id, variant_id) pair in the
    judged subset rather than reversing the hash.
    """
    from dealpoint.config import JUDGE_SCORES_PATH
    from dealpoint.eval.blinding import _packet_id

    rows = _load_jsonl(path or JUDGE_SCORES_PATH)
    subset = judged_subset_variants()
    case_ids = subset.get("case_ids", [])
    variant_ids = [v["variant_id"] for v in subset.get("variants", [])]

    packet_to_trace: dict[str, tuple[str, str]] = {}
    for cid in case_ids:
        for vid in variant_ids:
            packet_to_trace[_packet_id(cid, vid)] = (cid, vid)

    dims = ("reasoning", "evidence", "trajectory", "professional")
    by_trace: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        if not row.get("ok", True):
            continue
        packet_id = row.get("packet_id")
        trace_key = packet_to_trace.get(packet_id) if packet_id else None
        if trace_key is None:
            continue
        entry = by_trace.setdefault(trace_key, {d: [] for d in dims})
        for d in dims:
            value = row.get(d)
            if value is not None:
                entry[d].append(value)

    return {
        key: {d: (sum(vals) / len(vals) if vals else None) for d, vals in dims_map.items()}
        for key, dims_map in by_trace.items()
    }


def _paired_spearman_and_n(xs: list, ys: list) -> dict:
    from dealpoint.eval.agreement import spearman

    n = sum(1 for a, b in zip(xs, ys, strict=True) if a is not None and b is not None)
    return {"spearman": spearman(xs, ys), "n": n}


def compare_to_judge(deepeval_scores: list[dict], judge_means_by_trace: dict) -> dict:
    """DeepEval metrics <-> all four calibrated judge dimensions, per trace.

    `n` is the count of traces where BOTH sides are non-None (paired,
    matching `compare_to_deterministic`'s convention) -- not
    `len(common_keys)`, which only requires the trace to exist in both
    dicts and over-counts whenever either side's score is null.
    """
    by_trace_key = {(r.get("case_id"), r.get("variant_id")): r for r in deepeval_scores}
    common_keys = sorted(set(by_trace_key) & set(judge_means_by_trace))

    step_eff = [by_trace_key[k].get("step_efficiency", {}).get("score") for k in common_keys]
    task_completion = [by_trace_key[k].get("task_completion", {}).get("score") for k in common_keys]
    tool_correctness = [by_trace_key[k].get("tool_correctness", {}).get("score") for k in common_keys]
    argument_correctness = [
        by_trace_key[k].get("argument_correctness", {}).get("score") for k in common_keys
    ]
    trajectory_dim = [judge_means_by_trace[k].get("trajectory") for k in common_keys]
    reasoning_dim = [judge_means_by_trace[k].get("reasoning") for k in common_keys]
    evidence_dim = [judge_means_by_trace[k].get("evidence") for k in common_keys]
    professional_dim = [judge_means_by_trace[k].get("professional") for k in common_keys]

    return {
        "step_efficiency_vs_judge_trajectory": _paired_spearman_and_n(step_eff, trajectory_dim),
        "task_completion_vs_judge_reasoning": _paired_spearman_and_n(task_completion, reasoning_dim),
        "tool_correctness_vs_judge_evidence": _paired_spearman_and_n(tool_correctness, evidence_dim),
        "argument_correctness_vs_judge_professional": _paired_spearman_and_n(
            argument_correctness, professional_dim
        ),
    }


DISAGREEMENT_METRIC_NAME = "task_completion_vs_grounded_accuracy"


def find_disagreements(per_trace_scores: list[dict], det_by_row: list[dict]) -> list[dict]:
    """Traces where DeepEval and the deterministic signal disagree.

    Pure function, no I/O, no model calls: `per_trace_scores[i]` and
    `det_by_row[i]` MUST refer to the same trace (same index -- callers pass
    the parallel lists `main()` already builds).

    Rule (minimum, per spec): `task_completion >= 0.5` with
    `grounded_accuracy is False`, or `task_completion <= 0.5` with
    `grounded_accuracy is True`. `None` on either side is never coerced --
    such traces are skipped entirely, not counted as agreement OR
    disagreement.
    """
    disagreements: list[dict] = []
    for score_row, det_row in zip(per_trace_scores, det_by_row, strict=True):
        tc = (score_row.get("task_completion") or {}).get("score")
        ga = det_row.get("grounded_accuracy")
        if tc is None or ga is None:
            continue
        if tc >= 0.5 and ga is False:
            direction = "deepeval_pass_deterministic_fail"
        elif tc <= 0.5 and ga is True:
            direction = "deepeval_fail_deterministic_pass"
        else:
            continue
        disagreements.append(
            {
                "case_id": score_row.get("case_id"),
                "variant_id": score_row.get("variant_id"),
                "metric": DISAGREEMENT_METRIC_NAME,
                "deepeval_score": tc,
                "deterministic_score": ga,
                "direction": direction,
            }
        )
    return disagreements


def load_human_dimension_means(path=None) -> dict[tuple[str, str], dict[str, float | None]]:
    """`(case_id, variant_id) -> {dimension: mean-of-scorers}`, read from
    `data/eval/calibration/human_scores.jsonl` (one row per
    `(packet_id, scorer)`), joined back to `(case_id, variant_id)` through
    `data/eval/calibration/variant_key.json` -- unlike judge packet ids
    (which are re-derived by hashing), the human calibration packet ids are
    recorded directly in that lookup file, so no hash reconstruction is
    needed here.
    """
    from dealpoint.config import CALIBRATION_DIR

    human_path = Path(path) if path is not None else CALIBRATION_DIR / "human_scores.jsonl"
    rows = _load_jsonl(human_path)
    if not rows:
        return {}

    variant_key_path = CALIBRATION_DIR / "variant_key.json"
    variant_key: dict = {}
    if variant_key_path.exists():
        try:
            variant_key = json.loads(variant_key_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            variant_key = {}

    dims = ("reasoning", "evidence", "trajectory", "professional")
    by_trace: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        packet_id = row.get("packet_id")
        key_entry = variant_key.get(packet_id) if packet_id else None
        if key_entry is None:
            continue
        trace_key = (key_entry.get("case_id"), key_entry.get("variant_id"))
        if trace_key[0] is None or trace_key[1] is None:
            continue
        entry = by_trace.setdefault(trace_key, {d: [] for d in dims})
        for d in dims:
            value = row.get(d)
            if value is not None:
                entry[d].append(value)

    return {
        key: {d: (sum(vals) / len(vals) if vals else None) for d, vals in dims_map.items()}
        for key, dims_map in by_trace.items()
    }


def compare_to_human(deepeval_scores: list[dict] | None = None, human_means_by_trace: dict | None = None) -> dict:
    """Human comparison: real `{n, spearman}` statistics per the same
    paired-count convention `compare_to_deterministic`/`compare_to_judge`
    use, computed from `data/eval/calibration/human_scores.jsonl` joined
    through `variant_key.json`. Falls back to the `"pending"` shape (n=0,
    no statistic) only when the human file is genuinely empty -- never a
    fabricated number, and never a placeholder string once real rows exist.

    `task_completion_vs_human` compares DeepEval's `task_completion` against
    the human `reasoning` dimension; `step_efficiency_vs_human_trajectory_quality`
    compares DeepEval's `step_efficiency` against the human `trajectory`
    dimension -- the same rubric-dimension pairing `compare_to_judge` uses
    for the calibrated judge panel.
    """
    if human_means_by_trace is None:
        human_means_by_trace = load_human_dimension_means()

    n_human_rows = len(human_means_by_trace)
    if n_human_rows == 0 or not deepeval_scores:
        return {
            "n": 0,
            "status": "pending",
            "task_completion_vs_human": "pending",
            "step_efficiency_vs_human_trajectory_quality": "pending",
        }

    by_trace_key = {(r.get("case_id"), r.get("variant_id")): r for r in deepeval_scores}
    common_keys = sorted(set(by_trace_key) & set(human_means_by_trace))

    task_completion = [by_trace_key[k].get("task_completion", {}).get("score") for k in common_keys]
    step_eff = [by_trace_key[k].get("step_efficiency", {}).get("score") for k in common_keys]
    reasoning_dim = [human_means_by_trace[k].get("reasoning") for k in common_keys]
    trajectory_dim = [human_means_by_trace[k].get("trajectory") for k in common_keys]

    task_completion_vs_human = _paired_spearman_and_n(task_completion, reasoning_dim)
    step_efficiency_vs_human = _paired_spearman_and_n(step_eff, trajectory_dim)

    return {
        "n": n_human_rows,
        "status": "available",
        "task_completion_vs_human": task_completion_vs_human,
        "step_efficiency_vs_human_trajectory_quality": step_efficiency_vs_human,
    }


def classify(comparisons: dict, *, resolved_evaluator: dict | None = None) -> tuple[str, str]:
    """One of `KEEP_CORE_DIAGNOSTIC` / `KEEP_OPTIONAL_ANALYSIS` /
    `REMOVE_NO_ADDED_SIGNAL`, derived from the measured correlations.

    Rule (documented, deterministic): if DeepEval's task_completion tracks
    grounded_accuracy strongly (|rho| >= 0.6, n >= 10) it adds no
    independent signal over the deterministic score -> REMOVE_NO_ADDED_SIGNAL.
    If it tracks moderately (0.3 <= |rho| < 0.6) or the sample is too small
    to tell, it is a plausible secondary cross-check -> KEEP_OPTIONAL_ANALYSIS.
    Otherwise (weak/no correlation, meaning it measures something distinct,
    or judge-comparison data exists showing added value) -> KEEP_CORE_DIAGNOSTIC.

    If `resolved_evaluator` shares a model with the M5 judge trio, any
    result that would otherwise be KEEP_CORE_DIAGNOSTIC on the strength of
    vs_judge agreement is downgraded to KEEP_OPTIONAL_ANALYSIS -- shared-model
    bias means high judge agreement is not trustworthy independent evidence.
    vs_deterministic-driven outcomes (REMOVE_NO_ADDED_SIGNAL, and
    KEEP_OPTIONAL_ANALYSIS from weak power) are unaffected: the deterministic
    score shares no model with the evaluator.
    """
    det = comparisons.get("vs_deterministic", {})
    rho_info = det.get("task_completion_vs_grounded_accuracy", {})
    rho = rho_info.get("spearman")
    n = rho_info.get("n", 0)
    shares_model = bool((resolved_evaluator or {}).get("shares_model_with_judge_trio"))

    if rho is None or n < 10:
        msg = (
            f"Correlation with grounded_accuracy is not computable or under-powered "
            f"(rho={rho}, n={n}); too little evidence to remove or promote to core."
        )
        return "KEEP_OPTIONAL_ANALYSIS", msg
    if abs(rho) >= 0.6:
        msg = (
            f"DeepEval task_completion correlates strongly with grounded_accuracy "
            f"(rho={rho:.3f}, n={n}) -- it tracks the deterministic score closely "
            f"enough to add no independent diagnostic signal."
        )
        return "REMOVE_NO_ADDED_SIGNAL", msg
    if abs(rho) >= 0.3:
        msg = (
            f"DeepEval task_completion correlates moderately with grounded_accuracy "
            f"(rho={rho:.3f}, n={n}) -- plausible as a secondary cross-check but not "
            f"strong enough evidence to call it a core diagnostic."
        )
        return "KEEP_OPTIONAL_ANALYSIS", msg
    if shares_model:
        msg = (
            f"DeepEval task_completion correlates weakly with grounded_accuracy "
            f"(rho={rho:.3f}, n={n}), which would otherwise support "
            "KEEP_CORE_DIAGNOSTIC -- but the resolved evaluator shares a model with "
            "the M5 judge trio it is compared against (see decisions), so any "
            "apparent independence from vs_judge agreement is not trustworthy. "
            "Downgraded to KEEP_OPTIONAL_ANALYSIS pending a genuinely independent "
            "evaluator model."
        )
        return "KEEP_OPTIONAL_ANALYSIS", msg
    msg = (
        f"DeepEval task_completion correlates weakly with grounded_accuracy "
        f"(rho={rho:.3f}, n={n}) -- it appears to measure something distinct from the "
        f"deterministic score, which is exactly the independent cross-check the milestone "
        f"wants."
    )
    return "KEEP_CORE_DIAGNOSTIC", msg


BRIEF_DIFFERENCES = [
    {
        "id": 1,
        "topic": "deepeval_brief_exclusion",
        "difference": (
            "Brief section 3 lists DeepEval under 'Not used' and section 4 excludes it "
            "outright. M7a's authority line amends section 4 as of 2026-09-05; the "
            "classification field this module produces is the mechanism by which that "
            "amendment is tested rather than assumed -- REMOVE_NO_ADDED_SIGNAL is a "
            "legitimate outcome of this cross-check, not a foregone conclusion."
        ),
    },
    {
        "id": 2,
        "topic": "dual_tracing_risk",
        "difference": (
            "Brief section 4 excludes dual tracing paths. DeepEval ships OpenTelemetry "
            "and posthog telemetry. This module opts out of DeepEval telemetry "
            "(DEEPEVAL_TELEMETRY_OPT_OUT=1) before any import, never logs to a second "
            "trace store, and writes results only to local JSON plus deepeval/-namespaced "
            "Braintrust scores -- Braintrust remains primary observability."
        ),
    },
    {
        "id": 3,
        "topic": "scale_inherited",
        "difference": None,  # filled in by _brief_differences() with the real human-calibration n
    },
    {
        "id": 4,
        "topic": "score_budget",
        "difference": (
            "Brief section 2.7 caps Braintrust logging at <= 6 scores/case. M7a permits "
            "<= 12 scores/case on subsets <= 60 cases as a ceiling, not a target, with M4/M6 "
            "sweeps still logging exactly six -- enforced at sync time by "
            "dealpoint.eval.braintrust_sync.assert_score_budget."
        ),
    },
    {
        "id": 5,
        "topic": "evaluator_independence",
        "difference": (
            "The spec calls DeepEval an 'independent agent evaluator'. The resolved "
            "evaluator model is drawn from judge_slate.json's verified, "
            "non-candidate-family judges -- which is exactly the M5 judge trio, so the "
            "resolved model is always one of the three judges it is compared against, "
            "never a fourth independent model. This is a consequence of the spec's own "
            "resolution rule ('the resolved provider/model/version... read from the "
            "existing judge configuration, never hardcoded'), not a bug in this repair; "
            "the caveat is recorded here, in the report's `decisions`, and factored "
            "into `classify()` because it is not otherwise visible from the numbers."
        ),
    },
]

BRIEF_SCALE_INHERITED_HUMAN_TARGET_N = 30  # brief section 2.5's hand-scored-trace target


def _brief_differences() -> list[dict]:
    """`BRIEF_DIFFERENCES` with the `scale_inherited` entry's human-calibration
    text filled in from the ACTUAL row count on disk -- never the stale
    hardcoded "n=0 ('pending')" the pre-repair version carried once real
    human scores existed, and never silently wrong if the count changes
    again later.
    """
    n_human = len(load_human_dimension_means())
    if n_human == 0:
        human_sentence = (
            "human calibration is n=0 ('pending'), not the brief's "
            f"{BRIEF_SCALE_INHERITED_HUMAN_TARGET_N} hand-scored traces. Every "
            "DeepEval<->human comparison in this report is 'pending'."
        )
    else:
        human_sentence = (
            f"human calibration is n={n_human} hand-scored traces, {n_human} of the brief's "
            f"{BRIEF_SCALE_INHERITED_HUMAN_TARGET_N}, not 'pending' -- DeepEval<->human "
            "comparisons in this report carry a real {n, spearman} statistic wherever a "
            "trace has both a DeepEval score and a human score."
        )
    differences = [dict(d) for d in BRIEF_DIFFERENCES]
    for d in differences:
        if d["topic"] == "scale_inherited":
            d["difference"] = (
                "The judged subset is 18 cases x 6 variants (108 traces), not the brief "
                f"section 2.5's 40 x 6; {human_sentence} Already recorded in judges.json; "
                "restated here because this report's own vs_human field is the thing that "
                "changes when human calibration grows."
            )
    return differences


def build_report(
    *,
    resolved_evaluator: dict,
    subset: dict,
    per_trace_scores: list[dict],
    comparisons: dict,
    disagreements: list[dict],
    spend: dict,
    framework_versions: dict,
    coverage: dict | None = None,
    decisions: list[dict] | None = None,
    omitted_metrics: tuple[dict, ...] = OMITTED_RAG_METRICS,
) -> dict:
    classification, rationale = classify(comparisons, resolved_evaluator=resolved_evaluator)
    return {
        "schema_version": 1,
        "resolved_evaluator": resolved_evaluator,
        "framework_versions": framework_versions,
        "subset": subset,
        "metrics": {
            "used": ["task_completion", "tool_correctness", "argument_correctness", "step_efficiency"],
            "step_efficiency_basis": per_trace_scores[0].get("step_efficiency_basis")
            if per_trace_scores
            else None,
            "step_efficiency_geval_criteria": STEP_EFFICIENCY_GEVAL_CRITERIA,
            "step_efficiency_geval_criteria_hash": hashlib.sha256(
                STEP_EFFICIENCY_GEVAL_CRITERIA.encode()
            ).hexdigest()[:16],
            "omitted_rag_metrics": list(omitted_metrics),
        },
        "coverage": coverage or {},
        "per_trace_scores": per_trace_scores,
        "comparisons": comparisons,
        "disagreements": disagreements,
        "classification": classification,
        "classification_rationale": rationale,
        "spend": spend,
        "decisions": decisions or [],
        "brief_differences": _brief_differences(),
    }


def render_markdown(report: dict) -> str:
    lines = ["# M7a -- DeepEval independent agent-eval cross-check\n"]
    ev = report.get("resolved_evaluator", {})
    lines.append(
        f"Resolved evaluator: `{ev.get('model')}` (provider `{ev.get('provider')}`, "
        f"family `{ev.get('family')}`, deepeval `{ev.get('deepeval_version')}`, "
        f"source `{ev.get('source')}`)\n"
    )
    if ev.get("shares_model_with_judge_trio"):
        lines.append(
            "**Independence caveat:** this evaluator model is one of the three M5 judge-trio "
            "models, not a fourth independent model -- see Decisions.\n"
        )
    subset = report.get("subset", {})
    lines.append(
        f"Subset: {len(subset.get('case_ids', []))} cases x "
        f"{len(subset.get('variants', []))} variants = {subset.get('n_traces')} traces\n"
    )
    lines.append("## Metrics used\n")
    for name in report.get("metrics", {}).get("used", []):
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append(f"step_efficiency basis: {report.get('metrics', {}).get('step_efficiency_basis')}\n")

    coverage = report.get("coverage") or {}
    if coverage:
        lines.append("## Coverage (per-metric scored/null counts)\n")
        lines.append("| metric | n_scored | n_null | null reasons |")
        lines.append("|---|---|---|---|")
        for name, c in coverage.items():
            reasons = ", ".join(f"{k}={v}" for k, v in (c.get("null_reasons") or {}).items()) or "-"
            lines.append(f"| {name} | {c.get('n_scored')} | {c.get('n_null')} | {reasons} |")
        lines.append("")

    spend = report.get("spend") or {}
    if spend:
        lines.append("## Spend\n")
        lines.append(f"```json\n{json.dumps(spend, indent=2, sort_keys=True)}\n```\n")

    decisions = report.get("decisions") or []
    if decisions:
        lines.append("## Decisions\n")
        for d in decisions:
            lines.append(f"- **{d['topic']}**: {d['decision']}")
        lines.append("")

    lines.append("## Omitted RAG metrics (deliberate)\n")
    for m in report.get("metrics", {}).get("omitted_rag_metrics", []):
        lines.append(f"- **{m['name']}**: {m['reason']}")
    lines.append("")
    lines.append("## Comparisons\n")
    lines.append(f"```json\n{json.dumps(report.get('comparisons', {}), indent=2, sort_keys=True)}\n```\n")
    disagreements = report.get("disagreements", [])
    lines.append(f"## Disagreement cases ({len(disagreements)})\n")
    if not disagreements:
        lines.append("None found (or none computable -- both sides require a non-None value).")
    else:
        lines.append("| case_id | variant_id | metric | deepeval_score | deterministic_score | direction |")
        lines.append("|---|---|---|---|---|---|")
        for d in disagreements:
            lines.append(
                f"| {d['case_id']} | {d['variant_id']} | {d['metric']} | "
                f"{d['deepeval_score']} | {d['deterministic_score']} | {d['direction']} |"
            )
    lines.append("")
    lines.append(f"## Classification: `{report.get('classification')}`\n")
    lines.append(report.get("classification_rationale", ""))
    lines.append("")
    lines.append("## Brief-vs-spec differences\n")
    for d in report.get("brief_differences", []):
        lines.append(f"- **{d['topic']}**: {d['difference']}")
    lines.append("")
    return "\n".join(lines) + "\n"


# --- disk-driven pipeline -----------------------------------------------------


def _resolve_document_for_case(case: dict, doc_cache: dict):
    from dealpoint.corpus.document import load_document
    from dealpoint.eval.cases import resolve_document_id

    doc_id = resolve_document_id(case)
    if doc_id not in doc_cache:
        doc_cache[doc_id] = load_document(doc_id)
    return doc_cache[doc_id]


def _score_traces(rows, variant_ids, metrics, find_case, doc_cache) -> list[dict]:
    """Score one contiguous slice of (row, variant_id) pairs, in order."""
    out: list[dict] = []
    for row, variant_id in zip(rows, variant_ids, strict=True):
        case = find_case(row["case_id"])
        doc = _resolve_document_for_case(case, doc_cache)
        payload = row_to_test_case(row, case, doc)
        test_case = dict_to_llm_test_case(payload)
        scores = measure_test_case(metrics, test_case)
        scores["case_id"] = row["case_id"]
        scores["variant_id"] = variant_id
        scores["step_efficiency_basis"] = metrics.get("step_efficiency_basis")
        out.append(scores)
    return out


def _load_all_det_rows(case_ids: list[str], variants: list[dict]) -> tuple[list[dict], list[str]]:
    """Every det row for the subset x variants, in the exact (case_id,
    variant) order `main()`'s scoring loop uses -- shared by the metered
    path and the offline `--from-cache` path so both agree on row order.
    """
    all_rows: list[dict] = []
    all_variant_ids: list[str] = []
    for variant in variants:
        rows_by_case = {r["case_id"]: r for r in _load_jsonl(variant["results_path"])}
        for cid in case_ids:
            row = rows_by_case.get(cid)
            if row is None:
                continue
            all_rows.append(row)
            all_variant_ids.append(variant["variant_id"])
    return all_rows, all_variant_ids


DECISIONS_TEMPLATE: tuple[dict, ...] = (
    {
        "id": 1,
        "topic": "step_efficiency_geval_fallback",
        "decision": (
            "step_efficiency always uses the documented GEval definition, never "
            "DeepEval-native StepEfficiencyMetric, because that metric requires "
            "DeepEval's @observe trace capture (a second tracing path the brief "
            "excludes); see build_metrics's docstring for the measured failure mode "
            "this avoids (93/108 traces scoring 0.0 with 'no trace provided')."
        ),
    },
    {
        "id": 2,
        "topic": "evaluator_independence_caveat",
        "decision": (
            "The resolved evaluator ({model}) is one of the three M5 "
            "judge-trio models (judge_slate.json), not a fourth independent model -- "
            "all three verified, non-candidate-family judges ARE the trio, so there "
            "is no alternative to switch to under the existing resolution rule (which "
            "this repair does not change, per the milestone spec). "
            "task_completion_vs_judge_reasoning and any other vs_judge comparison are "
            "therefore contaminated by shared-model bias: DeepEval's 'independent "
            "agent evaluator' shares a model with one third of the panel it is being "
            "compared against. Factored into the classification below."
        ),
    },
)


def _write_report(
    *,
    resolved: dict,
    case_ids: list[str],
    variants: list,
    n_traces: int,
    per_trace_scores: list[dict],
    det_by_row: list[dict],
    spend: dict,
    coverage: dict,
    decisions: list[dict],
    framework_versions_payload: dict,
) -> dict:
    """Build comparisons/disagreements/report from already-scored traces and
    write the JSON + markdown. Shared by the metered path (fresh scores) and
    the offline `--from-cache` path (scores already on disk, zero spend) --
    the ONLY thing that legitimately differs between the two call sites is
    `vs_human`, since `compare_to_human` now computes a real statistic
    wherever `data/eval/calibration/human_scores.jsonl` has rows; every
    other comparison is a pure function of `per_trace_scores`/`det_by_row`,
    which are identical either way.
    """
    judge_means_by_trace = load_judge_dimension_means()
    human_means_by_trace = load_human_dimension_means()
    comparisons = {
        "vs_deterministic": compare_to_deterministic(per_trace_scores, det_by_row),
        "vs_judge": compare_to_judge(per_trace_scores, judge_means_by_trace),
        "vs_human": compare_to_human(per_trace_scores, human_means_by_trace),
    }
    disagreements = find_disagreements(per_trace_scores, det_by_row)

    variant_ids = [v["variant_id"] if isinstance(v, dict) else v for v in variants]

    report = build_report(
        resolved_evaluator=resolved,
        subset={"case_ids": case_ids, "variants": variant_ids, "n_traces": n_traces},
        per_trace_scores=per_trace_scores,
        comparisons=comparisons,
        disagreements=disagreements,
        spend=spend,
        framework_versions=framework_versions_payload,
        coverage=coverage,
        decisions=decisions,
    )

    DEEPEVAL_CROSSCHECK_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEEPEVAL_CROSSCHECK_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        fh.write("\n")
    with open(DEEPEVAL_CROSSCHECK_MD_PATH, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))

    print(f"Wrote {DEEPEVAL_CROSSCHECK_JSON_PATH} and {DEEPEVAL_CROSSCHECK_MD_PATH}")
    return report


def _main_from_cache() -> int:
    """Offline regeneration: reuse the per-trace scores, spend, coverage and
    decisions ALREADY stored in `data/reports/deepeval_crosscheck.json` --
    no metric is re-scored, no OpenRouterClient is constructed, no money is
    spent. Only `vs_human` (and, as an automatic consequence, anything
    downstream of it -- nothing currently is) can change; `classification`
    depends only on `vs_deterministic` and the resolved evaluator's
    `shares_model_with_judge_trio` flag, neither of which this path touches,
    so it comes out identical to the cached report by construction.
    """
    if not DEEPEVAL_CROSSCHECK_JSON_PATH.exists():
        print(f"{DEEPEVAL_CROSSCHECK_JSON_PATH} does not exist -- cannot regenerate from cache")
        return 1

    cached = json.loads(DEEPEVAL_CROSSCHECK_JSON_PATH.read_text(encoding="utf-8"))
    resolved = cached["resolved_evaluator"]
    subset = cached["subset"]
    case_ids = subset["case_ids"]
    variant_ids = subset["variants"]
    per_trace_scores = cached["per_trace_scores"]

    subset_payload = judged_subset_variants()
    variants_by_id = {v["variant_id"]: v for v in subset_payload["variants"]}
    variants = [variants_by_id[vid] for vid in variant_ids if vid in variants_by_id]

    all_rows, _all_variant_ids = _load_all_det_rows(case_ids, variants)
    det_by_row = [row.get("scores") or {} for row in all_rows]

    if len(det_by_row) != len(per_trace_scores):
        print(
            f"row-count mismatch: {len(det_by_row)} det rows on disk vs "
            f"{len(per_trace_scores)} cached per_trace_scores -- refusing to regenerate "
            "from a stale cache; run without --from-cache to re-score"
        )
        return 1

    _write_report(
        resolved=resolved,
        case_ids=case_ids,
        variants=variant_ids,
        n_traces=subset["n_traces"],
        per_trace_scores=per_trace_scores,
        det_by_row=det_by_row,
        spend=cached["spend"],
        coverage=cached["coverage"],
        decisions=cached["decisions"],
        framework_versions_payload=cached["framework_versions"],
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if "--from-cache" in argv:
        return _main_from_cache()

    from dealpoint.eval.cases import find_case
    from dealpoint.eval.framework_versions import framework_versions
    from dealpoint.eval.spend import assert_within_cap, realized_usd
    from dealpoint.llm.client import OpenRouterClient

    subset_payload = judged_subset_variants()
    case_ids = subset_payload["case_ids"]
    variants = subset_payload["variants"]

    all_rows, all_variant_ids = _load_all_det_rows(case_ids, variants)
    n_traces = len(all_rows)

    resolved = resolve_evaluator_model()
    client = OpenRouterClient(milestone_tag="m7a")
    model = _make_deepeval_model(client, resolved["model"])
    metrics = build_metrics(model)
    doc_cache: dict = {}

    # Metered discipline (spec section 4.4): estimate -> <=3-trace calibration
    # sample -> compare projected vs realised -> assert_within_cap -> continue
    # scoring the REMAINING traces (never re-score the calibration slice, so
    # no trace is ever paid for twice).
    n_calib = min(3, n_traces)
    before_calib = realized_usd()
    calib_scores = _score_traces(
        all_rows[:n_calib], all_variant_ids[:n_calib], metrics, find_case, doc_cache
    )
    after_calib = realized_usd()
    calib_realized = round(after_calib - before_calib, 6)
    per_trace_calib = calib_realized / n_calib if n_calib else 0.0
    projected_full_usd = round(per_trace_calib * n_traces, 6)
    print(
        f"DeepEval calibration: {n_calib} traces cost ${calib_realized:.6f} "
        f"(${per_trace_calib:.6f}/trace); projected full run (n={n_traces}): "
        f"${projected_full_usd:.6f}"
    )
    assert_within_cap(projected_full_usd - calib_realized)

    remaining_scores = _score_traces(
        all_rows[n_calib:], all_variant_ids[n_calib:], metrics, find_case, doc_cache
    )
    per_trace_scores = calib_scores + remaining_scores

    det_by_row = [row.get("scores") or {} for row in all_rows]

    from dealpoint.eval.spend import read_ledger

    captured_at_iso = __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat()
    all_ledger_rows = read_ledger()
    deepeval_realized_usd = round(
        sum(float(r.get("usd") or 0) for r in all_ledger_rows if r.get("purpose") == "deepeval"), 6
    )
    m7a_realized_usd = round(
        sum(float(r.get("usd") or 0) for r in all_ledger_rows if r.get("milestone_tag") == "m7a"), 6
    )
    global_realized_usd = realized_usd()
    spend = {
        "captured_at": captured_at_iso,
        "deepeval_realized_usd": deepeval_realized_usd,
        "m7a_realized_usd": m7a_realized_usd,
        "global_realized_usd": global_realized_usd,
        "n_traces": n_traces,
        "calibration_n": n_calib,
        "calibration_realized_usd": calib_realized,
        "projected_full_usd": projected_full_usd,
    }

    coverage = compute_coverage(per_trace_scores)

    decisions = [dict(d, decision=d["decision"].format(model=resolved["model"])) for d in DECISIONS_TEMPLATE]

    _write_report(
        resolved=resolved,
        case_ids=case_ids,
        variants=variants,
        n_traces=n_traces,
        per_trace_scores=per_trace_scores,
        det_by_row=det_by_row,
        spend=spend,
        coverage=coverage,
        decisions=decisions,
        framework_versions_payload=framework_versions(),
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
