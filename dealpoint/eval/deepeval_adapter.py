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

    if env_value:
        family = _family_for_model_id(env_value)
        if family in CANDIDATE_FAMILIES:
            raise RuntimeError(
                f"{DEEPEVAL_MODEL_ENV}={env_value!r} is in a candidate-agent family "
                f"({family!r}) -- refusing per the M5 non-candidate-family policy"
            )
        return {
            "model": env_value,
            "family": family,
            "source": f"env:{DEEPEVAL_MODEL_ENV}",
            "price_basis": "n/a (explicit override)",
            "policy_checked": True,
        }

    for judge in verified_judges:
        model = judge.get("model")
        family = judge.get("family") or _family_for_model_id(model)
        if family in CANDIDATE_FAMILIES:
            continue
        return {
            "model": model,
            "family": family,
            "source": "judge_slate.json (verified judges, non-candidate-family)",
            "price_basis": price_basis,
            "policy_checked": True,
        }

    raise RuntimeError(
        "no evaluator model available: every verified judge-slate entry is in a "
        "candidate-agent family, and DEEPEVAL_MODEL is not set"
    )


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

    `step_efficiency` uses DeepEval-native `StepEfficiencyMetric` when the
    installed version has one (checked at runtime, since the milestone plan
    treats this as a moving target); otherwise falls back to a documented
    `GEval` definition whose criteria string is `STEP_EFFICIENCY_GEVAL_CRITERIA`
    (recorded verbatim, hashed, in the report).
    """
    from deepeval.metrics import (
        ArgumentCorrectnessMetric,
        TaskCompletionMetric,
        ToolCorrectnessMetric,
    )

    metrics = {
        "task_completion": TaskCompletionMetric(model=evaluator_model, async_mode=False, include_reason=True),
        "tool_correctness": ToolCorrectnessMetric(model=evaluator_model, async_mode=False, include_reason=True),
        "argument_correctness": ArgumentCorrectnessMetric(
            model=evaluator_model, async_mode=False, include_reason=True
        ),
    }

    try:
        from deepeval.metrics import StepEfficiencyMetric

        metrics["step_efficiency"] = StepEfficiencyMetric(
            model=evaluator_model, async_mode=False, include_reason=True
        )
        metrics["step_efficiency_basis"] = "deepeval-native:StepEfficiencyMetric"
    except ImportError:
        from deepeval.metrics import GEval
        from deepeval.test_case.llm_test_case import SingleTurnParams

        metrics["step_efficiency"] = GEval(
            name="StepEfficiency",
            criteria=STEP_EFFICIENCY_GEVAL_CRITERIA,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=evaluator_model,
        )
        metrics["step_efficiency_basis"] = "documented GEval"

    return metrics


def measure_test_case(metrics: dict, test_case) -> dict:
    """Run every metric on one test case, returning `{name: {score, reason}}`.

    `argument_correctness` only applies where the case has a checkable
    argument (a defined-term case) -- DeepEval itself scores 1.0/"no tool
    calls" when nothing applicable is present, which this records rather
    than hides.
    """
    out: dict = {}
    for name in ("task_completion", "tool_correctness", "argument_correctness", "step_efficiency"):
        metric = metrics[name]
        try:
            score = metric.measure(test_case)
            out[name] = {"score": score, "reason": getattr(metric, "reason", None)}
        except Exception as exc:  # noqa: BLE001 - a metric failure is a recorded row, not a crash
            out[name] = {"score": None, "reason": f"{type(exc).__name__}: {exc}"}
    return out


# --- 4.4: comparisons and conclusion -----------------------------------------


def compare_to_deterministic(deepeval_scores: list[dict], det_rows: list[dict]) -> dict:
    """DeepEval task_completion vs `grounded_accuracy`/`required_evidence_met`/tool cap."""
    from dealpoint.eval.agreement import spearman

    tc = [r.get("task_completion", {}).get("score") for r in deepeval_scores]
    grounded = [
        (1.0 if v is True else 0.0 if v is False else None)
        for v in (r.get("grounded_accuracy") for r in det_rows)
    ]
    req_met = [
        (1.0 if v is True else 0.0 if v is False else None)
        for v in (r.get("required_evidence_met") for r in det_rows)
    ]
    tool_calls = [r.get("tool_calls") for r in det_rows]
    cap_hits = sum(1 for tc_ in tool_calls if tc_ is not None and tc_ >= MAX_TOOL_CALLS)

    return {
        "task_completion_vs_grounded_accuracy": {
            "spearman": spearman(tc, grounded),
            "n": sum(1 for a, b in zip(tc, grounded, strict=True) if a is not None and b is not None),
        },
        "tool_correctness_vs_required_evidence_met": {
            "n": sum(1 for v in req_met if v is not None),
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


def compare_to_judge(deepeval_scores: list[dict], judge_means_by_trace: dict) -> dict:
    """DeepEval step_efficiency <-> judge `trajectory` dimension, per trace."""
    from dealpoint.eval.agreement import spearman

    by_trace_key = {(r.get("case_id"), r.get("variant_id")): r for r in deepeval_scores}
    common_keys = sorted(set(by_trace_key) & set(judge_means_by_trace))

    step_eff = [by_trace_key[k].get("step_efficiency", {}).get("score") for k in common_keys]
    task_completion = [by_trace_key[k].get("task_completion", {}).get("score") for k in common_keys]
    trajectory_dim = [judge_means_by_trace[k].get("trajectory") for k in common_keys]
    reasoning_dim = [judge_means_by_trace[k].get("reasoning") for k in common_keys]

    return {
        "step_efficiency_vs_judge_trajectory": {
            "spearman": spearman(step_eff, trajectory_dim),
            "n": len(common_keys),
        },
        "task_completion_vs_judge_reasoning": {
            "spearman": spearman(task_completion, reasoning_dim),
            "n": len(common_keys),
        },
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


def compare_to_human() -> dict:
    """Human comparison: `data/eval/calibration/human_scores.jsonl` is 0
    bytes -- every cell here is `"pending"`, n=0, never fabricated.
    """
    from dealpoint.config import CALIBRATION_DIR

    human_path = CALIBRATION_DIR / "human_scores.jsonl"
    n = len(_load_jsonl(human_path))
    status = "pending" if n == 0 else "available"
    return {
        "n": n,
        "status": status,
        "task_completion_vs_human": status,
        "step_efficiency_vs_human_trajectory_quality": status,
    }


def classify(comparisons: dict) -> tuple[str, str]:
    """One of `KEEP_CORE_DIAGNOSTIC` / `KEEP_OPTIONAL_ANALYSIS` /
    `REMOVE_NO_ADDED_SIGNAL`, derived from the measured correlations.

    Rule (documented, deterministic): if DeepEval's task_completion tracks
    grounded_accuracy strongly (|rho| >= 0.6, n >= 10) it adds no
    independent signal over the deterministic score -> REMOVE_NO_ADDED_SIGNAL.
    If it tracks moderately (0.3 <= |rho| < 0.6) or the sample is too small
    to tell, it is a plausible secondary cross-check -> KEEP_OPTIONAL_ANALYSIS.
    Otherwise (weak/no correlation, meaning it measures something distinct,
    or judge-comparison data exists showing added value) -> KEEP_CORE_DIAGNOSTIC.
    """
    det = comparisons.get("vs_deterministic", {})
    rho_info = det.get("task_completion_vs_grounded_accuracy", {})
    rho = rho_info.get("spearman")
    n = rho_info.get("n", 0)

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
        "difference": (
            "The judged subset is 18 cases x 6 variants (108 traces), not the brief "
            "section 2.5's 40 x 6; human calibration is n=0 ('pending'), not the brief's "
            "30 hand-scored traces. Already recorded in judges.json; restated here because "
            "every DeepEval<->human comparison in this report is 'pending'."
        ),
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
]


def build_report(
    *,
    resolved_evaluator: dict,
    subset: dict,
    per_trace_scores: list[dict],
    comparisons: dict,
    disagreements: list[dict],
    spend: dict,
    framework_versions: dict,
    omitted_metrics: tuple[dict, ...] = OMITTED_RAG_METRICS,
) -> dict:
    classification, rationale = classify(comparisons)
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
        "per_trace_scores": per_trace_scores,
        "comparisons": comparisons,
        "disagreements": disagreements,
        "classification": classification,
        "classification_rationale": rationale,
        "spend": spend,
        "decisions": [],
        "brief_differences": BRIEF_DIFFERENCES,
    }


def render_markdown(report: dict) -> str:
    lines = ["# M7a -- DeepEval independent agent-eval cross-check\n"]
    ev = report.get("resolved_evaluator", {})
    lines.append(
        f"Resolved evaluator: `{ev.get('model')}` (family `{ev.get('family')}`, "
        f"source `{ev.get('source')}`)\n"
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


def main(argv: list[str] | None = None) -> int:
    from dealpoint.eval.cases import find_case
    from dealpoint.eval.framework_versions import framework_versions
    from dealpoint.eval.spend import assert_within_cap, realized_usd
    from dealpoint.llm.client import OpenRouterClient

    subset_payload = judged_subset_variants()
    case_ids = subset_payload["case_ids"]
    variants = subset_payload["variants"]

    # Load all det rows for the subset x variants.
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
    judge_means_by_trace = load_judge_dimension_means()
    comparisons = {
        "vs_deterministic": compare_to_deterministic(per_trace_scores, det_by_row),
        "vs_judge": compare_to_judge(per_trace_scores, judge_means_by_trace),
        "vs_human": compare_to_human(),
    }
    disagreements = find_disagreements(per_trace_scores, det_by_row)

    before = realized_usd()
    spend = {
        "m7a_ledger_before": before,
        "n_traces": n_traces,
    }

    report = build_report(
        resolved_evaluator=resolved,
        subset={"case_ids": case_ids, "variants": [v["variant_id"] for v in variants], "n_traces": n_traces},
        per_trace_scores=per_trace_scores,
        comparisons=comparisons,
        disagreements=disagreements,
        spend=spend,
        framework_versions=framework_versions(),
    )

    DEEPEVAL_CROSSCHECK_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEEPEVAL_CROSSCHECK_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        fh.write("\n")
    with open(DEEPEVAL_CROSSCHECK_MD_PATH, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))

    print(f"Wrote {DEEPEVAL_CROSSCHECK_JSON_PATH} and {DEEPEVAL_CROSSCHECK_MD_PATH}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
