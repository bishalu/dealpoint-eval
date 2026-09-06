# M7a -- DeepEval independent agent-eval cross-check

Resolved evaluator: `mistralai/mistral-small-3.2-24b-instruct` (provider `mistralai`, family `Mistral`, deepeval `4.2.1`, source `judge_slate.json (verified judges, non-candidate-family)`)

**Independence caveat:** this evaluator model is one of the three M5 judge-trio models, not a fourth independent model -- see Decisions.

Subset: 18 cases x 6 variants = 108 traces

## Metrics used

- `task_completion`
- `tool_correctness`
- `argument_correctness`
- `step_efficiency`

step_efficiency basis: documented GEval

## Coverage (per-metric scored/null counts)

| metric | n_scored | n_null | null reasons |
|---|---|---|---|
| argument_correctness | 105 | 3 | RateLimitError=3 |
| step_efficiency | 107 | 1 | RateLimitError=1 |
| task_completion | 105 | 3 | RateLimitError=3 |
| tool_correctness | 107 | 1 | RateLimitError=1 |

## Spend

```json
{
  "calibration_n": 3,
  "calibration_realized_usd": 0.00104,
  "captured_at": "2026-09-06T18:51:16.331455+00:00",
  "deepeval_realized_usd": 0.082826,
  "global_realized_usd": 3.75583,
  "m7a_realized_usd": 0.14189,
  "n_traces": 108,
  "projected_full_usd": 0.03744
}
```

## Decisions

- **step_efficiency_geval_fallback**: step_efficiency always uses the documented GEval definition, never DeepEval-native StepEfficiencyMetric, because that metric requires DeepEval's @observe trace capture (a second tracing path the brief excludes); see build_metrics's docstring for the measured failure mode this avoids (93/108 traces scoring 0.0 with 'no trace provided').
- **evaluator_independence_caveat**: The resolved evaluator (mistralai/mistral-small-3.2-24b-instruct) is one of the three M5 judge-trio models (judge_slate.json), not a fourth independent model -- all three verified, non-candidate-family judges ARE the trio, so there is no alternative to switch to under the existing resolution rule (which this repair does not change, per the milestone spec). task_completion_vs_judge_reasoning and any other vs_judge comparison are therefore contaminated by shared-model bias: DeepEval's 'independent agent evaluator' shares a model with one third of the panel it is being compared against. Factored into the classification below.

## Omitted RAG metrics (deliberate)

- **FaithfulnessMetric**: Measures whether the answer is supported by retrieved context using an LLM judge over paraphrased claims -- weaker than exact-quote gold-span overlap, which is expert-labelled and character-exact.
- **AnswerRelevancyMetric**: Measures topical relevance of the answer to the question -- already implied by answer_correct against the fixed MAUD option list, and less precise for a closed-option-set task.
- **ContextualPrecisionMetric**: Measures whether relevant context is ranked above irrelevant context -- the M7a LlamaIndex RAG lab (li_rag_eval.json) already answers this directly against the same gold-span truth, at retriever granularity.
- **ContextualRecallMetric**: Measures whether all needed context was retrieved -- same duplication concern as ContextualPrecisionMetric; the RAG lab's hit_rate/mrr already cover this against gold spans.

## Comparisons

```json
{
  "vs_deterministic": {
    "max_tool_calls": 8,
    "n_at_or_over_tool_cap": 41,
    "task_completion_vs_grounded_accuracy": {
      "n": 45,
      "spearman": 0.2108247230110294
    },
    "tool_correctness_vs_required_evidence_met": {
      "agreement_rate": 0.8888888888888888,
      "n": 54,
      "note": "available_tools (search_agreement, get_section, lookup_defined_term) is now passed to ToolCorrectnessMetric so both the tool-CALL and tool-SELECTION halves of the metric are exercised.",
      "spearman": 0.3952847075210474
    }
  },
  "vs_human": {
    "n": 24,
    "status": "available",
    "step_efficiency_vs_human_trajectory_quality": {
      "n": 24,
      "spearman": 0.355900254571165
    },
    "task_completion_vs_human": {
      "n": 24,
      "spearman": 0.8710153084573097
    }
  },
  "vs_judge": {
    "argument_correctness_vs_judge_professional": {
      "n": 105,
      "spearman": 0.10991312619527821
    },
    "step_efficiency_vs_judge_trajectory": {
      "n": 107,
      "spearman": 0.5365550834618981
    },
    "task_completion_vs_judge_reasoning": {
      "n": 105,
      "spearman": 0.8391613113478562
    },
    "tool_correctness_vs_judge_evidence": {
      "n": 107,
      "spearman": 0.28214900558598555
    }
  }
}
```

## Disagreement cases (18)

| case_id | variant_id | metric | deepeval_score | deterministic_score | direction |
|---|---|---|---|---|---|
| contract_32__q04 | A@haiku | task_completion_vs_grounded_accuracy | 0.7 | False | deepeval_pass_deterministic_fail |
| contract_32__q06 | A@haiku | task_completion_vs_grounded_accuracy | 0.7 | False | deepeval_pass_deterministic_fail |
| contract_103__q10 | A@haiku | task_completion_vs_grounded_accuracy | 0.6 | False | deepeval_pass_deterministic_fail |
| contract_144__q09 | A@haiku | task_completion_vs_grounded_accuracy | 0.7 | False | deepeval_pass_deterministic_fail |
| contract_39__q01 | A@haiku | task_completion_vs_grounded_accuracy | 0.9 | False | deepeval_pass_deterministic_fail |
| contract_32__q06 | D@haiku | task_completion_vs_grounded_accuracy | 0.3 | True | deepeval_fail_deterministic_pass |
| contract_39__q01 | D@haiku | task_completion_vs_grounded_accuracy | 0.95 | False | deepeval_pass_deterministic_fail |
| contract_32__q06 | D@glm | task_completion_vs_grounded_accuracy | 1.0 | False | deepeval_pass_deterministic_fail |
| contract_103__q10 | D@glm | task_completion_vs_grounded_accuracy | 0.7 | False | deepeval_pass_deterministic_fail |
| contract_32__q06 | D@deepseek-v4-flash | task_completion_vs_grounded_accuracy | 0.5 | True | deepeval_fail_deterministic_pass |
| contract_144__q09 | D@deepseek-v4-flash | task_completion_vs_grounded_accuracy | 0.9 | False | deepeval_pass_deterministic_fail |
| contract_4__q12 | D@deepseek-v4-flash | task_completion_vs_grounded_accuracy | 1.0 | False | deepeval_pass_deterministic_fail |
| contract_32__q03 | D@qwen3.7-flash | task_completion_vs_grounded_accuracy | 1.0 | False | deepeval_pass_deterministic_fail |
| contract_39__q01 | D@qwen3.7-flash | task_completion_vs_grounded_accuracy | 0.7 | False | deepeval_pass_deterministic_fail |
| contract_4__q12 | D@qwen3.7-flash | task_completion_vs_grounded_accuracy | 1.0 | False | deepeval_pass_deterministic_fail |
| contract_32__q06 | D@gemini-3.1-flash-lite | task_completion_vs_grounded_accuracy | 0.7 | False | deepeval_pass_deterministic_fail |
| contract_144__q09 | D@gemini-3.1-flash-lite | task_completion_vs_grounded_accuracy | 1.0 | False | deepeval_pass_deterministic_fail |
| contract_39__q01 | D@gemini-3.1-flash-lite | task_completion_vs_grounded_accuracy | 1.0 | False | deepeval_pass_deterministic_fail |

## Classification: `KEEP_OPTIONAL_ANALYSIS`

DeepEval task_completion correlates weakly with grounded_accuracy (rho=0.211, n=45), which would otherwise support KEEP_CORE_DIAGNOSTIC -- but the resolved evaluator shares a model with the M5 judge trio it is compared against (see decisions), so any apparent independence from vs_judge agreement is not trustworthy. Downgraded to KEEP_OPTIONAL_ANALYSIS pending a genuinely independent evaluator model.

## Brief-vs-spec differences

- **deepeval_brief_exclusion**: Brief section 3 lists DeepEval under 'Not used' and section 4 excludes it outright. M7a's authority line amends section 4 as of 2026-09-05; the classification field this module produces is the mechanism by which that amendment is tested rather than assumed -- REMOVE_NO_ADDED_SIGNAL is a legitimate outcome of this cross-check, not a foregone conclusion.
- **dual_tracing_risk**: Brief section 4 excludes dual tracing paths. DeepEval ships OpenTelemetry and posthog telemetry. This module opts out of DeepEval telemetry (DEEPEVAL_TELEMETRY_OPT_OUT=1) before any import, never logs to a second trace store, and writes results only to local JSON plus deepeval/-namespaced Braintrust scores -- Braintrust remains primary observability.
- **scale_inherited**: The judged subset is 18 cases x 6 variants (108 traces), not the brief section 2.5's 40 x 6; human calibration is n=24 hand-scored traces, 24 of the brief's 30, not 'pending' -- DeepEval<->human comparisons in this report carry a real {n, spearman} statistic wherever a trace has both a DeepEval score and a human score. Already recorded in judges.json; restated here because this report's own vs_human field is the thing that changes when human calibration grows.
- **score_budget**: Brief section 2.7 caps Braintrust logging at <= 6 scores/case. M7a permits <= 12 scores/case on subsets <= 60 cases as a ceiling, not a target, with M4/M6 sweeps still logging exactly six -- enforced at sync time by dealpoint.eval.braintrust_sync.assert_score_budget.
- **evaluator_independence**: The spec calls DeepEval an 'independent agent evaluator'. The resolved evaluator model is drawn from judge_slate.json's verified, non-candidate-family judges -- which is exactly the M5 judge trio, so the resolved model is always one of the three judges it is compared against, never a fourth independent model. This is a consequence of the spec's own resolution rule ('the resolved provider/model/version... read from the existing judge configuration, never hardcoded'), not a bug in this repair; the caveat is recorded here, in the report's `decisions`, and factored into `classify()` because it is not otherwise visible from the numbers.

