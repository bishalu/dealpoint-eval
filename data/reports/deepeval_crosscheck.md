# M7a -- DeepEval independent agent-eval cross-check

Resolved evaluator: `mistralai/mistral-small-3.2-24b-instruct` (family `Mistral`, source `judge_slate.json (verified judges, non-candidate-family)`)

Subset: 18 cases x 6 variants = 108 traces

## Metrics used

- `task_completion`
- `tool_correctness`
- `argument_correctness`
- `step_efficiency`

step_efficiency basis: deepeval-native:StepEfficiencyMetric

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
      "n": 42,
      "spearman": 0.3814307332915905
    },
    "tool_correctness_vs_required_evidence_met": {
      "n": 54
    }
  },
  "vs_human": {
    "n": 0,
    "status": "pending",
    "step_efficiency_vs_human_trajectory_quality": "pending",
    "task_completion_vs_human": "pending"
  },
  "vs_judge": {
    "step_efficiency_vs_judge_trajectory": {
      "n": 108,
      "spearman": -0.009681525404957505
    },
    "task_completion_vs_judge_reasoning": {
      "n": 108,
      "spearman": 0.8423373253364367
    }
  }
}
```

## Disagreement cases (17)

| case_id | variant_id | metric | deepeval_score | deterministic_score | direction |
|---|---|---|---|---|---|
| contract_32__q04 | A@haiku | task_completion_vs_grounded_accuracy | 0.7 | False | deepeval_pass_deterministic_fail |
| contract_32__q06 | A@haiku | task_completion_vs_grounded_accuracy | 0.5 | False | deepeval_pass_deterministic_fail |
| contract_144__q09 | A@haiku | task_completion_vs_grounded_accuracy | 0.7 | False | deepeval_pass_deterministic_fail |
| contract_39__q01 | A@haiku | task_completion_vs_grounded_accuracy | 0.9 | False | deepeval_pass_deterministic_fail |
| contract_144__q09 | D@haiku | task_completion_vs_grounded_accuracy | 0.7 | False | deepeval_pass_deterministic_fail |
| contract_39__q01 | D@haiku | task_completion_vs_grounded_accuracy | 0.9 | False | deepeval_pass_deterministic_fail |
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
| contract_39__q01 | D@gemini-3.1-flash-lite | task_completion_vs_grounded_accuracy | 0.9 | False | deepeval_pass_deterministic_fail |

## Classification: `KEEP_OPTIONAL_ANALYSIS`

DeepEval task_completion correlates moderately with grounded_accuracy (rho=0.381, n=42) -- plausible as a secondary cross-check but not strong enough evidence to call it a core diagnostic.

## Brief-vs-spec differences

- **deepeval_brief_exclusion**: Brief section 3 lists DeepEval under 'Not used' and section 4 excludes it outright. M7a's authority line amends section 4 as of 2026-09-05; the classification field this module produces is the mechanism by which that amendment is tested rather than assumed -- REMOVE_NO_ADDED_SIGNAL is a legitimate outcome of this cross-check, not a foregone conclusion.
- **dual_tracing_risk**: Brief section 4 excludes dual tracing paths. DeepEval ships OpenTelemetry and posthog telemetry. This module opts out of DeepEval telemetry (DEEPEVAL_TELEMETRY_OPT_OUT=1) before any import, never logs to a second trace store, and writes results only to local JSON plus deepeval/-namespaced Braintrust scores -- Braintrust remains primary observability.
- **scale_inherited**: The judged subset is 18 cases x 6 variants (108 traces), not the brief section 2.5's 40 x 6; human calibration is n=0 ('pending'), not the brief's 30 hand-scored traces. Already recorded in judges.json; restated here because every DeepEval<->human comparison in this report is 'pending'.
- **score_budget**: Brief section 2.7 caps Braintrust logging at <= 6 scores/case. M7a permits <= 12 scores/case on subsets <= 60 cases as a ceiling, not a target, with M4/M6 sweeps still logging exactly six -- enforced at sync time by dealpoint.eval.braintrust_sync.assert_score_budget.

