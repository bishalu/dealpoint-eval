# M5 judges report

These are **model-judged secondary scores, budget-scaled**, and are never presented as objective truth. Human calibration is **pending** unless stated otherwise below.

`rubric_version`: `cfda9f8cc401`  
`git_sha7`: `b9eac78`

## Judged subset

18 cases x 6 variants = 108 traces; 324 judge calls. subset_hash=`5918ef10a7e6`.

## Judges

| model | family | n_scored | n_failed | substituted_for |
|---|---|---|---|---|
| bytedance-seed/seed-2.0-mini | ByteDance | 108 | 0 | None |
| mistralai/mistral-small-3.2-24b-instruct | Mistral | 108 | 0 | None |
| nvidia/nemotron-3-super-120b-a12b | NVIDIA | 108 | 0 | None |

## Per-dimension aggregates (mean of judges)

| dimension | mean | n |
|---|---|---|
| reasoning | 2.9012345679012346 | 108 |
| evidence | 2.611111111111111 | 108 |
| trajectory | 3.2376543209876547 | 108 |
| professional | 2.9043209876543212 | 108 |

## Correlation with grounded_accuracy

| dimension | pearson | spearman | n | caveat |
|---|---|---|---|---|
| reasoning | 0.31518222734965157 | 0.2838445400671935 | 48 |  |
| evidence | 0.4968275423500662 | 0.42040715868343176 | 48 |  |
| trajectory | 0.015921225599434503 | -0.019874319670401073 | 48 |  |
| professional | 0.378998976113394 | 0.3563293874919645 | 48 |  |

## Human calibration

**pending.** Pending human input: `score data/eval/calibration/form.md`.

## Recorded decisions

- **D1**: Judged subset is 18 cases (test_subset_v1 tranche_1), not the spec's 12-case default. -- The spec's default is 12 cases (1 per question). The same spec's latitude floor requires 'every question keeps >= 1 case and counterfactuals stay in'. A 12-case, one-per-question set drops every counterfactual, so it cannot satisfy the floor. tranche_1 (12 test cases + 4 redacted + 2 out-of-scope = 18) is the smallest set that satisfies both; the 6 extra cases cost ~$0.05.
- **D2**: Three variants judged (A@Haiku, D@Haiku, D@GLM), not the spec's two. -- Spec deliverable 4 names arm A@Haiku and arm D@Haiku. The engineer's 'Judge trio' instruction separately says 'judge the default and GLM in M5'. Both are satisfied by judging A@Haiku, D@Haiku, D@GLM: 18 cases x 3 variants = 54 traces, 54 x 3 judges = 162 calls. This costs ~$0.045 more and gives M6 a judged data point for a model it reuses rather than re-runs.
- **D3**: Agreement statistics (Spearman, quadratic-weighted kappa, Pearson) are pure Python, not scipy. -- scipy is not installed on this CPU-only VM (~3.5 GB free disk) and must not be added; the brief's architecture table also keeps scorers framework-free.
- **D4**: M5 targets a realised spend well under its $0.60 guide. -- Only a small fraction of the whole $4.00 M1-M6 envelope remained by the time M5 ran; sizing M5's judging (162 calls at the measured per-call cost) leaves headroom for M6, which is recorded under `spend.envelope_headroom_after_m5` below.
- **D5**: Mean-of-judges is rounded to the nearest integer for kappa only, not for Spearman. -- Weighted Cohen's kappa is defined on integer ratings; the mean of three judges is not one. Spearman rho uses the raw mean; kappa uses round() of it (Python's round-half-to-even).
- **D6**: M6 extended the judged subset from 3 variants (54 traces, 162 calls) to 3 + k variants, one per completed Pareto sweep model. -- M6 spec deliverable 3 requires the M5 harness to judge arm D at every completed Pareto model on the same 18-case judged subset, so results are comparable to the three M5 variants. The 18-case universe, its ranking and its subset_hash are unchanged (fixed before any M6 run); only `variants` grows, in pareto_manifest.json order.

## Brief-vs-spec differences

- **scale**: Brief section 2.5 specifies 40 stratified cases, 6 variants, 240 traces, 720 judge calls, 30 hand-scored traces, est. $15-40. This milestone runs 18 cases, 3 variants, 54 traces, 162 calls at a small fraction of that cost. Every judged number here is budget-scaled and is never presented as the brief's benchmark.
- **judge_families**: Brief section 2.5 gives 'e.g. GPT + Grok + Qwen' as the example judge trio while also requiring families not in the candidate-agent slate. The M6 candidate slate now contains OpenAI, xAI and Alibaba/Qwen, so the brief's own illustrative example would violate the brief's own rule. The engineer's trio (Mistral, NVIDIA, ByteDance; spare Amazon) satisfies the rule; the brief's rule is honoured, its illustrative example is not, because the candidate slate changed after the brief was written.
- **human_calibration_count**: Brief section 2.5 mandates 30 hand-scored traces (~5 per variant); this milestone's spec sets 'whatever is judged, target >= 12'. 54 packets are produced, and form.md names a deterministic suggested minimum of 12. The gap between 12 and the brief's 30 is a reported difference, carried in this milestone's pending human input.
- **output_keys**: Brief Appendix C fixes the judge's JSON as {reasoning:int, evidence:int, trajectory:int, professional:int, notes:str}. Those exact key names are used, with no renaming and no extra model-supplied keys.

## Blinding limitations

- The packet blinds arm and model identity (the spec's requirement) but cannot blind the SHAPE of the trajectory: arm A is a single-shot pipeline with no tool calls, arm D is an agent loop, so a knowledgeable reader can often infer which family of system produced a trace from trajectory length and tool-call pattern alone.
- An empty `options` list hints that a case is an out-of-scope probe, even though the exact question_id and category are never shown.

## Spend

```json
{
  "absolute_usd": 1.2,
  "assumed_input_tokens_per_call": 8000,
  "envelope_headroom_after_m5": 0.286712,
  "envelope_headroom_after_m6": 0.286712,
  "envelope_realized_usd": 3.713288,
  "envelope_usd": 4.0,
  "est_basis": "tokens_per_call x live",
  "est_usd": 0.12852,
  "m6_realized_usd": 0.742655,
  "measured_input_tokens_per_call": 4510.9,
  "realized_usd": 0.110038,
  "target_usd": 0.6
}
```

## Caveats

- Judge scores are secondary and model-judged; they are never presented as objective truth.
- Budget-scaled: 18 cases x 3 variants x 3 judges, not the brief's 40 x 6 x 3.
- Human calibration is pending until data/eval/calibration/human_scores.jsonl carries scores; no judge-human agreement figure exists yet.
