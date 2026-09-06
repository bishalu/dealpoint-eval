# Four-arm experiment on the frozen test set (M4)

**Version:** v2 after harness repair.

This report covers a budget-scaled, 32-case frozen discriminative subset (18 cases at Haiku) of MAUD's test set -- not the brief's full 167-case test set. Task here is document -> (answer, citation); MAUD's published task is span -> answer, so scores are not comparable to MAUD leaderboard numbers. Every score in this report is deterministic Python over expert labels; there is no model judging in M4.

## Config diff between adjacent arms

| from | to | key | from value | to value |
|---|---|---|---|---|
| A | B | loop | `pipeline` | `agent` |
| B | C | retriever | `{'kind': 'dense', 'name': 'dense', 'fetch_k': 5, 'rerank_model': None, 'multi_query': False, 'rrf_k': 60}` | `{'fetch_k': 20, 'kind': 'hybrid_rrf', 'multi_query': False, 'name': 'hybrid_rrf', 'rerank_model': None, 'rrf_k': 60}` |
| C | D | skill | `False` | `True` |

## Harness repair (M4.1)

Probe results: `data/reports/m4_1_probes.json`.

- Capture failure_detail (exception class + first 300 chars, or the validation error) and raw_final_text (first 1500 chars of the model's final response) on every EXECUTION_FAILED record, plus finish_reasons per metered call (previously discarded).
- Tolerant JSON extraction (extract_json_object): fenced code blocks, prose-wrapped JSON, a balanced-brace scan -- truncated/unbalanced JSON still fails, never 'repaired'.
- Option normalisation (match_option): canonicalise + casefold + strip a wrapping quote pair before comparing a model's answer string to the option list; on a match, finding.answer is rewritten to the exact option string (answer_correct's exact-string comparison unaffected).
- Raised MAX_TOKENS_FINAL 600 -> 1200 uniformly for every arm/model: the v1 ledger showed every one of the 21 GLM arm-A schema failures dying at output_tokens == 1200 == 2x the old ceiling.
- response_format capability table + resolver (response_format_for): json_schema where pinned, json_object otherwise; a 400 naming response_format/json_schema triggers exactly one downgrade retry to json_object, recorded in failure_detail.
- Assistant tool-call messages send content: '' instead of content: null when the model's turn had no text -- several providers reject a null content field alongside tool_calls.
- Real exponential backoff with jitter (API_RETRY_BASE_DELAY_S, default ~1.0s, honouring Retry-After) replacing the old 0.05*2^n backoff, which totalled 0.35s across 3 retries.

Majority-baseline overall accuracy: 0.0%

Majority-baseline accuracy on this subset is 0.0% by construction: data/eval/test_subset_v1.json's selection rule prefers cases where gold_answer != majority_answer (the majority already gets these wrong), so a near-zero subset baseline is not evidence the majority baseline is generally weak. For context, the majority baseline on the full 167-case MAUD test set is 46.1%.

## Execution-failure rate: v1 -> v2

v1 result files are preserved (see v1_artifacts below); v2 is this report's live arms/ data.

| model | arm | n v1 | n v2 | EXECUTION_FAILED v1 | EXECUTION_FAILED v2 | delta | CAP_HIT v1 | CAP_HIT v2 |
|---|---|---|---|---|---|---|---|---|
| anthropic/claude-haiku-4.5 | A | 18 | 18 | 5.6% | 0.0% | -5.6% | 0.0% | 0.0% |
| anthropic/claude-haiku-4.5 | D | 18 | 18 | 88.9% | 5.6% | -83.3% | 0.0% | 50.0% |
| z-ai/glm-5.3-flash | A | 32 | 32 | 65.6% | 25.0% | -40.6% | 0.0% | 0.0% |
| z-ai/glm-5.3-flash | B | 32 | 32 | 18.8% | 0.0% | -18.8% | 31.2% | 43.8% |
| z-ai/glm-5.3-flash | C | 32 | 32 | 12.5% | 0.0% | -12.5% | 25.0% | 31.2% |
| z-ai/glm-5.3-flash | D | 32 | 32 | 34.4% | 15.6% | -18.8% | 12.5% | 15.6% |

## Residual failures (v2, above the 10% threshold)

- **z-ai/glm-5.3-flash/A** (provider): 8/32 (25.0%) EXECUTION_FAILED; most common failure_detail class: 'empty response content' (4/8). 8/8 of these end with finish_reason == 'length' on both the initial attempt and the schema retry -- the model spends its whole MAX_TOKENS_FINAL=1200 budget (already doubled from 600 per this milestone's fix) on prose reasoning before ever emitting the JSON object, so raw_final_text is a truncated reasoning preamble with no JSON in it at all. This is the model's own verbosity, not a request-shape defect the harness can fix without an unbounded token ceiling (out of scope: CAP_HIT-style caps are deliberate, not to be raised without limit).
- **z-ai/glm-5.3-flash/D** (provider): 5/32 (15.6%) EXECUTION_FAILED; most common failure_detail class: 'empty response content' (4/5). 5/5 of these end with finish_reason == 'length' on both the initial attempt and the schema retry -- the model spends its whole MAX_TOKENS_FINAL=1200 budget (already doubled from 600 per this milestone's fix) on prose reasoning before ever emitting the JSON object, so raw_final_text is a truncated reasoning preamble with no JSON in it at all. This is the model's own verbosity, not a request-shape defect the harness can fix without an unbounded token ceiling (out of scope: CAP_HIT-style caps are deliberate, not to be raised without limit). 5/5 of these failed cases (and 27/32 of this leg's cases overall) finalised immediately after a tool-calling turn whose finish_reason was 'length' -- MAX_TOKENS_TOOL_TURN (300) cut that turn off before the model could keep searching, so the loop moved to finalisation with fewer tool calls used than it would otherwise have made. This is left-at-300, unfixed for this v2 report (no budget for another sweep); it is a contributing cause alongside, not instead of, final-answer verbosity.

## v1 artefacts (preserved before the v2 re-run)

- `four_arm_json`: `/home/exedev/repos/dealpoint-eval/data/reports/four_arm_v1_execution_defects.json`
- `four_arm_md`: `/home/exedev/repos/dealpoint-eval/data/reports/four_arm_v1_execution_defects.md`
- `four_arm_manifest`: `/home/exedev/repos/dealpoint-eval/data/reports/four_arm_manifest_v1.json`

## Model: `anthropic/claude-haiku-4.5`

| arm | n_cases | n_scored | grounded_accuracy | answer_correct | fabrication | abstain_correct | skill_adherence | execution_failed | cap_hit | tool_calls | usd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 18 | 10 | 50.0% (10/18) | 45.5% | 0.0% | 83.3% | 50.9% | 0.0% | 0.0% | 0.00 | $0.0034 |
| D | 18 | 8 | 62.5% (8/18) | 62.5% | 12.5% | 66.7% | 71.5% | 5.6% | 50.0% | 6.44 | $0.0458 |

## Model: `z-ai/glm-5.3-flash`

| arm | n_cases | n_scored | grounded_accuracy | answer_correct | fabrication | abstain_correct | skill_adherence | execution_failed | cap_hit | tool_calls | usd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 32 | 17 | 76.5% (17/32) | 65.0% | 0.0% | 81.2% | 53.5% | 25.0% | 0.0% | 0.00 | $0.0010 |
| B | 32 | 17 | 58.8% (17/32) | 61.1% | 11.1% | 75.0% | 64.4% | 0.0% | 43.8% | 5.38 | $0.0024 |
| C | 32 | 21 | 66.7% (21/32) | 66.7% | 19.0% | 78.1% | 67.2% | 0.0% | 31.2% | 5.03 | $0.0024 |
| D | 32 | 17 | 64.7% (17/32) | 65.0% | 35.0% | 75.0% | 64.6% | 15.6% | 15.6% | 4.81 | $0.0028 |

## Per-question metrics

One table per (model, arm), question_id x metric, with the scored n alongside every mean (corrective task #4). The majority-baseline row (label-only, never model output) is printed once below, shared by every arm.

Majority-baseline accuracy per question: q01=0.0%, q02=0.0%, q03=0.0%, q04=0.0%, q05=0.0%, q06=0.0%, q07=0.0%, q08=0.0%, q09=0.0%, q10=0.0%, q11=0.0%, q12=0.0%

### anthropic/claude-haiku-4.5 / arm A

| question_id | grounded_accuracy | answer_correct | citation_gold_overlap | gold_seen | required_evidence_met | skill_adherence | execution_failed | cap_hit |
|---|---|---|---|---|---|---|---|---|
| oos04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 33.3% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| oos09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 50.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q01 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 75.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q02 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 60.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q03 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 75.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q04 | 0.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) | n/a (n=0) | 75.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q05 | n/a (n=0) | n/a (n=0) | 0.0% (n=1) | 0.0% (n=1) | 0.0% (n=2) | 29.8% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q06 | 0.0% (n=1) | 0.0% (n=2) | 0.0% (n=1) | 100.0% (n=1) | 50.0% (n=2) | 46.7% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q07 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=2) | 44.3% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q08 | n/a (n=0) | n/a (n=0) | 0.0% (n=1) | 0.0% (n=1) | 0.0% (n=2) | 35.7% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q09 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 60.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q10 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 50.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q11 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q12 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 75.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |

### anthropic/claude-haiku-4.5 / arm D

| question_id | grounded_accuracy | answer_correct | citation_gold_overlap | gold_seen | required_evidence_met | skill_adherence | execution_failed | cap_hit |
|---|---|---|---|---|---|---|---|---|
| oos04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 66.7% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| oos09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 66.7% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| q01 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 75.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q02 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=1) | n/a (n=0) | 66.7% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| q03 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 100.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q04 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 80.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q05 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 79.2% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q06 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 58.3% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q07 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=1) | 100.0% (n=2) | 70.8% (n=2) | 50.0% (n=2) | 50.0% (n=2) |
| q08 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=1) | 100.0% (n=2) | 62.5% (n=2) | 0.0% (n=2) | 100.0% (n=2) |
| q09 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 60.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q10 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 80.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q11 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=1) | 100.0% (n=1) | 75.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| q12 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 75.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |

### z-ai/glm-5.3-flash / arm A

| question_id | grounded_accuracy | answer_correct | citation_gold_overlap | gold_seen | required_evidence_met | skill_adherence | execution_failed | cap_hit |
|---|---|---|---|---|---|---|---|---|
| oos04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 50.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| oos09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 50.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q01 | 50.0% (n=2) | 50.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 75.0% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q02 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 67.5% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q03 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 75.0% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q04 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=2) | n/a (n=0) | 62.5% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q05 | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=2) | 50.0% (n=2) | 0.0% (n=3) | 45.4% (n=3) | 0.0% (n=3) | 0.0% (n=3) |
| q06 | 0.0% (n=2) | 0.0% (n=3) | 0.0% (n=2) | 100.0% (n=2) | 33.3% (n=3) | 56.7% (n=3) | 0.0% (n=3) | 0.0% (n=3) |
| q07 | 100.0% (n=2) | 66.7% (n=3) | 100.0% (n=2) | 100.0% (n=2) | 66.7% (n=3) | 51.1% (n=3) | 0.0% (n=3) | 0.0% (n=3) |
| q08 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=2) | 33.3% (n=3) | 27.8% (n=3) | 66.7% (n=3) | 0.0% (n=3) |
| q09 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 47.2% (n=3) | 66.7% (n=3) | 0.0% (n=3) |
| q10 | 100.0% (n=1) | 50.0% (n=2) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 58.3% (n=3) | 33.3% (n=3) | 0.0% (n=3) |
| q11 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | 50.0% (n=2) | 29.2% (n=2) | 100.0% (n=2) | 0.0% (n=2) |
| q12 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 67.5% (n=2) | 0.0% (n=2) | 0.0% (n=2) |

### z-ai/glm-5.3-flash / arm B

| question_id | grounded_accuracy | answer_correct | citation_gold_overlap | gold_seen | required_evidence_met | skill_adherence | execution_failed | cap_hit |
|---|---|---|---|---|---|---|---|---|
| oos04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 66.7% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| oos09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 33.3% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| q01 | 50.0% (n=2) | 50.0% (n=2) | 50.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 75.0% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q02 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 55.0% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q03 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 70.8% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q04 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 63.3% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q05 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 61.1% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q06 | 0.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=2) | 100.0% (n=3) | 60.0% (n=3) | 0.0% (n=3) | 66.7% (n=3) |
| q07 | 50.0% (n=2) | 50.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 67.2% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q08 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 61.1% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q09 | 0.0% (n=2) | 0.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 68.9% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q10 | 100.0% (n=1) | 50.0% (n=2) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 68.9% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q11 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | 100.0% (n=2) | 58.3% (n=2) | 0.0% (n=2) | 100.0% (n=2) |
| q12 | 50.0% (n=2) | 100.0% (n=2) | 50.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 77.5% (n=2) | 0.0% (n=2) | 0.0% (n=2) |

### z-ai/glm-5.3-flash / arm C

| question_id | grounded_accuracy | answer_correct | citation_gold_overlap | gold_seen | required_evidence_met | skill_adherence | execution_failed | cap_hit |
|---|---|---|---|---|---|---|---|---|
| oos04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 66.7% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| oos09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 33.3% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| q01 | 50.0% (n=2) | 50.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 67.5% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q02 | 50.0% (n=2) | 50.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 70.0% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q03 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 83.3% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q04 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=2) | n/a (n=0) | 73.3% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q05 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 61.1% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q06 | 0.0% (n=2) | 0.0% (n=2) | 0.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 65.6% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q07 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 72.7% (n=3) | 0.0% (n=3) | 0.0% (n=3) |
| q08 | 50.0% (n=2) | 50.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 63.9% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q09 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 68.9% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q10 | 50.0% (n=2) | 50.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 62.2% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q11 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 100.0% (n=2) | 62.5% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q12 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 77.5% (n=2) | 0.0% (n=2) | 0.0% (n=2) |

### z-ai/glm-5.3-flash / arm D

| question_id | grounded_accuracy | answer_correct | citation_gold_overlap | gold_seen | required_evidence_met | skill_adherence | execution_failed | cap_hit |
|---|---|---|---|---|---|---|---|---|
| oos04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 66.7% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| oos09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 33.3% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| q01 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 90.0% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q02 | n/a (n=0) | n/a (n=0) | 0.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 63.3% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q03 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 77.5% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q04 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 73.3% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q05 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 69.4% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q06 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 100.0% (n=3) | 55.6% (n=3) | 33.3% (n=3) | 33.3% (n=3) |
| q07 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 72.7% (n=3) | 0.0% (n=3) | 0.0% (n=3) |
| q08 | 0.0% (n=1) | 50.0% (n=2) | 0.0% (n=1) | 100.0% (n=2) | 100.0% (n=3) | 55.6% (n=3) | 33.3% (n=3) | 0.0% (n=3) |
| q09 | 50.0% (n=2) | 33.3% (n=3) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 66.7% (n=3) | 0.0% (n=3) | 0.0% (n=3) |
| q10 | 0.0% (n=2) | 33.3% (n=3) | 50.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 56.7% (n=3) | 0.0% (n=3) | 0.0% (n=3) |
| q11 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 100.0% (n=2) | 55.0% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q12 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 60.0% (n=2) | 0.0% (n=2) | 0.0% (n=2) |

## Honesty-rule verdicts

- **anthropic/claude-haiku-4.5 / A_to_D**: At anthropic/claude-haiku-4.5, arm D improves grounded_accuracy over arm A by 12.5% (50.0% -> 62.5%).
- **z-ai/glm-5.3-flash / A_to_B**: At z-ai/glm-5.3-flash, arm B does NOT improve grounded_accuracy over arm A (76.5% -> 58.8%, delta -17.6%). See adherence, fabrication, abstention, trajectory and efficiency deltas below instead.
- **z-ai/glm-5.3-flash / B_to_C**: At z-ai/glm-5.3-flash, arm C improves grounded_accuracy over arm B by 7.8% (58.8% -> 66.7%).
- **z-ai/glm-5.3-flash / C_to_D**: At z-ai/glm-5.3-flash, arm D does NOT improve grounded_accuracy over arm C (66.7% -> 64.7%, delta -2.0%). See adherence, fabrication, abstention, trajectory and efficiency deltas below instead.
- **z-ai/glm-5.3-flash / A_to_D**: At z-ai/glm-5.3-flash, arm D does NOT improve grounded_accuracy over arm A (76.5% -> 64.7%, delta -11.8%). See adherence, fabrication, abstention, trajectory and efficiency deltas below instead.

## Adherence / fabrication / abstention / trajectory / efficiency deltas

### anthropic/claude-haiku-4.5 / A_to_D

| metric | from | to | delta |
|---|---|---|---|
| skill_adherence | 0.508994708994709 | 0.7148148148148148 | 0.2058201058201058 |
| fabrication | 0.0 | 0.125 | 0.125 |
| abstain_correct | 0.8333333333333334 | 0.6666666666666666 | -0.16666666666666674 |
| tool_calls | 0.0 | 6.444444444444445 | 6.444444444444445 |
| wall_ms | 3693.0 | 25830.5 | 22137.5 |
| usd | 0.003355777777777778 | 0.045795999999999996 | 0.04244022222222222 |

### z-ai/glm-5.3-flash / A_to_B

| metric | from | to | delta |
|---|---|---|---|
| skill_adherence | 0.5352678571428572 | 0.6442708333333333 | 0.10900297619047616 |
| fabrication | 0.0 | 0.1111111111111111 | 0.1111111111111111 |
| abstain_correct | 0.8125 | 0.75 | -0.0625 |
| tool_calls | 0.0 | 5.375 | 5.375 |
| wall_ms | 12858.34375 | 27416.46875 | 14558.125 |
| usd | 0.0009901984375 | 0.00242948859375 | 0.00143929015625 |

### z-ai/glm-5.3-flash / B_to_C

| metric | from | to | delta |
|---|---|---|---|
| skill_adherence | 0.6442708333333333 | 0.6723214285714285 | 0.028050595238095166 |
| fabrication | 0.1111111111111111 | 0.19047619047619047 | 0.07936507936507936 |
| abstain_correct | 0.75 | 0.78125 | 0.03125 |
| tool_calls | 5.375 | 5.03125 | -0.34375 |
| wall_ms | 27416.46875 | 24015.15625 | -3401.3125 |
| usd | 0.00242948859375 | 0.002404935 | -2.4553593750000196e-05 |

### z-ai/glm-5.3-flash / C_to_D

| metric | from | to | delta |
|---|---|---|---|
| skill_adherence | 0.6723214285714285 | 0.6462797619047619 | -0.02604166666666663 |
| fabrication | 0.19047619047619047 | 0.35 | 0.1595238095238095 |
| abstain_correct | 0.78125 | 0.75 | -0.03125 |
| tool_calls | 5.03125 | 4.8125 | -0.21875 |
| wall_ms | 24015.15625 | 33117.71875 | 9102.5625 |
| usd | 0.002404935 | 0.002775360859375 | 0.00037042585937500016 |

### z-ai/glm-5.3-flash / A_to_D

| metric | from | to | delta |
|---|---|---|---|
| skill_adherence | 0.5352678571428572 | 0.6462797619047619 | 0.1110119047619047 |
| fabrication | 0.0 | 0.35 | 0.35 |
| abstain_correct | 0.8125 | 0.75 | -0.0625 |
| tool_calls | 0.0 | 4.8125 | 4.8125 |
| wall_ms | 12858.34375 | 33117.71875 | 20259.375 |
| usd | 0.0009901984375 | 0.002775360859375 | 0.001785162421875 |

## Paired per-case flips

- **anthropic/claude-haiku-4.5 / A_to_D**: gained ['contract_32__q04', 'contract_32__q06'], lost [], unchanged_correct=2, unchanged_wrong=3
- **z-ai/glm-5.3-flash / A_to_B**: gained [], lost ['contract_103__q02', 'contract_77__q12', 'contract_7__q07'], unchanged_correct=8, unchanged_wrong=3
- **z-ai/glm-5.3-flash / B_to_C**: gained ['contract_144__q09', 'contract_39__q09', 'contract_77__q12', 'contract_7__q07'], lost ['contract_32__q08'], unchanged_correct=9, unchanged_wrong=3
- **z-ai/glm-5.3-flash / C_to_D**: gained [], lost ['contract_103__q08', 'contract_103__q10', 'contract_39__q09'], unchanged_correct=10, unchanged_wrong=2
- **z-ai/glm-5.3-flash / A_to_D**: gained [], lost ['contract_103__q08', 'contract_103__q10'], unchanged_correct=9, unchanged_wrong=2

## Skill-adherence per-rule detail

Note (honesty, spec §3): arm A lacks several tools rules 2/3/4/6 need, so its applicable-rule set is structurally smaller than arms B/C/D's. n_applicable and n_satisfied are printed per rule per arm, so an A-vs-D adherence ratio comparison is never read as like-for-like.

### anthropic/claude-haiku-4.5

| arm | rule | n_applicable | n_satisfied | rate |
|---|---|---|---|---|
| A | 1 | 18 | 18 | 100.0% |
| A | 2 | 9 | 0 | 0.0% |
| A | 3 | 18 | 0 | 0.0% |
| A | 4 | 10 | 0 | 0.0% |
| A | 5 | 11 | 11 | 100.0% |
| A | 6 | 7 | 0 | 0.0% |
| A | 7 | 7 | 5 | 71.4% |
| A | 8 | 18 | 13 | 72.2% |
| D | 1 | 18 | 18 | 100.0% |
| D | 2 | 9 | 9 | 100.0% |
| D | 3 | 18 | 13 | 72.2% |
| D | 4 | 14 | 0 | 0.0% |
| D | 5 | 8 | 7 | 87.5% |
| D | 6 | 0 | 0 | n/a |
| D | 7 | 0 | 0 | n/a |
| D | 8 | 8 | 7 | 87.5% |

### z-ai/glm-5.3-flash

| arm | rule | n_applicable | n_satisfied | rate |
|---|---|---|---|---|
| A | 1 | 32 | 32 | 100.0% |
| A | 2 | 14 | 0 | 0.0% |
| A | 3 | 32 | 0 | 0.0% |
| A | 4 | 13 | 0 | 0.0% |
| A | 5 | 20 | 20 | 100.0% |
| A | 6 | 4 | 0 | 0.0% |
| A | 7 | 4 | 3 | 75.0% |
| A | 8 | 24 | 21 | 87.5% |
| B | 1 | 32 | 32 | 100.0% |
| B | 2 | 14 | 13 | 92.9% |
| B | 3 | 32 | 12 | 37.5% |
| B | 4 | 24 | 0 | 0.0% |
| B | 5 | 18 | 16 | 88.9% |
| B | 6 | 0 | 0 | n/a |
| B | 7 | 0 | 0 | n/a |
| B | 8 | 18 | 17 | 94.4% |
| C | 1 | 32 | 32 | 100.0% |
| C | 2 | 14 | 14 | 100.0% |
| C | 3 | 32 | 13 | 40.6% |
| C | 4 | 27 | 2 | 7.4% |
| C | 5 | 21 | 17 | 81.0% |
| C | 6 | 1 | 1 | 100.0% |
| C | 7 | 1 | 1 | 100.0% |
| C | 8 | 22 | 21 | 95.5% |
| D | 1 | 32 | 32 | 100.0% |
| D | 2 | 14 | 14 | 100.0% |
| D | 3 | 32 | 13 | 40.6% |
| D | 4 | 25 | 0 | 0.0% |
| D | 5 | 20 | 13 | 65.0% |
| D | 6 | 2 | 2 | 100.0% |
| D | 7 | 2 | 1 | 50.0% |
| D | 8 | 22 | 21 | 95.5% |

## Cost: estimate vs realised

Target: $1.5, absolute: $3.0, envelope: $4.0, M4 ledger total: $0.751879
M4.1 ledger total: $1.420234 (target $1.0, absolute $1.5); total ledger: $3.713288.
Estimate before the v2 sweeps: `{"arms": ["A", "B", "C", "D"], "basis": "ledger:measured(arm,model)", "calls": 656, "cases": 164, "est_usd": 0.907372, "legs": [{"arms": ["A", "B", "C", "D"], "models": ["z-ai/glm-5.3-flash"], "n_cases": 32}, {"arms": ["A", "D"], "models": ["anthropic/claude-haiku-4.5"], "n_cases": 18}], "model": ["z-ai/glm-5.3-flash", "anthropic/claude-haiku-4.5"], "per_call_usd": 0.001383, "per_case_usd": 0.005533, "sweep": "four_arm"}`

Caveat: this estimate's `ledger:measured(arm,model)` basis is biased LOW for Haiku arm D, because v1's arm-D cases aborted early (16/18 EXECUTION_FAILED) -- the mean per-case cost measured from those short, mostly-failed executions understates what a full 8-tool-call run costs. This is why the Haiku arm-D leg estimated $0.673436 and realised $0.824328 -- the only leg in the v2 sweeps to overrun its own estimate.

| leg | arm | model | tranche | n_cases | est_usd | realized_usd |
|---|---|---|---|---|---|---|
| headline_glm | A | z-ai/glm-5.3-flash | full | 32 | 0.048405 | 0.031689 |
| headline_glm | B | z-ai/glm-5.3-flash | full | 32 | 0.13938 | 0.077744 |
| headline_glm | C | z-ai/glm-5.3-flash | full | 32 | 0.117363 | 0.076958 |
| headline_glm | D | z-ai/glm-5.3-flash | full | 32 | 0.086601 | 0.088812 |
| replication_haiku | A | anthropic/claude-haiku-4.5 | tranche_1 | 18 | 0.060393 | 0.060404 |
| replication_haiku | D | anthropic/claude-haiku-4.5 | tranche_1 | 18 | 0.673436 | 0.824328 |

## Prompt-caching diagnosis

```json
{
  "any_cached": false,
  "cached_tokens_observed": [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
  ],
  "cost_per_call_usd_after": 0.003092,
  "cost_per_call_usd_before": 0.001914,
  "decision": "Not padded. The spec explicitly forbids padding a prompt to reach a cacheable minimum.",
  "finding": "The static prefix (system prompt + question spec + tool schemas) measures ~511-609 tokens across the 12 questions, well below Anthropic's 2048-token minimum cacheable prefix for the Haiku class. cached_tokens was 0 on every call in this probe (and on all 96 pre-existing ledger rows), consistent with the prefix being below the provider's minimum -- not a request-shape bug. The request shape is already correct (dealpoint.llm.client.build_system_message sends the system message as a content-block list with cache_control: {'type': 'ephemeral'}), so no block-placement fix was needed or applied.",
  "measured_static_prefix_tokens_range": "511-609 (approx, chars/4.8 across the 12 questions)",
  "model": "anthropic/claude-haiku-4.5",
  "n_calls_in_probe": 13,
  "provider_minimum_cacheable_tokens": 2048
}
```

## Notes

- Arm C is labelled `agent-hybrid-rrf` here, not the brief's `agent-hybrid-rerank`: the M3 tournament winner is hybrid RRF *without* rerank (hit@5 0.9138 vs 0.8103 for dense; also beat hybrid_rrf_rerank on MRR). The measurement governs per the milestone spec; see dealpoint/config.py for the full rationale.
- Spec-internal inconsistency (not spec-vs-brief): the re-sized deliverable 5 and the Definition of Done both name the GLM headline as the constant-model comparison with Haiku as the A/D replication; an older paragraph in the same spec says the reverse. The later, more specific instruction (deliverable 5 + DoD) is followed.
- Brief §7's relative gate ('no arm regresses vs its previously recorded dev result') is vacuous this milestone: M2's only recorded run is 3 dev cases at arm B, not a four-arm baseline to compare against.
- Arms B and C at Haiku: not run (budget). Arithmetic: 2 arms x 18 cases at the measured Haiku per-case cost (~$0.0227/case) is ~$0.82; adding that to the $0.75 already realised for M4 would take the milestone to ~$1.57, past the $1.50 target (though still under the $3.00 absolute) -- so they were deliberately skipped.
- `dealpoint/eval/run.py` gained a per-process retriever cache after the final sweep finished. This is a runner-only fix (it prevents a second `QdrantClient` from wedging on the same on-disk index within one process) and does not change what any model saw or any score computed -- no test case was re-run because of it.
- Every headline number in this report is from a 32-case frozen subset (18 cases at Haiku), not the brief's full 167-case test set -- budget-scaled per the engineer's 2026-09-04 instruction, never the full benchmark.
- M4.1 request-shape decisions: `MAX_TOKENS_FINAL` raised 600 -> 1200 uniformly for every arm/model (the v1 evidence for this: every one of the 21 GLM arm-A schema failures died at output_tokens == 1200, exactly 2x the old ceiling). 
- `MAX_TOKENS_TOOL_TURN` was LEFT at 300 despite evidence that it DOES truncate tool turns and force premature finalisation. In data/reports/m4_1_probes.json's after2 round: z-ai/glm-5.3-flash/B (contract_32__q08, contract_39__q01) and anthropic/claude-haiku-4.5/D (contract_7__q07) each have a finish_reasons entry of 'length' immediately after their last 'tool_calls' turn, with a matching data/results/spend_ledger.jsonl row at exactly output_tokens == 300, and the loop then finalised with only 6/8, 4/8 and 3/8 tool calls used respectively. Across the v2 sweeps the same pattern recurs at scale (computed from the result rows): 17/32 (z-ai/glm-5.3-flash B), 19/32 (z-ai/glm-5.3-flash C), 27/32 (z-ai/glm-5.3-flash D), 7/18 (anthropic/claude-haiku-4.5 D) of this milestone's cases finalised immediately after a tool turn that ended on 'length', and 83 v2 sweep calls ended at exactly 300 output tokens (data/results/spend_ledger.jsonl). The constant was NOT raised because the M4.1 budget ($1.4202 of the $1.50 absolute) leaves no room for another sweep -- this is a recorded, unfixed limitation of the v2 numbers, not a finding that truncation is harmless.
