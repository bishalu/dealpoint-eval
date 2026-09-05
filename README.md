# dealpoint-eval

M&A deal-point review agent evaluation harness (MAUD v1, CC BY 4.0, Zenodo 7500064).

## Results

<!-- BEGIN RESULTS -->
**v2 after harness repair.** Results below are from a **budget-scaled** 32-case frozen discriminative subset of MAUD's test set (18 cases at `anthropic/claude-haiku-4.5`), not the brief's full 167-case design; the task here is document -> (answer, citation) while MAUD's published task is span -> answer, so scores are **not comparable** to MAUD leaderboard numbers, and every metric below is **objective** (deterministic Python over expert labels) -- M4 has no model judging.

Majority-baseline overall accuracy: 0.0%.

Majority-baseline accuracy on this subset is 0.0% by construction: data/eval/test_subset_v1.json's selection rule prefers cases where gold_answer != majority_answer (the majority already gets these wrong), so a near-zero subset baseline is not evidence the majority baseline is generally weak. For context, the majority baseline on the full 167-case MAUD test set is 46.1%.

**Model `anthropic/claude-haiku-4.5`:**

| arm | n_cases | n_scored | grounded_accuracy |
|---|---|---|---|
| A | 18 | 10 | 50.0% (10/18) |
| D | 18 | 8 | 62.5% (8/18) |

Overall EXECUTION_FAILED rate by arm: A=0.0%, D=5.6%. Overall CAP_HIT rate by arm: A=0.0%, D=50.0%.

**Model `z-ai/glm-5.3-flash`:**

| arm | n_cases | n_scored | grounded_accuracy |
|---|---|---|---|
| A | 32 | 17 | 76.5% (17/32) |
| B | 32 | 17 | 58.8% (17/32) |
| C | 32 | 21 | 66.7% (21/32) |
| D | 32 | 17 | 64.7% (17/32) |

Overall EXECUTION_FAILED rate by arm: A=25.0%, B=0.0%, C=0.0%, D=15.6%. Overall CAP_HIT rate by arm: A=0.0%, B=43.8%, C=31.2%, D=15.6%.

**v1 -> v2 EXECUTION_FAILED rate by (model, arm):**

| model | arm | v1 | v2 | delta |
|---|---|---|---|---|
| anthropic/claude-haiku-4.5 | A | 5.6% | 0.0% | -5.6% |
| anthropic/claude-haiku-4.5 | D | 88.9% | 5.6% | -83.3% |
| z-ai/glm-5.3-flash | A | 65.6% | 25.0% | -40.6% |
| z-ai/glm-5.3-flash | B | 18.8% | 0.0% | -18.8% |
| z-ai/glm-5.3-flash | C | 12.5% | 0.0% | -12.5% |
| z-ai/glm-5.3-flash | D | 34.4% | 15.6% | -18.8% |

- Residual failure **z-ai/glm-5.3-flash/A** (provider): 8/32 (25.0%) EXECUTION_FAILED; most common failure_detail class: 'empty response content' (4/8). 8/8 of these end with finish_reason == 'length' on both the initial attempt and the schema retry -- the model spends its whole MAX_TOKENS_FINAL=1200 budget (already doubled from 600 per this milestone's fix) on prose reasoning before ever emitting the JSON object, so raw_final_text is a truncated reasoning preamble with no JSON in it at all. This is the model's own verbosity, not a request-shape defect the harness can fix without an unbounded token ceiling (out of scope: CAP_HIT-style caps are deliberate, not to be raised without limit).
- Residual failure **z-ai/glm-5.3-flash/D** (provider): 5/32 (15.6%) EXECUTION_FAILED; most common failure_detail class: 'empty response content' (4/5). 5/5 of these end with finish_reason == 'length' on both the initial attempt and the schema retry -- the model spends its whole MAX_TOKENS_FINAL=1200 budget (already doubled from 600 per this milestone's fix) on prose reasoning before ever emitting the JSON object, so raw_final_text is a truncated reasoning preamble with no JSON in it at all. This is the model's own verbosity, not a request-shape defect the harness can fix without an unbounded token ceiling (out of scope: CAP_HIT-style caps are deliberate, not to be raised without limit). 5/5 of these failed cases (and 27/32 of this leg's cases overall) finalised immediately after a tool-calling turn whose finish_reason was 'length' -- MAX_TOKENS_TOOL_TURN (300) cut that turn off before the model could keep searching, so the loop moved to finalisation with fewer tool calls used than it would otherwise have made. This is left-at-300, unfixed for this v2 report (no budget for another sweep); it is a contributing cause alongside, not instead of, final-answer verbosity.

- At z-ai/glm-5.3-flash, arm D does NOT improve grounded_accuracy over arm C (66.7% -> 64.7%, delta -2.0%). See adherence, fabrication, abstention, trajectory and efficiency deltas below instead.
<!-- END RESULTS -->
