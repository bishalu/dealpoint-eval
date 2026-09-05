# dealpoint-eval

M&A deal-point review agent evaluation harness (MAUD v1, CC BY 4.0, Zenodo 7500064).

## Results

<!-- BEGIN RESULTS -->
Results below are from a **budget-scaled** 32-case frozen discriminative subset of MAUD's test set (18 cases at `anthropic/claude-haiku-4.5`), not the brief's full 167-case design; the task here is document -> (answer, citation) while MAUD's published task is span -> answer, so scores are **not comparable** to MAUD leaderboard numbers, and every metric below is **objective** (deterministic Python over expert labels) -- M4 has no model judging.

Majority-baseline overall accuracy: 0.0%.

**Model `anthropic/claude-haiku-4.5`:**

| arm | n_cases | n_scored | grounded_accuracy |
|---|---|---|---|
| A | 18 | 9 | 44.4% (9/18) |
| D | 18 | 2 | 100.0% (2/18) |

Overall EXECUTION_FAILED rate by arm: A=5.6%, D=88.9%. Overall CAP_HIT rate by arm: A=0.0%, D=0.0%.

**Model `z-ai/glm-5.3-flash`:**

| arm | n_cases | n_scored | grounded_accuracy |
|---|---|---|---|
| A | 32 | 7 | 85.7% (7/32) |
| B | 32 | 14 | 57.1% (14/32) |
| C | 32 | 17 | 64.7% (17/32) |
| D | 32 | 13 | 61.5% (13/32) |

Overall EXECUTION_FAILED rate by arm: A=65.6%, B=18.8%, C=12.5%, D=34.4%. Overall CAP_HIT rate by arm: A=0.0%, B=31.2%, C=25.0%, D=12.5%.

- At z-ai/glm-5.3-flash, arm D does NOT improve grounded_accuracy over arm C (64.7% -> 61.5%, delta -3.2%). See adherence, fabrication, abstention, trajectory and efficiency deltas below instead.
<!-- END RESULTS -->
