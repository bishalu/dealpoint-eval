# Four-arm experiment on the frozen test set (M4)

This report covers a budget-scaled, 32-case frozen discriminative subset (18 cases at Haiku) of MAUD's test set -- not the brief's full 167-case test set. Task here is document -> (answer, citation); MAUD's published task is span -> answer, so scores are not comparable to MAUD leaderboard numbers. Every score in this report is deterministic Python over expert labels; there is no model judging in M4.

## Config diff between adjacent arms

| from | to | key | from value | to value |
|---|---|---|---|---|
| A | B | loop | `pipeline` | `agent` |
| B | C | retriever | `{'kind': 'dense', 'name': 'dense', 'fetch_k': 5, 'rerank_model': None, 'multi_query': False, 'rrf_k': 60}` | `{'fetch_k': 20, 'kind': 'hybrid_rrf', 'multi_query': False, 'name': 'hybrid_rrf', 'rerank_model': None, 'rrf_k': 60}` |
| C | D | skill | `False` | `True` |

Majority-baseline overall accuracy: 0.0%

## Model: `anthropic/claude-haiku-4.5`

| arm | n_cases | n_scored | grounded_accuracy | answer_correct | fabrication | abstain_correct | skill_adherence | execution_failed | cap_hit | tool_calls | usd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 18 | 9 | 44.4% (9/18) | 40.0% | 0.0% | 83.3% | 49.4% | 5.6% | 0.0% | 0.00 | $0.0034 |
| D | 18 | 2 | 100.0% (2/18) | 100.0% | 0.0% | 66.7% | 38.0% | 88.9% | 0.0% | 2.11 | $0.0090 |

## Model: `z-ai/glm-5.3-flash`

| arm | n_cases | n_scored | grounded_accuracy | answer_correct | fabrication | abstain_correct | skill_adherence | execution_failed | cap_hit | tool_calls | usd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 32 | 7 | 85.7% (7/32) | 85.7% | 0.0% | 75.0% | 44.4% | 65.6% | 0.0% | 0.00 | $0.0010 |
| B | 32 | 14 | 57.1% (14/32) | 60.0% | 6.7% | 71.9% | 66.4% | 18.8% | 31.2% | 5.25 | $0.0018 |
| C | 32 | 17 | 64.7% (17/32) | 63.2% | 21.1% | 78.1% | 66.1% | 12.5% | 25.0% | 4.50 | $0.0019 |
| D | 32 | 13 | 61.5% (13/32) | 66.7% | 20.0% | 81.2% | 60.1% | 34.4% | 12.5% | 4.47 | $0.0025 |

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
| q07 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=1) | 50.0% (n=2) | 31.0% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q08 | n/a (n=0) | n/a (n=0) | 0.0% (n=1) | 0.0% (n=1) | 0.0% (n=2) | 35.7% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q09 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 60.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q10 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 50.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q11 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q12 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | n/a (n=0) | 75.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |

### anthropic/claude-haiku-4.5 / arm D

| question_id | grounded_accuracy | answer_correct | citation_gold_overlap | gold_seen | required_evidence_met | skill_adherence | execution_failed | cap_hit |
|---|---|---|---|---|---|---|---|---|
| oos04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 33.3% (n=1) | 100.0% (n=1) | 0.0% (n=1) |
| oos09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 50.0% (n=1) | 100.0% (n=1) | 0.0% (n=1) |
| q01 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=1) | n/a (n=0) | 50.0% (n=1) | 100.0% (n=1) | 0.0% (n=1) |
| q02 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=1) | n/a (n=0) | 33.3% (n=1) | 100.0% (n=1) | 0.0% (n=1) |
| q03 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=1) | n/a (n=0) | 50.0% (n=1) | 100.0% (n=1) | 0.0% (n=1) |
| q04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 0.0% (n=1) | n/a (n=0) | 0.0% (n=1) | 100.0% (n=1) | 0.0% (n=1) |
| q05 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 66.7% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q06 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=2) | 33.3% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q07 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=1) | 100.0% (n=2) | 58.3% (n=2) | 100.0% (n=2) | 0.0% (n=2) |
| q08 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=1) | 50.0% (n=2) | 25.0% (n=2) | 100.0% (n=2) | 0.0% (n=2) |
| q09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 0.0% (n=1) | n/a (n=0) | 0.0% (n=1) | 100.0% (n=1) | 0.0% (n=1) |
| q10 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 0.0% (n=1) | n/a (n=0) | 0.0% (n=1) | 100.0% (n=1) | 0.0% (n=1) |
| q11 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=1) | 100.0% (n=1) | 0.0% (n=1) |
| q12 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 0.0% (n=1) | n/a (n=0) | 50.0% (n=1) | 100.0% (n=1) | 0.0% (n=1) |

### z-ai/glm-5.3-flash / arm A

| question_id | grounded_accuracy | answer_correct | citation_gold_overlap | gold_seen | required_evidence_met | skill_adherence | execution_failed | cap_hit |
|---|---|---|---|---|---|---|---|---|
| oos04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 50.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| oos09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 50.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q01 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 62.5% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q02 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | n/a (n=0) | 41.7% (n=2) | 100.0% (n=2) | 0.0% (n=2) |
| q03 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 75.0% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q04 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=2) | n/a (n=0) | 62.5% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q05 | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=2) | 50.0% (n=2) | 0.0% (n=3) | 39.4% (n=3) | 33.3% (n=3) | 0.0% (n=3) |
| q06 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | 33.3% (n=3) | 30.6% (n=3) | 100.0% (n=3) | 0.0% (n=3) |
| q07 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 66.7% (n=3) | 39.4% (n=3) | 66.7% (n=3) | 0.0% (n=3) |
| q08 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 50.0% (n=2) | 33.3% (n=3) | 25.0% (n=3) | 100.0% (n=3) | 0.0% (n=3) |
| q09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | n/a (n=0) | 38.9% (n=3) | 100.0% (n=3) | 0.0% (n=3) |
| q10 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 58.3% (n=3) | 66.7% (n=3) | 0.0% (n=3) |
| q11 | n/a (n=0) | n/a (n=0) | 0.0% (n=1) | 100.0% (n=2) | 50.0% (n=2) | 29.2% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q12 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | n/a (n=0) | 41.7% (n=2) | 100.0% (n=2) | 0.0% (n=2) |

### z-ai/glm-5.3-flash / arm B

| question_id | grounded_accuracy | answer_correct | citation_gold_overlap | gold_seen | required_evidence_met | skill_adherence | execution_failed | cap_hit |
|---|---|---|---|---|---|---|---|---|
| oos04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 66.7% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| oos09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 66.7% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| q01 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | n/a (n=0) | 83.3% (n=2) | 100.0% (n=2) | 0.0% (n=2) |
| q02 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | n/a (n=0) | 33.3% (n=2) | 50.0% (n=2) | 50.0% (n=2) |
| q03 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 83.3% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q04 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 50.0% (n=2) | n/a (n=0) | 63.3% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q05 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 65.6% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q06 | n/a (n=0) | n/a (n=0) | 0.0% (n=1) | 100.0% (n=2) | 100.0% (n=3) | 53.2% (n=3) | 33.3% (n=3) | 33.3% (n=3) |
| q07 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 100.0% (n=3) | 75.0% (n=3) | 33.3% (n=3) | 33.3% (n=3) |
| q08 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 100.0% (n=3) | 61.1% (n=3) | 33.3% (n=3) | 33.3% (n=3) |
| q09 | 0.0% (n=2) | 0.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 68.9% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q10 | 50.0% (n=2) | 33.3% (n=3) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 73.3% (n=3) | 0.0% (n=3) | 0.0% (n=3) |
| q11 | 0.0% (n=2) | 0.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 66.7% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q12 | 50.0% (n=2) | 100.0% (n=2) | 50.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 70.0% (n=2) | 0.0% (n=2) | 0.0% (n=2) |

### z-ai/glm-5.3-flash / arm C

| question_id | grounded_accuracy | answer_correct | citation_gold_overlap | gold_seen | required_evidence_met | skill_adherence | execution_failed | cap_hit |
|---|---|---|---|---|---|---|---|---|
| oos04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 33.3% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| oos09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 66.7% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| q01 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 70.8% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q02 | 50.0% (n=2) | 50.0% (n=2) | 50.0% (n=2) | 50.0% (n=2) | n/a (n=0) | 46.7% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q03 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 83.3% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q04 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 70.0% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q05 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 72.2% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q06 | 0.0% (n=1) | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=2) | 100.0% (n=3) | 65.6% (n=3) | 33.3% (n=3) | 33.3% (n=3) |
| q07 | 100.0% (n=2) | 66.7% (n=3) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 81.1% (n=3) | 0.0% (n=3) | 0.0% (n=3) |
| q08 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 100.0% (n=3) | 63.9% (n=3) | 33.3% (n=3) | 33.3% (n=3) |
| q09 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 58.3% (n=3) | 33.3% (n=3) | 33.3% (n=3) |
| q10 | 0.0% (n=2) | 33.3% (n=3) | 50.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 60.0% (n=3) | 0.0% (n=3) | 0.0% (n=3) |
| q11 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | 100.0% (n=2) | 70.8% (n=2) | 0.0% (n=2) | 100.0% (n=2) |
| q12 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 65.0% (n=2) | 0.0% (n=2) | 0.0% (n=2) |

### z-ai/glm-5.3-flash / arm D

| question_id | grounded_accuracy | answer_correct | citation_gold_overlap | gold_seen | required_evidence_met | skill_adherence | execution_failed | cap_hit |
|---|---|---|---|---|---|---|---|---|
| oos04 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 66.7% (n=1) | 0.0% (n=1) | 0.0% (n=1) |
| oos09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 33.3% (n=1) | 0.0% (n=1) | 100.0% (n=1) |
| q01 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | n/a (n=0) | 55.0% (n=2) | 50.0% (n=2) | 0.0% (n=2) |
| q02 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | n/a (n=0) | 33.3% (n=2) | 100.0% (n=2) | 0.0% (n=2) |
| q03 | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 77.5% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q04 | 50.0% (n=2) | 100.0% (n=2) | 50.0% (n=2) | 50.0% (n=2) | n/a (n=0) | 60.0% (n=2) | 0.0% (n=2) | 0.0% (n=2) |
| q05 | 100.0% (n=2) | 66.7% (n=3) | 100.0% (n=2) | 100.0% (n=2) | 100.0% (n=3) | 61.1% (n=3) | 0.0% (n=3) | 0.0% (n=3) |
| q06 | 0.0% (n=1) | 0.0% (n=2) | 100.0% (n=1) | 100.0% (n=2) | 100.0% (n=3) | 66.7% (n=3) | 33.3% (n=3) | 0.0% (n=3) |
| q07 | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 100.0% (n=3) | 68.3% (n=3) | 33.3% (n=3) | 0.0% (n=3) |
| q08 | 0.0% (n=1) | 100.0% (n=1) | 0.0% (n=1) | 100.0% (n=2) | 100.0% (n=3) | 58.3% (n=3) | 33.3% (n=3) | 33.3% (n=3) |
| q09 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | n/a (n=0) | 55.6% (n=3) | 100.0% (n=3) | 0.0% (n=3) |
| q10 | 50.0% (n=2) | 50.0% (n=2) | 100.0% (n=2) | 100.0% (n=2) | n/a (n=0) | 75.6% (n=3) | 0.0% (n=3) | 33.3% (n=3) |
| q11 | 0.0% (n=1) | 0.0% (n=1) | 100.0% (n=1) | 100.0% (n=2) | 100.0% (n=2) | 66.7% (n=2) | 0.0% (n=2) | 50.0% (n=2) |
| q12 | n/a (n=0) | n/a (n=0) | n/a (n=0) | 100.0% (n=2) | n/a (n=0) | 41.7% (n=2) | 100.0% (n=2) | 0.0% (n=2) |

## Honesty-rule verdicts

- **anthropic/claude-haiku-4.5 / A_to_D**: At anthropic/claude-haiku-4.5, arm D vs arm A on grounded_accuracy is not comparable: 9 vs 2 cases scored, 1 in common (the rest EXECUTION_FAILED/CAP_HIT).
- **z-ai/glm-5.3-flash / A_to_B**: At z-ai/glm-5.3-flash, arm B does NOT improve grounded_accuracy over arm A (85.7% -> 57.1%, delta -28.6%). See adherence, fabrication, abstention, trajectory and efficiency deltas below instead.
- **z-ai/glm-5.3-flash / B_to_C**: At z-ai/glm-5.3-flash, arm C improves grounded_accuracy over arm B by 7.6% (57.1% -> 64.7%).
- **z-ai/glm-5.3-flash / C_to_D**: At z-ai/glm-5.3-flash, arm D does NOT improve grounded_accuracy over arm C (64.7% -> 61.5%, delta -3.2%). See adherence, fabrication, abstention, trajectory and efficiency deltas below instead.
- **z-ai/glm-5.3-flash / A_to_D**: At z-ai/glm-5.3-flash, arm D does NOT improve grounded_accuracy over arm A (85.7% -> 61.5%, delta -24.2%). See adherence, fabrication, abstention, trajectory and efficiency deltas below instead.

## Adherence / fabrication / abstention / trajectory / efficiency deltas

### anthropic/claude-haiku-4.5 / A_to_D

| metric | from | to | delta |
|---|---|---|---|
| skill_adherence | 0.4941798941798942 | 0.3796296296296296 | -0.11455026455026462 |
| fabrication | 0.0 | 0.0 | 0.0 |
| abstain_correct | 0.8333333333333334 | 0.6666666666666666 | -0.16666666666666674 |
| tool_calls | 0.0 | 2.111111111111111 | 2.111111111111111 |
| wall_ms | 4454.111111111111 | 5118.555555555556 | 664.4444444444443 |
| usd | 0.0033545000000000003 | 0.008961166666666666 | 0.005606666666666666 |

### z-ai/glm-5.3-flash / A_to_B

| metric | from | to | delta |
|---|---|---|---|
| skill_adherence | 0.44375 | 0.6639136904761904 | 0.22016369047619044 |
| fabrication | 0.0 | 0.06666666666666667 | 0.06666666666666667 |
| abstain_correct | 0.75 | 0.71875 | -0.03125 |
| tool_calls | 0.0 | 5.25 | 5.25 |
| wall_ms | 12197.3125 | 32010.59375 | 19813.28125 |
| usd | 0.0009680828125 | 0.00181874 | 0.0008506571874999999 |

### z-ai/glm-5.3-flash / B_to_C

| metric | from | to | delta |
|---|---|---|---|
| skill_adherence | 0.6639136904761904 | 0.6614583333333334 | -0.002455357142857051 |
| fabrication | 0.06666666666666667 | 0.21052631578947367 | 0.14385964912280702 |
| abstain_correct | 0.71875 | 0.78125 | 0.0625 |
| tool_calls | 5.25 | 4.5 | -0.75 |
| wall_ms | 32010.59375 | 30356.15625 | -1654.4375 |
| usd | 0.00181874 | 0.00187741734375 | 5.867734375000007e-05 |

### z-ai/glm-5.3-flash / C_to_D

| metric | from | to | delta |
|---|---|---|---|
| skill_adherence | 0.6614583333333334 | 0.6014880952380952 | -0.05997023809523816 |
| fabrication | 0.21052631578947367 | 0.2 | -0.010526315789473661 |
| abstain_correct | 0.78125 | 0.8125 | 0.03125 |
| tool_calls | 4.5 | 4.46875 | -0.03125 |
| wall_ms | 30356.15625 | 33826.40625 | 3470.25 |
| usd | 0.00187741734375 | 0.0025176973296875 | 0.0006402799859375002 |

### z-ai/glm-5.3-flash / A_to_D

| metric | from | to | delta |
|---|---|---|---|
| skill_adherence | 0.44375 | 0.6014880952380952 | 0.15773809523809523 |
| fabrication | 0.0 | 0.2 | 0.2 |
| abstain_correct | 0.75 | 0.8125 | 0.0625 |
| tool_calls | 0.0 | 4.46875 | 4.46875 |
| wall_ms | 12197.3125 | 33826.40625 | 21629.09375 |
| usd | 0.0009680828125 | 0.0025176973296875 | 0.0015496145171875002 |

## Paired per-case flips

- **anthropic/claude-haiku-4.5 / A_to_D**: gained ['contract_32__q06'], lost [], unchanged_correct=0, unchanged_wrong=0
- **z-ai/glm-5.3-flash / A_to_B**: gained [], lost [], unchanged_correct=4, unchanged_wrong=1
- **z-ai/glm-5.3-flash / B_to_C**: gained ['contract_77__q12'], lost ['contract_103__q10'], unchanged_correct=7, unchanged_wrong=2
- **z-ai/glm-5.3-flash / C_to_D**: gained ['contract_103__q10'], lost ['contract_103__q08', 'contract_32__q04'], unchanged_correct=5, unchanged_wrong=2
- **z-ai/glm-5.3-flash / A_to_D**: gained [], lost [], unchanged_correct=6, unchanged_wrong=1

## Skill-adherence per-rule detail

Note (honesty, spec §3): arm A lacks several tools rules 2/3/4/6 need, so its applicable-rule set is structurally smaller than arms B/C/D's. n_applicable and n_satisfied are printed per rule per arm, so an A-vs-D adherence ratio comparison is never read as like-for-like.

### anthropic/claude-haiku-4.5

| arm | rule | n_applicable | n_satisfied | rate |
|---|---|---|---|---|
| A | 1 | 18 | 18 | 100.0% |
| A | 2 | 9 | 0 | 0.0% |
| A | 3 | 18 | 0 | 0.0% |
| A | 4 | 10 | 0 | 0.0% |
| A | 5 | 10 | 10 | 100.0% |
| A | 6 | 7 | 0 | 0.0% |
| A | 7 | 7 | 5 | 71.4% |
| A | 8 | 17 | 12 | 70.6% |
| D | 1 | 18 | 13 | 72.2% |
| D | 2 | 9 | 7 | 77.8% |
| D | 3 | 13 | 1 | 7.7% |
| D | 4 | 8 | 0 | 0.0% |
| D | 5 | 2 | 2 | 100.0% |
| D | 6 | 0 | 0 | n/a |
| D | 7 | 0 | 0 | n/a |
| D | 8 | 2 | 2 | 100.0% |

### z-ai/glm-5.3-flash

| arm | rule | n_applicable | n_satisfied | rate |
|---|---|---|---|---|
| A | 1 | 32 | 32 | 100.0% |
| A | 2 | 14 | 0 | 0.0% |
| A | 3 | 32 | 0 | 0.0% |
| A | 4 | 13 | 0 | 0.0% |
| A | 5 | 7 | 7 | 100.0% |
| A | 6 | 4 | 0 | 0.0% |
| A | 7 | 4 | 2 | 50.0% |
| A | 8 | 11 | 11 | 100.0% |
| B | 1 | 32 | 32 | 100.0% |
| B | 2 | 14 | 14 | 100.0% |
| B | 3 | 32 | 15 | 46.9% |
| B | 4 | 27 | 1 | 3.7% |
| B | 5 | 15 | 14 | 93.3% |
| B | 6 | 1 | 1 | 100.0% |
| B | 7 | 1 | 0 | 0.0% |
| B | 8 | 16 | 15 | 93.8% |
| C | 1 | 32 | 31 | 96.9% |
| C | 2 | 14 | 14 | 100.0% |
| C | 3 | 30 | 11 | 36.7% |
| C | 4 | 21 | 0 | 0.0% |
| C | 5 | 19 | 15 | 78.9% |
| C | 6 | 1 | 1 | 100.0% |
| C | 7 | 1 | 1 | 100.0% |
| C | 8 | 20 | 20 | 100.0% |
| D | 1 | 32 | 32 | 100.0% |
| D | 2 | 14 | 14 | 100.0% |
| D | 3 | 32 | 10 | 31.2% |
| D | 4 | 26 | 0 | 0.0% |
| D | 5 | 15 | 12 | 80.0% |
| D | 6 | 2 | 2 | 100.0% |
| D | 7 | 2 | 2 | 100.0% |
| D | 8 | 17 | 15 | 88.2% |

## Cost: estimate vs realised

Target: $1.5, absolute: $3.0, envelope: $4.0, M4 ledger total: $0.751879

| leg | arm | model | tranche | n_cases | est_usd | realized_usd |
|---|---|---|---|---|---|---|
| headline_glm | A | z-ai/glm-5.3-flash | full | 32 | 0.030976 | 0.030979 |
| headline_glm | B | z-ai/glm-5.3-flash | full | 32 | 0.180726 | 0.0582 |
| headline_glm | C | z-ai/glm-5.3-flash | full | 32 | 0.154305 | 0.060077 |
| headline_glm | D | z-ai/glm-5.3-flash | full | 32 | 0.082256 | 0.080566 |
| replication_haiku | A | anthropic/claude-haiku-4.5 | tranche_1 | 18 | 0.060381 | 0.060381 |
| replication_haiku | D | anthropic/claude-haiku-4.5 | tranche_1 | 18 | 0.22334 | 0.161301 |

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
