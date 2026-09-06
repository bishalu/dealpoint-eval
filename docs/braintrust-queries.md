# Braintrust BTQL investigations

Exact reproducible queries, `curl` and `bt sql` forms, and observed results (dates below). Run `just braintrust-sync` first so there is fresh data to query.

## 1. Retrieval rescue

**Question:** Which cases did a later search_agreement call surface gold evidence that the first search missed?

```
from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | select: id, input, metadata.arm, metadata.secondary_diagnostics."obj/tool_calls" as tool_calls, scores."obj/grounded_accuracy" as grounded_accuracy | filter: metadata.secondary_diagnostics."obj/tool_calls" >= 2 and scores."obj/grounded_accuracy" = 1 | limit: 50
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('"'"'A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc'"'"') | select: id, input, metadata.arm, metadata.secondary_diagnostics.\"obj/tool_calls\" as tool_calls, scores.\"obj/grounded_accuracy\" as grounded_accuracy | filter: metadata.secondary_diagnostics.\"obj/tool_calls\" >= 2 and scores.\"obj/grounded_accuracy\" = 1 | limit: 50"}'
```

```bash
bt sql --non-interactive --json "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | select: id, input, metadata.arm, metadata.secondary_diagnostics.\"obj/tool_calls\" as tool_calls, scores.\"obj/grounded_accuracy\" as grounded_accuracy | filter: metadata.secondary_diagnostics.\"obj/tool_calls\" >= 2 and scores.\"obj/grounded_accuracy\" = 1 | limit: 50"
```

**Executed at:** 2026-09-06T20:38:22.513989+00:00  
**Row count:** 0

**Notes:** 0 rows -- verified data-supported cause (not retention purge): every row in the target agent experiment has metadata.secondary_diagnostics."obj/tool_calls" == 0 in this run, so the filter's tool_calls >= 2 predicate cannot match any case. No case in this run triggered a second search_agreement call before a grounded-correct answer.


## 2. Failure attribution

**Question:** EXECUTION_FAILED / CAP_HIT counts grouped by model and arm.

```
from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | filter: metadata.secondary_diagnostics."obj/execution_failed" = 1 or metadata.secondary_diagnostics."obj/cap_hit" = 1 | dimensions: metadata.model, metadata.arm | measures: count(1) as n
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('"'"'A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc'"'"') | filter: metadata.secondary_diagnostics.\"obj/execution_failed\" = 1 or metadata.secondary_diagnostics.\"obj/cap_hit\" = 1 | dimensions: metadata.model, metadata.arm | measures: count(1) as n"}'
```

```bash
bt sql --non-interactive --json "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | filter: metadata.secondary_diagnostics.\"obj/execution_failed\" = 1 or metadata.secondary_diagnostics.\"obj/cap_hit\" = 1 | dimensions: metadata.model, metadata.arm | measures: count(1) as n"
```

**Executed at:** 2026-09-06T20:38:23.088858+00:00  
**Row count:** 1

```json
[
  {
    "arm": "A",
    "model": "z-ai/glm-5.3-flash",
    "n": 24
  }
]
```


## 3. Reasoning slices

**Question:** grounded_accuracy by reasoning_type (direct, numeric, structured, defined-term, cross-ref, carve-out).

```
from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | dimensions: metadata.reasoning_type | measures: avg(scores."obj/grounded_accuracy") as grounded_accuracy, count(1) as n
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('"'"'A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc'"'"') | dimensions: metadata.reasoning_type | measures: avg(scores.\"obj/grounded_accuracy\") as grounded_accuracy, count(1) as n"}'
```

```bash
bt sql --non-interactive --json "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | dimensions: metadata.reasoning_type | measures: avg(scores.\"obj/grounded_accuracy\") as grounded_accuracy, count(1) as n"
```

**Executed at:** 2026-09-06T20:38:23.904232+00:00  
**Row count:** 7

```json
[
  {
    "grounded_accuracy": null,
    "n": 8,
    "reasoning_type": "out-of-scope"
  },
  {
    "grounded_accuracy": 1,
    "n": 8,
    "reasoning_type": "numeric"
  },
  {
    "grounded_accuracy": 0.6666666666666666,
    "n": 48,
    "reasoning_type": "defined-term"
  },
  {
    "grounded_accuracy": 1,
    "n": 16,
    "reasoning_type": "carve-out"
  },
  {
    "grounded_accuracy": 0.5,
    "n": 24,
    "reasoning_type": "cross-ref"
  }
]
```


## 4. DeepEval disagreement

**Question:** Traces where deepeval/task_completion and obj/grounded_accuracy disagree.

```
from: experiment('deepeval-crosscheck') | select: id, input, scores."deepeval/task_completion" as task_completion, scores."obj/grounded_accuracy" as grounded_accuracy | filter: (scores."deepeval/task_completion" >= 0.5 and scores."obj/grounded_accuracy" = 0) or (scores."deepeval/task_completion" < 0.5 and scores."obj/grounded_accuracy" = 1) | limit: 50
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('"'"'deepeval-crosscheck'"'"') | select: id, input, scores.\"deepeval/task_completion\" as task_completion, scores.\"obj/grounded_accuracy\" as grounded_accuracy | filter: (scores.\"deepeval/task_completion\" >= 0.5 and scores.\"obj/grounded_accuracy\" = 0) or (scores.\"deepeval/task_completion\" < 0.5 and scores.\"obj/grounded_accuracy\" = 1) | limit: 50"}'
```

```bash
bt sql --non-interactive --json "from: experiment('deepeval-crosscheck') | select: id, input, scores.\"deepeval/task_completion\" as task_completion, scores.\"obj/grounded_accuracy\" as grounded_accuracy | filter: (scores.\"deepeval/task_completion\" >= 0.5 and scores.\"obj/grounded_accuracy\" = 0) or (scores.\"deepeval/task_completion\" < 0.5 and scores.\"obj/grounded_accuracy\" = 1) | limit: 50"
```

**Executed at:** 2026-09-06T20:38:24.110824+00:00  
**Row count:** 17

```json
[
  {
    "grounded_accuracy": 0,
    "id": "cb97ec1b-1bfe-4cbc-a1fd-4bc57456077c",
    "input": "contract_39__q01",
    "task_completion": 1
  },
  {
    "grounded_accuracy": 0,
    "id": "4e1a705c-d53b-4e7b-bf83-01f4c0419bc8",
    "input": "contract_144__q09",
    "task_completion": 1
  },
  {
    "grounded_accuracy": 0,
    "id": "162bc618-1144-40e3-b6ca-d681c30273d8",
    "input": "contract_32__q06",
    "task_completion": 0.7
  },
  {
    "grounded_accuracy": 0,
    "id": "a584c0ad-6f32-478d-96ad-91372515ecb8",
    "input": "contract_4__q12",
    "task_completion": 1
  },
  {
    "grounded_accuracy": 0,
    "id": "a6eb820e-e25f-4fbd-92d6-ae81cf08d89c",
    "input": "contract_39__q01",
    "task_completion": 0.7
  }
]
```


## 5. Model economics

**Question:** Grounded accuracy and average cost per case, for one M6 Pareto model experiment.

```
from: experiment('pareto-anthropic_claude-haiku-4.5') | dimensions: metadata.model | measures: avg(scores."obj/grounded_accuracy") as grounded_accuracy, avg(metadata.secondary_diagnostics."obj/usd") as avg_usd
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('"'"'pareto-anthropic_claude-haiku-4.5'"'"') | dimensions: metadata.model | measures: avg(scores.\"obj/grounded_accuracy\") as grounded_accuracy, avg(metadata.secondary_diagnostics.\"obj/usd\") as avg_usd"}'
```

```bash
bt sql --non-interactive --json "from: experiment('pareto-anthropic_claude-haiku-4.5') | dimensions: metadata.model | measures: avg(scores.\"obj/grounded_accuracy\") as grounded_accuracy, avg(metadata.secondary_diagnostics.\"obj/usd\") as avg_usd"
```

**Executed at:** 2026-09-06T20:38:24.621075+00:00  
**Row count:** 1

```json
[
  {
    "avg_usd": null,
    "grounded_accuracy": null,
    "model": "anthropic/claude-haiku-4.5"
  }
]
```


## 6. Trajectory inefficiency

**Question:** Tool calls vs outcome -- does more searching correlate with a worse result?

```
from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | select: id, metadata.arm, metadata.secondary_diagnostics."obj/tool_calls" as tool_calls, scores."obj/grounded_accuracy" as grounded_accuracy | filter: metadata.secondary_diagnostics."obj/tool_calls" is not null | limit: 50
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('"'"'A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc'"'"') | select: id, metadata.arm, metadata.secondary_diagnostics.\"obj/tool_calls\" as tool_calls, scores.\"obj/grounded_accuracy\" as grounded_accuracy | filter: metadata.secondary_diagnostics.\"obj/tool_calls\" is not null | limit: 50"}'
```

```bash
bt sql --non-interactive --json "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | select: id, metadata.arm, metadata.secondary_diagnostics.\"obj/tool_calls\" as tool_calls, scores.\"obj/grounded_accuracy\" as grounded_accuracy | filter: metadata.secondary_diagnostics.\"obj/tool_calls\" is not null | limit: 50"
```

**Executed at:** 2026-09-06T20:38:25.106428+00:00  
**Row count:** 50

```json
[
  {
    "arm": "A",
    "grounded_accuracy": null,
    "id": "14fbe113-b71c-4fe1-9992-e980f2b1a9e3",
    "tool_calls": 0
  },
  {
    "arm": "A",
    "grounded_accuracy": null,
    "id": "5639ddec-1b22-48f2-bb52-7cdb8e6b30ea",
    "tool_calls": 0
  },
  {
    "arm": "A",
    "grounded_accuracy": 1,
    "id": "e95ba2b5-0abc-4284-8985-c8a7fe575459",
    "tool_calls": 0
  },
  {
    "arm": "A",
    "grounded_accuracy": 1,
    "id": "b3fefa26-e98d-460b-bfe9-858ecbfa097a",
    "tool_calls": 0
  },
  {
    "arm": "A",
    "grounded_accuracy": 1,
    "id": "28d7d9f1-b2a9-4c77-96e0-11e536d61ab0",
    "tool_calls": 0
  }
]
```


