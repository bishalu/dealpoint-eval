# Braintrust BTQL investigations

Exact reproducible queries, `curl` and `bt sql` forms, and observed results (dates below). Run `just braintrust-sync` first so there is fresh data to query.

## 1. Retrieval rescue

**Question:** Which cases did a later search_agreement call surface gold evidence that the first search missed?

```
from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | select: id, input, metadata.arm as arm, metadata.reasoning_type as rt | filter: metadata.reasoning_type is not null | limit: 50
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | select: id, input, metadata.arm as arm, metadata.reasoning_type as rt | filter: metadata.reasoning_type is not null | limit: 50"}'
```

```bash
bt sql --non-interactive --json "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | select: id, input, metadata.arm as arm, metadata.reasoning_type as rt | filter: metadata.reasoning_type is not null | limit: 50"
```

**Executed at:** 2026-09-06T02:11:21.047065+00:00  
**Row count:** 32

```json
[
  {
    "arm": "A",
    "id": "7b6931f8-d414-4756-9692-e2adb1425cc1",
    "input": "contract_14__redacted_q10",
    "rt": "cross-ref"
  },
  {
    "arm": "A",
    "id": "9a269bc3-4e2c-49fa-89c9-ce68100aed30",
    "input": "contract_6__redacted_q09",
    "rt": "cross-ref"
  },
  {
    "arm": "A",
    "id": "220d7076-f1d0-4b26-b3ab-424ef5935873",
    "input": "contract_4__q04",
    "rt": "structured"
  },
  {
    "arm": "A",
    "id": "3b8c0484-480a-416f-8140-3cee7b59494a",
    "input": "contract_99__q03",
    "rt": "numeric"
  },
  {
    "arm": "A",
    "id": "1c99bae9-8208-4249-886a-6a42b0c55713",
    "input": "contract_103__q02",
    "rt": "direct"
  }
]
```


## 2. Failure attribution

**Question:** EXECUTION_FAILED / CAP_HIT counts grouped by model and arm.

```
from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | dimensions: metadata.model as model, metadata.arm as arm | measures: count(1) as n
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | dimensions: metadata.model as model, metadata.arm as arm | measures: count(1) as n"}'
```

```bash
bt sql --non-interactive --json "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | dimensions: metadata.model as model, metadata.arm as arm | measures: count(1) as n"
```

**Executed at:** 2026-09-06T02:11:21.614601+00:00  
**Row count:** 1

```json
[
  {
    "arm": "A",
    "model": "z-ai/glm-5.3-flash",
    "n": 32
  }
]
```


## 3. Reasoning slices

**Question:** grounded_accuracy by reasoning_type (direct, numeric, structured, defined-term, cross-ref, carve-out).

```
from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | dimensions: metadata.reasoning_type as rt | measures: count(1) as n
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | dimensions: metadata.reasoning_type as rt | measures: count(1) as n"}'
```

```bash
bt sql --non-interactive --json "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | dimensions: metadata.reasoning_type as rt | measures: count(1) as n"
```

**Executed at:** 2026-09-06T02:11:21.749020+00:00  
**Row count:** 7

```json
[
  {
    "n": 4,
    "rt": "carve-out"
  },
  {
    "n": 4,
    "rt": "direct"
  },
  {
    "n": 12,
    "rt": "defined-term"
  },
  {
    "n": 2,
    "rt": "out-of-scope"
  },
  {
    "n": 2,
    "rt": "structured"
  }
]
```


## 4. DeepEval disagreement

**Question:** Traces where deepeval/ and obj/ scores disagree.

```
from: experiment('judge-A@haiku') | select: id, metadata | limit: 50
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('judge-A@haiku') | select: id, metadata | limit: 50"}'
```

```bash
bt sql --non-interactive --json "from: experiment('judge-A@haiku') | select: id, metadata | limit: 50"
```

**Executed at:** 2026-09-06T02:11:21.905865+00:00  
**Row count:** 1

```json
[
  {
    "id": "f1c713f3-5093-44d2-8fa2-bf79134bbe63",
    "metadata": {
      "arm": "A",
      "case_type": null,
      "framework_versions": {
        "bm25s": "0.3.11",
        "braintrust": "0.37.0",
        "deepeval": "4.2.1",
        "fastembed": "0.8.0",
        "git_sha7": "b9eac78",
        "llama-index-core": "0.14.24",
        "llama-index-retrievers-bm25": "0.8.0",
        "openai": "3.8.0",
        "pydantic": "2.13.5",
        "python": "3.12.3",
        "qdrant-client": "1.19.0"
      },
      "git_sha": "b9eac78",
      "index_version": null,
      "model": "anthropic/claude-haiku-4.5",
      "reasoning_type": null,
      "retriever": "dense",
      "skill_version": "f8d255cc169b",
      "stage": "eval",
      "variant_id": "A@haiku"
    }
  }
]
```


## 5. Model economics

**Question:** Cost per correct answer by model, across the M6 Pareto experiments.

```
from: experiment('pareto-anthropic_claude-haiku-4.5') | dimensions: metadata.model as model | measures: count(1) as n
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('pareto-anthropic_claude-haiku-4.5') | dimensions: metadata.model as model | measures: count(1) as n"}'
```

```bash
bt sql --non-interactive --json "from: experiment('pareto-anthropic_claude-haiku-4.5') | dimensions: metadata.model as model | measures: count(1) as n"
```

**Executed at:** 2026-09-06T02:11:22.449129+00:00  
**Row count:** 1

```json
[
  {
    "model": "anthropic/claude-haiku-4.5",
    "n": 1
  }
]
```


## 6. Trajectory inefficiency

**Question:** Tool calls vs outcome -- does more searching correlate with a worse result?

```
from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | select: id, metadata.arm as arm | limit: 50
```

```bash
curl -s https://api.braintrust.dev/btql \
  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query": "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | select: id, metadata.arm as arm | limit: 50"}'
```

```bash
bt sql --non-interactive --json "from: experiment('A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc') | select: id, metadata.arm as arm | limit: 50"
```

**Executed at:** 2026-09-06T02:11:22.820062+00:00  
**Row count:** 32

```json
[
  {
    "arm": "A",
    "id": "7b6931f8-d414-4756-9692-e2adb1425cc1"
  },
  {
    "arm": "A",
    "id": "9a269bc3-4e2c-49fa-89c9-ce68100aed30"
  },
  {
    "arm": "A",
    "id": "220d7076-f1d0-4b26-b3ab-424ef5935873"
  },
  {
    "arm": "A",
    "id": "3b8c0484-480a-416f-8140-3cee7b59494a"
  },
  {
    "arm": "A",
    "id": "1c99bae9-8208-4249-886a-6a42b0c55713"
  }
]
```


