# Demo walkthrough (draft) — M7a

A 5–10 minute technical walkthrough, in the engineer's order. Every name, id and
number below resolves to a local artifact on disk (`data/reports/`,
`data/eval/`, `data/results/`) as of `git_sha7 b9eac78`. Sections that need a
live, authenticated Braintrust MCP session (after the Claude Code restart) are
marked **[M7b]** — this milestone (M7a) produces everything reproducible from
code/SDK/`bt` CLI, checkpointed so M7b resumes from it, never reruns it.

## Role diagram

```
custom Python  --> benchmark truth      (parser, canonical offsets, MAUD gold spans, scorers)
LlamaIndex     --> RAG lab / RAG eval   (retriever composition, RetrieverEvaluator, synthetic queries)
DeepEval       --> independent agent-eval cross-check (task completion, tool correctness, step efficiency)
Braintrust     --> traces / experiments / comparison (surface, not source of truth)
local reports + Git --> permanent evidence (data/reports/, data/results/, specs/)
```

Nothing above ever decides a winner except the first row. MAUD gold-span
overlap (`overlap_chars` against `MIN_GOLD_OVERLAP_CHARS=50`) is the only
truth; LlamaIndex node ids and DeepEval scores are cross-checks, never a
replacement.

## 1. Datasets

Frozen case sets, all under `data/eval/`:

- `dev.jsonl` — 58 cases, all with gold spans, across 5 documents; the RAG
  lab's canonical dev retrieval set.
- `test.jsonl` — 167 cases (the frozen holdout); `test_subset_v1.json` is the
  32-case budget-scaled subset the four-arm and Pareto sweeps ran on
  (`subset_hash` recorded in that file; sha256 asserted against
  `li_rag_eval.json`'s `frozen_assertions` so it is provably untouched by M7a).
- `counterfactual.jsonl` — 40 cases (30 redacted, 10 out-of-scope).
- `judged_subset.json` — the M5/M6 judged/calibration subset: 18 cases
  (`subset_hash 5918ef10a7e6`) × 6 variants (`A@haiku`, `D@haiku`, `D@glm`,
  `D@deepseek-v4-flash`, `D@qwen3.7-flash`, `D@gemini-3.1-flash-lite`) = 108
  traces. This is exactly the subset DeepEval cross-checks in section 6 below.
- `synthetic_dev_queries.jsonl` (new, M7a) — the frozen LlamaIndex-generated
  synthetic query set: 106 questions over 73 distinct gold-bearing dev
  chunks, model `z-ai/glm-5.3-flash`, `prompt_hash 93801419e0abfb5b`.

`just braintrust-sync` mirrors all five as Braintrust datasets
(`maud-dealpoint-dev`, `-test`, `-counterfactual`, `-judged_calibration`,
`-synthetic_query`) — never hand-edited, always rebuilt from these files.

## 2. RAG Lab

`data/reports/li_rag_eval.md` — LlamaIndex's `RetrieverEvaluator` run over all
six frozen M3 candidates on the full 58-case dev set, `chunk_version
8e5e8ba56765` / `index_version e2b4a2b97561` (asserted to match
`data/reports/versions.json` before anything runs).

| config | li/hit_rate | obj/gold_span_hit@5 |
|---|---|---|
| dense | 81.0% | 81.0% |
| bm25 | 86.2% | 86.2% |
| **hybrid_rrf** (M3 winner) | **91.4%** | **91.4%** |
| hybrid_rrf_rerank | 91.4% | 91.4% |
| multi_query_fusion | 82.8% | 82.8% |
| multi_query_fusion_rerank | 87.9% | 87.9% |

Ranking-order agreement (Spearman, `dealpoint.eval.agreement.spearman`) is
`rho = 1.0` on both hit_rate/hit@5 and MRR — LI and obj agree on every
retriever's rank, and disagreement count is 0 in both directions. That is
expected by construction (`expected_ids` for LlamaIndex ARE the gold-bearing
chunk ids), reported plainly rather than manufactured.

Synthetic-query robustness (secondary, never benchmark truth): the frozen
winner `hybrid_rrf` scores 94.3% li/hit_rate on the 106 synthetic queries vs
88.7% for dense — the winner stays strong under a different query
distribution. Rag Lab experiments in Braintrust: `rag-m3-dense`,
`rag-m3-bm25`, `rag-m3-hybrid`, `rag-m3-hybrid-rerank`, `rag-m3-fusion`,
`rag-m3-fusion-rerank`, `rag-m7-li-crosscheck`, `rag-m7-synthetic`.

## 3. Agent Systems

Four versions (A–D) differing by exactly one config key each, plus the M6
Pareto sweep varying only the model at arm D. Representative experiment:
`A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc` (32 cases, arm A). The frozen
winner retriever is `hybrid_rrf` (`ARM_C_RETRIEVER["name"]`), hit@5 `0.9138`
vs dense `0.8103` (M3 tournament, `data/reports/tournament.json`).

## 4. Logs trace

Example case: `contract_39__redacted_q05` (a redacted, defined-term case).
`just braintrust-sync` replays `case -> agent -> search_agreement (retrieval
stages as child spans) -> lookup_defined_term/get_section -> final_answer ->
scoring`, entirely from the stored `record.trajectory` — **zero model
calls**. Representative cases, chosen by documented rule (see
`data/reports/representative_cases.json`):

| category | case_id | variant_id |
|---|---|---|
| successful_direct | `contract_0__q01` | `D@openai/gpt-5.6-luna-pro` |
| retrieval_rescue | `contract_0__q01` | `C@z-ai/glm-5.3-flash` |
| defined_term_cross_ref | `contract_99__redacted_q06` | `D@anthropic/claude-haiku-4.5` |
| inefficient_trajectory | `contract_99__q03` | `D@qwen/qwen3.7-flash` |
| wrong_answer | `contract_144__q09` | `A@anthropic/claude-haiku-4.5` |
| abstention_counterfactual | `contract_75__oos09` | `A@anthropic/claude-haiku-4.5` |

**[M7b]** Custom trace view / saved views over these spans.

## 5. Scorers

Braintrust functions import the canonical implementations directly —
`grounded_accuracy`, `answer_correct`, `citation_verbatim`,
`citation_gold_overlap`, `required_evidence_met`, `skill_adherence`
(`dealpoint.eval.scorers`). Namespaces: `obj/` (canonical), `li/` (LlamaIndex
hit_rate/mrr), `judge/` (calibrated judge dimensions), `deepeval/` (task
completion, tool correctness, argument correctness, step efficiency),
`procedure/` (reserved). Every M7 experiment logs ≤ 12 scores/case on
subsets ≤ 60 cases; M4/M6 sweeps keep six — enforced by
`dealpoint.eval.braintrust_sync.assert_score_budget`.

## 6. Review

**[M7b]** The 12-trace blinded review set is synced (`review_set(12)` in
`dealpoint.eval.braintrust_sync`) and ready — 12 packets spanning 4+
reasoning types, ≥ 2 abstentions, 5 distinct variants — but scoring it in
Braintrust Review happens after the restart. Local human calibration is
`n = 0` (`data/eval/calibration/human_scores.jsonl` is empty) — reported as
`"pending"` everywhere, never fabricated.

## 7. Eval of Evals

`data/reports/deepeval_crosscheck.md` — DeepEval's independent read on the
same 108 judged-subset traces. Resolved evaluator (never hardcoded, read from
`judge_slate.json`'s verified non-candidate-family judges):
`mistralai/mistral-small-3.2-24b-instruct`. Metrics: `task_completion`,
`tool_correctness`, `argument_correctness`, `step_efficiency` (DeepEval-native
`StepEfficiencyMetric`). Comparisons: DeepEval `task_completion` vs
`grounded_accuracy` — Spearman `rho ≈ 0.38` (n=42, moderate); vs the
calibrated judge `reasoning` dimension — `rho ≈ 0.84` (n=108, strong); vs
human — `pending` (n=0). **Classification: `KEEP_OPTIONAL_ANALYSIS`** — the
moderate deterministic correlation and small deterministic-overlap sample
mean this is a plausible secondary cross-check, not yet strong enough
evidence to call it either core or redundant.

## 8. Loop / SQL

**[M7b]** Braintrust Loop investigation thread. The underlying data is ready:
`docs/braintrust-queries.md` has the six BTQL investigations (retrieval
rescue, failure attribution, reasoning slices, DeepEval disagreement, model
economics, trajectory inefficiency), executed against the live project and
recorded in `data/reports/btql_investigations.json` with the exact query, the
`curl`/`bt sql` reproduction commands, and the observed row counts (e.g.
"Reasoning slices" over `A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc" returned 7
rows: `direct=4, defined-term=12, cross-ref=6, structured=2, carve-out=4,
numeric=2, out-of-scope=2`).

## 9. Debugger

**[M7b]** Topics/Patterns/Debugger use over the synced traces.

## 10. Model Economics

`data/reports/pareto.md` — arm D held fixed, only the model varies. Frontier
(min $/case, max grounded_accuracy): `qwen/qwen3.7-flash` and
`deepseek/deepseek-v4-flash`. Braintrust experiments tagged `stage=economics`:
`pareto-deepseek_deepseek-v4-flash`, `pareto-qwen_qwen3.7-flash`,
`pareto-google_gemini-3.1-flash-lite`, `pareto-anthropic_claude-haiku-4.5`,
`pareto-z-ai_glm-5.3-flash`.

## 11. Dashboard

**[M7b]** A saved dashboard over the above experiments/scores.

## Braintrust tools — decision, not a stub

`search_agreement`/`get_section`/`lookup_defined_term` are **not** exposed as
Braintrust tools. Measured reason: they need the 68 MB Qdrant index
(`data/index/`), the 64 MB derived canonical corpus (`data/derived/`) and the
fastembed ONNX weights loaded in-process — none of which can run inside
Braintrust's function execution environment without duplicating the entire
index/data layer this project already owns. That duplication is exactly what
the no-bloat rule and the brief's "no second tracing/data architecture"
guidance forbid. This is a documented "no", not an omission.

## Reproduce this walkthrough

```bash
uv run python -m dealpoint.rag_lab.report            # li_rag_eval.json/.md
uv run python -m dealpoint.eval.deepeval_adapter      # deepeval_crosscheck.json/.md (METERED)
just braintrust-sync                                  # datasets, experiments, replayed traces (idempotent)
uv run python -m dealpoint.eval.btql                  # btql_investigations.json + docs/braintrust-queries.md
```
