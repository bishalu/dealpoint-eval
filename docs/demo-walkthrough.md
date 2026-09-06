# Demo walkthrough (draft) — M7a

A 5–10 minute technical walkthrough, in the engineer's order. Every name, id and
number below resolves to a local artifact on disk (`data/reports/`,
`data/eval/`, `data/results/`) as of `git_sha7 199ddba`. Sections that need a
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
  traces. This is exactly the subset DeepEval cross-checks in section 7 below.
- `synthetic_dev_queries.jsonl` (new, M7a) — the frozen LlamaIndex-generated
  synthetic query set: 106 questions over 73 distinct gold-bearing dev
  chunks (113 chunks attempted; 40 dropped by the keep filter), model
  `z-ai/glm-5.3-flash`, `prompt_hash 93801419e0abfb5b`.

`just braintrust-sync` mirrors all five as Braintrust datasets
(`maud-dealpoint-dev`, `-test`, `-counterfactual`, `-judged_calibration`,
`-synthetic_query`) — never hand-edited, always rebuilt from these files, with
a stable per-row id so a second sync updates rows in place rather than
duplicating them.

## 2. RAG Lab

`data/reports/li_rag_eval.md` — LlamaIndex's `RetrieverEvaluator` run over all
six frozen M3 candidates on the full 58-case dev set, `chunk_version
8e5e8ba56765` / `index_version e2b4a2b97561` (asserted to match
`data/reports/versions.json` before anything runs). `obj/` is evaluated at
`TOURNAMENT_K=10` (the LlamaIndex side stays at `top_k=5`), so hit@5, hit@10
and MRR reproduce `data/reports/tournament.json` exactly.

| config | li/hit_rate | obj/gold_span_hit@5 | obj/gold_span_hit@10 | obj/gold_span_mrr |
|---|---|---|---|---|
| dense | 81.0% | 81.0% | 84.5% | 0.626 |
| bm25 | 86.2% | 86.2% | 94.8% | 0.673 |
| **hybrid_rrf** (M3 winner) | **91.4%** | **91.4%** | **96.6%** | **0.732** |
| hybrid_rrf_rerank | 91.4% | 91.4% | 94.8% | 0.727 |
| multi_query_fusion | 82.8% | 82.8% | 91.4% | 0.690 |
| multi_query_fusion_rerank | 87.9% | 87.9% | 91.4% | 0.718 |
| li_native_bm25 (native LlamaIndex `BM25Retriever`, independent of `bm25s`) | 84.5% | 86.2% | 94.8% | 0.673 |

The six adapter-wrapped configs agree with `obj/` by construction
(Spearman `rho = 1.0`; `expected_ids` for the LlamaIndex `RetrieverEvaluator`
ARE the gold-bearing chunk ids), reported plainly rather than manufactured. A
seventh, genuinely native config (`li_native_bm25`, LlamaIndex's own
tokenizer/scoring over the same chunks) surfaces 3 real disagreements against
`obj/`'s `bm25` config: 1 LI-hit/MAUD-miss, 2 MAUD-hit/LI-miss. Worked
examples: `contract_0__q02` (LI-hit/MAUD-miss, cause `duplicate_relevant_chunks`)
and `contract_3__q06` (MAUD-hit/LI-miss, cause `duplicate_relevant_chunks`).

Synthetic-query robustness (secondary, never benchmark truth): the frozen
winner `hybrid_rrf` scores 94.3% li/hit_rate / 97.2% obj/hit@5 on the 106
synthetic queries vs 88.7%/n-a for dense — the winner stays strong under a
different query distribution. Cost reconciles three ways: `kept_questions_usd`
(only the 73 kept chunks), `frozen_run_usd` (the real cost of all 113
generation calls), `ledger_purpose_total_usd` (every `purpose=synthetic_query`
ledger row, including superseded earlier attempts) — see `li_rag_eval.json`'s
`synthetic.cost_note`. Rag Lab experiments in Braintrust, now logging one row
per case with real `li/`/`obj/` scores rather than a single placeholder row:
`rag-m3-dense`, `rag-m3-bm25`, `rag-m3-hybrid`, `rag-m3-hybrid-rerank`,
`rag-m3-fusion`, `rag-m3-fusion-rerank`, `rag-m7-li-crosscheck`,
`rag-m7-synthetic`.

## 3. Agent Systems

Four versions (A–D) differing by exactly one config key each, plus the M6
Pareto sweep varying only the model at arm D. Representative experiment:
`A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc` (32 cases, arm A). The frozen
winner retriever is `hybrid_rrf` (`ARM_C_RETRIEVER["name"]`), hit@5 `0.9138`
vs dense `0.8103` (M3 tournament, `data/reports/tournament.json`).

## 4. Logs trace

Example case: `contract_39__redacted_q05` (a redacted, defined-term case).
`just braintrust-sync` replays `case -> agent -> search_agreement (with
dense/bm25/rrf/rerank stage children, derived from the arm's static retriever
config) -> lookup_defined_term/get_section -> final_answer -> scoring`,
entirely from the stored `record.trajectory` — **zero model calls**. Scoring
spans now carry `judge/` and `deepeval/` provenance alongside `obj/` where a
representative case matches a judged/DeepEval-scored variant, not just
`obj/`. Spans are rooted on a dedicated `m7-representative-traces` experiment
made "current" at init time, so they land as real Braintrust spans rather
than the SDK's `_NoopSpan` (which is what a `braintrust.start_span()` call
with no current experiment silently returns). Representative cases, chosen
by documented rule (see `data/reports/representative_cases.json`, now
actually written by `just braintrust-sync` rather than a stale orphan):

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
(`dealpoint.eval.scorers`); the handler bridge from Braintrust's
`(input, output, expected, metadata)` shape back to the canonical
`(case, finding, record, canonical_text)` signature is exercised end to end
by an offline test. Namespaces: `obj/` (canonical), `li/` (LlamaIndex
hit_rate/mrr), `judge/` (calibrated judge dimensions, rescaled from their
native 1–5 scale into Braintrust's required [0, 1] score range),
`deepeval/` (task completion, tool correctness, argument correctness, step
efficiency). The `procedure/` namespace was dropped — it never had any score
mapped to it, and inventing one to fill it would have been exactly the kind
of placeholder the no-bloat rule forbids. Every M7 experiment logs ≤ 12
scores/case on subsets ≤ 60 cases; M4/M6 sweeps keep six — enforced twice by
`dealpoint.eval.braintrust_sync.assert_score_budget` (the plan's declaration)
and `assert_actual_score_budget` (what is actually logged; secondary
diagnostics like `usd`/`wall_ms`/`tool_calls` go to
`metadata.secondary_diagnostics`, never into `scores`).

**Scorer/prompt publishing, honestly scoped:** Braintrust 0.37.0's own
`Project.publish()` refuses code functions outright ("Code functions cannot
be published directly. Use `braintrust push` instead") and only actually
registers prompts/parameters via its `insert-functions` API. This sync
therefore declares the six scorer handlers (proving the canonical-import
bridge works) but does not claim they are live in Braintrust; prompts and
parameters ARE published. Tested and rejected, documented rather than
hidden — see `SCORER_PUBLISH_LIMITATION` in `braintrust_sync.py`.

## 6. Review

**[M7b]** The 12-trace blinded review set is synced (`review_set(12)` in
`dealpoint.eval.braintrust_sync`) and ready — the diversity rule (spans
reasoning types, ≥ 1 abstention, ≥ 2 variants) is now enforced by
construction (verified against the plain seeded-order prefix first, with an
explicit fallback construction only if that check fails) rather than holding
by luck of the current case mix. `variant_id` is deliberately never pushed to
the synced dataset — it would un-blind the set — and the packet's own
blinded body (`packet_text`, the same text judges/humans read) is pushed as
the row's `expected` field so there is something to actually read in
Braintrust Review. Scoring it there happens after the restart. Local human
calibration now has real data: `data/eval/calibration/human_scores.jsonl`
carries 24 scored rows (added by the engineer during this session) — DeepEval
vs human comparisons in section 7 report `n=24, status=available`, not
`pending`; existing scores are canonical and displayed, never re-labelled.

## 7. Eval of Evals

`data/reports/deepeval_crosscheck.md` — DeepEval's independent read on the
same 108 judged-subset traces. Resolved evaluator (never hardcoded, read from
`judge_slate.json`'s verified non-candidate-family judges):
`mistralai/mistral-small-3.2-24b-instruct` (provider `mistralai`, deepeval
`4.2.1`). **Independence caveat:** this model is one of the three M5
judge-trio models, not a fourth independent one — all three verified
non-candidate-family judges ARE the trio, so there is no alternative under
the existing (unchanged) resolution rule. `vs_judge` comparisons are
therefore contaminated by shared-model bias; factored into the
classification below. Metrics: `task_completion`, `tool_correctness`,
`argument_correctness`, `step_efficiency` (always the documented **GEval**
definition, never DeepEval-native `StepEfficiencyMetric` — that metric
requires DeepEval's own `@observe` trace capture, a second tracing path the
brief excludes; the pre-repair run paid real money to score 93/108 traces
`0.0` with "no trace provided"). Coverage is now reported honestly per
metric (a handful of transient `RateLimitError` nulls survive a bounded
retry; none are silently hidden). Comparisons: DeepEval `task_completion` vs
`grounded_accuracy` — Spearman `rho ≈ 0.21` (n=45, weak/under-powered); vs
the calibrated judge `reasoning` dimension — `rho ≈ 0.84` (n=105, strong, but
see the independence caveat); vs human — now **available** (n=24, not
pending). **Classification: `KEEP_OPTIONAL_ANALYSIS`** — the correlation
with `grounded_accuracy` is under-powered to promote to core diagnostic, and
the shared-evaluator-model caveat means the strong judge correlation cannot
be read as trustworthy independent corroboration either.

## 8. Loop / SQL

**[M7b]** Braintrust Loop investigation thread. The underlying data is ready:
`docs/braintrust-queries.md` has the six BTQL investigations (retrieval
rescue, failure attribution, reasoning slices, DeepEval disagreement, model
economics, trajectory inefficiency), executed against the live project AFTER
`just braintrust-sync`, every query referencing the actual quantity its
question names (`scores."obj/grounded_accuracy"`,
`scores."deepeval/task_completion"`, `metadata.secondary_diagnostics."obj/tool_calls"`,
etc. — not one of them was a `count(1)`-only placeholder). Recorded in
`data/reports/btql_investigations.json` with the exact query, the `curl`/
`bt sql` reproduction commands (the `curl` form now actually runs — its
payload's embedded single quotes are shell-escaped rather than breaking the
command), and the observed row counts: retrieval rescue 0 rows (verified
data-supported cause, not retention purge: every row in the target agent
experiment has `secondary_diagnostics."obj/tool_calls" == 0` in this run, so
no case triggered a second `search_agreement` call before a
grounded-correct answer), failure attribution 1 row, reasoning slices 7
rows, DeepEval disagreement **17 rows** (real disagreement, found live),
model economics 1 row, trajectory inefficiency 50 rows. This is the same
explanation `btql_investigations.json`'s own `notes` field for query 1
carries -- the two were previously out of sync (one blamed 14-day
retention purge, the other blamed the data) and are now reconciled to the
one the data actually supports.

## 9. Debugger

**[M7b]** Topics/Patterns/Debugger use over the synced traces.

## 10. Model Economics

`data/reports/pareto.md` — arm D held fixed, only the model varies. Frontier
(min $/case, max grounded_accuracy): `qwen/qwen3.7-flash` and
`deepseek/deepseek-v4-flash`. Braintrust experiments tagged `stage=economics`:
`pareto-deepseek_deepseek-v4-flash`, `pareto-qwen_qwen3.7-flash`,
`pareto-google_gemini-3.1-flash-lite`, `pareto-anthropic_claude-haiku-4.5`,
`pareto-z-ai_glm-5.3-flash`. **Code path, not yet a live fact:** `_economics_experiment_plans`
now BUILDS one row per case (from the stored M6 result JSONLs named in
`PARETO_MANIFEST_PATH`) rather than a single placeholder row, and this is
covered by an offline fake-client test -- but no completed live sync has
landed those rows in the actual Braintrust project yet, because the live
run was blocked by the workspace's `num_scores_calendar_months` plan limit
(see the `live_run_status` field in `data/reports/braintrust_sync.json`)
before it reached the economics experiments. `just braintrust-sync-dry-run` exercises this exact
code path with zero network calls and confirms the per-case rows are built
correctly; only a live run (after the quota resets) will confirm they land.

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
uv run python -m dealpoint.eval.disk_guard            # m7a_disk_guard.json (re-measured)
uv run python -m dealpoint.rag_lab.report             # li_rag_eval.json/.md
uv run python -m dealpoint.eval.deepeval_adapter      # deepeval_crosscheck.json/.md (METERED)
just braintrust-sync                                  # datasets, experiments, replayed traces (idempotent)
uv run python -m dealpoint.eval.btql                  # btql_investigations.json + docs/braintrust-queries.md
uv run python -m dealpoint.eval.framework_versions    # framework_versions.json
just pareto-report                                    # regenerates the README PARETO block
```

**Known environment limitation (2026-09-06):** a full live `just
braintrust-sync` run in this environment hit the Braintrust workspace's
`num_scores_calendar_months` plan limit (11016 of 11000) partway through —
an account-level quota, not a code defect. The mapping is nonetheless fully
covered by 23 offline tests against a fake client (spec: "offline tests with
a fake client cover the mapping"), including the score-normalization fix
(judge dimensions rescaled from their native 1–5 scale into [0, 1], which the
same run surfaced live before the quota was hit) and the score-budget/
per-case-rows/idempotency/blinding fixes this repair made. Re-running `just
braintrust-sync` after the quota resets (or on an upgraded plan) is expected
to complete cleanly against this same code.
