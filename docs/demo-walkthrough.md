# Demo walkthrough

Every name, id and number below resolves to `data/reports/demo_manifest.json`, which `just braintrust-cockpit` regenerates from Git/local sources. Sections needing a human operator in an MCP-enabled session are marked **[cockpit session]**; everything else is produced by code, checked by the `gate_m7b` offline suite.

## Role diagram

```
custom Python  --> benchmark truth      (parser, canonical offsets, MAUD gold spans, scorers)
LlamaIndex     --> RAG lab / RAG eval   (retriever composition, RetrieverEvaluator, synthetic queries)
DeepEval       --> independent agent-eval cross-check (task completion, tool correctness, step efficiency)
Braintrust     --> traces / experiments / comparison (surface, not source of truth)
local reports + Git --> permanent evidence (data/reports/, data/results/, specs/)
```

Only the first row ever decides a winner: MAUD gold-span overlap
(`overlap_chars` against `MIN_GOLD_OVERLAP_CHARS=50`). LlamaIndex and
DeepEval are cross-checks, never a replacement.

## The hero path

**Hero case `contract_32__q04`** (arms A@haiku/D@haiku disagree on `obj/grounded_accuracy`: `False` vs `True`; max pairwise judge spread `4` among 2 candidates). Replayed as `m7b-hero-case`: `case -> agent -> ... -> scoring -> {judge/mistral, judge/nvidia, judge/bytedance, judge/aggregate}` for both variants, zero model calls.

## 1. Datasets

Frozen case sets under `data/eval/`: `dev.jsonl` (58 cases, 5 documents),
`test.jsonl` (167 cases; `test_subset_v1.json` is the 32-case budget-scaled
subset the four-arm/Pareto sweeps ran on), `counterfactual.jsonl` (40 cases),
`judged_subset.json` (18 cases, `subset_hash 5918ef10a7e6`, x 6 variants =
108 traces), `synthetic_dev_queries.jsonl` (106 LlamaIndex-generated
questions, `prompt_hash 93801419e0abfb5b`). `just braintrust-sync` mirrors
all five as Braintrust datasets (`maud-dealpoint-*`) with stable per-row ids,
never hand-edited.

## 2. RAG Lab

`data/reports/li_rag_eval.md`: LlamaIndex's `RetrieverEvaluator` over all six
frozen M3 candidates on the 58-case dev set (`index_version e2b4a2b97561`).

| config | li/hit_rate | obj/hit@5 | obj/hit@10 | obj/mrr |
|---|---|---|---|---|
| dense | 81.0% | 81.0% | 84.5% | 0.626 |
| bm25 | 86.2% | 86.2% | 94.8% | 0.673 |
| **hybrid_rrf** (M3 winner) | **91.4%** | **91.4%** | **96.6%** | **0.732** |
| hybrid_rrf_rerank | 91.4% | 91.4% | 94.8% | 0.727 |

The six adapter-wrapped configs agree with `obj/` by construction (Spearman
`rho = 1.0`). A seventh, genuinely native config (`li_native_bm25`) surfaces
3 real disagreements against `obj/`'s `bm25` (cause: `duplicate_relevant_chunks`
in both worked examples, `contract_0__q02` and `contract_3__q06`).
Synthetic-query robustness: the frozen winner scores 94.3% li/hit_rate on the
106 synthetic queries — stays strong under a different query distribution.
Braintrust experiments: `rag-m3-dense`, `rag-m3-bm25`, `rag-m3-hybrid`,
`rag-m3-hybrid-rerank`, `rag-m3-fusion`, `rag-m3-fusion-rerank`,
`rag-m7-li-crosscheck`, `rag-m7-synthetic`.

## 3. Agent Systems

Four versions (A–D), one config key each, plus M6's Pareto sweep (arm D,
model varies). Representative experiment: `A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc`
(32 cases, arm A). Winner retriever `hybrid_rrf`, hit@5 `0.9138` vs dense
`0.8103` (`data/reports/tournament.json`).

## 4. Logs trace

Example: `contract_39__redacted_q05` (redacted, defined-term). `just
braintrust-sync` replays `case -> agent -> search_agreement (dense/bm25/rrf/
rerank stage children) -> lookup_defined_term/get_section -> final_answer ->
scoring`, entirely from stored `record.trajectory` — zero model calls.
Scoring spans carry `judge/`/`deepeval/` provenance alongside `obj/` where a
representative case matches a judged/DeepEval-scored variant. Representative
cases (`data/reports/representative_cases.json`), chosen by rule:

| category | case_id | variant_id |
|---|---|---|
| successful_direct | `contract_0__q01` | `D@openai/gpt-5.6-luna-pro` |
| retrieval_rescue | `contract_0__q01` | `C@z-ai/glm-5.3-flash` |
| defined_term_cross_ref | `contract_99__redacted_q06` | `D@anthropic/claude-haiku-4.5` |
| inefficient_trajectory | `contract_99__q03` | `D@qwen/qwen3.7-flash` |
| wrong_answer | `contract_144__q09` | `A@anthropic/claude-haiku-4.5` |
| abstention_counterfactual | `contract_75__oos09` | `A@anthropic/claude-haiku-4.5` |

Saved views over these spans, idempotent by name: `Judged traces by variant` (70a5af97-633a-4437-8bcf-fa576dfa1a63), `Judge disagreement` (00bd8ca6-7a82-4c6f-b2b3-f0d7344078a1), `Retrieval rescue` (922c48dd-5634-41fe-9362-635df1ca4504), `Failure attribution` (86c17089-a244-4757-bdd6-7def89c7f6f0), `DeepEval vs judge disagreement` (3b40b355-eeea-4400-8f47-03031d7e946c), `Trajectory inefficiency` (d72191ea-8253-42e3-8a0d-37861c65776f), `Review set (12)` (7cf518e1-6271-4551-892d-4a28497b0e95). [cockpit session] A Loop-generated custom trace view (judge spans as a 3x4 grid, human row beneath); its `tv` param via `just demo-manifest-record --step 4`.

## 5. Scorers

Braintrust functions import the canonical implementations directly
(`grounded_accuracy`, `answer_correct`, `citation_verbatim`,
`citation_gold_overlap`, `required_evidence_met`, `skill_adherence`).
Namespaces: `obj/`, `li/`, `judge/` (rescaled 1-5 into [0,1]), `deepeval/`.
Every M7 experiment logs <= 12 scores/case on subsets <= 60 cases; M4/M6
sweeps keep six — enforced by `assert_score_budget`/`assert_actual_score_budget`.
Braintrust 0.37.0's `Project.publish()` refuses code functions directly, so
this sync declares the six scorer handlers but does not claim they are live;
prompts and parameters ARE published (`SCORER_PUBLISH_LIMITATION`).

## 6. Review

Human-scoring path: **local_form** -- Starter plan allows one configured review score, not the four rubric dimensions needed, so scoring stays at `data/eval/calibration/form.md` (M5) and `just braintrust-cockpit` pushes 96 `human/<dimension>` scores onto the matching `judge-<variant_id>` rows (0 pushed in this run). Shown in the experiment table and trace, not Review mode.

[cockpit session] Scoring the review set at `form.md`, then `just calibration` and `just braintrust-cockpit` (`--step 1`).

[cockpit session] Playground: the hero case's blinded packet against the calibrated judge prompt, three judge models side by side (`--step 3`).

## 7. Eval of Evals

`data/reports/deepeval_crosscheck.md`: DeepEval's independent read on the
108 judged-subset traces. Evaluator: `mistralai/mistral-small-3.2-24b-instruct`
(one of the three M5 judge-trio models, not a fourth independent one —
factored into the classification). Metrics: task_completion,
tool_correctness, argument_correctness, step_efficiency (GEval, never
DeepEval-native). Comparisons: vs `grounded_accuracy` Spearman `rho ~= 0.21`
(n=45, weak); vs judge `reasoning` `rho ~= 0.84` (n=105, contaminated by
shared model); vs human n=24, available. **Classification:
`KEEP_OPTIONAL_ANALYSIS`.**

## 8. Loop / SQL

The six BTQL investigations are saved as four of the seven cockpit views (`Retrieval rescue`, `Failure attribution`, `DeepEval vs judge disagreement`, `Trajectory inefficiency`), matching `docs/braintrust-queries.md` verbatim.

[cockpit session] A Loop thread over the hero case asking query 2's question in plain words; URL via `just demo-manifest-record --step 2`.

## 9. Debugger

Topics (`trace-outcome-summary` facet, clustering on) render each `case > agent` span as question/tools/answer/`obj/grounded_accuracy` text.

[cockpit session] One Pattern, `Trajectory inefficiency: cap-hit or >=6 tool calls without a correct answer` (5 supporting trace ids from the same predicate as BTQL query 6) is defined in code but has no public REST creation route on this platform, so it is created during the cockpit session (`new_pattern`), not by `braintrust_cockpit.py`.

[cockpit session] Cluster names surfaced by Topics, and whether any maps to a query-2 failure cause (`--step 2`).

## 10. Model Economics

`data/reports/pareto.md`: arm D fixed, model varies. Frontier (min $/case,
max grounded_accuracy): `qwen/qwen3.7-flash`, `deepseek/deepseek-v4-flash`.
Experiments tagged `stage=economics`: `pareto-deepseek_deepseek-v4-flash`,
`pareto-qwen_qwen3.7-flash`, `pareto-google_gemini-3.1-flash-lite`,
`pareto-anthropic_claude-haiku-4.5`, `pareto-z-ai_glm-5.3-flash`, one row per
case from the stored M6 result JSONLs, offline-tested; the last live sync was
blocked by the workspace's `num_scores_calendar_months` plan limit before it
reached these experiments (`data/reports/braintrust_sync.json`'s
`live_run_status`).

## 11. Dashboard

A saved dashboard, `DealPoint eval overview` (b3122524-c555-4599-8544-f302bc284b52), five charts: obj/grounded_accuracy by arm/model; judge/<dimension> mean by variant; judge vs human on the review set (captioned "pending human calibration" until scores exist); $/case by model; DeepEval vs obj/ agreement rate.

## Braintrust tools — decision, not a stub

`search_agreement`/`get_section`/`lookup_defined_term` are not exposed as
Braintrust tools: they need the 68 MB Qdrant index, the 64 MB derived
canonical corpus and fastembed's ONNX weights loaded in-process — duplicating
this project's entire index/data layer, which the no-bloat rule and the
brief's "no second tracing/data architecture" guidance forbid. Documented,
not omitted.

## Reproduce this walkthrough

```bash
uv run python -m dealpoint.eval.disk_guard            # m7a_disk_guard.json
uv run python -m dealpoint.rag_lab.report             # li_rag_eval.json/.md
uv run python -m dealpoint.eval.deepeval_adapter      # deepeval_crosscheck.json/.md (METERED)
just braintrust-sync                                  # datasets, experiments, replayed traces
uv run python -m dealpoint.eval.btql                  # btql_investigations.json + docs/braintrust-queries.md
just braintrust-cockpit                               # views, dashboard, Topics, one Pattern, hero-case replay
just demo-walkthrough                                 # regenerates this document from the manifest
```

**Known environment limitation (2026-09-06):** a full live `just
braintrust-sync` run hit the Braintrust workspace's `num_scores_calendar_months`
plan limit (11016 of 11000) partway through — an account-level quota, not a
code defect. The mapping is fully covered by offline tests against a fake
client. Re-running after the quota resets (or on an upgraded plan) is
expected to complete cleanly against this same code.

---

Every link and id above is one `data/reports/demo_manifest.json` can regenerate: re-run `just braintrust-cockpit` then `just demo-walkthrough`. This doc has no content that only exists because a person clicked it once.
