# M7a resume plan — framework roles: LlamaIndex RAG lab, DeepEval cross-check, Braintrust sync

**Authority.** `specs/grilled-product-brief.md` is the requirements document. `specs/milestones/m7a.md`
bounds this work. Both are **read-only** — do not edit either. Where the brief and the milestone spec
differ on a product decision, the brief wins and the difference is *recorded* in the relevant report's
`brief_differences` list, never resolved silently.

**Situation.** M7a is not a greenfield build. Three previous attempts landed a large, mostly working
implementation (last WIP commit `23acff1`). `uv run pytest -m "gate_m7 and not needs_network" -q` is
already green (61 passed), the full offline suite is green (512 passed), `ruff` and `pyright` are clean,
and $0.099348 of M7a spend is already on the ledger. **The gate passing does not mean the milestone is
done.** A file-by-file audit found that several Definition-of-done bullets are satisfied only on the
surface: artifacts exist but are empty, fabricated, degenerate, or silently discarded. This plan fixes
exactly those, and nothing else.

**Do not redo what is already correct.** In particular, never regenerate the frozen synthetic query set
(`data/eval/synthetic_dev_queries.jsonl`, 106 rows, $0.059 already spent), never re-run M1–M6 sweeps,
never retune the M3 winner, never touch `data/eval/test*.json*`.

---

## 0. Ground truth established by the audit

Verified on disk at HEAD `29bf205`. Trust these numbers; you do not need to re-derive them.

**Already correct — leave alone:**

| Thing | Evidence |
|---|---|
| `pyproject.toml` optional groups `rag-lab`, `deepeval`; both installed, no torch/transformers | `pyproject.toml:19,21`; `llama-index-core` 0.14.24, `llama-index-retrievers-bm25` 0.8.0, `deepeval` 4.2.1 |
| README **RESULTS** block already reshaped to the M7a rule, and `tests/test_readme_results.py` already updated | `README.md:40-64`; `dealpoint/eval/report.py:1108`; `tests/test_readme_results.py:24,70-87` |
| `data/reports/four_arm.md` carries the complete moved-out diagnostics | `four_arm.md:31,44,191` |
| `data/reports/pareto.md` **already** carries all seven sections that must leave the README | `pareto.md:19,30,37,41,64,70,76` |
| Synthetic query set: 106 questions, 73 chunks, <=2/chunk, real metered generation, sha256 matches in two places | `synthetic_generation_run.json`, `li_rag_eval.json.synthetic` |
| DeepEval subset is exactly the M5 judged subset x 6 judged variants (18 x 6 = 108), no short rows | `deepeval_crosscheck.json.subset` |
| Evaluator resolution honours `DEEPEVAL_MODEL`, policy-checks the family, defaults from `judge_slate.json` | `dealpoint/eval/deepeval_adapter.py:86-139` |
| 29 Braintrust experiment **names** all present and correctly staged/tagged | `braintrust_sync.json.experiments` |
| Representative-case selection rule, six categories, all resolve to real case ids; replay imports no LLM client | `braintrust_sync.py:633,647,685` |
| Braintrust tools non-exposure decision, documented with a concrete reason, no stubs | `braintrust_sync.py:907` |
| `claude mcp list` recorded verbatim | `docs/milestones/m7a.md:62-63` |
| `docs/demo-walkthrough.md` order, `[M7b]` markers, role diagram, and every spot-checked id/number resolves | `docs/demo-walkthrough.md:11-17` + `tests/test_m7_artifacts.py:105,117` |
| `gate_m7` marker registered; **zero** `gate_m7` tests carry `needs_model` | `pyproject.toml:48`; the task's gate command and `just gate-m7` select the identical 61 tests |

**Spend and headroom (measured from `data/results/spend_ledger.jsonl`, 4139 rows):**

```
m4_1 $1.420234   m4 $0.751879   m6 $0.742655   m1 $0.419120
m2   $0.170014   m5 $0.110038   m7a $0.099348
GLOBAL $3.713288 / $6.00  ->  headroom $2.286712
```
M7a is at $0.0993 against a $1.00 soft target and $2.50 absolute. All 908 M7a rows carry
`milestone_tag` and `purpose`; all models are in-slate; no router endpoints. **This plan's new metered
work is one DeepEval re-run at ~$0.04-0.06.** Everything else below is free (no model calls).

---

## 1. What is actually broken (the work)

Eight defect clusters, each mapped to a Definition-of-done bullet. Do them in this order; later items
depend on earlier ones.

---

### D1 — Disk guard artifact is hand-authored, not measured
*DoD bullet: "Optional dependency groups installed within the disk guard; `framework_versions` recorded."*

`data/reports/m7a_disk_guard.json` was written by hand, not by `record_guard()`: it has an extra `notes`
key the writer cannot emit, unsorted keys (the writer uses `sort_keys=True`), a placeholder
`measured_at` of `2026-09-05T00:00:00+00:00`, and free-space deltas of 2.56 MB / 1.65 MB against a prose
claim that `.venv` grew 111 MB. `dry_run_install()` (`dealpoint/eval/disk_guard.py:29-39`) has **zero
callers anywhere in the repo**, and `main()` (`:56-61`) is a stub that ignores argv and prints
`free_bytes_now`. No test covers the guard; `M7A_DISK_GUARD_PATH` is missing from `M7A_REPORT_PATHS`
(`tests/test_m7_artifacts.py:34-42`).

**Do:**
1. Make `disk_guard.main()` real: run `dry_run_install()` for both groups (`rag-lab`, `deepeval`),
   capture free space before/after, evaluate against the 500 MB floor (`dealpoint/config.py:320`), and
   write via `record_guard()` so the artifact is machine-produced.
2. The groups are already installed, so `uv pip install --dry-run` will report already-satisfied. Record
   that honestly: a `re_measured_at` field plus the dry-run stdout, and keep the original install-time
   deltas under a clearly labelled `install_time_observation` key rather than passing them off as this
   measurement. Do **not** uninstall anything to force a cleaner measurement — never destabilise the
   M1-M6 environment.
3. Add a `just disk-guard` recipe.
4. Add `M7A_DISK_GUARD_PATH` to `M7A_REPORT_PATHS` in `tests/test_m7_artifacts.py:34-42` and add a
   `gate_m7` test asserting the artifact was written by `record_guard()` (sorted keys, real
   `measured_at`, free space above the floor).

Current free space is 1.5 GB against a 500 MB floor — no scope-down is needed.

---

### D2 — LlamaIndex RAG lab: `obj/` hit@10 and MRR are wrong
*DoD bullet: "`li_rag_eval.json`: per-retriever LI hit_rate/mrr vs `obj/` hit@k/mrr on all dev cases..."*

`dealpoint/rag_lab/report.py:346` calls `evaluate_retriever_obj(..., k=5)`, which passes `k=5` into
`tournament.evaluate_config`. That retrieves only 5 chunks, so `hit_at_10` is derived from a 5-item list
and MRR is truncated at rank 5. Result: `gold_span_hit_at_10 == gold_span_hit_at_5` for all six configs,
and both hit@10 and MRR **contradict the frozen M3 tournament**:

```
config                     m3 hit@5  li hit@5   m3 h@10  li h@10   m3 mrr   li mrr
bm25                        0.8621    0.8621    0.9483   0.8621   0.6727   0.6635
dense                       0.8103    0.8103    0.8448   0.8103   0.6260   0.6207
hybrid_rrf                  0.9138    0.9138    0.9655   0.9138   0.7319   0.7244
hybrid_rrf_rerank           0.9138    0.9138    0.9483   0.9138   0.7271   0.7227
multi_query_fusion          0.8276    0.8276    0.9138   0.8276   0.6900   0.6784
multi_query_fusion_rerank   0.8793    0.8793    0.9138   0.8793   0.7180   0.7126
```

**Do:** evaluate the `obj/` side at `TOURNAMENT_K` (`dealpoint/config.py:184` = 10) so hit@5, hit@10 and
MRR all reproduce `data/reports/tournament.json` **exactly**. Keep the LlamaIndex side at top_k=5 and
compare `li/hit_rate` against `obj/gold_span_hit@5`; state that k explicitly in the report and the .md.
Add a `gate_m7` test asserting every config's `obj` triple equals the corresponding frozen
`tournament.json` value — that is the cheapest possible guard against this class of drift.

---

### D3 — LlamaIndex RAG lab: the cross-check is tautological, so there is no evidence
*DoD bullet: "...ranking-order agreement, disagreement cases with causes"; reviewer criterion: "no
duplicate canonical truth".*

`disagreements: []`, `worked_examples: []`, `li_hit_maud_miss: 0`, `maud_hit_li_miss: 0`, Spearman 1.0
on both metric pairs. This is structural, not incidental: `build_li_retriever` wraps the *same*
`build_retriever` output (`adapters.py:130-145`) and `gold_bearing_chunk_ids` uses the *same* overlap
rule as `tournament._first_hit_rank` (`evaluate.py:78-95`). LlamaIndex and `obj/` are two readings of one
ranking, so agreement is guaranteed by construction. The report says so honestly, but the spec's
cross-framework diagnostic produces zero evidence, zero cause labels and zero worked examples — and a
pure re-derivation of canonical truth is exactly what the reviewer's "no duplicate canonical truth"
criterion targets.

The spec anticipates this. Section 1A permits wrapping the project retrievers **"or compose native
equivalents where LlamaIndex has them"**. Take the second branch for one retriever.

**Do:**
1. Add a genuinely native LlamaIndex retriever: `llama_index.retrievers.bm25.BM25Retriever`
   (`from_defaults(nodes=..., similarity_top_k=5)`) built over the *same* canonical chunks. **Verified
   working offline in this venv, CPU-only, zero model calls.** Register it as a seventh evaluated config,
   e.g. `li_native_bm25`, and evaluate it with the same `RetrieverEvaluator` on the same 58 dev cases.
2. Compare `li_native_bm25`'s LlamaIndex hit/miss against the project `bm25` config's `obj/` hit/miss
   per case. LlamaIndex's own tokenizer and scoring differ from `bm25s`, so this produces **real
   disagreements in both directions** — which is the whole point of the deliverable.
3. Run the existing `find_disagreements` cause rule over those cases and fill `disagreements` and
   `worked_examples` (1-2 with case ids, per the spec).
4. Keep the six adapter-wrapped configs and keep the honest note that their agreement is 1.0 by
   construction. Report both: the tautological result *and* the independent one. Do not delete the
   existing note — it is accurate and it explains the contrast.
5. Fix the unreachable `reranking` cause label. `find_disagreements` (`evaluate.py:286-292`) never
   computes or passes `rerank_would_hit`/`current_hit`, so the branch at `evaluate.py:230-237` can only
   fire from a test. Either wire the pre-rerank ranking through for rerank configs, or remove
   `reranking` from `cause_rule`. **Do not describe a rule the pipeline cannot apply.**

---

### D4 — LlamaIndex RAG lab: provenance and reporting gaps

1. **Cost is under-reported three ways.** `synthetic.total_usd` is `0.015215` (only the 73 chunks that
   yielded kept questions), the frozen generation run itself cost `$0.02412` (113 calls), and total
   ledger `purpose=synthetic_query` spend is `$0.059064` (349 rows, including earlier attempts). Nothing
   reconciles them. Report all three under distinct, self-explaining keys
   (`kept_questions_usd`, `frozen_run_usd`, `ledger_purpose_total_usd`) with a one-line note that the
   ledger total includes superseded attempts.
2. **The 113 -> 73 chunk drop is unmentioned.** 40 gold-bearing chunks (35%) yielded no kept question
   after `_looks_like_question` filtering (`synthetic.py:167-180`). Record `n_chunks_attempted: 113`,
   `n_chunks_kept: 73` and the filter's name. `n_chunks` currently means two different things in two
   files; disambiguate.
3. **Hardcoded winner.** `synthetic_eval.py:81` hardcodes `winner_name = "hybrid_rrf"`. Read the frozen
   winner from `tournament.json` instead.
4. **No test touches the frozen artifact.** `tests/test_synthetic_queries.py:55-84` round-trips a
   hand-built row in `tmp_path`. Add `gate_m7` tests asserting the *real*
   `data/eval/synthetic_dev_queries.jsonl`: <=2 questions per chunk, every chunk id is a dev chunk, every
   row carries generator/generator_version/prompt_hash/model/usd, and the file's sha256 matches the value
   recorded in `li_rag_eval.json`. Add a test asserting `li_rag_eval.json` covers all six frozen M3
   candidates (plus the new native one).

The hash assertions for the frozen winner and test set (`frozen_assertions.test_subset_v1_sha256`,
`tournament_json_sha256`, `tests/test_m7_artifacts.py:74-80`) already exist and are correct — keep them.

---

### D5 — DeepEval: `step_efficiency` never ran
*DoD bullet: "metrics on the judged subset x variants ... step efficiency"*

Measured distribution over 108 traces: `0.0` x93, `1.0` x1, `None` x14. 93 of the reasons are literally
*"No task or trace provided for evaluation. Efficiency cannot be assessed without input."*

Root cause: `StepEfficiencyMetric` sets `requires_trace = True` and reads `test_case._trace_dict`
(`.venv/.../deepeval/metrics/step_efficiency/step_efficiency.py:49,156`), which is populated only by
DeepEval's `@observe` tracing. `dict_to_llm_test_case` never sets a trace
(`deepeval_adapter.py:238-256`). `TaskCompletionMetric` survives only because it has a no-trace fallback.
So every step-efficiency number — and the `step_efficiency_vs_judge_trajectory` Spearman of
`-0.0097` — is an artifact of empty input paid for with real model calls.

**Do:** switch `step_efficiency` to the **documented GEval definition already written** at
`deepeval_adapter.py:337-346`. The spec explicitly allows this: *"step efficiency (DeepEval-native if
present, else a documented GEval definition)"*. `STEP_EFFICIENCY_GEVAL_CRITERIA` and its hash
`befc63f450cebd6e` are already recorded in the report. Set
`step_efficiency_basis = "documented GEval"` and record **why** the native metric was rejected (it
requires `@observe` trace capture, which would mean adopting DeepEval's tracer — a second tracing path
the brief §4 excludes and the spec's no-bloat rule forbids). Put that in the report's `decisions` list.

Prefer this over populating `_trace_dict`: constructing DeepEval's internal trace shape by hand is
fragile across versions and edges toward the dual-tracing architecture the brief bans.

---

### D6 — DeepEval: degraded run and comparison defects

1. **47 of 432 metric cells are `429 RateLimitError` nulls, entirely unsurfaced.** Spread over 22
   traces (`task_completion` 15, `argument_correctness` 18, `step_efficiency` 14, `tool_correctness` 0).
   Swallowed by the deliberate bare `except` at `deepeval_adapter.py:365-366` — correct by design, but
   the `.md` states "18 cases x 6 variants = 108 traces" with no caveat and the word "429" appears
   nowhere outside per-trace reasons. **Add a `coverage` block** to the JSON and a paragraph to the
   `.md`: per metric, `n_scored`, `n_null`, and the null reasons grouped by exception type. Add a bounded
   retry (a few seconds' backoff, at most two retries) so a transient 429 does not silently become a
   missing datum.
2. **`tool_correctness_vs_required_evidence_met` is a bare count** `{"n": 54}`
   (`deepeval_adapter.py:394-396`). The spec asks for a comparison. Emit an agreement rate and a
   statistic alongside `n`. Note the metric sits at ceiling (`1.0` x96, `0.5` x12) because
   `available_tools` is never passed, so the tool-*selection* half reports "No available tools were
   provided" on every row — either pass the three tool names or record that limitation explicitly.
3. **`resolved_evaluator` has no provider and no version.** The spec says "provider/model/version".
   Add `provider` (from the model-id prefix) and the DeepEval library version. `build_report` currently
   passes the dict through unchanged at `:634`.
4. **`vs_judge.n` is overstated.** It reports `len(common_keys)` = 108 (`:461,:465`), but the paired
   non-None counts are 93 (task_completion vs reasoning) and 94 (step_efficiency vs trajectory). Use the
   paired non-None count, as `vs_deterministic` already does correctly at `:392`.
5. **The `spend` block is a mislabelled placeholder.** `{"m7a_ledger_before": 3.713288, "n_traces": 108}`
   — `3.713288` is the **global** total across all milestones (`realized_usd()` sums every row,
   `spend.py:214-225`), and it is captured *after* scoring (`:798`), so the key name is wrong twice, and
   no DeepEval-attributable cost appears anywhere. Replace with `{m7a_realized_usd,
   deepeval_realized_usd, global_realized_usd, captured_at}`, computed by tag/purpose. Persist the
   calibration projection printed at `:777-781` too.
6. **`decisions: []` is empty** (`:654`). Populate it — at minimum the GEval decision from D5 and the
   evaluator-independence caveat below.
7. **Judge dimensions `evidence` and `professional` are loaded but never compared** (`:425` collects all
   four; `compare_to_judge` at `:446-467` uses only `trajectory` and `reasoning`). Either compare all
   four or stop loading the unused two.
8. **Independence caveat — must be recorded.** `resolve_evaluator_model` picks the first verified judge
   from `judge_slate.json`, which is `mistralai/mistral-small-3.2-24b-instruct` — **one third of the M5
   judge trio**. So `task_completion_vs_judge_reasoning` (rho 0.842) is contaminated by shared-model
   bias, and the "independent agent evaluator" shares a model with the panel it is being compared to. All
   three verified judges are in the trio, so there is no alternative to switch to. **Do not change the
   resolution rule** (the spec mandates resolving from the existing judge configuration). Record the
   caveat prominently in the JSON, the `.md`, and `brief_differences`, and factor it into the
   classification.
9. **Re-derive the classification** from the repaired evidence (working step_efficiency, honest
   coverage, corrected `n`s, the independence caveat). `KEEP_OPTIONAL_ANALYSIS` may well remain correct —
   but it must follow from the repaired numbers, not the broken ones. `REMOVE_NO_ADDED_SIGNAL` is a
   legitimate outcome.
10. **No test asserts run quality.** Add `gate_m7` tests: 108 traces scored, all four metrics present per
    trace, and a coverage floor (e.g. every metric has a non-null score on a stated majority of traces)
    so a silently-degraded run fails the gate instead of passing it.

**Metered:** re-run `just deepeval-crosscheck` end to end after these fixes. There is no per-trace cache
and building one is not worth it — a full re-run is ~$0.04-0.06, taking M7a to roughly $0.15 against a
$1.00 soft target. Follow the existing estimate -> 3-trace calibration -> `assert_within_cap` discipline
already in `main()`; it is correct, keep it.

---

### D7 — Braintrust sync: the largest gap
*DoD bullet: "`just braintrust-sync` recreates datasets, experiments (tags/namespaces), replayed
representative traces, scorers, prompts, parameters, the 12-trace review set; idempotent; offline tests
with a fake client cover the mapping; score budget assertion (<= 12 on <= 60, six elsewhere)."*

The on-disk `data/reports/braintrust_sync.json` has exactly the top-level keys of the **pre-`23acff1`**
version. The current code returns five more (`scorers`, `prompts`, `parameters`, `tools`, `review_set`).
**The scorers/prompts/parameters/tools code added in the interrupted attempt has never been executed
against the real SDK.** Re-running the sync live is therefore itself part of the work.

**D7a — Score budget assertion is decorative, and the ceiling is violated in the live product.**
`assert_score_budget()` (`braintrust_sync.py:141`, called at `:569`) validates `plan["score_names"]`, a
*declared* list with no relationship to what `sync()` actually logs. `_log_scores_for_row()` (`:937`)
namespaces **every** numeric/bool key in the stored scores dict. Measured on the live agent experiments:
6 declared, **17 actually logged** on 32-case subsets — `obj/abstain_correct`, `obj/cap_hit`,
`obj/citation_gold_overlap`, `obj/execution_failed`, `obj/gold_seen`, `obj/input_tokens`,
`obj/output_tokens`, `obj/required_evidence_met`, `obj/skill_adherence`, `obj/tool_calls`, `obj/usd`,
`obj/wall_ms`, ... That is over the 12 ceiling, and the spec says plainly: *"Secondary diagnostics go in
metadata."*
**Do:** log only the plan's declared score names as scores; route everything else (`input_tokens`,
`output_tokens`, `usd`, `wall_ms`, `tool_calls`, ...) to metadata. Then assert the budget against what is
**actually logged**, not the declaration. Add a test comparing logged-score counts to the budget.

**D7b — 13 of 29 experiments are placeholder shells.** All 8 RAG plans and all 5 pareto plans log a
single synthetic row `{"case_id": None, ...}` with **zero scores** (`:302,:323,:335,:549`):

```
rag-m3-* (6), rag-m7-li-crosscheck, rag-m7-synthetic   ->  58 or 106 cases declared, 0 scores logged
pareto-* (5)                                            ->  18 or 32 cases declared, 0 scores logged
9 agent experiments                                     ->  6 declared, 17 logged  (see D7a)
6 judge experiments + deepeval-crosscheck               ->  correct (4 and 5)
```
This is why BTQL investigation 5 returned `{"model": "anthropic/claude-haiku-4.5", "n": 1}`.
**Do:** log real per-case rows. RAG experiments get one row per dev case with `li/` and `obj/` scores
from `li_rag_eval.json`; pareto experiments get one row per case from the stored M6 result JSONLs. All
from canonical local rows — **no re-runs, no model calls.**

**D7c — The six replayed traces were silently discarded.** `_emit_span_tree` falls back to
`client.start_span(name=...)` at `:1095`. With the real `braintrust` module and no active
logger/experiment, `braintrust.start_span()` returns `braintrust.logger._NoopSpan` (verified directly in
this venv). `"replayed_traces": 6` counts local dict builds, not logged traces. **Do:** start the span
tree on an experiment or logger object so the hierarchy actually lands, and assert the returned span is
not a no-op.

**D7d — Log hierarchy gaps.** (i) "retrieval stages as child spans" is not implemented: children of the
`search_agreement` node are the N-th search *call*, not the dense/BM25/RRF/rerank stages. (ii) Scoring
spans carry only `obj` provenance (`:601-605`, `:1086`); `judge/` and `deepeval/` provenance is absent,
contrary to the spec. Fix both.

**D7e — `representative_cases.json` is a stale orphan.** Nothing writes it. `REPRESENTATIVE_CASES_PATH`
(`dealpoint/config.py:313`) is referenced only by config and `tests/test_m7_artifacts.py:26,40`; `sync()`
writes only `braintrust_sync.json` via `_record_sync_run()` (`:1188`). The committed file predates the
current code (no `results_path` field), so *"re-creatable from Git/local sources by one command"* fails
for it. **Do:** have `sync()`/`main()` write it.

**D7f — Datasets: metadata incomplete, and inserts are not idempotent.** `_dataset_row_metadata()`
(`:175`) returns only `question_id, agreement_id, reasoning_type, case_type` — the spec also requires
**source hashes**, which only `judged_calibration` (`subset_hash`, `:218`) and `synthetic_query`
(`prompt_hash`, `:239`) carry. The synthetic set is also missing `reasoning_type` and `case_type`
(`:230-243`). And `dataset.insert(...)` is called without `id=` (`:1000`), so the real SDK mints a fresh
row uuid every run and **a second live sync duplicates every row**. **Do:** add source hashes to all five
sets, complete the synthetic set's metadata, and pass a stable deterministic `id=` per row.

**D7g — Idempotency is not implemented, and the test is vacuous.** `_init_experiment` (`:925`) calls
`braintrust.init_experiment(project=..., experiment=name)` without `update=True`; `braintrust.init`
documents `update: If the experiment already exists, continue logging to it` with default `None`, so a
name collision creates a new experiment. `tests/test_braintrust_sync.py:129` only asserts that
`len(client.datasets)` / `len(client.experiments)` do not grow — but the fake client dedupes by name in
its own dict, so the test **can only ever pass**, and `FakeExperiment.log` appends again on the second
call. The milestone record's claim *"ran twice live, second run created nothing new"* is unverifiable and
contradicted by the SDK default. **Do:** pass `update=True`; strengthen the fake-client test to assert
**row and log counts** do not grow on a second sync, not just container counts; then verify live by
running `just braintrust-sync` twice and diffing the result.

**D7h — Scorers and prompts are declarations that are never published.** `SCORER_PLAN` (`:790`) has
exactly the six required names and `_make_scorer_handler()` (`:819`) genuinely imports and calls the
canonical `dealpoint.eval.scorers.*` functions (signatures verified, including the `doc=` kwarg). But
`_sync_scorers` (`:1112`) calls `braintrust.projects.create(name=...).scorers.create(...)`, which in
braintrust 0.37.0 returns a `CodeFunction | CodePrompt` — a **declaration for `braintrust push`**, not an
API call. `sync()` never calls `project.publish()` and there is no push step in the justfile. Same for
prompts (`:1138`). **Do:** either publish (add the push step and a `just` recipe) or, if publishing is
not reachable from this environment, say so plainly in the sync result, the report and the walkthrough —
the spec's rule is *"tested-and-rejected features are documented, not hidden"*. Also: `parameters=`
(`:1123`) is a bare dict of `object` types, not the Pydantic model the SDK documents for a code scorer.
And `_make_scorer_handler` is **never called by any test** — the bridge from
`(input, output, expected, metadata)` back to `(case, finding, record, canonical_text)` is completely
unexercised. Add an offline test that drives a real handler end to end.

**D7i — Parameters schema.** `parameters_schema()` (`:888`) is missing the spec's **skill version** and
**model alias**, and is not "compact": it inlines the entire `ARMS` dict with full retriever configs.
Trim it and add the two missing keys.

**D7j — Review set: not enforced, not blinded, not scoreable.** `review_set()` (`:748`) sorts by
`_seeded_key` and takes the first 12 (`:786`) — the diversity rule holds **by luck** on current data
(4 reasoning types, 5 variants, >=1 abstention), and a data change breaks it silently. Worse: the synced
row is `input=packet_id` with metadata only (`:1032-1044`), so the blinded packet **body** (question,
answer, trajectory) is never pushed and there is nothing for the engineer to read in Braintrust Review —
and exposing `variant_id` in metadata **un-blinds the set**. **Do:** enforce the rule by construction
(select to satisfy it, then fall back to seeded order within each bucket); push the blinded packet body;
move `variant_id` to a blinded key the reviewer cannot see. Keep the n=30 prefix-stability property and
its test (`:233`).

**D7k — `procedure/` is a dead namespace** (declared at `:36`, nothing maps to it). Either map the
procedural scores that belong there or drop it from `SCORE_NAMESPACES`. Also note the ten common metadata
keys are present but frequently `null` on non-agent rows (`case_type`, `reasoning_type`, `index_version`)
because the eval/rag/economics plans used synthetic rows — D7b fixes this at the source.

---

### D8 — BTQL: six shape-valid placeholders that do not answer their questions
*DoD bullet: "`docs/braintrust-queries.md` + `btql_investigations.json` with executed results."*

The queries **were** genuinely executed live against `https://api.braintrust.dev/btql` on
2026-09-06T02:11 with real row uuids — not simulated. The problem is what they ask.
`_require_from_and_shape` (`btql.py:25`) only checks for `from:` plus one of
`select:`/`dimensions:`/`measures:`. **No query in the file references `scores.*` at all.**

| # | Question | Query as built | Verdict |
|---|---|---|---|
| 1 | which cases did a later search surface gold the first missed | `select: id, input, metadata.arm, metadata.reasoning_type \| filter: reasoning_type is not null \| limit 50` | no rank or search-count predicate |
| 2 | EXECUTION_FAILED / CAP_HIT counts by model and arm | `dimensions: model, arm \| measures: count(1)` | never touches either field; returned 1 group |
| 3 | grounded_accuracy by reasoning_type | `dimensions: reasoning_type \| measures: count(1)` | no grounded_accuracy anywhere |
| 4 | traces where `deepeval/` and `obj/` disagree | `from: experiment('judge-A@haiku') \| select: id, metadata \| limit 50` | **wrong experiment** and no comparison |
| 5 | cost per correct answer by model | `dimensions: model \| measures: count(1)` | no cost, no correctness; returned `n: 1` |
| 6 | tool calls vs outcome | `select: id, metadata.arm \| limit 50` | neither field present |

**Do:**
1. Rewrite all six to actually answer their questions, using `scores.*` and the correct metadata now that
   D7a/D7b make those fields real. Each must reference the quantity its question names.
2. Fix `_experiment_names()` (`btql.py:76-79`): it picks the **first** eval-stage plan, which is a judge
   experiment. Query 4 must target `deepeval-crosscheck`.
3. Fix the `curl` rendering (`btql.py:222-227`): the BTQL string contains single quotes
   (`experiment('...')`) embedded inside a single-quoted `-d '...'` payload, so the shell terminates the
   string early and **the printed curl commands do not run**. Build the payload with `json.dumps` and a
   heredoc, or switch the outer quoting. The `bt sql --non-interactive --json "..."` forms are already
   fine.
4. **Order matters:** run `just braintrust-sync` (D7) *before* `just btql`. The recorded results
   currently predate the sync that produced the data they describe (02:11 vs 02:38).
5. Add an offline `gate_m7` test asserting each query references the field its question is about (e.g.
   query 3 must mention `grounded_accuracy`, query 5 must mention a cost field), so a placeholder cannot
   pass the gate again.

---

### D9 — README Pareto block
*DoD bullet: "`docs/demo-walkthrough.md` draft; README updated"; spec section "README results block".*

The RESULTS block is **already done**. The PARETO block is untouched: `_readme_pareto_block`
(`pareto_report.py:667`) just calls `render_markdown(report)` and strips the leading H1, so the README
receives the entire 80-line `pareto.md` body — an 11-column table plus seven H2 sections whose headings
collide with the README's own outline.

**Target shape** (spec: *"per-model table of grounded accuracy 'x% (k of n scored)' and $/case only"*):

- the `budget-scaled` / `objective` / `not comparable` disclosure sentence — **keep**
- "Arm D is held fixed (frozen index, skill, tools, cases); only the model varies." — **keep**
- one table: `model | grounded_accuracy | $/case`, accuracy rendered `x% (k of n scored)` to match the
  RESULTS convention (the block currently renders `(17/32)`)
- the **Frontier** line — keep; it is not in the move-out list
- a link line to `data/reports/pareto.md` and `data/reports/pareto.json`, mirroring the RESULTS block
- **nothing else**

Drop from the README table: `n_cases` (folded into the accuracy denominator), `abstain_recall`,
`execution_failed`, `cap_hit`, median and p90 latency, `total $`, `frontier` column.

Move out (**pure deletion from the generator — `data/reports/pareto.md` already carries all seven,
verbatim, at `:19,:30,:37,:41,:64,:70,:76`; nothing needs adding there**): `## Not run`,
`## Partially run`, `## Comparability note`, `## Spend`, `## Recorded decisions`,
`## Brief-vs-spec differences`, `## Caveats`.

**Placement.** Both generated blocks must sit under `## Latest numbers` (`README.md:38`). Today the
PARETO block is at `README.md:144-223`, at the very bottom under `## Model cost/quality Pareto (M6)`
(`:142`), *after* `## Scope` and `## Layout`. Move it directly below the RESULTS block as an
`### Model cost/quality (M6)` subsection, delete the trailing H2 at `:142`, and update the generator's
fallback heading insertion at `pareto_report.py:691` to match.

**Tests.**
- `tests/test_readme_pareto.py`: delete the latency assertions (`:56-58`) and the `not_run` model
  assertions (`:60-61`); change the denominator format at `:50-52` to `({n} of {n} scored)`; add
  assertions for the two links and for the **absence** of all seven moved H2 headings; add `gate_m7`
  alongside the existing `gate_m6` marker (`:16`).
- **`tests/test_spend_m6.py:149-171` will break** — it greps the README PARETO block for
  ``` `<model>`: not run ```. Repoint it at `data/reports/pareto.md`, where that content now lives.
- `tests/test_readme_results.py` needs no change.

Then regenerate with `just pareto-report`. Note the README block is *already* one row-order stale versus
its generator (the generator now sorts models alphabetically), so regeneration is required regardless.

---

### D10 — Staleness and final regeneration

- `data/reports/framework_versions.json` records `git_sha7: b9eac78` against HEAD `29bf205`, and the
  three reports disagree three ways (`b9eac78` / `ebfc41d` / HEAD). Nothing regenerates the standalone
  file: it is written only by `python -m dealpoint.eval.framework_versions` (`:54-57`), with no justfile
  recipe and no test. **Add a `just framework-versions` recipe** and regenerate. The embedded copies
  refresh automatically when each report is regenerated.
- `docs/demo-walkthrough.md` header claims artifacts "as of `git_sha7 b9eac78`". Refresh it, and update
  **every number this plan changes** — the RAG lab table (D2/D3 change hit@10, MRR, and the
  disagreement count from 0), the DeepEval section (D5/D6 change step_efficiency and possibly the
  classification), the Braintrust experiment/score description (D7), and the BTQL results (D8). Fix the
  unbalanced quote in section 8 (``...glm-5.3-flash-e2b4a2b97561-e3ee9cc" returned 7 rows``). The role
  diagram, the section order and the `[M7b]` markers are correct — leave them.
- Add `just braintrust-sync` to the README's `## Try it` recipe list (`README.md:96-109`); the recipe
  exists and both the README's Framework roles section and the walkthrough already reference it.
- **Do not edit** `docs/milestones/m7a.md` (the documenter's record for the previous attempt) or
  `specs/mvp/state.json` (factory state). Both currently overstate M7a's status — `docs/milestones/m7a.md`
  marks every DoD row "Met" and claims "506 passed / 55 gate_m7" (actuals: 512 / 61). The documenter
  regenerates that file; just don't rely on it, and mention the discrepancy in your handoff.

---

## 2. Order of work

1. **D1** disk guard (free, isolated)
2. **D2** obj-metric k fix, **D3** native LlamaIndex BM25 + real disagreements, **D4** provenance
   -> re-run `just li-rag-eval` (free, no model calls)
3. **D5** + **D6** DeepEval -> re-run `just deepeval-crosscheck` (**the only metered step, ~$0.04-0.06**)
4. **D7** Braintrust sync -> run `just braintrust-sync` **twice**, verify the second run adds nothing
5. **D8** BTQL rewrite -> run `just btql` (must come **after** step 4)
6. **D9** README Pareto block -> `just pareto-report`
7. **D10** `just framework-versions`, refresh the walkthrough, README `## Try it`

---

## 3. Files you will touch

```
dealpoint/eval/disk_guard.py            D1
dealpoint/rag_lab/report.py             D2, D3, D4
dealpoint/rag_lab/evaluate.py           D2, D3
dealpoint/rag_lab/adapters.py           D3   (native BM25Retriever composition)
dealpoint/rag_lab/synthetic_eval.py     D4   (frozen winner, not hardcoded)
dealpoint/eval/deepeval_adapter.py      D5, D6
dealpoint/eval/braintrust_sync.py       D7
dealpoint/eval/btql.py                  D8
dealpoint/eval/pareto_report.py         D9
README.md                               D9, D10   (regenerated blocks + Try it)
docs/demo-walkthrough.md                D10
docs/braintrust-queries.md              D8   (regenerated)
justfile                                D1, D10   (disk-guard, framework-versions recipes)
tests/test_m7_artifacts.py              D1, D6
tests/test_rag_lab_eval.py              D2, D3
tests/test_synthetic_queries.py         D4
tests/test_deepeval_adapter.py          D5, D6
tests/test_braintrust_sync.py           D7
tests/test_btql_queries.py              D8
tests/test_readme_pareto.py             D9
tests/test_spend_m6.py                  D9   (repoint the "not run" grep at pareto.md)
data/reports/*                          regenerated artifacts
```

---

## 4. Budget and disk discipline

- Global cap `MAX_OPENROUTER_SPEND_USD=6.00`; realised $3.713288; headroom **$2.286712**.
- M7a realised $0.099348 against $1.00 soft / $2.50 absolute.
- **Only D5/D6 spends.** Estimated $0.04-0.06 (one full DeepEval re-run, 108 traces x 4 metrics at
  `mistralai/mistral-small-3.2-24b-instruct`). M7a lands near $0.15 — comfortably inside both caps.
- Before that run: use the existing estimate -> <=3-trace calibration -> projected-vs-realised ->
  `assert_within_cap` path already in `deepeval_adapter.main()`. It is correct; do not rewrite it.
- Every metered call keeps `milestone_tag: m7a` and `purpose: deepeval`. No router endpoints; models only
  from the slate, the M5 judge trio, or the configured evaluator.
- D3 (native BM25), D7 (sync) and D8 (BTQL) make **zero model calls**. Keep it that way — trace replay is
  from stored trajectories only.
- Disk: 1.5 GB free against the 500 MB floor. Do not install anything new; do not uninstall anything.

---

## 5. Verification

Every command is judged by **exit status**, never by grepping its output for words like `error`.

```bash
uv run pytest -m "gate_m7 and not needs_network" -q     # the milestone gate
uv run pytest -m "not needs_network and not needs_model" -q
uv run ruff check .
uv run pyright
```

Then confirm each Definition-of-done bullet against the artifact, not against the code:

- [ ] `data/reports/m7a_disk_guard.json` was written by `record_guard()` and carries a real dry-run
      measurement; `framework_versions.json` git_sha7 matches HEAD.
- [ ] `li_rag_eval.json`: `obj` hit@5/hit@10/MRR equal `tournament.json` exactly for all six frozen
      configs; `disagreements` and `worked_examples` are **non-empty**, in both directions, with cause
      labels and case ids; the synthetic block reconciles its three cost figures and reports the
      113 -> 73 drop; `frozen_assertions` still match the files on disk.
- [ ] `deepeval_crosscheck.json`: `step_efficiency` has real, varied scores; a `coverage` block reports
      nulls honestly; `resolved_evaluator` carries provider/model/version; `vs_judge` `n`s are paired
      counts; `spend` names m7a and deepeval totals separately; `decisions` is non-empty and includes the
      evaluator-independence caveat; the classification follows from the repaired numbers.
- [ ] `just braintrust-sync` run twice adds nothing the second time; RAG and pareto experiments carry
      real per-case rows with scores; the six replayed traces actually land (not `_NoopSpan`); logged
      scores per case are <= 12 on <= 60-case subsets and six on the M4/M6 sweeps, asserted against what
      is logged; `representative_cases.json` is written by the command.
- [ ] `docs/braintrust-queries.md` + `btql_investigations.json`: each of the six queries references the
      quantity its question names, was executed after the sync, and the printed `curl` form actually runs.
- [ ] README: both blocks under `## Latest numbers`; the Pareto block is the reduced table plus the
      frontier line and links; the seven moved sections appear only in `data/reports/pareto.md`.
- [ ] `docs/demo-walkthrough.md`: every name, id and number resolves to a local artifact **after** the
      regenerations above.
- [ ] Spend: M7a <= $1.00 target, global <= $6.00, ledger-asserted.

### Two known environment quirks

1. **The offline suite is not working-tree-clean.** Running it rewrites `data/reports/judges.md`
   (updating `git_sha7`) and emits untracked `data/results/pareto_probe_*_offline_<sha>.{jsonl,json}`
   files. This is pre-existing behaviour, not something you introduced. The `judges.md` change is a
   legitimate regeneration and can be committed; the `pareto_probe_*` files are test byproducts and
   should not be. Check `git status` before committing and do not be alarmed by them.
2. **`tests/test_btql_queries.py::test_execute_investigations_live`** is marked `needs_network` and hits
   the live API (read-only, no spend). The task's gate command
   (`-m "gate_m7 and not needs_network"`) excludes it; `just gate-m7` also excludes it. Confirmed: there
   are **zero** tests carrying both `gate_m7` and `needs_model`, so no gate invocation can spend money.

---

## 6. Boundaries

- **Out of scope:** any later milestone, the product UI, the MCP layer, and every M7b item (custom trace
  view, saved views, dashboard, Loop, Review scoring session, Playground, Topics/Patterns/Debugger,
  walkthrough finalisation).
- **Read-only:** `specs/grilled-product-brief.md`, `specs/milestones/m7a.md`, `docs/milestones/m7a.md`,
  `specs/mvp/state.json`.
- **Frozen:** M1-M6 results, the frozen subsets, the skill, arm configs, scorers, the M3 winner. MAUD
  gold-span overlap stays the only benchmark truth; LlamaIndex node ids, DeepEval scores and Braintrust
  scores are cross-checks that never decide anything.
- **No-bloat:** no new agent runtime, no second tracing backend (this is why D5 takes the GEval route
  rather than adopting DeepEval's tracer), no metric weaker than gold labels unless it adds a distinct
  diagnostic, no hardcoded judge-model stack. Features tested and rejected get documented, not hidden.
- Report any brief-vs-spec difference you hit in the relevant report's `brief_differences` list. Do not
  resolve one silently.
