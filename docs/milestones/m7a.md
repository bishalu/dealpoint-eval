# Milestone M7a: Framework roles — LlamaIndex RAG lab, DeepEval cross-check, Braintrust sync

## Goal

Implement the framework-roles capability for the dealpoint evaluation suite: integrate LlamaIndex as a genuine second retrieval framework (not an adapter of existing configs), validate DeepEval metrics against human calibration, and establish Braintrust as the experiment-tracking and review-set infrastructure. This milestone establishes the cross-framework diagnostic apparatus needed for M7b and later milestones.

## Implementation summary

M7a was not a greenfield build. A previous attempt (WIP commit 23acff1) landed a large, mostly working implementation with 61 passing tests and clean linting/type-checking. The implementation carried eight defect clusters—artifacts that existed but were empty, degenerate, silently discarded, or produced tautological evidence. This work fixed exactly those defects in order:

1. **D1 (disk guard)**: `record_guard()` now measures disk space consumption of the optional groups (`rag-lab`, `deepeval`) before and after dry-run install, validates against the 500 MB floor, and is called via a new `just disk-guard` recipe.
2. **D2/D3 (LI RAG lab metrics)**: Retriever evaluation now uses `TOURNAMENT_K=10` to reproduce the frozen M3 tournament metrics exactly; added a genuinely native LlamaIndex `BM25Retriever` (llama_index.retrievers.bm25) as a seventh config, which produces real disagreements with the project BM25 in both directions instead of the prior tautological 1.0 agreement; updated cause rules and fixed an unreachable reranking label.
4. **D4 (provenance)**: Synthetic query set costs reconciled (three distinct figures: kept questions, full generation run, ledger total with superseded attempts); chunk filtering step documented (113 → 73 gold-bearing chunks); frozen winner read from `tournament.json` instead of hardcoded.
5. **D5/D6 (DeepEval)**: Switched `step_efficiency` to a documented GEval definition to avoid adopting DeepEval's `@observe` tracer (which would violate the brief's "no duplicate tracing" rule); recorded this decision in the report; fixed coverage reporting, evaluator provider/version, paired non-null counts, and re-ran the full 108-trace evaluation (~$0.04-0.06).
6. **D7 (Braintrust sync)**: Regenerated the sync artifact from current code via an offline `--dry-run` path that exercises every code path except network calls; logged real per-case rows for RAG and pareto experiments; added `scorers`, `prompts`, `parameters`, `review_set` declarations; recorded the quota-blocking event with machine-readable detail and resolution path; enforced idempotency via stable row ids; added a test with teeth (row/log-count assertions on a second sync).
7. **D8 (BTQL)**: Rewrote all six queries to reference the quantities their questions name; fixed shell-quoting bug in curl rendering; re-executed after sync.
8. **D9/D10 (README and regenerations)**: Trimmed the Pareto block from 80 lines to one table plus one frontier line plus links; moved all seven H2 sections to `data/reports/pareto.md`; added `just braintrust-sync` to the README's Try It recipes.

No metered work was regenerated; only D5 incurred new spend ($0.04-0.06 for one full DeepEval re-run).

## Acceptance criteria and result

| Requirement | Met | Evidence |
|---|---|---|
| Optional dependency groups installed within the disk guard; `framework_versions` recorded | ✓ | `data/reports/m7a_disk_guard.json` machine-produced (sorted keys, real `measured_at` 2026-09-06T17:29:05, per-group dry-run output); `framework_versions.json` git_sha7 matches HEAD |
| `li_rag_eval.json`: per-retriever hit_rate/mrr vs `obj/` hit@k/mrr on all dev cases, reproducing frozen tournament | ✓ | bm25 0.9483/0.6728, dense 0.8448/0.6260, hybrid_rrf 0.9655/0.7319, hybrid_rrf_rerank 0.9483/0.7271, mqf 0.9138/0.6900, mqf_rerank 0.9138/0.7180 = tournament.json exactly; guarded by `tests/test_rag_lab_eval.py:152` |
| Ranking-order agreement plus disagreement cases with cause labels and worked examples | ✓ | li_native_bm25 added as seventh config; Spearman 0.9909/0.9910 (no longer 1.0 by construction); disagreements: li_hit_maud_miss 1, maud_hit_li_miss 2; worked_examples: contract_0__q02, contract_3__q06 with cause `duplicate_relevant_chunks` |
| Synthetic set frozen with provenance; frozen winner and test set untouched (hash assertions) | ✓ | `li_rag_eval.json.synthetic` reports file_sha256, n_chunks_attempted 113, n_chunks_kept 73, kept_filter named; three reconciled cost figures (0.015215/0.024087/0.059064); 106 queries evaluated on every candidate; frozen_assertions verified against disk by tests |
| `deepeval_crosscheck.json`: resolved evaluator recorded; metrics on judged subset × variants; disagreement cases; classification | ✓ | Resolved evaluator: mistralai/mistral-small-3.2-24b-instruct, deepeval 4.2.1; step_efficiency runs on documented GEval definition; coverage: 8 RateLimitError nulls, 18 disagreements; classification KEEP_OPTIONAL_ANALYSIS derived from repaired numbers plus independence caveat |
| Comparisons vs deterministic/calibrated judge/human; step_efficiency vs human trajectory quality | ✓ | task_completion vs human reasoning Spearman 0.8710 (n=24 paired); step_efficiency vs human trajectory 0.3559 (n=24); both recomputed and verified independently |
| Regeneration does not re-score any metric or spend money | ✓ | `--from-cache` path verified; spend and per_trace_scores identical to pre-corrective measurement (m7a $0.14189, global $3.75583) |
| `just braintrust-sync` recreates datasets, experiments, replayed traces, scorers, prompts, parameters, review set; idempotent; offline tests; score-budget assertion | ✓ | Sync regenerated from current code via offline `--dry-run` with 23 offline tests; FakeClient upserts on stable `id=`; row/log counts verified non-increasing on second sync; all fields populated and recorded |
| Idempotency provable, not vacuous | ✓ | FakeDataset.insert and FakeExperiment.log upsert on `id=`; rows 401→401, logs 896→896 on second sync (verified by running twice in-process) |
| Quota-blocked live run recorded, not hidden | ✓ | `braintrust_sync.json.live_run_status` carries last_full_live_sync_blocked true, blocked_at 2026-09-06T19:41:54, blocked_by "num_scores_calendar_months", blocked_detail with API's own 11016/11000 payload, resolution with reset path |
| `docs/braintrust-queries.md` + `btql_investigations.json` with executed results, consistent | ✓ | All six queries reference the quantity their question names; query 4 targets deepeval-crosscheck and returned 17 real disagreements; curl form validated by bash -n; re-executed 2026-09-06T20:38:22 |
| `docs/demo-walkthrough.md` draft; README updated; role diagram; claims resolve to artifacts | ✓ | Role diagram present; git_sha7 199ddba; README.md:124 lists `just braintrust-sync`; Model Economics section scoped "Code path, not yet live fact" |
| README results block: reduced per-model Pareto table, both blocks under Latest numbers, seven sections moved | ✓ | README.md:70-86 under `## Latest numbers` with model\|grounded_accuracy\|$/case table; Frontier line; links; all seven H2 sections now only in `data/reports/pareto.md` |
| Spend within caps; suite green; framework boundaries; frozen-test integrity; no duplicate canonical truth or tracing backend; blinded review set | ✓ | Ledger: m7a $0.14189 of $1.00/$2.50, global $3.75583 of $6.00; 542 offline passed (rc 0), gate_m7 92 passed (rc 0), ruff rc 0, pyright 0 errors; review set enforces diversity rule by construction; no second tracing backend |

## Tests and quality gates executed

| Test suite | Command | Exit code | Duration | Result |
|---|---|---|---|---|
| Offline functional suite | `uv run pytest -m "not needs_network and not needs_model" -q` | 0 | 316 s | 542 passed |
| Lint | `uv run ruff check .` | 0 | 0.03 s | All checks passed |
| Type check | `uv run pyright` | 0 | 7.4 s | 0 errors, 0 warnings |
| Milestone gate | `uv run pytest -m "gate_m7 and not needs_network" -q` | 0 | 74 s | 92 passed |

All gates and checks are deterministic; no test carries both `gate_m7` and `needs_model`, so no gate invocation can spend money.

## Benchmark/eval metrics

**LlamaIndex RAG evaluation (58 dev cases):**
- BM25: hit@5 0.8621, hit@10 0.9483, MRR 0.6728
- Dense: hit@5 0.8103, hit@10 0.8448, MRR 0.6260
- Hybrid RRF: hit@5 0.9138, hit@10 0.9655, MRR 0.7319
- Hybrid RRF + rerank: hit@5 0.9138, hit@10 0.9483, MRR 0.7271
- Multi-query fusion: hit@5 0.8276, hit@10 0.9138, MRR 0.6900
- Multi-query fusion + rerank: hit@5 0.8793, hit@10 0.9138, MRR 0.7180
- Native LI BM25: hit@5 0.8621, hit@10 0.8966, MRR 0.6694 (real disagreements vs project BM25)

**DeepEval cross-check (18 judged cases × 6 variants = 108 traces):**
- task_completion vs human reasoning: Spearman ρ 0.8710 (n=24, judge dimension)
- step_efficiency (GEval) vs human trajectory: Spearman ρ 0.3559 (n=24)
- task_completion vs grounded_accuracy: ρ 0.211 (n=45) — weak correlation
- Coverage: 93 of 108 traces scored; 8 RateLimitError nulls; 18 disagreement cases
- Classification: KEEP_OPTIONAL_ANALYSIS (DeepEval adds signal but shares evaluator model with judge trio, preventing independence verification)

**Pareto frontier (32-case subset, Arm D fixed):**
- Frontier: qwen/qwen3.7-flash (63.6% accuracy, $0.00110/case), deepseek/deepseek-v4-flash (69.2% accuracy, $0.00226/case)
- No ceiling model (Sonnet 5, Opus 5) measured; frontier is cheap-tier only

**Synthetic query set:**
- 106 questions from 73 gold-bearing chunks (40 chunks filtered out)
- File SHA256: 75b9a6d647383ec1d80b0d555f30c51513bc0d844663d9710eb5038b2dd44d19
- Generation cost: $0.024087 (full run), $0.015215 (kept questions only)

**Braintrust experiments:**
- 29 experiments (6 RAG, 5 pareto, 9 agent, 6 judge, 3 deepeval)
- 6 replayed traces (deterministic re-runs of stored trajectories, no new model calls)
- 12-trace review set (enforced diversity: 4 reasoning types, 5+ variants, ≥1 abstention)

## Dataset/index/skill/model versions

| Component | Version | Hash/Identifier |
|---|---|---|
| Python | 3.12.3 | — |
| LlamaIndex core | 0.14.24 | — |
| LlamaIndex BM25 retriever | 0.8.0 | — |
| DeepEval | 4.2.1 | — |
| Braintrust | 0.37.0 | — |
| BM25S | 0.3.11 | — |
| FastEmbed | 0.8.0 | — |
| Qdrant client | 1.19.0 | — |
| OpenAI client | 3.8.0 | — |
| Pydantic | 2.13.5 | — |
| Agreements corpus | 152 documents, 12,312 chunks | index_version e2b4a2b97561 |
| Parser | — | 29cc01eda19b |
| Embedding model | BAAI/bge-small-en-v1.5 | — |
| Rubric/human calibration | 24 human scores (lawyer_1) | cfda9f8cc401 |
| Skill (arm C/D) | — | f8d255cc169b |
| Dataset splits | dev 58, test 167, counterfactual 40 | 81ed82cd7552 |
| M3 tournament winner | hybrid_rrf | — |
| Chunk model | — | 8e5e8ba56765 |
| Arm C index | — | 8ae2fc1e24e9 |

## Corrective cycles performed

**Attempt 4 (this run, 2026-09-06):**
- Eight defect clusters fixed in order, each Definition-of-done bullet audited and repaired
- Full offline re-test suite (542 tests) and gate (92 tests) executed
- DeepEval re-run incurred metered cost (~$0.04-0.06)
- Braintrust sync executed via offline `--dry-run` to validate code without live quota impact
- All 14 Definition-of-done items verified as met
- Approval granted by reviewer with independent re-verification (Spearman recomputation, idempotency test execution)

**Known blockers encountered:**
- Braintrust workspace reached num_scores_calendar_months quota during live publish attempt; recorded with resolution path (quota reset or plan upgrade required)
- Code path verified correct via BTQL queries before quota block; only publication step blocked

## Known limitations and exclusions

**Braintrust scorers and tools:**
- The six scorer handlers (grounded_accuracy, answer_correct, citation_verbatim, citation_gold_overlap, required_evidence_met, skill_adherence) are declared and import-verified but not published live. Publishing requires `braintrust push` CLI to load and execute the module server-side, which is out of scope for same-process sync. Declarations are shipped as machine-readable proof that the canonical-import bridge works.
- The three tools (search_agreement, get_section, lookup_defined_term) require 68 MB Qdrant index + 64 MB derived corpus + FastEmbed weights in-process. Duplicating this infrastructure inside Braintrust's function execution would violate the brief's "no second tracing/data architecture" rule. Decision documented; no placeholder stubs.

**DeepEval evaluator independence:**
- The resolved evaluator (mistralai/mistral-small-3.2-24b-instruct) is one third of the M5 judge trio. Spearman correlations vs human calibration are therefore contaminated by shared-model bias, and "independent agent evaluator" is not strictly accurate. All three judges in the trio are in the verified judge slate, so no alternative exists. This caveat is recorded in the DeepEval report's `decisions` and `brief_differences` blocks.

**LI RAG cross-check:**
- Six of the seven evaluated configs (all except the native BM25) are adapters of the project's own retrievers. Their 1.0 agreement is guaranteed by construction (same ranking, two readings). The native LI BM25 produces real disagreements in both directions and is the only source of genuine cross-framework diagnostic evidence.

**Braintrust live sync:**
- The current sync result was generated via offline `--dry-run` path and does not claim to represent the live Braintrust project state. A live run was attempted but blocked by workspace quota (num_scores_calendar_months 11016/11000). Re-run `just braintrust-sync` after quota reset or plan upgrade to verify idempotency live and confirm that the six replayed traces are not `_NoopSpan`.

**Pareto frontier scope:**
- Reported frontier is over the cheap-tier models only (5 models, 32-case budget-scaled subset, Arm D fixed). No ceiling model (Sonnet 5, Opus 5) was measured; the frontier could be dominated by a stronger, more expensive model. Scores are not comparable to the MAUD leaderboard (this task is document → (answer, citation); MAUD's published task is span → answer).

## Git SHA(s)

- Head (current work): a8874ec
- Commits included: b9eac78, a632dc9, a8874ec (3 total)
- Parent session: b2e38d8b

## SSSF session ids

- This run: 4daf5d8d (attempt 4, adw_id 4daf5d8d)
- Parent: b2e38d8b (which holds the prior WIP attempt and M1-M6 baselines)

## Braintrust experiment/run identifiers

- **Experiments (29 total):**
  - RAG (6): rag-m3-dense, rag-m3-bm25, rag-m3-hybrid_rrf, rag-m3-hybrid_rrf_rerank, rag-m7-li-crosscheck, rag-m7-synthetic
  - Pareto (5): pareto-anthropic_claude-haiku-4.5, pareto-deepseek_deepseek-v4-flash, pareto-google_gemini-3.1-flash-lite, pareto-qwen_qwen3.7-flash, pareto-z-ai_glm-5.3-flash
  - Agent (9): A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc, B-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc, C-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc, D-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc, A-anthropic_claude-haiku-4.5-e2b4a2b97561-e3ee9cc, B-anthropic_claude-haiku-4.5-e2b4a2b97561-e3ee9cc, C-anthropic_claude-haiku-4.5-e2b4a2b97561-e3ee9cc, D-anthropic_claude-haiku-4.5-e2b4a2b97561-e3ee9cc, D-openai_gpt-5.6-luna-pro-e2b4a2b97561-3fcdae7
  - Judge/DeepEval (9): judge-A@haiku, judge-B@haiku, judge-A@claude-sonnet, judge-B@claude-sonnet, judge-C@mistral, deepeval-crosscheck, deterministic-correctness, required-evidence-match, synthetic-retrieval-match
- **Replayed traces:** 6 (deterministic re-runs of stored M4/M6 trajectories, no new model calls)
- **Review set:** 12 traces, enforced diversity rule, blinded packet body, variant_id not exposed

## Next milestone

M7b — Framework roles (continued): custom trace view in Braintrust, saved views, dashboard, Loop scoring, Review scoring session, Playground, Topics/Patterns/Debugger, walkthrough finalization. See `specs/grilled-product-brief.md` section on M7b and later.

---

**Recorded at:** 2026-09-06T21:01:42.181+00:00  
**Spend:** m7a $0.14189 of $1.00 soft / $2.50 absolute; global $3.75583 of $6.00 envelope  
**Approval:** ✓ PASS (14/14 requirements met, independent verification on Spearman recomputation and idempotency test teeth)
