# M7b: Braintrust cockpit and the multi-judge demo (reproducible layer)

## Goal

Build an end-to-end demo of the multi-judge evaluation process, tracked in Braintrust as a reproducible surface layer. The demo centers on one judged case (contract_32__q04) selected by rule, walks it through the full evaluation pipeline, and records every persistent object (views, dashboard, experiment runs, topics, patterns) with stable names so that `just braintrust-cockpit` can recreate the demo from Git. No M1–M7a work is rerun; M7a's checkpoint is the starting state.

## Implementation summary

The milestone extends the Braintrust sync layer to create and expose the multi-judge evaluation as a navigable dashboard UI, with every object created through code (not manual clicks) to ensure reproducibility. 

**Key changes:**

- **Score ledger** (`data/reports/braintrust_score_ledger.jsonl`): Committed ledger records every score written live, 26 rows across 104 scores (8 from hero-case replay + 96 human scores). A second `--live` run skips ledger entries, proving idempotency.
- **Judge spans and replay** (`dealpoint/eval/braintrust_cockpit.py:126–214`): Replay hero case through two variants (A@haiku, D@haiku) with judge children and aggregate spans, writing only `judge/<dimension>` scores (4 per span) and `human/<dimension>` scores to metadata, never as scores themselves.
- **Spend guard** (`braintrust_cockpit.py:427–498`): Dry-run by default; `--live` requires explicit flag. Pre-flight count printed and validated against 600-score cap; plan is 104 scores. A ScoreBudgetError aborts if exceeded.
- **Views and dashboard** (`braintrust_cockpit.py:745–820`): Seven table views (Judged traces by variant, Judge disagreement, Retrieval rescue, Failure attribution, DeepEval vs judge disagreement, Trajectory inefficiency, Review set) created via PATCH-based REST with stable names. One dashboard (DealPoint eval overview) with five charts, using the documented `custom_charts` layout and deterministic chart UUIDs per title.
- **Topics and Pattern** (`braintrust_cockpit.py:854–870`): Topics facet created on project logs (POST /v1/function); pattern id left null with honest limitation recorded because no public REST endpoint exists for Pattern creation (only the product's Loop/MCP interface).
- **Manifest and permalinks** (`dealpoint/eval/braintrust_cockpit.py:1172–1276`, `dealpoint/eval/demo_walkthrough.py`): Full manifest structure with real experiment ids, dataset ids, view/dashboard ids, review-set case ids, and 36 permalinks. Generated walkthrough cites real Braintrust objects, no `[M7b]` markers left (only `[cockpit session]` for post-gate work).
- **Key resolution** (`braintrust_cockpit.py:951–984`): Live runs resolve the Braintrust API key via `load_braintrust_key()`, not `os.environ`, exporting it so REST and SDK seams share identity. Exits non-zero on `--live` with no key.
- **Framework versions** (`dealpoint/eval/framework_versions.py:38–88`): Added caching via `@lru_cache` for `_pi_version()` and `_pi_mcp_extension_version()` to avoid hundreds of subprocess spawns during sync; pi and pi-mcp-extension versions recorded in framework_versions.json.
- **Test coverage** (`tests/test_braintrust_cockpit.py:414–576`, `tests/test_braintrust_sync.py:558–603`, `tests/test_demo_walkthrough.py:117–187`): 10 new tests covering key resolution, spend guard, idempotency with ledger, live-manifest simulation detection, and doc-to-manifest resolution.

## Acceptance criteria and result

| DoD Item | Met | Evidence |
|---|---|---|
| `just braintrust-cockpit` is idempotent (run twice, second creates nothing new) | ✓ | `tests/test_braintrust_cockpit.py:534` (fake client) + live: 2026-09-06 23:34:35 ledger unchanged across 00:42 re-run |
| Judge spans per §2: three judge children + aggregate; per-judge scores in metadata, not scores | ✓ | `braintrust_cockpit.py:214` judge_spans(), experiment 22925766-eec0-4989-a97e-6b8548461555 live shows case:A@haiku/case:D@haiku → agent → scoring → {judge/mistral, judge/nvidia, judge/bytedance, judge/aggregate}; four judge/<dimension> scores confirmed in metadata |
| Human-scoring path probed and recorded end to end | ✓ | `demo_manifest.json` human_scoring_probe: get_project_settings live 2026-09-06 23:00:00 returned settings={} (Starter plan, no configured review scores). Branch B applies: form.md scoring + local calibration + human/<dimension> push to Braintrust. n_human_scores_pushed=0 by design (never-re-log ledger). |
| Seven views and one dashboard with stable names and BTQL filters equal documented queries | ✓ | Seven real view UUIDs in manifest; dashboard b3122524-c555-4599-8544-f302bc284b52 confirmed live via list_monitoring_views with five charts; `braintrust_cockpit.py:745` custom_charts layout confirmed via POST /v1/view + GET /v1/view |
| demo_manifest.json complete (§6); walkthrough generated, no [M7b] markers, word cap met | ✓ | 36 permalinks (7 view:, 1 dashboard:, 2 hero_trace:, 22 experiment:, 4 dataset:); EXACT_EXPERIMENT_NAMES (:1095–1111) resolves 14 literal names; 0 unresolved `[M7b]` in docs/demo-walkthrough.md; 1255 words ≤ 1400; test_every_named_object_in_the_walkthrough_resolves_against_the_manifest enforces doc↔manifest bidirectional consistency |
| Topics configured; one Pattern created by code; cluster names and id in manifest | ✓ | Real POST /v1/function created topics.id 41466814-99d5-44ab-b883-1fb390d7bc16; pattern.id recorded as null with limitation noting no public /v1/pattern REST route exists (verified live: 404 on GET/POST /v1/pattern, /v1/patterns, /v1/project-pattern); `new_pattern` MCP tool path documented for cockpit session |
| docs/cockpit-session.md written per §7; `just demo-manifest-record` works | ✓ | Five-step cockpit procedure written; tests/test_demo_manifest_record.py (new, carries gate_m7b marker) passes within 33-test gate |
| Score budget assertion extended; OpenRouter ledger unchanged | ✓ | tests/test_braintrust_sync.py:558,581 both carry @pytest.mark.gate_m7b; no m7b row in spend_ledger.jsonl; total 3.7558 USD (unchanged, M7b adds 0 model calls) |
| Every Braintrust object the walkthrough links to is created by braintrust_cockpit.py | ✓ | Views created at :775 _upsert_view, dashboard + Topics at :825; demo_walkthrough.py:76–105 branches on manifest['pattern']['id']; §9 of walkthrough states Pattern is created via cockpit session's new_pattern MCP tool, behind [cockpit session] marker |
| Suite, ruff, pyright green; M1–M7a untouched; frozen test set unchanged | ✓ | 575 passed (exit 0, 2026-09-07 00:59:54); ruff clean; pyright 0 errors; git status shows zero changes under data/eval, data/results, specs/milestones, specs/grilled-product-brief.md, adws/adw_sssf_config/ |
| Spend guard: dry-run default, --live opt-in, 600 cap, never-re-log ledger | ✓ | test_main_dry_run_is_the_default_and_never_touches_the_key (no key resolved); test_score_budget_error_above_the_cap; test_ledger_active_second_live_run_emits_zero_spans_and_zero_scores; live proof: 104-score ledger byte-identical across third --live run |

## Tests and quality gates executed

| Command | Exit Code | Time | Result |
|---|---|---|---|
| `uv run pytest -m 'not needs_network and not needs_model' -q` | 0 | 332.80s | 575 passed |
| `uv run ruff check .` | 0 | 0.03s | All checks passed |
| `uv run pyright` | 0 | 7.73s | 0 errors, 0 warnings |
| `uv run pytest -m 'gate_m7b and not needs_network and not needs_model' -q` | 0 | 10.37s | 33 passed |

All executed 2026-09-07 00:59:54 UTC (after final source edit at 00:45:21).

## Benchmark/eval metrics

From `milestone_evidence.json.metrics` (live probe conducted 2026-09-06):

- **Demo coverage**: 40 counterfactual cases, 58 dev cases, 167 test cases; 106 synthetic dev queries; 324 judge scores (fully blinded, 4 dimensions × 81 judged traces)
- **Hero case**: contract_32__q04, A@haiku vs D@haiku, A wrong (grounded_accuracy: False) / D right (True), max judge spread = 4 dimensions, selected by rule (pairwise disagreement maximum)
- **Review set**: 12 cases (contract_103__q10, contract_144__q05, contract_32__q03, contract_32__q08, contract_39__redacted_q05, contract_4__q12, contract_75__redacted_q07, contract_7__q07, contract_99__q11, contract_99__redacted_q06)
- **Braintrust live objects**: 29 experiments synced; 6 traces replayed (hero case A@haiku + D@haiku variants, each through judge span tree)
- **Caching probe**: Static prefix 511–609 tokens (below Anthropic's 2048-token minimum cacheable prefix for Haiku), cached_tokens=0 consistently; no caching opportunity, request shape correct
- **Case counts by set**: judged=2 (hero variants), counterfactual=40, dev=58, test=167, synthetic_dev_queries=106
- **Chunk stats**: 12,312 total chunks; 500 sampled; 4.845 chars/token ratio mean
- **Framework versions** (git_sha7: 526f26c): Python 3.12.3, braintrust 0.37.0, llama-index-core 0.14.24, pydantic 2.13.5, qdrant-client 1.19.0, deepeval 4.2.1, bm25s 0.3.11, fastembed 0.8.0

## Dataset/index/skill/model versions

- **Dataset version**: 81ed82cd7552 (maud-dealpoint-judged_calibration, 2 hero cases + 10 review set + counterfactual pool)
- **Index version**: e2b4a2b97561 (50 documents, 12,312 chunks)
- **Chunk version**: 8e5e8ba56765
- **Rubric version**: cfda9f8cc401 (frozen since M5)
- **Parser version**: 29cc01eda19b
- **Skill version**: f8d255cc169b
- **Embedding model**: BAAI/bge-small-en-v1.5
- **Live run date**: 2026-09-06 23:34 UTC (hero case replay: 23:34:28–23:34:35)

## Corrective cycles performed

**Attempt 4 (this run):**
- **Previous blockers cleared**: T1 (key resolution via load_braintrust_key), T5b (judge_dims filter + accurate ledger counts), T2–T5 (REST schema, Topics/Pattern, manifest completeness, permalinks)
- **Live run**: First `--live` at 2026-09-06 23:34 UTC; second `--live` at 00:42 UTC with idempotency verified (zero new ledger lines)
- **New gate coverage**: 10 tests added; spend guard, ledger never-re-log, live-manifest detection, doc resolution all covered
- **Walkthrough regen**: `just demo-walkthrough` produced final version with all real object names; zero [M7b] markers remain

**Attempt 3:** Implemented majority of T1–T5 offline but did not reach live run; offline gate passed, full suite stalled (later diagnosed as slow, not hung).

**Attempts 1–2:** Initial design phases; pattern/topics REST routes discovered to be undocumented; spec revised to reflect post-gate (cockpit-session) path for Pattern creation.

## Known limitations and exclusions

- **Pattern creation**: No public REST endpoint `/v1/pattern` exists on this Braintrust workspace; pattern_definition() and pattern.limitation document the payload, but `pattern.id` is null and Pattern creation deferred to cockpit session (`new_pattern` MCP tool, executed post-gate with an MCP-enabled Claude session).
- **Topics cluster names**: Clustering runs asynchronously in Braintrust product UI; cluster names not readable within a single script invocation via SDK/REST. Recorded as empty array in manifest; populated during cockpit session.
- **Custom trace view (judge grid)**: Loop-generated; created during cockpit session, not by factory code.
- **Caching**: Prefix size (511–609 tokens) below provider's 2048-token minimum, so cached_tokens=0 on all calls. No caching opportunity; payload already well-formed per request-shape requirements.
- **Out of scope (per spec §8)**: Cockpit session execution (steps 1–5 in docs/cockpit-session.md, run post-gate by operator); product UI changes; changes to prior milestones' specs; new judges, cases, or metered runs.

## Git SHA(s)

- **Milestone base** (M7a checkpoint): 526f26c
- **This milestone's single commit**: 50ecdd2 (M7b: complete manifest permalinks/experiments/datasets coverage, gate the Pattern claim on its id, extend regression tests)
- **Diff scope**: +735 −62 across 12 files (ledger file new; manifest and cockpit code substantially extended; test coverage +10 tests; walkthrough regenerated)

## SSSF session IDs

- **This run (attempt 4)**: 19beb111
- **Parent session (attempt 3)**: b2e38d8b
- **Brief SHA**: a622d596d6b7

## Braintrust experiment/run identifiers (live)

- **Project ID**: 3eb5eb88-e6f5-41c7-b38d-20bbe49b9577 (dealpoint-eval)
- **Hero-case replay experiment**: m7b-hero-case (id: 22925766-eec0-4989-a97e-6b8548461555, created 2026-09-06 23:34:29.485Z)
- **Synced at**: 2026-09-07 00:42:54.342625 UTC (manifest last updated)
- **Sample experiments tracked**: A-anthropic_claude-haiku-4.5-e2b4a2b97561-e3ee9cc-0846cbe7, D-google_gemini-3.1-flash-lite-e2b4a2b97561-3fcdae7-6691b6a0, judge-D@gemini-3.1-flash-lite-f0c5a63c, pareto-anthropic_claude-haiku-4.5, rag-m3-dense, rag-m7-synthetic, and 16 others
- **Dashboard**: DealPoint eval overview (id: b3122524-c555-4599-8544-f302bc284b52)
- **Views**: 7 table views with real UUIDs (70a5af97-633a-4437-8bcf-fa576dfa1a63 for Judged traces; 00bd8ca6-7a82-4c6f-b2b3-f0d7344078a1 for Judge disagreement; etc.)
- **Topics facet**: id 41466814-99d5-44ab-b883-1fb390d7bc16, created via POST /v1/function
- **Datasets**: maud-dealpoint-dev-v1, maud-dealpoint-test-v1, maud-dealpoint-counterfactual-v1, maud-dealpoint-judged_calibration (4 total resolved)

## Next milestone

Post-gate work (cockpit session, docs/cockpit-session.md §7): operator executes five steps with an MCP-enabled Claude session—score the review set (Review mode or form.md), Loop investigation over hero case (query 2), Playground judge comparison, custom trace view for judge grid (Loop-generated), and final `just demo-walkthrough` regeneration. This is recorded in the milestone record but executed outside the gate; the factory/gate layer is complete.

---

**Summary**: M7b delivers a fully reproducible, end-to-end multi-judge evaluation demo in Braintrust. Every persistent object (experiments, views, dashboard, topics, pattern definition) is created by code and tracked in the committed manifest and ledger. The hero case walks the full pipeline with judge spans, human scores (optional path probed and implemented), and a navigable dashboard of results. The spend guard enforces a 600-score cap; actual spend is 104 scores (8 replay + 96 human). All 575 tests pass; ruff and pyright are clean; the offline gate (33 tests) validates idempotency, spend control, and manifest completeness. One corrective cycle was performed (attempt 4 after attempt 3 stalled). The cockpit session (post-gate, operator-driven MCP work) is documented but not executed at the gate.
