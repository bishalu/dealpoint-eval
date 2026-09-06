# M7a — Framework roles made explicit: LlamaIndex RAG lab, DeepEval cross-check, Braintrust sync (scriptable layer)

## Goal

Make LlamaIndex's, DeepEval's and Braintrust's roles explicit and bounded: LlamaIndex owns retriever
composition/native evaluation and the synthetic-query study; DeepEval provides an independent
agent-eval cross-check on the M5/M6 judged subset; Braintrust holds traces/experiments/comparison,
entirely recreatable from Git/local sources by `just braintrust-sync`. MAUD gold-span overlap remains
benchmark truth throughout — nothing here rewrites, retunes or reruns M1–M6.

## Implementation summary

- **Dependency groups** (`pyproject.toml`): `rag-lab` (`llama-index-core`, `llama-index-retrievers-bm25`)
  and `deepeval` (`deepeval`), both optional, both lazily imported everywhere they are used. Disk
  re-measured with `uv pip install --dry-run` before installing (`data/reports/m7a_disk_guard.json`):
  free space stayed at ~1.25 GB after both groups, well above the 500 MB floor; `.venv` grew from
  340 MB to 451 MB; neither group pulled torch/transformers/nvidia wheels.
- **LlamaIndex RAG lab** (`dealpoint/rag_lab/`): `adapters.py` wraps the frozen M3
  `build_retriever` output in a `ProjectRetrieverAdapter(BaseRetriever)`, scoring nodes by reciprocal
  rank (the project retrievers return ranked lists, never similarity scores). `evaluate.py` runs
  LlamaIndex's `RetrieverEvaluator` on all 58 dev cases against `expected_ids` = gold-bearing chunk
  ids (the *same* `overlap_chars`/`MIN_GOLD_OVERLAP_CHARS` rule `tournament._first_hit_rank` uses),
  and separately calls `tournament.evaluate_config` for the `obj/` side — never a duplicate metric.
  `synthetic.py` generates the frozen synthetic-query set (106 questions, 73 distinct gold-bearing dev
  chunks, `z-ai/glm-5.3-flash`), routed through `OpenRouterClient` (`milestone_tag=m7a,
  purpose=synthetic_query`) via a thin `CustomLLM` shim, following estimate → 3-chunk calibration →
  `assert_within_cap` → generate. `report.py` writes `data/reports/li_rag_eval.json`/`.md`.
- **DeepEval cross-check** (`dealpoint/eval/deepeval_adapter.py`): `resolve_evaluator_model` reads
  `data/reports/judge_slate.json`'s verified judges and picks the first whose family is outside
  `judge_slate.CANDIDATE_FAMILIES` — resolves to `mistralai/mistral-small-3.2-24b-instruct` today,
  never hardcoded. `row_to_test_case` is a pure, framework-free dict-returning function (a thin shim
  converts it to a real `LLMTestCase`), so the mapping is gate-tested with DeepEval absent.
  `expected_tools` is derived purely from `required_evidence` (never from model output). Metrics:
  `task_completion`, `tool_correctness`, `argument_correctness`, `step_efficiency`
  (DeepEval-native `StepEfficiencyMetric`). `DEEPEVAL_TELEMETRY_OPT_OUT=1` set before any import — no
  second trace store, no dual tracing path. Ran on all 108 judged-subset traces (18 cases × 6
  variants), with a 3-trace calibration sample before the bulk run. Classification:
  **`KEEP_OPTIONAL_ANALYSIS`** (task_completion vs grounded_accuracy Spearman ρ≈0.38, n=42 —
  moderate, not strong enough evidence either way).
- **Braintrust sync** (`dealpoint/eval/braintrust_sync.py`): pure mapping functions
  (`score_namespace`, `common_metadata`, `dataset_rows`, `experiment_plan`, `log_hierarchy`,
  `representative_cases`, `review_set`, `assert_score_budget`) plus a thin `sync(client, dry_run=)`
  driver. `common_metadata` puts all ten common keys on every logged **row**, not just the experiment
  — the exact fix for the measured `{"arm": null, "n": 108}` failure mode from a live BTQL probe.
  `just braintrust-sync` recreates 5 datasets and 29 experiments (RAG/agent/eval/economics stages),
  replays 6 representative traces from stored trajectories with zero model calls, and prepares a
  12-trace blinded review set (spans 4 reasoning types, ≥1 abstention, 5 distinct variants). Verified
  idempotent: run twice against the live project, second run creates nothing new.
- **BTQL investigations** (`dealpoint/eval/btql.py`): all six queries built and executed against the
  live API; results in `data/reports/btql_investigations.json` and reproducible `curl`/`bt sql` forms
  in `docs/braintrust-queries.md`.
- **README results block** (engineer's instruction, 2026-09-05): `_readme_results_block` now writes,
  per model, an arm table with `grounded_accuracy` as `"x% (k of n scored)"` plus the majority-baseline
  note, and nothing else — no execution-failure/cap-hit line, no v1-vs-v2 table, no residual-failure
  bullets, no per-arm verdict sentences. Those diagnostics stay in `data/reports/four_arm.md`/`.json`,
  which the README block now links explicitly. `tests/test_readme_results.py` rewritten to assert the
  new shape and the absence of the trimmed sections.
- **Spend plumbing**: `SWEEP_DEFS["m7a"]` added (two `tokens_per_call` legs: synthetic-query
  generation at the workhorse, DeepEval at the resolved evaluator model) so `python -m
  dealpoint.eval.budget m7a` resolves. M7a realised: **$0.0993** (synthetic $0.059, DeepEval $0.040) —
  well under the $1.00 soft target and $2.50 absolute. Global ledger: **$3.7133** of $6.00.
- **`claude mcp list`** (recorded verbatim, per spec — not re-added, no MCP-dependent work attempted):
  `braintrust: https://api.braintrust.dev/mcp (HTTP) - ! Needs authentication`.

## Recorded decisions

1. **M6 envelope test scoping.** `tests/test_spend_m6.py::test_realized_usd_within_envelope`
   originally summed the *whole* ledger; once M7a's own rows exist that would go false for a reason
   unrelated to M6. Repaired to sum only the M1–M6 tags via the existing `realized_by_tag()` helper —
   this keeps M6's own DoD claim exactly as strong as measured, never weaker (the $4.00 number itself
   is untouched). M7a's own ledger discipline is asserted separately in `tests/test_spend_m7.py`
   against the $6.00 global cap.
2. **Braintrust tools decision.** `search_agreement`/`get_section`/`lookup_defined_term` are **not**
   exposed as Braintrust functions: they need the 68 MB Qdrant index, the 64 MB derived canonical
   corpus and fastembed's ONNX weights loaded in-process, none of which run in Braintrust's function
   environment without duplicating this project's entire index/data layer. Documented, no stubs shipped.

## Brief-vs-spec differences (recorded in the reports, not resolved silently)

1. **LlamaIndex scope widened** — brief §3 allows LlamaIndex only behind the `Retriever` interface;
   M7a additionally uses it for framework-native evaluation and synthetic-query generation
   (agents/workflows remain excluded). M3 declined it for retrieval composition; M7a adopts it for a
   different purpose only.
2. **DeepEval was a brief exclusion** — amended for M7a per the engineer's 2026-09-05 authority; the
   `classification` field is the mechanism that tests the amendment rather than assuming it.
3. **Score budget raised** — brief §2.7 caps at 6 scores/case; M7a permits ≤12 on subsets ≤60 cases
   (ceiling, not target); M4/M6 sweeps still at six, enforced by `assert_score_budget`.
4. **Dual-path tracing risk** — DeepEval ships OpenTelemetry/posthog; telemetry opted out, no second
   trace store created, results land only in local JSON + `deepeval/`-namespaced Braintrust scores.
5. **Scale, inherited from M5/M6** — judged subset is 18×6=108 traces, not the brief's 40×6; human
   calibration is n=0 ("pending"), not the brief's 30 hand-scored traces.
6. **M6 envelope test scoping** — see Recorded decisions above.

## Acceptance criteria and result

| DoD item | Status | Evidence |
|---|---|---|
| Optional groups installed within disk guard; framework_versions recorded | **Met** | `data/reports/m7a_disk_guard.json`, `data/reports/framework_versions.json` |
| `li_rag_eval.json`: LI vs obj on all 58 dev cases; ranking agreement; disagreements; synthetic set frozen+evaluated; frozen winner/test set untouched | **Met** | `data/reports/li_rag_eval.json`/`.md`; `frozen_assertions` sha256-matches `test_subset_v1.json`/`tournament.json` on disk |
| `deepeval_crosscheck.json`: resolved evaluator; metrics on judged subset×variants; comparisons; disagreements; classification | **Met** | `data/reports/deepeval_crosscheck.json`/`.md`; classification `KEEP_OPTIONAL_ANALYSIS` |
| `just braintrust-sync` recreates datasets/experiments/traces/scorers/prompts/parameters/review set; idempotent; offline fake-client tests; score budget | **Met** | `data/reports/braintrust_sync.json`; `tests/test_braintrust_sync.py` (9 tests); ran twice live, second run created nothing new |
| `docs/braintrust-queries.md` + `btql_investigations.json` with executed results | **Met** | both files present, six investigations executed live |
| `docs/demo-walkthrough.md` draft; README updated; role diagram present | **Met** | `docs/demo-walkthrough.md`; README "Framework roles" section + "Where to look next" links |
| Spend: M7a ≤ $1.00 target / ≤ $2.50 absolute; global ≤ $6.00 | **Met** | `$0.0993` m7a tag, `$3.7133` global — `tests/test_spend_m7.py` |
| Suite, ruff, pyright green | **Met** | 506 passed (full offline suite incl. 55 gate_m7); ruff 0 errors; pyright 0 errors |

## Tests executed

```bash
uv run pytest -m "gate_m7 and not needs_network" -q   # 55 passed
uv run pytest -m "not needs_network and not needs_model" -q  # 506 passed, 5 deselected
uv run ruff check .     # All checks passed
uv run pyright          # 0 errors, 0 warnings
uv run python -m dealpoint.eval.budget m7a   # resolves, exits 0
just braintrust-sync    # run twice; second run created nothing new
```
