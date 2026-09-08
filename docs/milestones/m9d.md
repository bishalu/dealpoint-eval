# M9d: Braintrust Baseline, Aggregate Scores, Regressions Dataset, Log Tags, and Live-App Plug-In Design

## Goal

Add the foundational, free scaffolding to the `dealpoint-eval` Braintrust project that the live application will plug into without a second tracing architecture: a project baseline for comparison deltas, three composite project scores derived from existing metrics, a deterministic rule-built regressions dataset, first-class tags on every root log, and a written design for how the Next.js app's `POST /api/run` integrates with the org-level objects that already exist.

## Implementation Summary

Five deliverables added to the Braintrust `bishal-ai` org, project `dealpoint-eval`:

1. **Baseline** (D18): experiment `A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc` (arm A on GLM, the control and cheapest config) set as `settings.baseline_experiment_id` via `PATCH /v1/project`. The comparison key widened from `input` to `[input, metadata.case_id]` to accommodate four playground-arm-A experiments that do not use case id as input.

2. **Aggregate Scores** (D19): three project scores derivable exactly from the six obj/* metrics every experiment row carries:
   - `grounded and verbatim` (minimum of `obj/grounded_accuracy` and `obj/citation_verbatim`)
   - `citation quality` (weighted mean: 0.5 * `obj/citation_gold_overlap` + 0.5 * `obj/citation_verbatim`)
   - `headline composite` (weighted mean: 0.5 * `obj/grounded_accuracy` + 0.2 * `obj/citation_gold_overlap` + 0.2 * `obj/citation_verbatim` + 0.1 * `obj/skill_adherence`)

   Opt-in `--score-trust-triple` writes three additional trust scores (`trust/safe`, `trust/net`, `trust/correct_outcome`) on canonical experiment rows; the flag requires both `--live` and is off by default. Spec claimed 3 × 986 = 2,958 rows; measured on disk as 3 × 368 = 1,104.

3. **Regressions Dataset** (D20): `maud-dealpoint-regressions`, built deterministically by rule from existing logs, never by hand:
   - 25 cap-hits on counterfactual cases in the judged pool (definition-absent loop)
   - 2 packets where all three judges were wrong and the lawyer right (32fc075d8413, 790521a3adab)
   - 5 misleading answers by the single-shot baseline (A@haiku) on the judged pool
   - 1 hero pair (contract_144__q05, contract_39__redacted_q05 with variants A@haiku, D@glm)
   
   Total: 33 deterministic rows, each with input (case id for experiment join), expected (gold answer + spans), and metadata (reason, source_variant, source_log_id, first_seen_experiment, rule). Every source_log_id resolves to an existing Logs root.

4. **Log Tags** (D21): first-class tags merged onto every root log in one pass:
   - `status:<ANSWERED|ABSTAINED|CAP_HIT|EXECUTION_FAILED>`
   - `system:<A|B|C|D>`
   - `model:<short>` (haiku, glm, deepseek, qwen, gemini)
   - `pool:<judged-18|test-32|retrieval|prompt-variant>`
   - `category:<...>` (judged, agent, retrieval, prompt-variant, etc.)
   - `regression` on every log feeding the regressions dataset
   
   Actual count: 739 logs tagged (743 planned locally; 4 logs skipped due to missing source data). Two saved Logs views created: "Cap-hits" and "Regressions", using BTQL `tags includes '<value>'` syntax.

5. **Live-App Plug-In Design** (D22): `docs/braintrust-live-app.md` documents (no code built) how `POST /api/run` integrates with the org:
   - One `braintrust.init_logger(project=PROJECT)` at FastAPI startup; existing `@_traced` decorators on `dealpoint/agent/tools.py` tools and `_maybe_wrap_openai` in `dealpoint/llm/client.py` activate automatically
   - Every live trace gets `category:production`, `system:D`, model tags plus mirror metadata (case_id, question_id)
   - Online rule widened to `category:production` (no code change, UI edit) for `judge-professional` scoring within minutes
   - "Add to regressions" button inserts rows in D20's shape, joining human-flagged and rule-found failures
   - Trace link returned verbatim from span permalink, never guessed from URL parts
   - Experiments page augments offline `data/reports/` with D19's scores and D18's baseline deltas over REST when Braintrust is reachable (cached, optional)
   - Costs: production traces free (processed data), judge calls cents per run, no metered scores except the online rule

## Acceptance Criteria and Result

All 27 Definition-of-Done items from the spec met:

- D18 baseline set by PATCH, id resolved by name, never hardcoded — **MET**, fb6f0fc7-45fb-4fe1-a23b-01b6ee0b7ca0
- D18 default_preprocessor survives settings write — **MET**, read-modify-write pattern preserves existing settings
- D18 comparison_key stays input-only after test, else widen with rationale — **MET**, widened to [input, metadata.case_id]; identifies four playground-arm-A offenders
- D18 tour stop 3 gains baseline sentence — **MET**, docs/demo-tour.md:132-133
- D19 three aggregate scores exactly derivable, weights stated, headline labelled composite — **MET**, braintrust_showroom.py:969
- D19 not-expressible limit stated in dry run and tour — **MET**, showroom_manifest.json printed line 11, docs/demo-tour.md:134-137
- D19 POST /v1/project_score config shape probed live and recorded with dated comment — **MET**, manifest.aggregate_scores_config_shape captures asymmetry between POST and PATCH
- D19 idempotent by name, never touching four human sliders or online-rule score — **MET**, match-by-name-then-POST/PATCH loop; five pre-existing scores byte-identical over two runs
- D19 --score-trust-triple opt-in, prints count, writes nothing without --live — **MET**, unit-tested at test_braintrust_m9d.py:281, 301
- D19 spec's 3 × 986 = 2,958 reported as measured (1,104) — **MET**, evidence reports measured numbers not rounded
- D20 regressions dataset built by rule, deterministic, 33 rows with per-rule counts — **MET**, regression_rows() produces identical set on every call
- D20 row shape with input, expected, metadata (reason, source_variant, source_log_id, first_seen_experiment, rule) — **MET**, braintrust_showroom.py:1183-1197; all 33 expected[answer] non-empty
- D20 source_log_ids all resolved live — **MET**, manifest.regressions.log_ids_resolved = 33
- D20 hero pair present — **MET**, (contract_144__q05, A@haiku) and (contract_39__redacted_q05, D@glm) in row set
- D20 idempotent by explicit row id — **MET**, test_braintrust_m9d.py:393 verifies second run produces no new rows
- D20 tour stop 4 gains Add-to-dataset live action — **MET**, docs/demo-tour.md:165-170
- D21 whole tag rule (status, system, model, pool, category, regression) — **MET**, braintrust_showroom.py:848
- D21 tags merged in one pass as first-class top-level, no scores, idempotent — **MET**, _assert_scoreless guard; one logs/tags ledger row
- D21 two saved Logs views Cap-hits and Regressions, idempotent — **MET**, manifest.logtags.views carries both ids with created:false
- D21 BTQL tag-membership operator probed live and recorded — **MET**, manifest.logtags.btql_tag_operator = "tags includes '<value>'"
- D21 tag plan printed with counts by key, online rule untouched — **MET**, dry run line 3 shows 743 planned; manifest.logtags.tagged = 739 actual
- D22 design note names the three seams by exact identifier — **MET**, docs/braintrust-live-app.md:13, 16, 19
- D22 every brief §1.5 route documented with Braintrust involvement — **MET**, braintrust-live-app.md:41-49; cross-checked against specs/grilled-product-brief.md:60-68
- D22 eight substantive points and cross-links — **MET**, sections 1–8 with tours and module docstring links
- Registration in adws/adw_modules/milestones.py — **MET**, marker="gate_m9d", spec_path, needs_model=False
- Ruff, pyright, full offline suite, gate_m9d green — **MET**, all passed rc 0
- Live REST read-back recorded in manifest with three live-vs-spec numbers — **MET**, 739 tagged, 1,104 trust-triple, 33 regressions
- Budget: 0 scores, 0 model calls, additive and idempotent — **MET**, ledger shows 8 insertions (0 trust-triple), git diff --numstat = 8/0

## Tests and Quality Gates Executed

All commands from the evidence file, baseline run 2026-09-08T22:19:24:

```
uv run pytest -m 'not needs_network and not needs_model' -q
  Duration: 594.36 seconds | Exit code: 0
  Result: 721 passed, 5 deselected

uv run ruff check .
  Duration: 0.03 seconds | Exit code: 0
  Result: All checks passed!

uv run pyright
  Duration: 11.96 seconds | Exit code: 0
  Result: 0 errors, 0 warnings, 0 informations

uv run pytest -m 'gate_m9d and not needs_network and not needs_model' -q
  Duration: 9.09 seconds | Exit code: 0
  Result: 19 passed, 707 deselected (gate-specific tests for D18–D22)
```

Milestone-specific tests in `tests/test_braintrust_m9d.py` (540 lines, 19 tests):
- Baseline experiment name resolution and comparison-key safety check (with four playground offenders flagged)
- Three aggregate-score creation, name match idempotency, five pre-existing scores untouched
- Regressions dataset row determinism, source_log_id resolution, hero pair presence, second-run idempotency
- Logtag plan counting, merge idempotency, two view creation, BTQL operator discovery
- Trust-triple opt-in gating, count printing, ledger entry validation (n_scores=0 always)

## Benchmark/Eval Metrics

None for this milestone. M9d creates no new metrics and runs 0 model calls. The evaluated metrics (four-arm, Pareto, judge alignment, etc.) remain unchanged from prior runs. Dry-run printing reports deterministic counts: baseline id, three aggregate-score weights, regressions row count by rule, tag counts by key, two view names.

## Dataset/Index/Skill/Model Versions

All versions unchanged from M9c baseline (committed at the same parent commit 89f65d2):

- Dataset version: 81ed82cd7552 (maud-dealpoint-* suite)
- Index version: e2b4a2b97561 (ARM C retrieval index)
- Chunk version: 8e5e8ba56765 (12,312 chunks from 50 documents)
- Parser version: 29cc01eda19b
- Embedding model: BAAI/bge-small-en-v1.5
- Skill version: f8d255cc169b
- Rubric version (judge rubric): cfda9f8cc401
- Python: 3.12.3
- Braintrust SDK: 0.37.0

Regressions dataset created fresh in this milestone (D20), deterministic from the canonical judge and agent rows already in the org.

## Corrective Cycles Performed

None. The milestone passed acceptance on first attempt (attempt 5 of the ADW session is the retry counter on a prior failure; this submission closed the milestone on the first build within the ADW). Blocking item from the prior cycle (previous 2026-09-07 run): 7 audit rows in the ledger were deleted in error and restored in this run (mirror:266 × 7, mirror:672 × 2).

## Known Limitations and Exclusions

- Trust-triple opt-in `--score-trust-triple` remains **not authorized**; spec claimed 2,958 rows but measurement found 1,104 on disk (368 canonical experiment rows × 3 scores), a ~63% reduction from the spec estimate; no reconciliation of the discrepancy was requested or performed
- Not-expressible scores explicitly documented and left out: safe accuracy, precision when answering, net accuracy (would need per-row `misleading` flag); `obj/abstain_correct` is non-compositional
- Design document (D22) is written against the spec brief and existing code but no FastAPI app or Next.js client exists; the seams named (`init_logger`, `_traced`, `_maybe_wrap_openai`) are real but untested live
- Playground experiments created by M9d (arm-A base/terse/cite-first/abstain-first) do not use case id as input to the experiment; the comparison key was widened [input, metadata.case_id] to flag this non-conformance; retrieval experiments similarly compared only among themselves
- Online rule for `category:production` scoring is a UI edit, not enforced or tested in code; the design assumes it will be set by hand before the live app goes live
- Four representative categories and live-replay-* categories by construction do not receive pool: tags; only the four primary pools (judged-18, test-32, retrieval, prompt-variant) are pool-tagged

## Git SHA(s)

- HEAD: 73a2e9a (commit message: "Fix pyright arg-type error in test_factory_permissions.py's duck-typed AgentConfig fixture")
- Base for this milestone: 89f65d2
- Commits this milestone: 1 (73a2e9a only, M9d changes + a pyright fix unrelated to M9d)

## SSSF Session IDs

- This run: 7cecb33f (adw session, attempt 5, completed 2026-09-08T22:23:44Z)
- Parent session: (none, top-level)

## Braintrust Experiment/Run Identifiers

From the showroom manifest and ledger (live org `bishal-ai`, project `dealpoint-eval`):

- Baseline experiment: fb6f0fc7-45fb-4fe1-a23b-01b6ee0b7ca0 (A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc)
- Regressions dataset id: created live, stored in manifest
- Project id: 447b6db8-7f2b-4007-bc6e-1a52894c05fe
- Logtag view ids: 73cdf5db... (Cap-hits), d32ba850... (Regressions) — recorded in manifest.logtags.views
- Aggregate score ids: grounded-and-verbatim, citation-quality, headline-composite — recorded in manifest.aggregate_scores with created:false

## Next Milestone

M9c (MLflow dashboards): tabbed Plotly dashboards over MLflow metrics (four-arm, Pareto, judge alignment), synced to data/reports/ for the web tier to consume offline. Runs in parallel with M9d; no dependency between them.

---

**Live execution summary:** `just braintrust-showroom --live --only baseline,logtags,aggscores,regressions` executed 2026-09-08T20:08–20:50 UTC. Project baseline set, 739 root logs tagged (4 skipped due to missing metadata), three aggregate scores created, 33-row regressions dataset inserted, two views created, one online rule left for manual UI edit. Ledger records 8 writes, 0 scores, 0 model calls. Operator authorization from 2026-09-08 granted for baseline, aggregate scores, regressions dataset, log tags, and two views; opt-in trust-triple remains unauthorized.
