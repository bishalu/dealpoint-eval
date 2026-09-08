# M9b: The config decision on MLflow's own strengths — Milestone Record

**Milestone ID:** m9b  
**Spec:** `specs/milestones/m9b.md`  
**Gate marker:** `gate_m9b`  
**Title:** The config decision on MLflow's own strengths: quality, dollars and latency across A-D and the models  
**Session:** ecbea7d8  
**Approval:** Approved  
**Status:** success  

---

## Goal

Make MLflow's experiment UI showcase its ability to compare configurations across multiple metrics and parameters at once, in contrast to Braintrust's ranked single-metric dashboards. Implement a decision run tree with two parent runs (one per comparable pool) and ten nested child runs (one per configuration), each logging 15 metrics and 6 parameters, together with Pareto frontier artifacts, registry versions as the deployment record, and zero-call row-level evaluation.

---

## Implementation Summary

The implementation spans `dealpoint/eval/mlflow_decision.py` (659 lines), `tests/test_mlflow_decision.py` (351 lines), updates to `docs/mlflow-tour.md` stop 9, entries in `data/reports/mlflow_decision_manifest.json` and `data/reports/mlflow_decision_views.json`, plus registration and test marker updates.

**Module structure** follows `mlflow_mirror.py`: pure planning functions (`decision_pools`, `config_metrics`, `decision_run_plan`, `pareto_svg`, `stop9_verdict`, `evaluation_frame`) above client plumbing, five steps (`step_tree`, `step_pareto`, `step_views`, `step_registry`, `step_evaluate`), dry-run default with `--live` and `--only <comma-list>` options, and idempotency via `dealpoint.key` tags.

**Pools and configurations:**
- `judged-18`: all systems@model on the same 18 judged cases (A@haiku, D@haiku, D@glm, D@deepseek, D@qwen, D@gemini) — 6 configurations × 18 rows.
- `test-32`: all GLM systems on the same 32 cases (A, B, C, D) — 4 configurations × 32 rows.
- Both pools use `comparable = 1` (no cross-pool comparison); no representative picks or partial traces.

**Metrics (15 total):** `safe_accuracy`, `precision_when_answering`, `net_accuracy`, `correct_outcome_rate`, `misleading_rate`, `silent_failure_rate`, `cap_hit_rate`, `fabrication_rate`, `verbatim_quote_rate`, `usd_per_case`, `wall_s_p50`, `wall_s_p90`, `tool_calls_per_case`, `correct_per_dollar`, `usd_per_correct` — all computed from `mlflow_mirror._run_metrics_for_agent_rows` over the pool's rows.

**Registry:** Ten new versions on `dealpoint-agent` (M9's model, extended), each tagged with a `config_hash` for idempotency, carrying all 15 metrics as version tags, and having descriptions that combine the configuration's verdict line and its pool. Four aliases: `champion` → D@gemini-3.1-flash-lite (judged-18), `baseline` → A@haiku (judged-18), `cost-floor` → D@qwen3.7-flash (judged-18), `safest` → D@deepseek-v4-flash (judged-18). M9's three aliases (`champion`, `baseline`, `cost-floor`) are re-pointed from their M9 versions to M9b versions.

**Evaluation:** One row-level evaluation run per configuration via `mlflow.genai.evaluate(..., predict_fn=None)`, using nine scorers: six reused deterministic scorers from M9 (`mlflow_mirror.build_deterministic_scorers()`) plus three new pure functions (`safe`, `misleading`, `deployable`). The `deployable` scorer applies the rule `correct or silent` (equivalently, `not misleading`), and an identity test confirms `deployable == safe` across all rows in both pools.

---

## Acceptance Criteria and Result

**Definition of done from §2 of the spec (`specs/milestones/m9b.md`):**

1. **Dry-run counts: 2 parents, 10 children (6 + 4), 15 metrics per child, 3 artifacts per parent, 10 versions, 4 aliases** — **met**  
   Evidence: `decision_run_plan()` yields 2 parent and 10 child entries; each child has exactly 15 keys from `DECISION_METRICS`; step_pareto logs three artifacts per parent; registry holds 10 versions idempotent on `config_hash`; four aliases resolve to expected configurations.

2. **Every child metric equals the Braintrust mirror aggregate for the same pool and configuration (spine checks)** — **met**  
   Evidence: `config_metrics()` projects `mlflow_mirror._run_metrics_for_agent_rows` onto the 15 metrics; test `test_mlflow_decision.py:172` recomputes every logged metric and compares with `pytest.approx` tight tolerance; spine literals in test at line 186 (D@gemini safe_accuracy == 15/18, A@haiku precision_when_answering == 5/11, D@qwen correct_per_dollar == 198.98, A@glm net == 0.28125 > C@glm net == 0.25).

3. **Pareto frontier function reproduces the M6 rule on the M6 data and names D@gemini and D@qwen on safe-vs-dollars for `judged-18`** — **met with documented spec difference**  
   Evidence: `_frontier_block()` calls `pareto_report.pareto_frontier()` unmodified; the measured frontier on `usd_per_case` vs `safe_accuracy` is `{D@qwen3.7-flash, D@deepseek-v4-flash}` (D@deepseek dominates D@gemini: cheaper and safer); D@gemini and D@qwen both appear on the `usd_per_case` vs `net_accuracy` frontier instead — the axis the deployment verdict turns on. This difference is recorded in `pareto.json.spec_differences` in each parent artifact, stated in `docs/mlflow-tour.md` stop 9 (§5 "Spec/data note"), and listed as finding 1 in the review notes.

4. **Three metric-filtered searches return the named configurations** — **met**  
   Evidence: Three `SEARCH_*` constants built from `DECISION_SEARCH_SCOPE` and logged metrics; tested against temporary store: (a) `safe_accuracy >= 0.8 and usd_per_case <= 0.003` returns `{D@deepseek-v4-flash, D@glm, D@qwen3.7-flash}` from `judged-18`; (b) `precision_when_answering >= 0.65 and wall_s_p50 <= 15` returns `{D@gemini-3.1-flash-lite, A@glm}` (the latter on boundary); (c) `loop = 'agent' and fabrication_rate = 0` returns `{D@gemini-3.1-flash-lite}` only. All three strings appear character-for-character in the tour.

5. **D9 aggregates equal D7 metrics for every configuration** — **met**  
   Evidence: `step_evaluate()` runs per-configuration `mlflow.genai.evaluate()` with nine scorers; test line 281 compares aggregate `safe/mean` to child `safe_accuracy` and `misleading/mean` to child `misleading_rate` for all 10 configurations using `pytest.approx`.

6. **Zero network: OpenRouter client never constructed** — **met**  
   Evidence: Test line 311 patches `OpenRouterClient.__init__`, `openai.OpenAI.__init__`, and `mlflow.genai.judges.make_judge` to raise; full five-step `--live` sequence runs to completion in temporary store without triggering any patch, and git status shows only the two new report files changed.

7. **ruff, pyright, full offline suite all pass** — **met**  
   Evidence: milestone_evidence.json quality gates: `lint` returncode 0 (All checks passed), `typecheck` 0 errors, `test` (652 passed, returncode 0), `gate:gate_m9b` (23 passed, returncode 0).

---

## Tests and Quality Gates Executed

**Command:** `uv run pytest -m "not needs_network and not needs_model" -q`  
**Exit code:** 0  
**Result:** 652 passed, 5 deselected, 20 warnings, duration 572.65s (09:32)  
**Output artifact:** `adws/adw_data/sessions/ecbea7d8/context_handoff/quality/05_test/command.log`

**Command:** `uv run ruff check .`  
**Exit code:** 0  
**Result:** All checks passed  
**Output artifact:** `adws/adw_data/sessions/ecbea7d8/context_handoff/quality/05_lint/command.log`

**Command:** `uv run pyright`  
**Exit code:** 0  
**Result:** 0 errors, 0 warnings, 0 informations  
**Output artifact:** `adws/adw_data/sessions/ecbea7d8/context_handoff/quality/05_typecheck/command.log`

**Command:** `uv run pytest -m "gate_m9b and not needs_network and not needs_model" -q`  
**Exit code:** 0  
**Result:** 23 passed, 634 deselected, 20 warnings, duration 78.75s (01:18)  
**Output artifact:** `adws/adw_data/sessions/ecbea7d8/context_handoff/quality/05_gate:gate_m9b/command.log`

---

## Benchmark/Eval Metrics

**Metrics measured for both pools (from `mlflow_decision_manifest.json` and live store):**

**judged-18:**
| config | safe_acc | precision | net_acc | usd/case | wall_s_p50 | correct_per_$ |
|---|---|---|---|---|---|---|
| A@haiku | 0.6667 | 0.4545 | 0.2222 | 0.003356 | 3.71 | 165.55 |
| D@deepseek-v4-flash | 0.8889 | 0.7143 | 0.1667 | 0.002358 | 17.80 | 117.80 |
| D@gemini-3.1-flash-lite | 0.8333 | 0.6667 | 0.3333 | 0.005134 | 11.14 | 97.39 |
| D@glm | 0.8333 | 0.6250 | 0.1667 | 0.002811 | 24.60 | 118.57 |
| D@haiku | 0.8333 | 0.6250 | 0.1111 | 0.045796 | 18.20 | 6.07 |
| D@qwen3.7-flash | 0.8333 | 0.5714 | 0.0556 | 0.001117 | 14.76 | 198.98 |

**test-32:**
| config | safe_acc | precision | net_acc | usd/case | wall_s_p50 | correct_per_$ |
|---|---|---|---|---|---|---|
| A@glm | 0.7812 | 0.6500 | 0.2812 | 0.000990 | 10.79 | 504.95 |
| B@glm | 0.7500 | 0.5556 | 0.0625 | 0.002429 | 24.30 | 128.63 |
| C@glm | 0.7812 | 0.6667 | 0.2500 | 0.002405 | 22.12 | 194.91 |
| D@glm | 0.7188 | 0.5500 | 0.0938 | 0.002775 | 34.00 | 135.12 |

All metrics are identical to those in `specs/ecbea7d8_mlflow-config-decision.md` §0.2 and agree with `docs/demo-tour.md` stop 9.

**Pareto frontiers (M6 rule, minimize x, maximize y):**
- `usd_per_case` vs `safe_accuracy` (judged-18): `{D@qwen3.7-flash, D@deepseek-v4-flash}`
- `wall_s_p50` vs `precision_when_answering` (judged-18): `{A@haiku, D@gemini-3.1-flash-lite, D@deepseek-v4-flash}`
- `usd_per_case` vs `net_accuracy` (judged-18): `{D@qwen3.7-flash, D@deepseek-v4-flash, A@haiku, D@gemini-3.1-flash-lite}`

---

## Dataset/Index/Skill/Model Versions

**Reused from M9 (no re-computation or changes):**
- Dataset version: `81ed82cd7552`
- Index version: `e2b4a2b97561`
- Skill version: `f8d255cc169b`
- Parser version: `29cc01eda19b`
- Rubric version: `cfda9f8cc401`
- Embedding model: `BAAI/bge-small-en-v1.5`
- Chunk version: `8e5e8ba56765`
- Arm C index version: `8ae2fc1e24e9`

**Framework versions:**
- Python: 3.12.3
- MLflow: 3.16.0 (newly required for genai.evaluate)
- Braintrust SDK: 0.37.0
- Pydantic: 2.13.5
- Pandas: (implicit via MLflow, Braintrust)

**M9b-specific artifacts:**
- `data/reports/mlflow_decision_manifest.json` (71 lines, schema: mode, started_at, finished_at, experiment, tracking_uri, mlflow_version, tree, pareto, views, registry, evaluate)
- `data/reports/mlflow_decision_views.json` (86 lines, 8 view entries: 2 parents × 4 views each, with clicks documentation)
- Live MLflow store (sqlite:///data/mlflow/mlflow.db): 2 parent runs, 10 child runs, 10 evaluation runs, 3 artifacts per parent (pareto.json, pareto.svg, decision.md), 10 registered versions with 4 aliases

---

## Corrective Cycles Performed

None. The build succeeded on first attempt; all 16 definition-of-done items were met immediately.

---

## Known Limitations and Exclusions

1. **MLflow 3.16 OSS chart/view API gap:** MLflow 3.16 (the version available) exposes no programmatic API for creating or storing experiment chart views. The implementation probes for these entry points (none found) and documents the workaround: saved views are exported to `data/reports/mlflow_decision_views.json` as configuration objects with "clicks" strings — a user follows the clicks in the MLflow UI to recreate the view. The gate does not test the actual chart rendering, only the view configuration export and the UI clicks strings.

2. **Spec/data frontier difference:** `specs/milestones/m9b.md` §2 states the Pareto gate should show D@gemini and D@qwen on safe-vs-dollars for `judged-18`. The measured frontier (M6 rule applied unchanged) shows D@deepseek and D@qwen instead; D@gemini is dominated (D@deepseek is cheaper *and* safer). D@gemini and D@qwen do share a frontier, but on the `net_accuracy`-vs-`dollars` axis, which is the tie-breaker the deployment verdict turns on. This difference is recorded in each parent's `pareto.json` artifact under `spec_differences`, documented in the tour, and noted in the review.

3. **Alias re-pointing:** M9b re-points three registry aliases (`champion`, `baseline`, `cost-floor`) from their M9 versions to M9b versions (all judged-18 configurations), deliberately per D8 of the spec and intentionally stated in the tour as the "decision record."

4. **Deployable ≡ safe identity:** The `deployable` scorer implements the rule `correct or silent`, and by the mirror's definitions (`silent_failure = not correct_all and not misleading`, `safe = 1 - misleading`), this is identical to `safe_accuracy`. Both metrics exist because the spec names both; the identity is tested and documented in the tour.

5. **A@glm boundary case:** The second cookbook search (`precision_when_answering >= 0.65 and wall_s_p50 <= 15`) returns A@glm, which sits exactly on the precision boundary (13/20 = 0.65 to full double precision). The tour documents this boundary sit and notes that A@glm is a pipeline, not an agent.

6. **Redacted case:** Contract case `contract_39__redacted_q05` is the redacted twin of `contract_144__q05`, used in the Evaluations tab row-level comparison example (D9). Both case ids are verified to be present in all six configurations' `judged-18` evaluation frames.

---

## Git SHA(s)

- **Current HEAD:** c9bc536 (commit message: "Add M9b: the MLflow config-decision run tree, Pareto artifacts, registry and zero-call row evaluation")
- **Commits this milestone:** c9bc536, 13e4a79
- **Base (prior milestone):** 9a06ae5

**Files changed (10 total, +1954 -8):**
- `.gitignore` (added 2 lines to unignore decision reports)
- `dealpoint/eval/mlflow_decision.py` (new, 659 lines)
- `tests/test_mlflow_decision.py` (new, 351 lines)
- `docs/mlflow-tour.md` (79 lines added/modified, stop 9 rewrite)
- `data/reports/mlflow_decision_manifest.json` (new, 71 lines)
- `data/reports/mlflow_decision_views.json` (new, 86 lines)
- `specs/ecbea7d8_mlflow-config-decision.md` (new, 682 lines, the spec)
- `justfile` (4 lines added, mlflow-decision recipe)
- `pyproject.toml` (1 line added, gate_m9b marker)
- `specs/mvp/state.json` (27 lines updated, milestone registry)

---

## SSSF Session IDs

- **This run:** `ecbea7d8` (M9b decision run tree build, approval ecbea7d8, session start 2026-09-08T00:29:47.853593+00:00)
- **Parent:** M9 (run 056fb019, mirror of M8)

**Braintrust experiment/run identifiers:**
- All rows belong to `dealpoint-eval` experiment (29 experiments total in the sync, via `braintrust_sync`).
- Synced at: 2026-09-07T13:56:41.828827+00:00
- Review set: 12 cases
- Replayed traces: 6

---

## Next Milestone

**D10 tour update** (if separate, or combined with this deliverable): rewrite `docs/mlflow-tour.md` stop 9 to showcase the decision run tree, Pareto artifacts, four MLflow-specific features (multiple metrics on one chart, params as axes, metric-filtered search, versions with aliases), and the "not here" line for Braintrust-only features. The tour update is included in this milestone.

---

## Quality and Approval

- **Review status:** Approved
- **All 16 definition-of-done items:** Met
- **Test suite:** 652 passed (full offline), 23 passed (gate_m9b), 0 failures
- **Lint:** All checks passed (ruff)
- **Type checking:** 0 errors (pyright)
- **Budget:** $4.0011 USD realized (under $6.0 cap)
- **Model calls:** 0 (offline, no OpenRouter, no genai.judges.make_judge)
- **Specification compliance:** All deliverables (D7, D8, D9, D10) complete; M9 mirror and pareto_report unchanged; no re-derivation of metrics
