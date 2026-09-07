# M9b: the config decision, on MLflow's own strengths

**Authority:** `specs/milestones/m9.md` (the mirror this builds on; its principles 1 to 6 apply unchanged),
`docs/demo-tour.md` stop 9 and the eight Braintrust dashboards' verdicts (the numbers this must reproduce),
and the engineer's directive of 2026-09-07: **bring out MLflow's unique features so it shines too, focused
on the user's interest, the A/B/C/D optimal configuration for quality, dollars and latency; copy as much
Braintrust data as possible, OpenRouter only as a last resort.**

M9b is additive to M9. It reads the same stored rows through the same pure functions (`stored_row_index`,
`mirror_for`, `classify_experiment`, `arm_parameter_sets`, `braintrust_cockpit.DASHBOARDS`) and the M9
mirror module. **It makes no model calls.** Every number equals its Braintrust counterpart for the same
(case, variant); a test asserts it.

## 0. The question, and MLflow's answer to it

The Braintrust dashboards answer "which system, which model" with ranked bar lists over one metric at a
time, because a monitor chart can only rank one measure. MLflow's experiment UI was built for the
multi-metric version of that question: every configuration is a run with **params** (loop, retriever,
skill, model) and **metrics** (safe accuracy, precision when answering, net accuracy, dollars per case, p50
and p90 latency, correct outcomes per dollar, cap-hit and misleading rates), and the runs table, the
parallel-coordinates chart, the scatter chart, metric-filtered search and the model registry compare them
all at once. That is the feature to make shine. The scoring definitions do not change: safe accuracy is the
accuracy point, precision when answering its check, net accuracy the tie-breaker (M9 §D2, `docs/demo-tour.md`).

## 1. Deliverables

### D7. The config-decision run tree

`just mlflow-decision` (dry run default, `--live`, idempotent by tag `dealpoint.key`), a step of the M9
module or a sibling module `dealpoint/eval/mlflow_decision.py`:

- **Two parent runs**, one per comparable pool, with the pool in the name and description:
  `decision/judged-18` (every system@model on the same 18 cases: A@haiku, D@haiku, D@glm, D@deepseek,
  D@qwen, D@gemini) and `decision/test-32` (the four GLM systems on the same 32 cases). No cross-pool
  comparison anywhere; the M9 `comparable = 1` rule holds.
- **One nested child run per configuration** under each parent: params `arm`, `loop`, `retriever`, `skill`,
  `model`, `system_label`; metrics `safe_accuracy`, `precision_when_answering`, `net_accuracy`,
  `correct_outcome_rate`, `misleading_rate`, `silent_failure_rate`, `cap_hit_rate`, `fabrication_rate`,
  `verbatim_quote_rate`, `usd_per_case`, `wall_s_p50`, `wall_s_p90`, `tool_calls_per_case`,
  `correct_per_dollar`, `usd_per_correct`; tags = the factorial schema; description = that configuration's
  line in the verdicts. Metrics come from `metadata_mirror` over the pool's rows, never re-derived.
- **Pareto frontiers as artifacts** on each parent run: `pareto.json` (the frontier under the M6 rule from
  `dealpoint/eval/pareto_report.py`, applied to safe accuracy vs `usd_per_case`, precision vs `wall_s_p50`,
  and net accuracy vs `usd_per_case`) and `pareto.svg` drawn by a pure function (stdlib only, no new
  dependency), plus `decision.md`, the verdict paragraph from `docs/demo-tour.md` stop 9 verbatim.
- **Saved chart views**, one set per parent, exported to `data/reports/mlflow_decision_views.json` and
  applied through whatever chart-view or experiment-view API MLflow 3.16 exposes (probe first; if none,
  the tour documents the three clicks): parallel coordinates `loop, retriever, skill, model ->
  safe_accuracy, precision_when_answering, usd_per_case, wall_s_p50`; scatter `usd_per_case` vs
  `safe_accuracy` coloured by model; scatter `wall_s_p50` vs `precision_when_answering`; bar
  `correct_per_dollar` by run.
- **Metric-filtered searches** written down as a cookbook section in `docs/mlflow-tour.md` and as tests:
  `metrics.safe_accuracy >= 0.8 and metrics.usd_per_case <= 0.003` (safe and cheap),
  `metrics.precision_when_answering >= 0.65 and metrics.wall_s_p50 <= 15` (right and fast),
  `params.loop = 'agent' and metrics.fabrication_rate = 0` (agents that never invent a clause); each
  returns the configurations the tour names.

### D8. The registry as the decision record

Extend M9's registered model `dealpoint-agent`: every child run of D7 registers a version (idempotent by
config hash), version tags carry the same metrics, aliases `champion` (D@gemini), `baseline` (A@haiku),
`cost-floor` (D@qwen), `safest` (D@deepseek); each version's description is its verdict line and the pool
it was measured on. The tour points at the registry's version-comparison view as the deployment record.

### D9. Row-level comparison without a model call

`mlflow.genai.evaluate(data=<the pool's traces>, predict_fn=None, scorers=[the six deterministic scorers
+ three pure scorers `safe`, `misleading`, `deployable`])` per configuration, where `deployable` is the row
rule `correct or silent` (never misleading). The Evaluations tab then compares any two configurations row by
row on `contract_144__q05` and the redacted twin, the MLflow form of the Braintrust Grid, at zero model
calls. Aggregates must equal D7's metrics (test).

### D10. Tour update

`docs/mlflow-tour.md` stop 9 becomes the showcase: open the `decision/judged-18` parent, the parallel
coordinates, the two scatters, the Pareto artifact, the three searches, the registry versions; one line each
on what MLflow does here that the Braintrust dashboards could not (many metrics on one chart; params as
axes; search by metric; versions with aliases). One "not here" line for anything Braintrust has and MLflow
does not (the ranked single-metric lists over logs, online scoring).

## 2. Gate (`gate_m9b`, offline, temporary SQLite store, real client)

- dry run counts: 2 parents, 10 children (6 + 4), 15 metrics per child, 3 artifacts per parent, 10 registered
  versions, 4 aliases; `--live` twice creates nothing new the second time;
- every child metric equals the Braintrust mirror's aggregate for the same pool and configuration (spine
  checks: D@gemini safe accuracy, A@haiku precision, D@qwen correct per dollar, GLM A vs C net accuracy);
- the Pareto frontier function reproduces the M6 rule on the M6 data and names D@gemini and D@qwen on
  safe-vs-dollars for `judged-18`;
- the three searches return the named configurations against the temporary store;
- D9 aggregates equal D7 metrics for every configuration;
- zero network: a test patches the OpenRouter client and asserts it is never constructed;
- ruff, pyright, the full offline suite.

## 3. Budget

$0. No OpenRouter calls anywhere in M9b; if a step believes it needs one, it stops and reports instead.
Disk: artifacts under `data/mlflow/` (gitignored), a few hundred KB.

## 4. Out of scope

Changes to the agent, the judges, the scoring definitions, any frozen artifact, or anything in Braintrust;
monitoring; re-running any configuration. M9's D3 and D4 results are used as they landed.

## 5. Registration

`adws/adw_modules/milestones.py`: `m9b`, `spec_path="specs/milestones/m9b.md"`, `gate_marker="gate_m9b"`,
`needs_model=False`. Launch after `m9` (run `056fb019`) has finished.
