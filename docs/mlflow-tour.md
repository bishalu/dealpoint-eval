# The DealPoint tour, MLflow edition

Same rows, second cockpit. This tour sits beside [`docs/demo-tour.md`](demo-tour.md) (the Braintrust
tour, which links back here); neither replaces the other, and no number below differs from the
Braintrust one for the same object -- both are read out of the same stored rows through the same pure
functions (`dealpoint/eval/braintrust_showroom.py`, `braintrust_cockpit.py`, and the M9 mirror,
`dealpoint/eval/mlflow_mirror.py`).

Tracking server: self-hosted, SQLite-backed, on this VM --
`just mlflow-server` (`sqlite:///data/mlflow/mlflow.db`, artifacts under `data/mlflow/artifacts/`,
gitignored like `data/index/`), reached locally at `http://127.0.0.1:5000`, or through the exe.dev HTTPS
proxy once the operator wires it (the factory documents the port, not the URL).

Rebuild everything in a fresh store with one idempotent command:

```
just mlflow-sync            # dry run: prints the plan, writes nothing
just mlflow-sync --live     # writes; a second --live creates nothing new (tag `dealpoint.key`)
```

33 runs in one experiment (`dealpoint-eval`), 738 traces (265 agent + 406 retrieval + 67 prompt-variant),
9 evaluation datasets, 8 registered prompts, four OpenRouter judges, six in-process deterministic
scorers, one registered model (`dealpoint-agent`) with three aliases. Manifest:
`data/reports/mlflow_manifest.json`.

## Act I. Build a measurable agent

### Stop 1. The question, and the trap

**Datasets tab**, `maud-dealpoint-dev` / `maud-dealpoint-test` / `maud-dealpoint-counterfactual`: the
same 58/167/40 cases as the Braintrust datasets, one row per case, expectations = the gold answer. The
counterfactual set is the trap Braintrust's tour opens with: MLflow holds the same 40 redacted/
out-of-scope rows, unchanged.

### Stop 2. Can it retrieve the right evidence?

The **`rag-*` runs** (`rag-m3-dense` through `rag-m3-fusion-rerank`, plus `rag-m7-li-crosscheck`): open
the run comparison (tag `axis = 'retrieval'`), chart `obj/hit_at_5`, `obj/hit_at_10`, `obj/mrr` and
`li/hit_rate`, `li/mrr` side by side (LlamaIndex's independent grader on the same 406 retrieval traces).
Hybrid (dense + BM25, RRF) wins on both scorers, exactly as `docs/demo-tour.md` reports.

### Stop 3. Does agency help? Does better RAG help? Does the skill help?

**Evaluations tab**, filter to the four GLM system runs (`A-z-ai_glm-...` through `D-z-ai_glm-...`),
sort by `dealpoint.key = 'contract_144__q05'`: A abstains, B/C/D answer "Actual knowledge" -- the same
row Braintrust's Grid shows. `mlflow.genai.evaluate` runs the six deterministic scorers in-process here
(Braintrust could not run them server-side); their means become the run metrics `obj/*`.

## Act II. Correctness is not enough

### Stop 4. Watch one agent think

**Traces tab**, filter `tags.case_id = 'contract_144__q05'`. Each trace is a replayed span tree with its
**original timestamps** (`start_time_ns`/`end_time_ns` from the stored row's `t_ms`/`wall_ms`) -- the one
thing the Braintrust replay could not do. Assessment columns show `obj/*`, `judge/<family>/<dim>` and,
for the 24 lawyer-scored packets, `human/<dim>`, right on the trace.

### Stop 5. What deterministic truth captures, and scoring in production

The six `@mlflow.genai.scorer` functions (`grounded_accuracy`, `answer_correct`,
`citation_gold_overlap`, `citation_verbatim`, `abstain_correct`, `skill_adherence`) call
`dealpoint.eval.scorers` directly and run inside this MLflow install -- the in-process scorer Braintrust
never had. `just mlflow-score-new --since <timestamp>` is the batch stand-in for the production
monitoring MLflow OSS does not ship (Databricks-only): it re-scores traces newer than the cutoff instead
of running continuously.

### Stop 6. Human review, three judges, and where they disagree

Assessment columns on the **Traces tab** for the `judge-*` runs' traces: `judge/mistral/<dim>`,
`judge/nvidia/<dim>`, `judge/bytedance/<dim>` next to `human/<dim>` for the 24 packets the lawyer scored
(96 HUMAN assessments). `just mlflow-align-judges --live` ran `judge.align()` for each of the four
dimension judges against those 24 packets (MLflow-unique; Braintrust has no judge alignment) and
re-scored all 432 of the 108 judged traces x 4 dimensions, writing `data/reports/judge_alignment.json`.
Closeness-to-lawyer, before -> after: reasoning 0.9375 -> 0.9375, evidence 0.9444 -> 0.9271, professional
0.9167 -> 0.9583, and **trajectory 0.7847 -> 0.7604** -- alignment moved trajectory, but backwards: the
one dimension no judge gets right on its own got slightly worse, not better, after `.align()` against the
lawyer's 24 packets. (An earlier run of this command lost all 108 trajectory re-scores because
`bytedance-seed/seed-2.0-mini`, a reasoning model, spent its `max_tokens=20` budget on hidden reasoning
and returned an empty response; the fix -- the same `reasoning.enabled=False` flag the base judge runner
already sends -- and a bounded retry with backoff for transient OpenRouter 429s let every re-score land.)

### Stop 7. Iterate on the prompt before touching code

**Prompt registry**: `agent-base-system-prompt` (`baseline`), `arm-a-prompt-cite-first` (`best`),
`judge-panel-rubric` (`judge-panel`), `judge-calibrated-rubric` (`rubric`), `agent-arm-d-skill-injection`
(`skill`), plus the four Playground chat prompts (`arm-a-prompt-{base,terse,cite-first,abstain-first}`).
The 67 `prompt-variant` traces carry the ledger's `(variant, case_id)` pairs and inputs, but **not** the
model's outputs or the judges' scores on them: those were never mirrored to disk outside Braintrust (see
"gaps" below), so the trace exists and is honestly empty rather than backfilled from a Braintrust fetch.
`just mlflow-optimize-prompt --live` ran GEPA/metaprompting on `arm-a-prompt-base` over the 18
`maud-dealpoint-playground-armA` rows with the evidence judge as the objective. **`arm-a-prompt-base`
version 2 is now registered with alias `optimized`** (verified in `data/mlflow/mlflow.db`) -- the fifth
row in the prompt comparison, reported against the Git-recorded scores of the four hand-written variants
rather than a re-run of them.

## Act III. Turn evidence into a deployment decision

### Stop 8. Diagnose the real failure

`mlflow.search_traces(filter_string="tags.status = 'CAP_HIT'")` finds the same trajectories Braintrust's
Debugger walks; the **Traces tab** filtered to `tags.case_id = 'contract_39__redacted_q05'` (the redacted
twin) shows the cap-hit span tree with real timestamps this time. There is no Topics, no Loop and no
Pattern object here -- said plainly in "what only Braintrust has" below -- so the failure-shape clustering
Braintrust's Topics facet does automatically has to be read off the assessment columns and tags by hand
(`tags.category`, `tags.arm`, `tags.model`).

### Stop 9. What should we deploy? (the config decision, M9b)

The **`Which model?`** run comparison (tag `axis = 'model'`, arm D held fixed, five models on the 18
judged cases): chart `net_accuracy`, `usd_per_case`, `wall_s_p50`/`wall_s_p90`, `cap_hit_rate`,
`correct_per_dollar`. That is M9's mirror of the Braintrust dashboards -- one ranked bar list per metric.
M9b (`just mlflow-decision --live`, no model calls, `dealpoint/eval/mlflow_decision.py`) turns the same
rows into a multi-metric decision object MLflow is built for: a run tree, params as axes, search by
metric, and a registry the deployment decision lives on.

**The two parent runs.** `decision/judged-18` (A@haiku, D@haiku, D@glm, D@deepseek-v4-flash,
`D@qwen3.7-flash`, D@gemini-3.1-flash-lite -- every system@model on the same 18 judged cases) and
`decision/test-32` (the four GLM systems A/B/C/D on the same 32 test cases). No chart ever crosses the
two: `comparable = 1` is the M9 rule (drops representative picks, live replays, the partial
`D@openai/gpt-5.6-luna-pro` trace), and the parent boundary keeps every comparison inside one pool.

**The nested run tree.** Six children under `decision/judged-18`, four under `decision/test-32`, each
with params `arm`, `loop`, `retriever`, `skill`, `model`, `system_label` and fifteen metrics
(`safe_accuracy`, `precision_when_answering`, `net_accuracy`, `correct_outcome_rate`,
`misleading_rate`, `silent_failure_rate`, `cap_hit_rate`, `fabrication_rate`, `verbatim_quote_rate`,
`usd_per_case`, `wall_s_p50`, `wall_s_p90`, `tool_calls_per_case`, `correct_per_dollar`,
`usd_per_correct`) -- every one of them `mlflow_mirror._run_metrics_for_agent_rows`'s own number, never
re-derived. Safe accuracy is the accuracy point (did not mislead), precision when answering its check,
net accuracy the tie-breaker (`docs/demo-tour.md` stop 9).

**parallel coordinates** -- `loop, retriever, skill, model -> safe_accuracy, precision_when_answering,
usd_per_case, wall_s_p50` (OSS 3.16 has no chart API, so this is three clicks, not a saved object:
Experiment > Chart view > + > Parallel coordinates; params `loop, retriever, skill, model`; metrics
`safe_accuracy, precision_when_answering, usd_per_case, wall_s_p50`). This is the thing a Braintrust
monitor dashboard cannot do: four metrics and four params on one chart instead of one ranked bar list
per metric.

**The two scatters.** `usd_per_case` vs `safe_accuracy` coloured by `model`; `wall_s_p50` vs
`precision_when_answering`. Params become axes.

**The Pareto artifact.** `pareto.json` and `pareto.svg` on each parent run: the M6 rule
(`dealpoint.eval.pareto_report.pareto_frontier`, applied unchanged) over three axis pairs. Measured,
`judged-18`'s safe-vs-dollars frontier is `D@qwen3.7-flash` and `D@deepseek-v4-flash`; its
net-accuracy-vs-dollars frontier is `D@qwen3.7-flash`, `D@deepseek-v4-flash`, `A@haiku` and
`D@gemini-3.1-flash-lite`. **Spec/data note:** `specs/milestones/m9b.md` section 2 expected D@gemini and
D@qwen on safe-vs-dollars; measured, D@deepseek-v4-flash dominates D@gemini-3.1-flash-lite there
(cheaper, $0.002358 vs $0.005134, and safer, 0.8889 vs 0.8333) -- D@gemini and D@qwen do share a
frontier, but on net-accuracy-vs-dollars instead, the tie-breaker axis the deployment verdict actually
turns on. This is recorded verbatim in `pareto.json`'s `spec_differences` block.

**Cookbook: three metric-filtered searches** (scoped to `` tags.`dealpoint.decision` = 'config' `` so
M9's 33 runs, which also carry `safe_accuracy` and `usd_per_case`, never leak in):

```
safe and cheap:                tags.`dealpoint.decision` = 'config' and metrics.safe_accuracy >= 0.8 and metrics.usd_per_case <= 0.003
  -> D@deepseek-v4-flash, D@glm, D@qwen3.7-flash (judged-18)
right and fast:                tags.`dealpoint.decision` = 'config' and metrics.precision_when_answering >= 0.65 and metrics.wall_s_p50 <= 15
  -> D@gemini-3.1-flash-lite (judged-18), A@glm (test-32, sitting exactly on the 0.65 boundary -- and A is a pipeline, not an agent)
agents that never invent a clause: tags.`dealpoint.decision` = 'config' and params.loop = 'agent' and metrics.fabrication_rate = 0
  -> D@gemini-3.1-flash-lite only
```

Search by metric -- a Braintrust dashboard ranks, it does not filter.

**The registry as the decision record.** Ten new `dealpoint-agent` versions, one per decision child,
version tags carrying the same fifteen metrics; aliases **`champion`** -> `D@gemini-3.1-flash-lite`,
**`baseline`** -> `A@haiku`, **`cost-floor`** -> `D@qwen3.7-flash`, **`safest`** -> `D@deepseek-v4-flash`
(new). M9b re-points M9's three `champion`/`baseline`/`cost-floor` aliases at the decision versions --
deliberate, idempotent, and the point of D8: the registry's version-comparison view is the deployment
record, not a chart someone has to remember to reopen.

**Row-level comparison at zero model calls.** `mlflow.genai.evaluate(data=<the pool's stored rows>,
predict_fn=None)` per configuration with the six deterministic scorers plus three pure row scorers
`safe`, `misleading` and **`deployable`** (`correct or silent`, never misleading). The Evaluations tab
then compares any two configurations row by row on `contract_144__q05` and its redacted twin
`contract_39__redacted_q05` -- the MLflow form of the Braintrust Grid, at zero model calls. Note:
by `metadata_mirror`'s own definitions `deployable` is identical to `safe` (`1 - misleading`); it is
stated the way a lawyer states it, not restated as an independent metric.

**Not here.** The eight Braintrust dashboards' ranked single-metric lists over *logs*, each with a
verdict paragraph attached to the dashboard object, and online scoring -- MLflow's run comparisons are
run-level, rebuilt from `data/reports/mlflow_decision_views.json` by hand because OSS 3.16 has no
chart/view API.

## Stop 10. Every tab filled, and the eight dashboards (M9c)

`just mlflow-tabs --live` (dry run by default, `dealpoint/eval/mlflow_tabs.py`) fills the six tabs M9/M9b
left empty; `just braintrust-dashboard-snapshot` and `just mlflow-dashboards --live`
(`dealpoint/eval/mlflow_dashboards.py`) port the eight Braintrust dashboards word for word. No model
calls except the opt-in `--gateway-smoke` (five cents, ledgered `milestone_tag: m9c`).

| Sidebar tab | Filled from (Braintrust object) | What MLflow does with it that Braintrust cannot |
| --- | --- | --- |
| Overview | The experiment description | The dashboard runs, decision tree, registry and review queues are one click away, not a separate app (`step_overview`). |
| Observability > Traces | The same 738 traces M9 mirrors | Real span timestamps and in-process deterministic scorers (see "What MLflow adds" below). |
| Observability > Sessions | `mlflow.trace.session`/`mlflow.trace.user`, set on every trace at re-log time (`session_plan`/`step_sessions`, D11) | One contract question (`contract_144__q05`) reads as ONE session across every configuration that answered it, its redacted twin beside it -- Braintrust logs carry no session grouping at all. |
| Evaluation > Judges | The four `make_judge` dimension judges, registered (`judge_registration_plan`/`step_judges`, D12) | The registered judges are startable as a continuous online-scoring rule server-side; Braintrust can only declare a scorer. |
| Evaluation > Review | Two review queues, five label schemas, fully populated (`review_queue_plan`/`step_review`, D13): the 12 open review-set traces and the 24 already-scored lawyer packets | A real assignment/status workflow (pending/complete) around the same 96 lawyer assessments, not just plain feedback rows -- OSS 3.16, no Databricks needed. |
| Evaluation > Datasets | The 9 evaluation datasets (M9) | Unchanged from M9. |
| Evaluation > Evaluation runs | `mlflow.genai.evaluate` row comparisons (M9b) | Zero-model-call row-level comparison, the MLflow form of the Braintrust Grid. |
| Prompts & versions > Playground | The 8 registered prompts, each with a `model_config` naming its gateway endpoint (`prompt_endpoint_plan`/`step_playground`, D15) | `arm-a-prompt-cite-first` loads pre-wired to `dealpoint-glm`, `judge-panel-rubric` to `judge-mistral`, sampling parameters filled in -- one click to run, not a separate Braintrust Playground tab. |
| Prompts & versions > Prompts | The 8 registered prompts (M9) | Unchanged from M9. |
| Agent versions | 14 `LoggedModel`s: the ten M9b decision-tree configurations plus the four `playground-arm-A-*` prompt variants (`logged_model_plan`/`step_models`, D14), comparable traces linked via `mlflow.modelId` at re-log time | `champion`/`baseline`/`cost-floor`/`safest` resolve to a version with traces attached, not a dashboard someone has to remember to open. |
| AI Gateway | One `openrouter` connection, five role-named endpoints, created through the store's gateway methods (`gateway_plan`/`step_gateway`, D15) | One place for every model key the demo uses, instead of a `base_url` baked into each call site. |

**Probed and corrected, not assumed.** Three of the spec's assumed API shapes/outcomes turned out
different on this install's MLflow 3.16.0, verified live and recorded in `mlflow_tabs_manifest.json`'s
`spec_differences`-shaped fields rather than silently forced to match:

- **The six deterministic `@scorer` functions cannot be registered off Databricks.**
  `Scorer._check_can_be_registered` raises `DECORATOR_SCORER_REGISTRATION_NOT_SUPPORTED_ERROR`
  ("Custom (@scorer) scorers use exec() during deserialization, which poses a code execution risk"),
  verified live. `step_judges` attempts every one, catches exactly that exception, and records the
  verbatim message per scorer under `judges.deterministic_registration`; the measured
  `mlflow.genai.scorers.list_scorers()` count is **4**, never assumed to be 10.
- **D13's review-queue workflow is OSS, not Databricks-only.** `mlflow.genai.review_queues`
  (`create_review_queue`, `add_items_to_review_queue`, `set_review_queue_item_status`) and
  `mlflow.genai.label_schemas.create_label_schema` (not `mlflow.genai.labeling`, which IS
  Databricks-only) all work against a plain SQLite tracking URI -- verified live. Both queues, all
  five schemas, and every resolvable item are created and set to their target status. This corrects
  an earlier draft of this tour that (wrongly) called review queues Databricks-only.
- **D12's online-scoring rule fails for a different, more specific reason than "Databricks-only":**
  `judge-professional.start(sampling_config=...)` raises "Scorer 'judge-professional' requires
  expectations, but scorers with expectations are not currently supported for automatic evaluation"
  -- an MLflow 3.16 limitation on judges built from expectation-bearing instructions, not a
  Databricks gate. `step_judges` records the verbatim message under `judges.online_rule_error`; the
  four judges are still registered either way, so the tab is filled even when the rule cannot start.
- **The AI Gateway is fully scriptable via the tracking store**, not Settings-UI-only:
  `MlflowClient()._tracking_client.store` exposes `create_gateway_secret`/
  `create_gateway_model_definition`/`create_gateway_endpoint`/`list_gateway_endpoints` directly
  (`mlflow.gateway.client` does not exist on this install). `step_gateway` uses them; the
  Settings-UI print plus `--gateway-ready` is a fallback for when the store call itself raises, not
  the default path.

**The eight dashboards.** `data/reports/braintrust_dashboard_values.json` was written by
`snapshot_dashboards` against the live Braintrust project (`just braintrust-dashboard-snapshot
--live`, run 2026-09-08) -- real, read-only BTQL, not the local evaluator checked against itself. The
local evaluator (`chart_values`, `eval_filter`, `eval_measure`) recomputes the same numbers from the
stored mirror rows with no network call, and the eight `dashboard/<name>` runs each carry a
self-contained HTML render (ranked bar list, exact titles, exact order) plus one metric per chart row.

| MLflow run | Braintrust dashboard | Link |
| --- | --- | --- |
| `dashboard/dealpoint-eval-overview` | DealPoint eval overview | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/58a7278a-71c8-4a71-9a5b-40ab019ee0c2 |
| `dashboard/which-system` | Which system? | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/8e0003f4-5840-45b3-bc9a-86fcb686ec1a |
| `dashboard/which-model` | Which model? | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/7c13d289-97cb-4db6-8340-6494e85608a3 |
| `dashboard/which-retriever` | Which retriever? | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/32c5a4be-7898-4630-af11-ca40e743e9e4 |
| `dashboard/judges-and-the-lawyer` | Judges and the lawyer | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/54eae7f9-0815-4497-8166-725929ddd416 |
| `dashboard/deepeval` | DeepEval | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards (DeepEval) |
| `dashboard/llamaindex` | LlamaIndex | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards (LlamaIndex) |
| `dashboard/which-prompt` | Which prompt? | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/9c9b9fa9-4a54-47d9-9c6a-57e60dd009c5 |

Every chart row on these eight runs was checked against `data/reports/braintrust_dashboard_values.json`
to 1e-9 by `tests/test_mlflow_dashboards.py`, with the two named exceptions below.

**Two gaps named, not resolved silently:**
- Three charts (`prompt_professional`, `prompt_evidence`, `prompt_reasoning`) cannot be reproduced by
  the local evaluator -- `prompt_variant_trace_plan`'s own docstring (M9) records that the 67
  prompt-variant traces' Playground judge scores were never mirrored to disk, so those three charts
  have no rows locally (`locally_evaluable: false` in the snapshot) and are ported from the Braintrust
  snapshot values directly.
- One chart, `mod_p90` (`percentile(metadata.wall_s, 0.9)`), has rows locally but they do not match
  Braintrust's value to 1e-9: pulling the identical 18 raw values behind its `haiku` group and
  computing both by hand gives 63.2187 (the local evaluator's exact linear-interpolation quantile) vs
  63.43960425028111 (Braintrust's own `percentile(...)`, verified live) -- Braintrust's percentile
  aggregator is evidently an approximate, sketch-based quantile. `PERCENTILE_APPROXIMATE` in
  `mlflow_dashboards.py` names this chart so the value gap is checked (labels and order still must
  match), not silently forced to agree, and its rendered value on the port is also read straight from
  the snapshot -- like the three prompt charts above, so "Which model?" shows Braintrust's number
  (63.4396) rather than the local exact quantile.

Both gaps are visible on the rendered page itself, not just in this document: `render_dashboard_html`
prints one provenance line under the title of every snapshot-sourced chart.

## What MLflow adds

- **In-process deterministic scorers.** The six `dealpoint.eval.scorers` functions run as real MLflow
  scorers inside `mlflow.genai.evaluate`; Braintrust could only declare them, never execute them
  server-side.
- **Real span timestamps.** Every replayed trace carries its original `start_time_ns`/`end_time_ns` from
  the stored row's `t_ms`/`wall_ms`, so duration and ordering are exact, not synthetic.
- **Judge alignment.** `judge.align()` against the lawyer's 24 packets is a Braintrust-absent capability
  (Stop 6, `just mlflow-align-judges`).
- **Prompt optimization.** GEPA/metaprompting over the arm-A packets, registered as a new prompt version
  with an alias (Stop 7, `just mlflow-optimize-prompt`).
- **The model registry.** One `dealpoint-agent` object with a version per system@model and named
  aliases, rather than a dashboard someone has to remember to open (Stop 9).

## What only Braintrust has here

- **Monitor dashboards over logs**, the live objects with a verdict paragraph each; M9c
  (Stop 10) ports their numbers word for word as a rendered HTML artifact plus metrics per run, since
  OSS 3.16 has no chart/dashboard API to hold a live equivalent.
- **Online scoring**, continuous, over live logs; MLflow OSS's nearest equivalent is the batch
  `just mlflow-score-new`, plus M9c's `judge-professional.start(sampling_config=...)` attempt (Stop 10:
  not Databricks-gated on this install -- it fails because expectation-bearing judges are not yet
  supported for automatic evaluation, verified live and recorded, not assumed).
- **Topics, Patterns and the Debugger** are Databricks-only or absent from OSS MLflow 3.16; the SQLite
  store is queryable directly and `search_traces`/`search_runs` filter syntax covers the same
  investigations, by hand rather than as saved objects.
- **The review app's assignment/status workflow's Databricks-hosted UI** may still be Databricks-only,
  but the OSS review-queue functions themselves are not (Stop 10: `mlflow.genai.review_queues.*` and
  `mlflow.genai.label_schemas.create_label_schema` work against a plain SQLite tracking URI, verified
  live) -- corrected from this tour's earlier reading of them as Databricks-only entirely: both queues,
  all five schemas and every resolvable item are created and set to their target status by M9c.
  MLflow's HUMAN assessments still hold the same 96 lawyer scores as plain feedback rows either way.
- **The Playground UI** is Databricks-only/absent from OSS MLflow 3.16; M9c still gives every registered
  prompt a `model_config` naming its gateway endpoint (Stop 10), so the wiring the Playground would use
  exists even though the page does not.
- **No chart/view/dashboard API** exists in OSS MLflow 3.16 (verified against the installed 3.16.0):
  the `views` step exports `data/reports/mlflow_views.json` (one entry per Braintrust chart: question,
  tag filter, metric) instead of creating a saved chart object; a run comparison is built by hand from
  that list (filter by tag `axis=<question>`, chart `<metric>`).
- **The 67 prompt-variant outputs and their judge scores** live only in Braintrust (the Playground
  pre-run's model text and LLM-judge scores were never mirrored to `data/reports/`); MLflow mirrors the
  row set (case, variant, ledger key) honestly, not the text.

Project: this VM's tracking store (`sqlite:///data/mlflow/mlflow.db`), reached at
`http://127.0.0.1:5000` or through the operator's exe.dev proxy URL, once wired.
