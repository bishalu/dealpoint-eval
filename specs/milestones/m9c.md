# M9c: every MLflow tab filled, and the eight dashboards ported word for word

**Authority:** `specs/milestones/m9.md` and `m9b.md` (principles unchanged: same rows, one command, idempotent,
nothing metered, gaps named), the Braintrust showroom as it stands on 2026-09-08 (`braintrust_cockpit.DASHBOARDS`,
`chart_catalogue()`, `braintrust_showroom.metadata_mirror`), and the engineer's directive of 2026-09-08: **bring
everything we have in Braintrust over to every MLflow tab (Overview, Observability: Traces, Sessions; Evaluation:
Judges, Review, Datasets, Evaluation runs; Prompts & versions: Playground, Prompts; Agent versions; AI Gateway),
and get the customised A/B/C/D dashboards into MLflow word for word, result for result.**

Additive to M9 and M9b. Reads the same stored rows through the same pure functions. **No model calls** except one
opt-in gateway smoke test. Verified on the live server on 2026-09-08 before this spec was written: 738 traces
with no session metadata, zero registered scorers, zero logged models (19 registry versions exist), 18 datasets,
8 prompts, 10 evaluation runs, no gateway endpoints, and MLflow 3.16 exposes no chart-view API (M9b's finding).

## 1. Deliverables, one per empty tab, then the dashboards

### D11. Sessions (Observability > Sessions)

A session is one contract question across every configuration that answered it: every trace gets
`mlflow.trace.session = <case_id>` and `mlflow.trace.user = <system_label>` (the M9 mirror's label, e.g.
"D: agent + hybrid + skill"), so the Sessions tab lists `contract_144__q05` with its six agent traces and its
redacted twin `contract_39__redacted_q05` beside it, and retrieval traces group under their query's case. Set
through `mlflow.update_current_trace` / the tracing context at logging time (`docs: mlflow.trace.session`); if the
metadata cannot be merged onto the 738 existing traces, re-log them (free) and delete the unlinked originals,
keeping every assessment. Test: the spine session holds exactly the comparable variants for that case.

### D12. Judges (Evaluation > Judges)

Register every scorer to the experiment so the tab lists them: the six deterministic `@scorer` functions from M9
and the four `make_judge` judges (`scorer.register(experiment_id=...)`; `mlflow.genai.scorers.list_scorers` must
return ten). Each carries its rubric text as description. Then the Braintrust online-scoring rule, mirrored:
`judge-professional` started with sampling over new traces (`.start(sampling_config=...)`), restricted, as in
Braintrust, to traces tagged `category = live-replay`; `just mlflow-replay CASE:VARIANT` re-logs one stored trace
as a new trace with that tag so the demo can show a score landing. If `.start()` proves Databricks-only on 3.16,
record that in the manifest and the tour and keep the registered judges (the tab is still filled). Judges call
OpenRouter through the gateway endpoints of D15 when those exist, else through `base_url` as in M9.

### D13. Review (Evaluation > Review)

Open-source review queues (MLflow >= 3.14): four label schemas, `reasoning`, `evidence`, `trajectory`,
`professional`, numeric 1 to 5, type feedback, descriptions = the M5 rubric anchors (the same four sliders
Braintrust's Human review carries), plus one expectation schema `gold_answer`. One queue, "Lawyer calibration
review", holding the 12 review-set traces (`braintrust_sync.review_set`), assigned to the operator's MLflow user;
a second queue, "Lawyer-scored packets (24)", holding the 24 traces the lawyer scored, already complete, so
the tab shows both an open queue and a finished one. The lawyer's 96 HUMAN assessments stay as they are.
Functions: `create_label_schema`, `create_review_queue`, `add_items_to_review_queue`, `set_review_queue_item_status`.

### D14. Agent versions (Versions tab)

One `LoggedModel` per comparable configuration (the ten of M9b's decision tree, named `system@model`), params =
the arm config (`arm_parameter_sets`) and model, metrics = M9b's fifteen, tags = the factorial schema; every trace
of that configuration linked to it (`mlflow.set_active_model` at re-log time, or the client's trace-to-model link
if 3.16 exposes one for existing traces). The registry versions of `dealpoint-agent` are re-pointed at these
logged models so `champion`, `baseline`, `cost-floor` and `safest` resolve to a version with linked traces.
The four `playground-arm-A-*` prompt variants are logged models too, linked to their 67 traces.

### D15. AI Gateway and Playground

One LLM connection, `openrouter` (OpenAI-compatible provider, base URL `https://openrouter.ai/api/v1`, key from
`OPENROUTER_API_KEY`), and five chat endpoints named for their role: `dealpoint-glm` (z-ai/glm-5.3-flash),
`dealpoint-gemini` (google/gemini-3.1-flash-lite), `dealpoint-haiku` (anthropic/claude-haiku-4.5), `judge-mistral`
(mistralai/mistral-small-3.2-24b-instruct), `judge-bytedance` (bytedance-seed/seed-2.0-mini). Created through
the gateway SDK/REST where 3.16 exposes it; if only the Settings UI can create them, the step prints the exact
fields for the operator, waits for `--gateway-ready`, and verifies by listing endpoints. Every registered prompt
gets a `model_config` naming its endpoint, so the Playground (`/playground`) loads `arm-a-prompt-cite-first` on
`dealpoint-glm` and `judge-panel-rubric` on `judge-mistral` with sampling parameters filled in. Opt-in
`--gateway-smoke`: one call per endpoint (cents), recorded in the spend ledger with `milestone_tag: m9c`.

### D16. The eight dashboards, word for word, result for result

MLflow has no chart API, so the port is a rendered artifact per dashboard plus the numbers as metrics, and it is
checked against Braintrust value for value.

1. **Snapshot** (`just braintrust-dashboard-snapshot`, read-only, REST BTQL, respects the 20-queries-per-minute
   limit with backoff): for every chart in `chart_catalogue()` as placed by `DASHBOARDS`, run its measure, group
   and filters against the Braintrust logs and write `data/reports/braintrust_dashboard_values.json`
   (dashboard, chart key, title, rows [group -> value] with the display names Braintrust shows). Committed.
2. **Local evaluator**: a pure function that turns a catalogue chart (measure expression, group_by, filters,
   unit) into the same numbers from the mirror rows on disk (`stored_row_index` + `mirror_for`, the retrieval and
   prompt-variant rows), covering `avg`, `sum/sum`, `percentile`, the `and`/`or`/`=`/`!=`/`>=` filters the
   catalogue uses, and the display-name and group-name mappings (`GROUP_DISPLAY`, measure `name`).
   **Test: every row of every chart equals the snapshot within 1e-9**, including the toplist ordering.
3. **Render**: one self-contained HTML page per dashboard (no external assets) with the dashboard's exact name,
   its VERDICT description verbatim, and each chart as a ranked horizontal bar list with the exact title, the
   same row labels, the same values formatted the way Braintrust's unit type formats them (percent, cost,
   duration, count), in the same order. Logged as artifacts on eight runs `dashboard/<name>` (tag
   `axis=dashboard`) whose metrics are every chart row (`<chart_key>.<row>`), so the runs table can chart them
   too. MLflow renders HTML artifacts inline; the tour links each run.
4. The experiment description gains a "Dashboards" line listing the eight runs; the eight Braintrust dashboard
   links stay beside them in the tour.

### D17. Tour and Overview

`docs/mlflow-tour.md` gains a tabs table: for each of the thirteen sidebar entries, what fills it, from which
Braintrust object, and the one line of what MLflow does with it that Braintrust cannot (Sessions: one question as
a session; Judges: registered and startable; Review: queues with schemas; Versions: traces linked to the
configuration; Gateway: one place for every model key; Playground: prompts with their endpoint pre-filled).
The experiment description (Overview) is updated to point at the dashboard runs, the decision tree, the
registry and the review queues.

## 2. Gate (`gate_m9c`, offline, temporary SQLite store, real client)

- dry run counts: 738 (or re-logged 738) traces all carrying `mlflow.trace.session`; 10 registered scorers;
  5 label schemas and 2 review queues with 12 and 24 items; 14 logged models with linked-trace counts
  (18/32 per configuration, 18 per prompt variant); 5 endpoint definitions; 8 prompts with `model_config`;
  8 dashboard runs with artifacts; `--live` twice creates nothing new the second time;
- D16 value equality for every chart row against `braintrust_dashboard_values.json`;
- the rendered HTML of each dashboard contains the dashboard name, the verdict text and every chart title
  verbatim from `DASHBOARDS` / `chart_catalogue()`;
- zero model calls unless `--gateway-smoke`: a test patches the OpenRouter client and the gateway invoke path
  and asserts neither is called by any step;
- ruff, pyright, full offline suite.

## 3. Budget

$0 by default. `--gateway-smoke` at most five calls, cents, ledgered. `braintrust-dashboard-snapshot` is
read-only against Braintrust (no scores, no data written).

## 4. Out of scope

Changes to the agent, the judges' rubric, the scoring definitions, any frozen artifact, or anything in Braintrust;
re-scoring traces with models; Databricks-only monitoring or labeling sessions (review queues are the OSS surface).

## 5. Registration

`adws/adw_modules/milestones.py`: `m9c`, `spec_path="specs/milestones/m9c.md"`, `gate_marker="gate_m9c"`,
`needs_model=False`. The tracking server is running (`just mlflow-server`) and reachable at
`http://127.0.0.1:5000`; `MLFLOW_TRACKING_URI` defaults to it.
