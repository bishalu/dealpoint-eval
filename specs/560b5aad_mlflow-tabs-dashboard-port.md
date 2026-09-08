# Build plan — M9c: every MLflow tab filled, and the eight dashboards ported word for word

**Spec (read-only requirement):** `specs/milestones/m9c.md`
**Authority above it (read-only):** `specs/grilled-product-brief.md` — §2.7 "Braintrust usage (surface, not
source of truth)". Its principle applies unchanged to MLflow: the observability tool is a *surface*, and the
reports in Git decide. Nothing in m9c contradicts the brief; the one thing to keep honouring is that the
committed JSON/artifact is the record and the MLflow UI is the display.

**Precedent to imitate:** `specs/milestones/m9.md` (the mirror), `m9b.md` (the decision tree),
`dealpoint/eval/mlflow_mirror.py`, `dealpoint/eval/mlflow_decision.py`, `tests/test_mlflow_decision.py`.

**Done means:** every item under the spec's "Definition of done" holds on disk;
`uv run pytest -m "gate_m9c and not needs_network" -q` is green; and the full offline suite,
`uv run ruff check .` and `uv run pyright` are green.

**Out of scope:** later milestones, the product UI, the MCP layer, any edit to
`specs/grilled-product-brief.md` or `specs/milestones/m9c.md`, changes to the agent, the judges' rubric,
the scoring definitions, any frozen artifact, or anything written to Braintrust.

---

## 0. Ground truth: what MLflow 3.16.0 actually exposes

Probed directly against the installed `.venv` package on 2026-09-08. **These findings override the spec's
assumed API names and shapes wherever they differ; the differences are to be reported, not silently
resolved.** Re-probe anything marked "probe first" before building on it.

| Need | Reality in 3.16.0 | Consequence |
|---|---|---|
| Session / user on a trace | `mlflow.trace.session` and `mlflow.trace.user` are **`TraceMetadataKey`s (metadata), not tags** | must be set **at trace creation**; see next row |
| Post-hoc metadata on an existing trace | **No API.** `MlflowClient.set_trace_tag` exists; `set_trace_metadata` does **not**. `mlflow.update_current_trace(session_id=, user=, model_id=)` only works inside a live trace context, which the mirror's `client.start_trace` path never enters | the 738 traces **must be re-logged**, exactly as the spec's fallback anticipates |
| Creating a trace with metadata, out of context | `mlflow.tracing.fluent.start_span_no_context(name, span_type=, parent_span=, inputs=, attributes=, tags=, metadata=, experiment_id=, start_time_ns=, ...)` — takes `metadata`, but **no `run_id`** | use it for the root span, then `MlflowClient.link_traces_to_run(trace_ids: list[str], run_id: str)` to restore the run link |
| Registering a `@scorer` (decorator) scorer | **Blocked on non-Databricks.** `Scorer._check_can_be_registered` raises `DECORATOR_SCORER_REGISTRATION_NOT_SUPPORTED_ERROR` when `kind == DECORATOR and not is_databricks_uri(get_tracking_uri())`. Verbatim rationale in-source: *"Custom (@scorer) scorers use exec() during deserialization, which poses a code execution risk."* | the six deterministic scorers **cannot** be registered → `list_scorers()` returns **4, not 10** |
| Registering a `make_judge` judge | `make_judge(...)` returns `InstructionsJudge`, `kind == ScorerKind.INSTRUCTIONS`, which **is** in `_ALLOWED_SCORERS_FOR_REGISTRATION` and has no Databricks guard. `Scorer.register(*, name=None, experiment_id=None)` | the four judges register; the tab is filled with 4 |
| `Scorer.start(sampling_config=...)` | Exists, and has **no Databricks guard**. The real constraint is a **gateway-model guard enforced server-side**: the OSS server routes "gateway provider/model discovery APIs **and scorer invocation**" together (`mlflow/server/handlers.py:7227`), so a started scorer resolves its model through a gateway endpoint | **D12's online rule depends on D15.** Build the judge that starts with `model="gateway:/judge-mistral"`, not `openai:/...`; `gateway` must run before `judges` |
| Label schemas | `mlflow.genai.label_schemas` (**not** `mlflow.genai.labeling`). `create_label_schema(name, *, type: Literal["feedback","expectation"], input, instruction=None, enable_comment=False, title=None, overwrite=False, experiment_id=None)`; `InputNumeric(min_value=None, max_value=None)`; also `InputText`, `InputCategorical`, `list_label_schemas`, `get_label_schema`. Docstring: *"By default the schema is created in the MLflow tracking store, scoped to experiment_id"* | **OSS, scriptable** |
| Review queues | `mlflow.genai.review_queues` (**not** `mlflow.genai.labeling`). `create_review_queue(name, *, queue_type: Literal["user","custom"], users=None, schema_ids=None, experiment_id=None)`; `add_items_to_review_queue(queue_id, *, item_ids)`; `set_review_queue_item_status(queue_id, *, item_id, status: Literal["pending","complete","declined"], completed_by=None)`; `list_review_queue_items`, `get_review_queue`, `list_review_queues` | **OSS, scriptable.** Note: a `"user"` queue's `name` **must be the user id** and inherits all schemas; a named queue like "Lawyer calibration review" must be `queue_type="custom"` with explicit `schema_ids`. `"default"` is reserved. There is **no `description` param** |
| Logged models | `mlflow.create_external_model(name=, source_run_id=, tags=, params=, model_type=, experiment_id=)`; `MlflowClient.create_logged_model(...)`; `mlflow.log_metric(..., model_id=)` / `MlflowClient.log_metric(run_id, ..., model_id=)`; `mlflow.set_active_model(*, name=None, model_id=None)`; `finalize_logged_model(model_id, status)`; `mlflow.log_model_params(params, model_id=)` | scriptable |
| Linking an **existing** trace to a logged model | No API. `mlflow.modelId` is `TraceMetadataKey.MODEL_ID` — metadata, so same constraint as sessions | set at re-log time via `start_span_no_context(metadata={...})`; one re-log serves D11 **and** D14 |
| Registry version from a logged model | `MlflowClient.create_model_version(name, source, run_id=None, tags=None, run_link=None, description=None, await_creation_for=300, model_id=None)` — **has `model_id`** | re-point versions with `source=f"models:/{model_id}"`, `model_id=model_id` |
| AI Gateway | **Fully present in the OSS server.** REST: `/api/3.0/mlflow/gateway/secrets/{create,get,update,delete,list}`, `/gateway/model-definitions/{create,get,list,update,delete}`, `/gateway/endpoints/{create,get,list,update,delete}`. Store methods on both `SqlAlchemyStore` and `RestStore`: `create_gateway_secret(secret_name, secret_value: dict[str,str], provider=None, auth_config=None, created_by=None)`, `create_gateway_model_definition(name, secret_id, provider, model_name, created_by=None)`, `create_gateway_endpoint(name, model_configs: list[GatewayEndpointModelConfig], created_by=None, routing_strategy=None, fallback_config=None, experiment_id=None, usage_tracking=True)`, `list_gateway_endpoints(provider=None, secret_id=None)`. `GatewayEndpointModelConfig(model_definition_id, linkage_type, weight=1.0, fallback_order=None)` | **D15 is scriptable.** The spec's "print the fields and wait for `--gateway-ready`" becomes a documented fallback, not the main path |
| Prompt `model_config` | `mlflow.genai.register_prompt(name, template, commit_message=None, tags=None, response_format=None, model_config: PromptModelConfig | dict | None = None)` | scriptable; pass a plain dict |
| Chart / view API | **None** (M9b's finding, unchanged) | D16 stays "rendered artifact + metrics", as the spec already says |

**Correction owed to the docs:** `docs/mlflow-tour.md` currently states that *"Labeling sessions and the
review app (human review with assignment/status workflow) are Databricks-only"*. That is **false in 3.16**
— review queues and label schemas are OSS and experiment-scoped. D17 must fix that line.

---

## 1. Files to create and touch

### New modules

| File | Purpose | Network / MLflow |
|---|---|---|
| `dealpoint/eval/dashboard_values.py` | **Pure.** D16's local chart evaluator, unit formatting and HTML rendering. No `mlflow`, no `requests`, no I/O beyond the mirror row loaders. This is the testable core of D16 | neither |
| `dealpoint/eval/braintrust_dashboard_snapshot.py` | D16.1 `just braintrust-dashboard-snapshot`: read-only BTQL against Braintrust → `data/reports/braintrust_dashboard_values.json` | Braintrust read-only |
| `dealpoint/eval/mlflow_tabs.py` | D11–D17: `just mlflow-tabs`, steps `sessions, judges, review, versions, gateway, playground, dashboards, overview`; plus `--replay`, `--gateway-smoke`, `--gateway-ready` | MLflow; OpenRouter only under `--gateway-smoke` |
| `tests/test_mlflow_tabs.py` | the `gate_m9c` suite | temp SQLite store |
| `tests/test_dashboard_port.py` | D16 value-equality + render assertions (`gate_m9c`) | none |

### Edited

| File | Change |
|---|---|
| `dealpoint/eval/mlflow_mirror.py` | extract the root-trace creation into one helper that can carry metadata (§2.1); no behaviour change to existing counts |
| `pyproject.toml` | add `"gate_m9c: milestone 9c gate",` to `[tool.pytest.ini_options] markers`, directly after the `gate_m9b` line |
| `.gitignore` | add `!data/reports/braintrust_dashboard_values.json` and `!data/reports/mlflow_tabs_manifest.json` next to the other `!data/reports/mlflow_*` allowlist entries. **`data/reports/*` is ignored by default with an allowlist — without this the committed snapshot silently is not committed** |
| `justfile` | three recipes (§7) |
| `docs/mlflow-tour.md` | D17 (§6) |
| `adws/adw_modules/milestones.py` | **already registered** (commit `332d670`): `m9c`, `spec_path="specs/milestones/m9c.md"`, `gate_marker="gate_m9c"`, `needs_model` defaulted false. Verify only; change nothing |

### House conventions to copy exactly (from `mlflow_mirror.py` / `mlflow_decision.py`)

- Hand-rolled argv parsing, no argparse. `dry_run = "--live" not in argv or "--dry-run" in argv`; `--dry-run` always wins.
- `STEPS = (...)` tuple, dispatch via `globals()[f"step_{name}"](client, live, manifest)`, `--only a,b`.
- Every step fills its `manifest[...]` section **before** `if not live: return`, so a dry run prints real counts.
- `mlflow`, `pandas`, `requests` imported **lazily inside functions** so the offline suite runs without the extra.
- Idempotency by the tag `dealpoint.key`; `_find_run` / `_find_trace` from `mlflow_mirror` are reusable.
- Disk guard: refuse (`return 2`) when `mlflow_mirror.mlflow_dir_bytes() > DISK_GUARD_BYTES`.
- Module-level `os.environ.setdefault("MLFLOW_SQLALCHEMYSTORE_POOL_SIZE", "20")` / `MAX_OVERFLOW="40"` before the first engine creation (`mlflow_decision.py:41-43`) — omitting it deadlocks SQLite under the async trace exporter.
- Manifest written only when live; path behind a `_manifest_path()` indirection so tests can redirect it.
- Periodic `mlflow.flush_trace_async_logging()` every ~20 traces — the export queue defaults to 1000 and silently drops beyond it.

---

## 2. D11 — Sessions

### 2.1 Refactor `mlflow_mirror` first (enabling change, no behaviour change)

`step_traces` currently creates roots with `client.start_trace(..., run_id=...)`, which cannot carry
metadata. Extract:

```python
def _start_root_trace(client, name, *, inputs, tags, metadata, exp_id, start_time_ns, run_id):
    """Root span with trace METADATA (session, user, modelId) set at creation -- MLflow 3.16 has no
    post-hoc trace-metadata setter, so this is the only place these keys can land."""
    from mlflow.tracing.fluent import start_span_no_context
    span = start_span_no_context(name, inputs=inputs, tags={...}, metadata=metadata,
                                 experiment_id=exp_id, start_time_ns=start_time_ns)
    if run_id:
        client.link_traces_to_run([span.trace_id], run_id)
    return span
```

Call it from all three trace loops in `step_traces` (agent, retrieval, prompt-variant), passing
`metadata=session_metadata_for(entry)`. Children and `client.end_trace` are unchanged.

**Verify the refactor before continuing:** `mlflow_mirror`'s existing `gate_m9` suite must stay green, and
a live re-run into a temp store must still produce the same trace/assessment counts and the same
run linkage. If `link_traces_to_run` turns out not to reproduce the linkage `start_trace(run_id=)` gave,
report it and fall back to keeping `client.start_trace` for retrieval traces (which need no session).

### 2.2 The session key

Pure function in `mlflow_tabs.py`:

```python
def session_metadata_for(entry: dict) -> dict[str, str]:
    """`mlflow.trace.session` = the case, `mlflow.trace.user` = the M9 mirror's system label."""
```

- Agent traces: `session = entry["case_id"]` (e.g. `contract_144__q05`); `user = mirror["system_label"]`
  from `braintrust_showroom.metadata_mirror` (e.g. `"D: agent + hybrid + skill"`).
- Retrieval traces: `session = entry["case_id"]` (the query's case, so retrieval groups under it);
  `user = entry["retriever"]`.
- Prompt-variant traces: `session = entry["case_id"]`; `user = entry["prompt_variant"]`.
- Use the constants, not string literals: `from mlflow.tracing.constant import TraceMetadataKey` →
  `TraceMetadataKey.TRACE_SESSION`, `TraceMetadataKey.TRACE_USER`, `TraceMetadataKey.MODEL_ID`.

The redacted twin (`contract_39__redacted_q05`) is a **separate case id**, therefore a separate session
that sits beside the spine session in the tab — which is what the spec asks for. Do not merge them.

### 2.3 `step_sessions`

1. Count traces in the experiment lacking `TraceMetadataKey.TRACE_SESSION` in `trace_info.trace_metadata`.
2. Dry run: print `<n> of <total> traces need re-logging`; fill the manifest; return.
3. Live: for each such trace, re-log it through the mirror's own plan + `_start_root_trace`, **carrying
   every assessment across** (read `trace.info.assessments` from the old trace, re-log each onto the new
   trace id with `mlflow.log_feedback` / `mlflow.log_expectation`, preserving name, value, source type,
   source id and rationale), then `client.delete_traces(experiment_id=..., trace_ids=[old_id])`.
4. Idempotent: a second `--live` finds nothing lacking session metadata and writes nothing.

**Assessments must survive.** The 96 HUMAN assessments in particular are irreplaceable — they are the
lawyer's, not re-derivable. Write the copy-then-delete in that order (copy, verify count on the new trace,
only then delete) so an interrupted run loses nothing.

---

## 3. D12 — Judges, D13 — Review, D14 — Versions, D15 — Gateway/Playground

### `step_judges` (D12)

- Build the four judges with `mlflow_mirror.build_judges()` (unchanged), then
  `judge.register(experiment_id=exp_id)` for each. Idempotent: skip a name already in
  `mlflow.genai.scorers.list_scorers()`. Set the rubric text as the description if the entity carries one;
  if it does not, put the rubric in the judge's `instructions` (it already is) and say so in the manifest.
- **The six deterministic scorers cannot be registered** (§0). Do not swallow this: attempt one
  registration inside a `try`, catch `MlflowException`, and record in the manifest
  `deterministic_registration: {"registered": false, "reason": "<the exception message verbatim>"}`.
  The manifest and the tour both carry it. `list_scorers()` therefore returns **4**.
- Online rule: `judge_professional.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0,
  filter_string="tags.category = 'live-replay'"))`.

  **This is where D12 and D15 meet.** `.start()` is not blocked by Databricks — it is blocked by a
  server-side gateway-model guard. So the judge that starts must be constructed against the gateway
  endpoint D15 creates: `make_judge(name="judge-professional", instructions=<the M5 rubric text>,
  model="gateway:/judge-mistral")` — no `base_url`, because the gateway holds the key now. That is the
  point the spec is making when it says judges "call OpenRouter through the gateway endpoints of D15 when
  those exist, else through `base_url` as in M9": keep `mlflow_mirror.build_judges()`'s `openai:/` +
  `base_url` form for the evaluate-time judges, and add the `gateway:/` form for the one that starts.
  Hence `gateway` precedes `judges` in `STEPS`.

  Still probe, and still degrade honestly: on failure record
  `online_scoring: {"started": false, "reason": "<the exception message verbatim>"}` and keep the
  registered judges — the spec permits this and the tab is filled either way.
- `just mlflow-replay CASE:VARIANT` → `mlflow_tabs.py --replay <case>:<variant>`: look up the stored row via
  `braintrust_showroom.stored_row_index()`, re-log it as a **new** trace (new `dealpoint.key` suffixed with a
  timestamp so idempotency does not suppress it) tagged `category = live-replay`. This creates a trace, not
  a model call. Zero spend.

### `step_review` (D13)

Five schemas via `mlflow.genai.label_schemas.create_label_schema(..., experiment_id=exp_id, overwrite=True)`:

| name | type | input | instruction |
|---|---|---|---|
| `reasoning` | `feedback` | `InputNumeric(min_value=1, max_value=5)` | the M5 rubric anchor text for reasoning |
| `evidence` | `feedback` | `InputNumeric(1, 5)` | the M5 anchor for evidence |
| `trajectory` | `feedback` | `InputNumeric(1, 5)` | the M5 anchor for trajectory |
| `professional` | `feedback` | `InputNumeric(1, 5)` | the M5 anchor for professional |
| `gold_answer` | `expectation` | `InputText()` | the gold answer for the case |

Anchor text comes from `dealpoint/eval/rubric.py` (the same source M5 and the Braintrust human-review
sliders use) — quote it, do not paraphrase, so the two tools show the same words.

Two queues, both `queue_type="custom"` with all four feedback schema ids attached (`"user"` queues force
`name == user id` and inherit every schema):

- `"Lawyer calibration review"` — items = the 12 traces of `braintrust_sync.review_set()`, `users=[<the
  operator's MLflow user>]`, every item left `pending`.
- `"Lawyer-scored packets (24)"` — items = the 24 traces that carry HUMAN assessments, every item
  `set_review_queue_item_status(..., status="complete", completed_by=<the same user>)`.

Resolve the operator's MLflow user from `MLFLOW_TRACKING_USERNAME`, else `getpass.getuser()`; put the
resolved value in the manifest. `add_items_to_review_queue(queue_id, item_ids=[trace_id, ...])`. Idempotent
by queue name via `get_review_queue(name=..., experiment_id=...)` / `list_review_queue_items`.

**The 96 HUMAN assessments are not touched** — they stay exactly as they are.

### `step_versions` (D14)

Fourteen logged models:

- Ten configurations from `mlflow_decision.decision_pools()` — name them `system@model` exactly as M9b
  labels them (`D@gemini-3.1-flash-lite`, `A@haiku`, `D@qwen3.7-flash`, `D@deepseek-v4-flash`, `D@glm`,
  `D@haiku`, plus the four GLM systems of `test-32`).
- Four prompt variants `playground-arm-A-{base,terse,cite-first,abstain-first}`.

For each: `mlflow.create_external_model(name=..., params=<arm config from
`braintrust_showroom.arm_parameter_sets()` + model>, tags=<the factorial schema from
`classify_experiment`>, experiment_id=exp_id)`, then `mlflow.log_metric(k, v, model_id=lm.model_id)` for
M9b's fifteen metrics (reuse `mlflow_decision`'s own metric computation — **never re-derive a number**),
then `finalize_logged_model(model_id, "READY")`.

Trace linkage: `mlflow.modelId` is metadata, so it lands at re-log time. Fold this into D11 — one re-log
pass sets session, user **and** modelId together. Order the steps so `versions` runs **before** `sessions`
and passes its `variant_id -> model_id` map through the manifest.

Registry: for each of the ten configurations, `client.create_model_version("dealpoint-agent",
source=f"models:/{model_id}", model_id=model_id, tags={...}, description=<its verdict line>)`, then
re-point the aliases `champion` / `baseline` / `cost-floor` / `safest` at the new versions with
`set_registered_model_alias`. Idempotent by a `config_hash` tag, exactly as
`mlflow_mirror.step_registry` already does.

**Expect the spec's per-configuration trace counts ("18/32") not to match measurement**: `D@glm` sits in
both pools, so it links 18 + 32. Likewise the spec says the four prompt variants have 67 traces *and*
"18 per prompt variant" (4 × 18 = 72 ≠ 67). **Derive every count from the plan functions, assert the
measured number, and record both the measured and the spec's number in the manifest under
`spec_differences`** — the same pattern `pareto.json` already uses (see `docs/mlflow-tour.md` stop 9).

### `step_gateway` and `step_playground` (D15)

Scriptable (§0). Through `mlflow.tracking._tracking_service.utils._get_store()` or, preferably, the REST
routes so the same code works against SQLite and the HTTP server:

1. One connection: `create_gateway_secret(secret_name="openrouter",
   secret_value={"api_key": os.environ["OPENROUTER_API_KEY"]}, provider="openai",
   auth_config={"base_url": "https://openrouter.ai/api/v1"})` — **probe the exact `provider` /
   `auth_config` field names against the installed proto before writing**; report what they turn out to be.
2. Five model definitions, one per model: `dealpoint-glm` → `z-ai/glm-5.3-flash`, `dealpoint-gemini` →
   `google/gemini-3.1-flash-lite`, `dealpoint-haiku` → `anthropic/claude-haiku-4.5`, `judge-mistral` →
   `mistralai/mistral-small-3.2-24b-instruct`, `judge-bytedance` → `bytedance-seed/seed-2.0-mini`.
3. Five endpoints, same names, each with one `GatewayEndpointModelConfig(model_definition_id=...,
   linkage_type=..., weight=1.0)`.
4. Verify by `list_gateway_endpoints()` and record the five ids in the manifest. Idempotent by name.
5. **Fallback only if a live probe fails:** print the exact Settings-UI fields, exit 0, and accept
   `--gateway-ready` on the next invocation to skip creation and verify by listing. Keep this path; the
   gate must cover it as a branch, not as the default.

`step_playground`: re-register each of the eight prompts with
`mlflow.genai.register_prompt(name, template, model_config={"endpoint": "<name>", "temperature": ...,
"max_tokens": ...})`. Mapping: the agent/arm-A prompts → `dealpoint-glm`; `judge-panel-rubric` and
`judge-calibrated-rubric` → `judge-mistral`; keep `agent-arm-d-skill-injection` on `dealpoint-glm`.
Preserve `mlflow_mirror.PROMPT_ALIASES` — a new version must keep its alias.

`--gateway-smoke` (opt-in, **off by default**): one 1-token chat call per endpoint through
`OpenRouterClient(milestone_tag="m9c")` (`dealpoint/llm/client.py` — the client appends to
`data/results/spend_ledger.jsonl` itself, so the ledger entry comes for free). Print the estimate before
the first call and cap at five calls. Never reachable from any step, any test, or any default run.

---

## 4. D16 — The eight dashboards, word for word

### 4.1 `dealpoint/eval/dashboard_values.py` — pure evaluator

Source of truth: `braintrust_cockpit.DASHBOARDS` (a `tuple[tuple[name, verdict_text, chart_keys], ...]`)
and `braintrust_cockpit.chart_catalogue()` (a `dict[key, {"title", "measure", "group_by", "filters",
optional "unit"}]`). Row source: `braintrust_showroom.stored_row_index()` + `mirror_for()` for agent rows,
`retrieval_log_rows()` for retrieval, `prompt_variant_log_rows()` for prompt variants — the same rows the
mirror uses, never re-read from raw files.

**Two measured traps in the row sources — both verified on disk, both silent if missed:**

1. **`stored_row_index()` double-keys every row.** It returns **426 keys over 262 unique rows**, because
   each row is keyed under *both* variant-id conventions (`D@glm` **and** `D@z-ai/glm-5.3-flash`; see
   `_judged_variant_id_for`). Iterating `.values()` counts most rows twice, which corrupts `avg` and
   `percentile` silently (`sum(a)/sum(b)` ratios happen to survive, which makes the bug harder to spot).
   **De-duplicate by row identity before evaluating any chart**, and assert the de-duplicated count in a
   test so a future change to the index cannot reintroduce it.
2. **Prompt-variant rows carry no judge scores on disk.** `prompt_variant_trace_plan()` yields 67 rows
   whose metadata keys are exactly `case_id, category, gold_answer, prompt_variant, question_id` — there
   is no `judge_professional` / `judge_evidence` / `judge_reasoning`. Those scores were never mirrored out
   of Braintrust (`docs/mlflow-tour.md` already records this gap). Therefore the local evaluator **cannot**
   reproduce the four charts that measure `avg(metadata.judge_*)` under `metadata.category =
   'prompt-variant'`: `prompt_professional`, `prompt_evidence`, `prompt_reasoning` — i.e. the whole
   "Which prompt?" dashboard — plus `prompt_evidence` on the overview dashboard.

   **Handle it explicitly, do not paper over it** (M9 principle 6: gaps are named). The evaluator returns
   an explicit "no local rows" result for those charts rather than an empty list that silently reads as
   zero. Render them from the **snapshot** values with a visible provenance line on the page — *"values
   read from Braintrust; the judges' scores on these 67 rows were never mirrored to disk"* — and exclude
   exactly those chart keys from the 1e-9 equality assertion, listing them by name in the test so the
   exclusion is auditable rather than a blanket skip. Record the same list in the manifest's
   `spec_differences`.

The grammar the catalogue actually uses (nothing more is needed; reject anything else loudly rather than
silently returning zero):

- **Measures** — `avg(metadata.X)`; `sum(metadata.A) / sum(metadata.B)`;
  `percentile(metadata.wall_s, 0.9)`. A measure may be a **string** or a **list of
  `{"btql": ..., "name": ...}`** (the judge charts), where `name` is the row label shown.
- **Filters** — a list of strings, ANDed. Within one string: `and`, `or`, parentheses, and the
  comparisons `=`, `!=`, `>=` against integer or single-quoted string literals. Every field is
  `metadata.<name>`.
- **Group by** — a list of `metadata.<field>` (or empty, for the multi-measure judge charts, where each
  measure is its own row). Display names via `braintrust_cockpit.GROUP_DISPLAY`
  (`metadata.system_label → "system"`, `metadata.model_label → "model"`,
  `metadata.variant_label → "system@model"`, `metadata.retriever → "retriever"`,
  `metadata.prompt_variant → "prompt variant"`), fallback `g.rsplit(".", 1)[-1]`.
- **Null handling** — `avg` and `percentile` skip `None`; `sum` treats `None` as 0. This must match
  Braintrust; if the snapshot disagrees on a chart, the snapshot is the evidence and the evaluator is wrong.
- **Ordering** — every chart is a toplist with `sortByOptions: {type: "value", direction: "desc"}`
  (`braintrust_cockpit._chart_rest_definition`). Ties: break by group name ascending, and note it.

API:

```python
def evaluate_chart(chart: dict, rows: list[dict]) -> list[tuple[str, float]]:  # ordered, desc by value
def evaluate_dashboards() -> dict   # {dashboard: {chart_key: {"title":..., "rows": [[label, value], ...]}}}
def format_value(value: float, unit: str) -> str
def render_dashboard_html(name: str, verdict: str, charts: list[dict]) -> str
```

Unit resolution copies `_chart_rest_definition` exactly:
`unit = chart.get("unit") or ("cost" if "usd" in str(chart["measure"]) else "percent")`.
Formatting: `percent` → `f"{v * 100:.1f}%"`; `cost` → `f"${v:.4f}"`; `duration` → `f"{v:.1f}s"`;
`count` → `f"{v:,.1f}"`. Put these four rules in one dict so the test and the renderer share them.

`render_dashboard_html` produces **one self-contained page, no external assets**: the dashboard's exact
name as `<h1>`, the VERDICT text verbatim as a `<p>`, then one section per chart with the exact `title`
verbatim and a ranked horizontal bar list (a `<div>` per row with an inline `width: <pct>%` style), same
labels, same order, formatted values. HTML-escape every interpolated string.

### 4.2 `braintrust_dashboard_snapshot.py` — the Braintrust side

`just braintrust-dashboard-snapshot`. **Read-only: it issues BTQL `SELECT`s and writes nothing to
Braintrust — no scores, no data, no meter.**

Reuse `dealpoint/eval/btql.py`: `BTQL_ENDPOINT = "https://api.braintrust.dev/btql"`,
`run_btql(query, *, api_key, timeout=60)` (bearer auth from `BRAINTRUST_API_KEY`),
`parse_btql_result(payload) -> (rows, count)`.

For every chart in `chart_catalogue()` as placed by `DASHBOARDS`, translate `measure` / `group_by` /
`filters` into one BTQL query over the project logs and record the rows. Respect the 20-queries-per-minute
limit: a simple token-bucket sleep (≥ 3 s between queries) plus bounded exponential backoff on HTTP 429,
at most three retries. There are ~60 chart placements, so budget a few minutes.

Output `data/reports/braintrust_dashboard_values.json`, committed:

```json
{"generated_at": "...", "project": "...",
 "dashboards": [{"dashboard": "Which model?", "chart_key": "mod_safe", "title": "<verbatim>",
                 "rows": [{"group": "<display name Braintrust shows>", "value": 0.8888888888888888}]}]}
```

The module is `needs_network`; nothing in the gate calls it. **The gate reads the committed file.**

### 4.3 `step_dashboards` in `mlflow_tabs.py`

Eight runs named `dashboard/<name>`, tagged `axis=dashboard` and `dealpoint.key=dashboard/<name>`:

- artifact: the rendered HTML, via `mlflow.log_text(html, "dashboard.html")` (or
  `client.log_artifact` on a temp file — MLflow renders `.html` artifacts inline);
- metrics: every chart row as `<chart_key>.<row_label>`. **Sanitize the metric key** — MLflow metric names
  reject some characters and row labels are long, human strings like `"D: agent + hybrid + skill"`. Define
  one `_metric_key(chart_key, row_label)` (replace disallowed characters with `_`, truncate to MLflow's
  limit) and write the full mapping into the run's `dashboard.html` and the manifest so a truncated key is
  still traceable back to its label. Probe MLflow's actual key rules before choosing the rule.
- run description (`mlflow.note.content`) = the dashboard's VERDICT text verbatim.

Idempotent by `dealpoint.key`; a second `--live` re-uses the run and re-logs nothing new.

### 4.4 `step_overview` (part of D17)

Set the experiment description to the existing overview text **plus** a "Dashboards" line listing the eight
run names, the decision tree (`decision/judged-18`, `decision/test-32`), the registry
(`dealpoint-agent` and its four aliases) and the two review queues.

---

## 5. Steps and order

`STEPS = ("gateway", "playground", "judges", "review", "versions", "sessions", "dashboards", "overview")`

`versions` before `sessions` (the re-log needs the `model_id` map); `gateway` before `playground` (prompts
name their endpoint); `judges` before `review` is irrelevant but keeps the tour's order.

---

## 6. D17 — Tour and Overview (`docs/mlflow-tour.md`)

1. **New section after the header block, before "Act I": a tabs table.** One row per sidebar entry —
   Overview; Observability: Traces, Sessions; Evaluation: Judges, Review, Datasets, Evaluation runs;
   Prompts & versions: Playground, Prompts; Agent versions; AI Gateway — with three columns: *what fills
   it*, *from which Braintrust object*, *what MLflow does with it that Braintrust cannot*. The last column
   for the six the spec names: Sessions — one question as a session; Judges — registered and startable;
   Review — queues with schemas; Versions — traces linked to the configuration; Gateway — one place for
   every model key; Playground — prompts with their endpoint pre-filled.
2. **A dashboards section** listing the eight `dashboard/<name>` runs with the eight Braintrust dashboard
   links beside them, and one line stating that every value was checked against
   `braintrust_dashboard_values.json` to 1e-9.
3. **Fix the "What only Braintrust has here" section**: delete the claim that labeling sessions and the
   review app are Databricks-only (false in 3.16 — see §0) and replace it with the two gaps that are
   real: the ranked single-metric lists over *logs* attached to a dashboard object, and continuous online
   scoring (if `.start()` proved unavailable, say that here too).
4. **Add the honest negatives** discovered here: the six deterministic `@scorer` scorers cannot be
   registered on a non-Databricks tracking URI (quote MLflow's reason), so the Judges tab lists four;
   and trace metadata cannot be set post-hoc, which is why the 738 traces were re-logged.
5. Do not change any existing number.

---

## 7. `justfile` recipes

Add under a new `# ── dealpoint (M9c every tab + dashboard port) ──` banner, matching the existing
comment-then-recipe style:

```
# M9c: fill every MLflow tab (sessions, judges, review queues, agent versions, gateway, playground,
# dashboards). Dry run by default; --live writes. --gateway-smoke is opt-in and costs cents.
mlflow-tabs *ARGS:
    uv run python -m dealpoint.eval.mlflow_tabs "$@"

# M9c: re-log one stored trace as a new `live-replay` trace so an online-scoring rule can land on it
mlflow-replay CASE_VARIANT:
    uv run python -m dealpoint.eval.mlflow_tabs --live --replay {{CASE_VARIANT}}

# M9c: read-only BTQL snapshot of every Braintrust dashboard chart value (writes nothing to Braintrust)
braintrust-dashboard-snapshot *ARGS:
    uv run python -m dealpoint.eval.braintrust_dashboard_snapshot "$@"

# milestone 9c acceptance gates only, offline
gate-m9c:
    uv run pytest -m "gate_m9c and not needs_network and not needs_model" -q
```

---

## 8. The gate — `tests/test_mlflow_tabs.py` + `tests/test_dashboard_port.py`

Copy `tests/test_mlflow_decision.py`'s proven shape exactly:

```python
mlflow = pytest.importorskip("mlflow")
pytestmark = pytest.mark.gate_m9c
```

with module-scoped `tracking_uri` (`f"sqlite:///{tmp_path_factory.mktemp('m9c')}/m.db"`), `client`,
`monkeypatch_module` and a `live_manifest` fixture that sets `MLFLOW_TRACKING_URI`, redirects every
artifact path to a tmp dir, and runs every step live once.

Required tests:

1. **Dry-run counts** — 738 (or the re-logged 738) traces all carrying `mlflow.trace.session`; **4**
   registered scorers with the six deterministic ones recorded as unregisterable and the reason present;
   5 label schemas; 2 review queues with 12 and 24 items; 14 logged models with their **measured**
   linked-trace counts; 5 endpoint definitions; 8 prompts each with a `model_config` naming an endpoint;
   8 dashboard runs with an HTML artifact.
2. **Idempotency** — count runs, traces, logged models, model versions, schemas, queue items, endpoints
   and prompt versions before and after a second `--live`; **zero new objects**.
3. **Session shape** — the spine session `contract_144__q05` holds exactly the comparable variants for
   that case (assert the set, derived from `stored_row_index`, not a hard-coded list), and
   `contract_39__redacted_q05` is its own session beside it.
4. **Assessment survival** — after the re-log, the spine trace's assessments equal the pre-re-log set by
   name and value, and the total HUMAN assessment count in the experiment is still 96.
5. **D16 value equality** — for every dashboard, every chart, every row: the local evaluator's value
   equals `braintrust_dashboard_values.json` within `1e-9`, **and the toplist ordering matches**. Compare
   the full ordered list of `(label, value)` pairs, not a set.
6. **Render fidelity** — each rendered HTML contains, verbatim, the dashboard name from `DASHBOARDS`, its
   full VERDICT description, and every chart title from `chart_catalogue()`. Assert with `in` against the
   literal strings pulled from the modules, so a wording change in the cockpit cannot drift the port.
7. **Formatting** — `format_value` for each of the four units, on known values.
8. **Zero model calls** — patch `dealpoint.llm.client.OpenRouterClient` **and** the gateway invoke path
   with a `Mock(side_effect=AssertionError)`, run every step live, and assert neither was constructed or
   called. Copy `test_mlflow_decision.py`'s existing patching test as the template.
9. **`--gateway-smoke` is off by default** — a run without the flag never reaches the smoke function.
10. **Tour** — `docs/mlflow-tour.md` contains a row for each of the thirteen sidebar entries and names all
    eight `dashboard/<name>` runs.

`step_sessions` re-logs traces and is the slow step. Mirror `mlflow_mirror`'s existing convention and
accept `--limit N` on it, dispatched as `fn(client, live, manifest, limit=limit)` exactly as `main()`
already does for `traces` and `assessments`; the live fixture then caps the re-log the way
`tests/test_mlflow_mirror.py` already caps its own. Counts asserted in test 1 come from the pure plan
functions, which are unaffected by the cap.

Nothing in the gate touches the network. `braintrust_dashboard_snapshot` tests, if any, are
`@pytest.mark.needs_network` and excluded by the gate command.

---

## 9. Budget

**$0 by default.** The only spend is `--gateway-smoke`: at most five calls, cents, ledgered with
`milestone_tag: m9c` via `OpenRouterClient`. `braintrust-dashboard-snapshot` is read-only against
Braintrust — no scores, no writes, no meter. MLflow artifacts stay under `data/mlflow/` (gitignored)
behind the 200 MB disk guard.

---

## 10. Verification, in order

```
uv run ruff check .
uv run pyright
uv run pytest -m "gate_m9" -q                      # the refactor of mlflow_mirror broke nothing
uv run pytest -m "gate_m9b" -q
uv run python -m dealpoint.eval.mlflow_tabs        # dry run: the counts of §8.1
just mlflow-tabs --live                            # against the running server
just mlflow-tabs --live                            # again: zero new objects
uv run pytest -m "gate_m9c and not needs_network" -q
uv run pytest -m "not needs_network and not needs_model" -q
```

Then open the UI at `http://127.0.0.1:5000` and confirm by eye: Sessions lists `contract_144__q05` with
its variants; Judges lists four; Review shows one open and one complete queue; Versions lists fourteen
logged models; AI Gateway lists five endpoints; Playground loads `arm-a-prompt-cite-first` on
`dealpoint-glm`; each `dashboard/<name>` run renders its HTML inline.

`just braintrust-dashboard-snapshot` is run **once**, by hand, with the Braintrust key present, before the
D16 tests can pass; its output is committed. If the key is absent the builder must stop and say so rather
than fabricating the file — a hand-written snapshot would make test 5 assert the evaluator against itself.

---

## 11. Differences between the spec and reality, to be reported (not silently resolved)

Record each in `data/reports/mlflow_tabs_manifest.json` under a `spec_differences` block **and** in one
line of `docs/mlflow-tour.md`, following the precedent of `pareto.json`'s `spec_differences` (M9b).

1. **D12, "ten registered scorers":** only the four `make_judge` judges are registerable; MLflow 3.16
   refuses `@scorer` decorator scorers on a non-Databricks tracking URI, by design, for code-execution
   safety. `list_scorers()` returns 4.
2. **D13, function locations:** `create_label_schema` lives in `mlflow.genai.label_schemas` and the review
   queue functions in `mlflow.genai.review_queues`, not in `mlflow.genai.labeling` as the spec writes. Both
   are OSS. `create_review_queue` has no `description` parameter, and named queues must be
   `queue_type="custom"`.
3. **D11/D14, "if the metadata cannot be merged":** it cannot — trace metadata is immutable after
   creation in 3.16 — so the 738 traces are re-logged, which is the spec's own stated fallback.
4. **D14, linked-trace counts:** "18/32 per configuration" does not hold for `D@glm`, which is in both
   pools; and "67 traces" for four prompt variants cannot also be "18 per prompt variant". Report the
   measured numbers.
5. **D15, "if only the Settings UI can create them":** it is not — the OSS server exposes
   `/api/3.0/mlflow/gateway/{secrets,model-definitions,endpoints}`, so the step is scripted and the
   `--gateway-ready` path is a fallback.
5b. **D12, "if `.start()` proves Databricks-only on 3.16":** it is not Databricks-gated at all. The guard is
   that a started scorer's model must resolve through a gateway endpoint, which makes the online rule a
   *dependent* of D15 rather than an unavailable feature. Report it that way.
6. **Documentation correction:** `docs/mlflow-tour.md` currently calls review queues Databricks-only. It
   is wrong for 3.16 and D17 fixes it.
7. **D16, "every row of every chart":** four chart placements measure judge scores on prompt-variant rows,
   and those scores exist only in Braintrust. They are ported from the snapshot with stated provenance and
   named as excluded from the local-equality assertion (§4.1). Every other chart is checked to 1e-9.

No difference with `specs/grilled-product-brief.md` was found: §2.7's "surface, not source of truth"
principle is honoured — the committed `braintrust_dashboard_values.json` and the rendered artifacts are
the record, and MLflow displays them.
