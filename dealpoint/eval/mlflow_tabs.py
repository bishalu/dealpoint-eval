"""M9c D11-D15: every empty MLflow sidebar tab filled from the same stored rows M9/M9b already mirror
(`specs/milestones/m9c.md`). Additive to `mlflow_mirror`/`mlflow_decision`: no model calls, no new
scoring, nothing re-derived that a pure function upstream already computed.

Dry run is the default; `--live` executes against `MLFLOW_TRACKING_URI` (default
`http://127.0.0.1:5000`); `--dry-run` always wins over `--live`, matching every other M9* module.
`mlflow` is imported lazily so the planning functions work without the `mlflow` extra installed.

Probed live against this install's MLflow 3.16.0 (2026-09-08), overriding the spec's assumed API
shapes wherever they differ (recorded below and in `spec_differences` on the manifest, never silently
resolved):

- `mlflow.trace.session`/`mlflow.trace.user` are trace METADATA keys (`TraceMetadataKey`), not tags,
  and 3.16 has no post-hoc metadata setter -- `MlflowClient.set_trace_metadata` does not exist. The
  738 traces are therefore re-logged via `mlflow.tracing.fluent.start_span_no_context(metadata=...)`
  (the only path that can set metadata at creation), with every assessment copied across and the
  count verified BEFORE the original is deleted, and the run link restored with
  `MlflowClient.link_traces_to_run`. This one re-log pass also carries D14's `mlflow.modelId`.
- The six deterministic `@scorer` functions from M9 cannot be registered on a non-Databricks tracking
  URI (`Scorer._check_can_be_registered` raises `DECORATOR_SCORER_REGISTRATION_NOT_SUPPORTED_ERROR`,
  "Custom (@scorer) scorers use exec() during deserialization, which poses a code execution risk.").
  Every attempt and its verbatim exception is recorded; `list_scorers()` (the measured count, never
  assumed) returns 4 -- the `make_judge` judges only.
- `mlflow.genai.label_schemas.create_label_schema`/`mlflow.genai.review_queues.*` (not
  `mlflow.genai.labeling`, which is Databricks-only) are OSS and scriptable on this install:
  `create_review_queue`, `add_items_to_review_queue`, `set_review_queue_item_status`,
  `list_review_queue_items` all work against a SQLite tracking URI.
- The AI Gateway's `create_{secret,model_definition,endpoint}`/`list_gateway_endpoints` store methods
  (reachable via `MlflowClient()._tracking_client.store`) are present and scriptable on this install
  -- `mlflow.gateway.client` does not exist. The Settings-UI print is a fallback for when the store
  call itself raises, not the default path.

    uv run python -m dealpoint.eval.mlflow_tabs                 # dry run, every step
    uv run python -m dealpoint.eval.mlflow_tabs --live          # execute
    uv run python -m dealpoint.eval.mlflow_tabs --live --only sessions,judges
    uv run python -m dealpoint.eval.mlflow_tabs --live --gateway-ready   # skip gateway creation, verify only
    uv run python -m dealpoint.eval.mlflow_tabs --live --gateway-smoke  # + one call per endpoint (cents)
    uv run python -m dealpoint.eval.mlflow_tabs replay CASE:VARIANT --live   # D12's mlflow-replay
"""

from __future__ import annotations

import getpass
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MLFLOW_EXPERIMENT = "dealpoint-eval"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
MANIFEST_PATH = Path("data/reports/mlflow_tabs_manifest.json")
REGISTERED_MODEL = "dealpoint-agent"

STEPS = ("models", "gateway", "playground", "judges", "review", "sessions", "overview")

JUDGE_DIMS = ("reasoning", "evidence", "trajectory", "professional")

# D15: the gateway connection and its five role-named endpoints (spec text, verbatim).
GATEWAY_CONNECTION = "openrouter"
GATEWAY_BASE_URL = "https://openrouter.ai/api/v1"
GATEWAY_ENDPOINTS = {
    "dealpoint-glm": "z-ai/glm-5.3-flash",
    "dealpoint-gemini": "google/gemini-3.1-flash-lite",
    "dealpoint-haiku": "anthropic/claude-haiku-4.5",
    "judge-mistral": "mistralai/mistral-small-3.2-24b-instruct",
    "judge-bytedance": "bytedance-seed/seed-2.0-mini",
}
# prompt name -> the endpoint its model_config should name (spec D15; every other registered prompt
# defaults to its own model's endpoint, or dealpoint-glm when the prompt has no single model of its own).
PROMPT_ENDPOINT_OVERRIDES = {
    "arm-a-prompt-cite-first": "dealpoint-glm",
    "judge-panel-rubric": "judge-mistral",
}


def _manifest_path() -> Path:
    return MANIFEST_PATH


# ================================================================================================
# D11: sessions
# ================================================================================================


def session_plan() -> list[dict]:
    """One `{case_id, variant_id, session, user, category}` per one of the 738 traces M9 already
    mirrors (`mlflow_mirror.agent_trace_plan`/`retrieval_trace_plan`/`prompt_variant_trace_plan`):
    `mlflow.trace.session = case_id` (one contract question across every configuration that answered
    it), `mlflow.trace.user = system_label` (the M9 mirror's label, e.g. "D: agent + hybrid + skill")
    for agent traces, the retriever name for retrieval traces, and `prompt:<variant>` for
    prompt-variant traces."""
    from dealpoint.eval import mlflow_mirror as mm

    ctx = mm.mirror_context()
    plan: list[dict] = []
    for e in mm.agent_trace_plan():
        row = e.get("row") or {}
        mirror = mm.mirror_for(e["case_id"], e["variant_id"], row, ctx, category=e["category"])
        plan.append({"case_id": e["case_id"], "variant_id": e["variant_id"], "category": e["category"],
                     "session": e["case_id"], "user": mirror["system_label"], "comparable": mirror["comparable"],
                     "key": f"{e['case_id']}:{e['variant_id']}:{e['category']}"})
    for r in mm.retrieval_trace_plan():
        plan.append({"case_id": r["case_id"], "variant_id": r["retriever"], "category": "retrieval",
                     "session": r["case_id"], "user": r["retriever"], "comparable": None,
                     "key": f"{r['case_id']}:{r['retriever']}:retrieval"})
    for pv in mm.prompt_variant_trace_plan():
        plan.append({"case_id": pv["case_id"], "variant_id": pv["variant"], "category": "prompt-variant",
                     "session": pv["case_id"], "user": f"prompt:{pv['variant']}", "comparable": None,
                     "key": f"{pv['case_id']}:{pv['variant']}:prompt-variant"})
    return plan


def sessions_by_case(plan: list[dict] | None = None) -> dict[str, list[dict]]:
    plan = plan if plan is not None else session_plan()
    out: dict[str, list[dict]] = {}
    for e in plan:
        out.setdefault(e["session"], []).append(e)
    return out


# --- D14 folded in here: mlflow.modelId is trace metadata too, so it lands in the same re-log pass ---


def _config_model_name(config: str, pool: str) -> str:
    """`LoggedModel.name` rejects '.', so `D@gemini-3.1-flash-lite (judged-18)` becomes
    `D@gemini-3_1-flash-lite (judged-18)` -- the only character `_validate_logged_model_name`
    forbids that this naming scheme ever produces."""
    return f"{config} ({pool})".replace(".", "_")


def trace_key_to_model_name() -> dict[str, str]:
    """Every session-plan trace key -> the `LoggedModel` name it should link to (D14), derived from
    the exact same rowsets `mlflow_decision.decision_pools()`/`logged_model_plan()` use -- never a
    separate re-derivation. `D@glm` sits in both `judged-18` and `test-32` (its `judged` rows are a
    subset of test-32's GLM rowset too), and `mlflow.modelId` is a single-valued metadata key, so a
    shared trace key resolves to `judged-18`'s model (the pool the four registry aliases use); the
    measured overlap is recorded by `step_models` as a `spec_difference`, not silently absorbed."""
    from dealpoint.eval import mlflow_decision as md

    pools = md.decision_pools()
    out: dict[str, str] = {}
    for pool in ("judged-18", "test-32"):
        for config, entries in pools.get(pool, {}).items():
            model_name = _config_model_name(config, pool)
            for e in entries:
                key = f"{e['case_id']}:{e['variant_id']}:{e['category']}"
                out.setdefault(key, model_name)
    for p in logged_model_plan():
        if p["kind"] == "prompt-variant":
            for key in p.get("trace_keys", []):
                out.setdefault(key, p["model_name"])
    return out


def step_sessions(client, live: bool, manifest: dict) -> None:
    plan = session_plan()
    print(f"sessions: {len(plan)} traces get mlflow.trace.session/mlflow.trace.user "
          f"({len(sessions_by_case(plan))} distinct sessions/case_ids)")
    manifest["sessions"] = {"traces": len(plan), "distinct_sessions": len(sessions_by_case(plan))}
    if not live:
        return

    import mlflow
    from mlflow.entities import AssessmentSource
    from mlflow.entities.assessment import Feedback
    from mlflow.tracing.constant import TraceMetadataKey
    from mlflow.tracing.fluent import start_span_no_context

    from dealpoint.eval import mlflow_mirror as mm

    exp_id = mm._ensure_experiment(client)
    key_to_model_name = trace_key_to_model_name()
    model_id_by_name = manifest.get("models", {}).get("model_id_by_name", {})

    relogged = 0
    linked_to_model = 0
    for e in plan:
        trace = mm._find_trace(client, exp_id, e["key"])
        if trace is None:
            continue
        have_meta = trace.info.trace_metadata or {}
        model_name = key_to_model_name.get(e["key"])
        target_model_id = model_id_by_name.get(model_name) if model_name else None
        already_ok = (have_meta.get(TraceMetadataKey.TRACE_SESSION) == e["session"]
                      and have_meta.get(TraceMetadataKey.TRACE_USER) == e["user"]
                      and (target_model_id is None or have_meta.get(TraceMetadataKey.MODEL_ID) == target_model_id))
        if already_ok:
            continue

        root = trace.data.spans[0]
        metadata = {TraceMetadataKey.TRACE_SESSION: e["session"], TraceMetadataKey.TRACE_USER: e["user"]}
        if target_model_id:
            metadata[TraceMetadataKey.MODEL_ID] = target_model_id
            linked_to_model += 1
        tags = {k: v for k, v in (trace.info.tags or {}).items() if not k.startswith("mlflow.")}
        start_ns = int(trace.info.request_time or 0) * 1_000_000
        new_span = start_span_no_context(root.name, inputs=root.inputs, tags=tags, metadata=metadata,
                                         experiment_id=exp_id, start_time_ns=start_ns)
        if root.outputs is not None:
            new_span.set_outputs(root.outputs)
        new_span.end()
        mlflow.flush_trace_async_logging()
        new_trace_id = new_span.trace_id

        old_assessments = trace.info.assessments or []
        for a in old_assessments:
            source = AssessmentSource(source_type=a.source.source_type, source_id=a.source.source_id)
            if isinstance(a, Feedback):
                mlflow.log_feedback(trace_id=new_trace_id, name=a.name, value=a.value, rationale=a.rationale,
                                    metadata=a.metadata, source=source)
            else:
                mlflow.log_expectation(trace_id=new_trace_id, name=a.name, value=a.value, metadata=a.metadata, source=source)

        new_trace = client.get_trace(new_trace_id)
        new_count = len(new_trace.info.assessments or [])
        if new_count != len(old_assessments):
            raise RuntimeError(f"assessment count mismatch re-logging {e['key']}: {len(old_assessments)} old -> {new_count} new; original left in place")

        source_run_id = (trace.info.trace_metadata or {}).get(TraceMetadataKey.SOURCE_RUN)
        if source_run_id:
            client.link_traces_to_run([new_trace_id], source_run_id)

        client.delete_traces(experiment_id=exp_id, trace_ids=[trace.info.trace_id])
        relogged += 1
    manifest["sessions"]["tagged_this_run"] = relogged
    manifest["sessions"]["linked_to_model_this_run"] = linked_to_model


# ================================================================================================
# D12: judges (Evaluation > Judges)
# ================================================================================================

DETERMINISTIC_SCORER_RUBRICS = {
    "grounded_accuracy": "Correct answer per the deterministic contract-review rubric: the finding matches the gold answer under the case's grading rule (dealpoint.eval.scorers.grounded_accuracy).",
    "answer_correct": "The system's ANSWERED/ABSTAINED verdict matches the gold verdict (dealpoint.eval.scorers.answer_correct).",
    "citation_gold_overlap": "The cited span overlaps a gold span in the agreement (dealpoint.eval.scorers.citation_gold_overlap).",
    "citation_verbatim": "The cited quote appears word for word in the agreement text (dealpoint.eval.scorers.citation_verbatim).",
    "abstain_correct": "On a counterfactual case (the defined term or clause is genuinely absent), the system correctly abstained (dealpoint.eval.scorers.abstain_correct).",
    "skill_adherence": "The agent's skill-injection trajectory followed the M4 skill's required steps (dealpoint.eval.scorers.skill_adherence).",
}


def judge_registration_plan() -> list[dict]:
    """The ten scorers `specs/milestones/m9c.md` D12 asks the Judges tab to list: the six
    deterministic `@scorer` functions from M9 (`mlflow_mirror.DETERMINISTIC_SCORER_NAMES`) plus the
    four `make_judge` dimension judges. Only the four judges can actually be REGISTERED on this
    non-Databricks install (`step_judges`'s `deterministic_registration` records why, verbatim, per
    attempt); this plan still names all ten so the attempt -- and its honest outcome -- is visible."""
    from dealpoint.eval import mlflow_mirror as mm
    from dealpoint.eval.rubric import rubric_text

    out = [{"kind": "deterministic", "name": name, "description": DETERMINISTIC_SCORER_RUBRICS[name]}
           for name in mm.DETERMINISTIC_SCORER_NAMES]
    rubric = rubric_text()
    for dim in JUDGE_DIMS:
        out.append({"kind": "judge", "name": f"judge-{dim}", "dimension": dim,
                    "description": f"{dim.title()} dimension, 1-5 rubric anchors (specs rubric, {dim} section verbatim below).\n\n{rubric}",
                    "model": mm.JUDGE_MODEL_FOR_DIM[dim]})
    return out


ONLINE_SCORING_RULE = {
    "name": "judge-professional",
    "sampling_filter": "tags.category = 'live-replay'",
    "mirrors": "the Braintrust online-scoring rule of the same name, restricted to the same tag",
}


def step_judges(client, live: bool, manifest: dict) -> None:
    plan = judge_registration_plan()
    print(f"judges: {len(plan)} scorers named in the plan "
          f"({sum(1 for p in plan if p['kind'] == 'deterministic')} deterministic + {sum(1 for p in plan if p['kind'] == 'judge')} judges)")
    manifest["judges"] = {"planned": len(plan), "names": [p["name"] for p in plan], "online_rule": ONLINE_SCORING_RULE}
    if not live:
        return
    import mlflow.genai
    from mlflow.exceptions import MlflowException
    from mlflow.genai.judges import make_judge

    from dealpoint.eval import mlflow_mirror as mm
    from dealpoint.eval.braintrust_showroom import judge_scorer_messages

    exp_id = mm._ensure_experiment(client)
    deterministic_registration: dict[str, str] = {}
    for p in plan:
        if p["kind"] != "deterministic":
            continue

        def _scorer(outputs=None, metadata=None):
            return None

        _scorer.__name__ = p["name"]
        scorer_obj = mlflow.genai.scorer(_scorer)
        scorer_obj.description = p["description"]
        try:
            scorer_obj.register(experiment_id=exp_id)
            deterministic_registration[p["name"]] = "registered"
        except MlflowException as exc:
            deterministic_registration[p["name"]] = str(exc)

    gateway_endpoints = manifest.get("gateway", {}).get("endpoints_created", [])
    for p in plan:
        if p["kind"] != "judge":
            continue
        dim = p["dimension"]
        msgs = judge_scorer_messages(dim)
        instructions = "\n\n".join(m["content"] for m in msgs).replace("{{input}}", "{{ inputs }}").replace("{{output}}", "{{ outputs }}").replace("{{expected}}", "{{ expectations }}")
        if "judge-mistral" in gateway_endpoints and p["name"] == "judge-professional":
            scorer_obj = make_judge(name=p["name"], instructions=instructions, model="gateway:/judge-mistral")
        else:
            scorer_obj = make_judge(name=p["name"], instructions=instructions, model=f"openai:/{p['model']}", base_url=mm.OPENROUTER_BASE_URL)
        try:
            scorer_obj.register(experiment_id=exp_id)
        except MlflowException:
            pass

    registered = mlflow.genai.scorers.list_scorers(experiment_id=exp_id)
    manifest["judges"]["registered"] = len(registered)
    manifest["judges"]["registered_names"] = sorted(s.name for s in registered)
    manifest["judges"]["deterministic_registration"] = deterministic_registration

    try:
        from mlflow.genai.scorers.base import ScorerSamplingConfig

        target = [s for s in registered if s.name == "judge-professional"]
        sampling_config = ScorerSamplingConfig(sample_rate=1.0, filter_string=ONLINE_SCORING_RULE["sampling_filter"])
        started = target[0].start(sampling_config=sampling_config) if target else None
        manifest["judges"]["online_rule_started"] = bool(started)
    except Exception as exc:  # noqa: BLE001 - `.start()` may require the gateway model it names (spec D12/D15)
        manifest["judges"]["online_rule_started"] = False
        manifest["judges"]["online_rule_error"] = str(exc)


def replay_trace(case_id: str, variant_id: str, *, live: bool) -> dict:
    """`just mlflow-replay CASE:VARIANT`: re-log one stored trace as a NEW trace tagged
    `category = live-replay`, the only traces the D12 online-scoring rule samples."""
    from dealpoint.eval import mlflow_mirror as mm

    ctx = mm.mirror_context()
    idx = None
    from dealpoint.eval.braintrust_showroom import stored_row_index

    idx = stored_row_index()
    row = idx.get((case_id, variant_id), {})
    mirror = mm.mirror_for(case_id, variant_id, row, ctx, category="live-replay")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    plan = {"case_id": case_id, "variant_id": variant_id, "category": f"live-replay-{stamp}", "mirror": mirror}
    print(f"replay: {case_id}:{variant_id} -> new trace, category live-replay-{stamp} (online scoring may pick it up)")
    if not live:
        return plan
    import mlflow

    mm._ensure_experiment(mm._client())
    key = f"{case_id}:{variant_id}:{plan['category']}"
    tags = {"category": "live-replay", "case_id": case_id, "variant_id": variant_id, "dealpoint.key": key}
    with mlflow.start_span(name=f"{case_id} | replay:{variant_id}") as span:
        span.set_inputs({"case_id": case_id, "variant_id": variant_id})
        span.set_outputs(mirror)
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id:
        for k, v in tags.items():
            mlflow.set_trace_tag(trace_id=trace_id, key=k, value=v)
    plan["trace_id"] = trace_id
    return plan


# ================================================================================================
# D13: review (Evaluation > Review)
# ================================================================================================

LABEL_SCHEMAS = [
    {"name": "reasoning", "type": "feedback", "numeric_range": (1, 5), "description": "Reasoning quality, 1-5 (M5 rubric anchors)."},
    {"name": "evidence", "type": "feedback", "numeric_range": (1, 5), "description": "Evidence quality, 1-5 (M5 rubric anchors)."},
    {"name": "trajectory", "type": "feedback", "numeric_range": (1, 5), "description": "Trajectory quality, 1-5 (M5 rubric anchors)."},
    {"name": "professional", "type": "feedback", "numeric_range": (1, 5), "description": "Professional-quality, 1-5 (M5 rubric anchors)."},
    {"name": "gold_answer", "type": "expectation", "numeric_range": None, "description": "The gold answer this case's finding should match."},
]


def review_queue_plan() -> list[dict]:
    """Two queues: the 12 open review-set traces (`braintrust_sync.review_set`, status PENDING) and
    the 24 already-scored lawyer packets (`braintrust_showroom.judge_packet_rows`, status DONE)."""
    from dealpoint.eval.braintrust_showroom import judge_packet_rows
    from dealpoint.eval.braintrust_sync import review_set

    open_items = [{"case_id": it["case_id"], "variant_id": it["variant_id"], "status": "PENDING"} for it in review_set()]
    done_items = []
    for r in judge_packet_rows():
        meta = r["metadata"]
        if not meta.get("case_id") or not meta.get("variant_id"):
            continue
        done_items.append({"case_id": meta["case_id"], "variant_id": meta["variant_id"], "status": "DONE", "packet_id": meta["packet_id"]})
    return [
        {"name": "Lawyer calibration review", "items": open_items, "schemas": [s["name"] for s in LABEL_SCHEMAS]},
        {"name": "Lawyer-scored packets (24)", "items": done_items, "schemas": [s["name"] for s in LABEL_SCHEMAS]},
    ]


def _operator_user() -> str:
    return os.environ.get("MLFLOW_TRACKING_USERNAME") or getpass.getuser()


def _find_traces_by_case_variant(exp_id: str, case_id: str, variant_id: str) -> Any:
    import mlflow

    return mlflow.search_traces(locations=[exp_id], filter_string=f"tags.case_id = '{case_id}' and tags.variant_id = '{variant_id}'",
                                max_results=5, return_type="list")


def step_review(client, live: bool, manifest: dict) -> None:
    schemas = LABEL_SCHEMAS
    queues = review_queue_plan()
    operator = _operator_user()
    print(f"review: {len(schemas)} label schemas, {len(queues)} queues "
          f"({', '.join(f'{q['name']} ({len(q['items'])})' for q in queues)}); operator = {operator}")
    manifest["review"] = {"schemas": [s["name"] for s in schemas], "queues": {q["name"]: len(q["items"]) for q in queues}, "operator": operator}
    if not live:
        return
    import mlflow.genai.label_schemas as ls
    import mlflow.genai.review_queues as rq
    from mlflow.exceptions import MlflowException

    from dealpoint.eval import mlflow_mirror as mm

    exp_id = mm._ensure_experiment(client)

    existing_schemas = {s.name: s.schema_id for s in ls.list_label_schemas(experiment_id=exp_id)}
    schema_ids: list[str] = []
    for s in schemas:
        if s["name"] in existing_schemas:
            schema_ids.append(str(existing_schemas[s["name"]]))
            continue
        input_type = ls.InputText() if s["type"] == "expectation" else ls.InputNumeric(min_value=s["numeric_range"][0], max_value=s["numeric_range"][1])
        created = ls.create_label_schema(s["name"], type=s["type"], input=input_type, instruction=s["description"], experiment_id=exp_id)
        schema_ids.append(str(created.schema_id))
    manifest["review"]["schema_ids"] = schema_ids

    queue_ids: dict[str, str] = {}
    unresolved: dict[str, int] = {}
    for q in queues:
        try:
            existing_queue = rq.get_review_queue(name=q["name"], experiment_id=exp_id)
        except MlflowException:
            existing_queue = None
        if existing_queue is None:
            existing_queue = rq.create_review_queue(q["name"], queue_type="custom", users=[operator], schema_ids=schema_ids, experiment_id=exp_id)
        queue_ids[q["name"]] = existing_queue.queue_id

        have_item_ids = {it.item_id for it in rq.list_review_queue_items(existing_queue.queue_id)}
        n_unresolved = 0
        for item in q["items"]:
            matches = _find_traces_by_case_variant(exp_id, item["case_id"], item["variant_id"])
            if not matches:
                n_unresolved += 1
                continue
            trace_id = matches[0].info.trace_id
            if trace_id not in have_item_ids:
                rq.add_items_to_review_queue(existing_queue.queue_id, item_ids=[trace_id])
            if item["status"] == "DONE":
                rq.set_review_queue_item_status(existing_queue.queue_id, item_id=trace_id, status="complete", completed_by=operator)
        if n_unresolved:
            unresolved[q["name"]] = n_unresolved
    manifest["review"]["queue_ids"] = queue_ids
    if unresolved:
        manifest["review"]["unresolved_items"] = unresolved


# ================================================================================================
# D14: agent versions (Versions tab)
# ================================================================================================


def logged_model_plan() -> list[dict]:
    """One `LoggedModel` per comparable configuration (the ten of M9b's decision tree, `system@model`)
    plus the four `playground-arm-A-*` prompt variants -- 14 total. Params = the arm config
    (`arm_parameter_sets`), metrics = M9b's fifteen `DECISION_METRICS`; trace linkage is folded into
    `step_sessions`'s re-log pass (D11), not attempted here."""
    from dealpoint.eval import mlflow_decision as md
    from dealpoint.eval import mlflow_mirror as mm
    from dealpoint.eval.braintrust_showroom import arm_parameter_sets

    arm_params = {p["arm"]: p for p in arm_parameter_sets()}
    out: list[dict] = []
    for entry in md.decision_run_plan():
        if entry["role"] != "config":
            continue
        arm = entry["config"].split("@", 1)[0]
        out.append({
            "kind": "agent-config", "name": entry["config"], "pool": entry["pool"],
            "model_name": _config_model_name(entry["config"], entry["pool"]),
            "params": {**arm_params.get(arm, {}), **entry["params"]},
            "metrics": entry["metrics"], "tags": entry["tags"],
        })
    variants = sorted({p["variant"] for p in mm.prompt_variant_trace_plan()})
    for v in variants:
        traces = [p for p in mm.prompt_variant_trace_plan() if p["variant"] == v]
        out.append({
            "kind": "prompt-variant", "name": f"playground-arm-A-{v}", "pool": None,
            "model_name": f"playground-arm-A-{v}", "params": {"arm": "A", "prompt_variant": v},
            "metrics": {"n_traces": len(traces)}, "tags": {"category": "prompt-variant", "prompt_variant": v},
            "trace_keys": [f"{p['case_id']}:{p['variant']}:prompt-variant" for p in traces],
        })
    return out


REGISTRY_ALIASES_D14 = {
    "champion": ("judged-18", "D@gemini-3.1-flash-lite"),
    "baseline": ("judged-18", "A@haiku"),
    "cost-floor": ("judged-18", "D@qwen3.7-flash"),
    "safest": ("judged-18", "D@deepseek-v4-flash"),
}


def step_models(client, live: bool, manifest: dict) -> None:
    plan = logged_model_plan()
    key_to_model_name = trace_key_to_model_name()
    from collections import Counter

    expected_linked_trace_counts = dict(Counter(key_to_model_name.values()))
    plan_model_names = {p["model_name"] for p in plan}
    overlap_note = None
    d_glm_judged = expected_linked_trace_counts.get("D@glm (judged-18)")
    d_glm_test32 = expected_linked_trace_counts.get("D@glm (test-32)")
    if d_glm_judged is not None and d_glm_test32 is not None:
        overlap_note = (f"D@glm appears in both pools; mlflow.modelId is single-valued, so its 18 "
                        f"judged-18 rows are excluded from test-32's count -- measured "
                        f"{d_glm_judged} (judged-18) + {d_glm_test32} (test-32, raw-agent-only), "
                        f"not the spec's assumed 18/32 for both.")
    print(f"models: {len(plan)} logged models "
          f"({sum(1 for p in plan if p['kind'] == 'agent-config')} agent configs + {sum(1 for p in plan if p['kind'] == 'prompt-variant')} prompt variants)")
    manifest["models"] = {"planned": len(plan), "names": sorted(plan_model_names),
                          "expected_linked_trace_counts": expected_linked_trace_counts}
    if overlap_note:
        manifest["models"]["spec_difference_d_glm_overlap"] = overlap_note
    if not live:
        return

    import mlflow

    from dealpoint.eval import mlflow_mirror as mm

    exp_id = mm._ensure_experiment(client)
    model_id_by_name: dict[str, str] = {}
    existing_models = {m.name: m for m in client.search_logged_models(experiment_ids=[exp_id])}
    for p in plan:
        lm = existing_models.get(p["model_name"])
        if lm is None:
            lm = client.create_logged_model(experiment_id=exp_id, name=p["model_name"], params={k: str(v) for k, v in p["params"].items()},
                                            tags={k: str(v) for k, v in p["tags"].items()})
            for k, v in p["metrics"].items():
                if v is not None:
                    mlflow.log_metric(k, float(v), model_id=lm.model_id)
            client.finalize_logged_model(lm.model_id, "READY")
        model_id_by_name[p["model_name"]] = lm.model_id
    manifest["models"]["model_id_by_name"] = model_id_by_name

    try:
        client.create_registered_model(REGISTERED_MODEL)
    except Exception:  # noqa: BLE001, S110 - may already exist
        pass
    for alias, (pool, config) in REGISTRY_ALIASES_D14.items():
        model_name = _config_model_name(config, pool)
        model_id = model_id_by_name.get(model_name)
        if not model_id:
            continue
        versions = [v for v in client.search_model_versions(f"name = '{REGISTERED_MODEL}'") if v.tags.get("dealpoint.logged_model") == model_id]
        if not versions:
            mv = client.create_model_version(REGISTERED_MODEL, source=f"models:/{model_id}", model_id=model_id, tags={"dealpoint.logged_model": model_id})
            version = mv.version
        else:
            version = versions[0].version
        client.set_registered_model_alias(REGISTERED_MODEL, alias, version)


# ================================================================================================
# D15: gateway and playground
# ================================================================================================


def gateway_plan() -> dict:
    return {"connection": GATEWAY_CONNECTION, "base_url": GATEWAY_BASE_URL, "provider": "openai-compatible",
            "endpoints": GATEWAY_ENDPOINTS}


def prompt_endpoint_plan() -> list[dict]:
    """8 registered prompts -> the endpoint their `model_config` should name (spec D15). Every prompt
    without an explicit override falls back to `dealpoint-glm`, the default model prompts were built
    against (`braintrust_sync.WORKHORSE_MODEL`)."""
    from dealpoint.eval import mlflow_mirror as mm

    return [{"name": p["name"], "endpoint": PROMPT_ENDPOINT_OVERRIDES.get(p["name"], "dealpoint-glm")}
           for p in mm.prompt_plan_m9()]


def step_gateway(client, live: bool, manifest: dict, *, gateway_ready: bool = False) -> None:
    plan = gateway_plan()
    print(f"gateway: connection {plan['connection']} ({plan['base_url']}), {len(plan['endpoints'])} endpoints")
    manifest["gateway"] = {"connection": plan["connection"], "endpoints": list(plan["endpoints"])}
    if not live:
        return

    from dealpoint.eval import mlflow_mirror as mm

    exp_id = mm._ensure_experiment(client)
    store = client._tracking_client.store

    if gateway_ready:
        try:
            existing = {e.name for e in store.list_gateway_endpoints()}
            manifest["gateway"]["endpoints_created"] = sorted(existing & set(plan["endpoints"]))
            manifest["gateway"]["gateway_ready_verified"] = True
        except Exception as exc:  # noqa: BLE001 - honest failure, not a crash
            manifest["gateway"]["gateway_ready_verified"] = False
            manifest["gateway"]["reason"] = str(exc)
        return

    try:
        from mlflow.entities.gateway_endpoint import (
            GatewayEndpointModelConfig,
            GatewayModelLinkageType,
        )

        # `get_secret_info`/`get_gateway_model_definition` accept a `name=`/`secret_name=` kwarg on
        # `SqlAlchemyStore` but NOT on `RestStore` (verified live: RestStore's `get_secret_info`
        # requires `secret_id`, raising `INVALID_PARAMETER_VALUE`, and its
        # `get_gateway_model_definition` has no `name` parameter at all, raising `TypeError`) --
        # `list_secret_infos`/`list_gateway_model_definitions` work identically on both backends, so
        # idempotency is checked by listing, never by the name-keyed getters.
        existing_secret = next((s for s in store.list_secret_infos() if s.secret_name == plan["connection"]), None)
        if existing_secret is None:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            existing_secret = store.create_gateway_secret(plan["connection"], {"api_key": api_key}, provider="openai",
                                                          auth_config={"base_url": plan["base_url"]})
        existing_endpoints = {e.name for e in store.list_gateway_endpoints()}
        existing_model_defs = {d.name: d for d in store.list_gateway_model_definitions()}
        created: list[str] = []
        for name, model in plan["endpoints"].items():
            if name in existing_endpoints:
                created.append(name)
                continue
            print(f"gateway: endpoint {name} not found yet, creating")
            model_def = existing_model_defs.get(name) or store.create_gateway_model_definition(name, existing_secret.secret_id, "openai", model)
            store.create_gateway_endpoint(name, [GatewayEndpointModelConfig(model_definition_id=model_def.model_definition_id,
                                                                            linkage_type=GatewayModelLinkageType.PRIMARY, weight=1.0)],
                                          experiment_id=exp_id)
            created.append(name)
        manifest["gateway"]["created_via"] = "store"
        manifest["gateway"]["endpoints_created"] = sorted(created)
    except Exception as exc:  # noqa: BLE001 - OSS 3.16 may expose the gateway only through the Settings UI (spec D15 fallback)
        manifest["gateway"]["settings_ui_required"] = True
        manifest["gateway"]["reason"] = str(exc)
        print("gateway: store/REST route unavailable; create these fields in Settings > AI Gateway, "
              "then re-run with --gateway-ready:")
        print(f"  connection: {plan['connection']}  provider: openai  base_url: {plan['base_url']}  key env: OPENROUTER_API_KEY")
        for name, model in plan["endpoints"].items():
            print(f"  endpoint: {name}  model: {model}  task: chat")


def step_playground(client, live: bool, manifest: dict) -> None:
    prompts = prompt_endpoint_plan()
    print(f"playground: {len(prompts)} prompts get a model_config naming their endpoint")
    manifest["playground"] = {"prompts": {p["name"]: p["endpoint"] for p in prompts}}
    if not live:
        return

    import mlflow.genai

    from dealpoint.eval import mlflow_mirror as mm

    created: dict[str, int] = {}
    for p in prompts:
        try:
            existing_versions = client.search_prompt_versions(p["name"])
        except Exception:  # noqa: BLE001 - a fresh prompt name has no versions yet, never fatal
            existing_versions = []
        latest = max(existing_versions, key=lambda v: v.version, default=None) if existing_versions else None
        if latest is not None and (latest.model_config or {}).get("endpoint") == p["endpoint"]:
            version = latest.version
        else:
            template = latest.template if latest is not None else p["name"]
            pv = mlflow.genai.register_prompt(name=p["name"], template=template, model_config={"endpoint": p["endpoint"], "temperature": 0.2, "max_tokens": 1024})
            version = pv.version
            created[p["name"]] = version
        alias = mm.PROMPT_ALIASES.get(p["name"])
        if alias:
            client.set_prompt_alias(name=p["name"], alias=alias, version=version)
    manifest["playground"]["new_versions"] = created


def gateway_smoke_test(*, live: bool) -> dict:
    """Opt-in `--gateway-smoke`: one call per endpoint (cents), ledgered `milestone_tag: m9c` through
    the same `OpenRouterClient` every other metered call in this repo uses. Never called by any
    dry-run or offline-test path, and never called unless the flag is passed on the command line."""
    plan = gateway_plan()
    if not live:
        return {"planned_calls": len(plan["endpoints"]), "endpoints": list(plan["endpoints"])}
    from dealpoint.llm.client import OpenRouterClient

    or_client = OpenRouterClient(milestone_tag="m9c")
    calls = []
    for name, model in plan["endpoints"].items():
        result = or_client.chat(model=model, messages=[{"role": "user", "content": "Reply with the single word: ok."}], max_tokens=5)
        calls.append({"endpoint": name, "model": model, "usd": result.cost_usd})
    return {"calls": calls}


# ================================================================================================
# D16.4/D17: overview
# ================================================================================================


def step_overview(client, live: bool, manifest: dict) -> None:
    from dealpoint.eval import mlflow_mirror as mm
    from dealpoint.eval.mlflow_dashboards import dashboard_run_plan

    dashboard_runs = [p["run_key"] for p in dashboard_run_plan()]
    aliases = sorted(REGISTRY_ALIASES_D14)
    queues = [q["name"] for q in review_queue_plan()]
    lines = [
        "",
        "Dashboards: " + ", ".join(dashboard_runs) + ".",
        f"Decision tree: decision/judged-18, decision/test-32. Registry: {REGISTERED_MODEL} ({', '.join(aliases)}).",
        "Review queues: " + ", ".join(queues) + ".",
    ]
    addition = "\n".join(lines)
    print(f"overview: experiment description gains a Dashboards line naming {len(dashboard_runs)} dashboard runs")
    manifest["overview"] = {"dashboard_runs": dashboard_runs, "registry_aliases": aliases, "review_queues": queues}
    if not live:
        return

    exp_id = mm._ensure_experiment(client)
    exp = client.get_experiment(exp_id)
    base = (exp.tags or {}).get("mlflow.note.content", "")
    if addition.strip() not in base:
        client.set_experiment_tag(exp_id, "mlflow.note.content", base + addition)


# ================================================================================================
# main
# ================================================================================================


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and argv[0] == "replay":
        if len(argv) < 2 or ":" not in argv[1]:
            print("usage: mlflow_tabs replay CASE:VARIANT [--live]", file=sys.stderr)
            return 2
        case_id, variant_id = argv[1].split(":", 1)
        replay_trace(case_id, variant_id, live=("--live" in argv))
        return 0

    dry_run = "--live" not in argv or "--dry-run" in argv
    live = not dry_run
    only = None
    gateway_ready = "--gateway-ready" in argv
    gateway_smoke = "--gateway-smoke" in argv
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            only = set(argv[i + 1].split(","))

    manifest: dict[str, Any] = {"mode": "live" if live else "dry-run", "started_at": datetime.now(UTC).isoformat(), "experiment": MLFLOW_EXPERIMENT}
    print(f"{'LIVE' if live else 'DRY RUN'}: mlflow tabs on {MLFLOW_EXPERIMENT}")

    from dealpoint.eval import mlflow_mirror as mm

    client = mm._client() if live else None
    for name in STEPS:
        if only and name not in only:
            continue
        if name == "gateway":
            step_gateway(client, live, manifest, gateway_ready=gateway_ready)
        else:
            globals()[f"step_{name}"](client, live, manifest)

    if live and gateway_smoke:
        print("gateway-smoke: one call per endpoint, ledgered milestone_tag=m9c")
        manifest["gateway_smoke"] = gateway_smoke_test(live=True)

    manifest["finished_at"] = datetime.now(UTC).isoformat()
    if live:
        _manifest_path().parent.mkdir(parents=True, exist_ok=True)
        _manifest_path().write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"manifest -> {_manifest_path()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
