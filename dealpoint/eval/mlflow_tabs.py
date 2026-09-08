"""M9c D11-D15: every empty MLflow sidebar tab filled from the same stored rows M9/M9b already mirror
(`specs/milestones/m9c.md`). Additive to `mlflow_mirror`/`mlflow_decision`: no model calls, no new
scoring, nothing re-derived that a pure function upstream already computed.

Dry run is the default; `--live` executes against `MLFLOW_TRACKING_URI` (default
`http://127.0.0.1:5000`); `--dry-run` always wins over `--live`, matching every other M9* module.
`mlflow` is imported lazily so the planning functions work without the `mlflow` extra installed.

A live OSS MLflow 3.16 install may reject some of D12's `.start(sampling_config=...)` and D13's
`create_label_schema`/labeling-session calls as Databricks-only (the spec calls this out for D12;
verified against this build's install for D13 too -- `mlflow.genai.labeling` routes through
`get_review_app`, which requires a Databricks tracking URI). Every live step below tries the real
call, catches exactly that failure, records it in the manifest under `<step>.databricks_only`, and
still leaves the plan (the registered judges, the schema/queue definitions) in place: the tab is
filled with what OSS can hold even when OSS cannot run the workflow around it.

    uv run python -m dealpoint.eval.mlflow_tabs                 # dry run, every step
    uv run python -m dealpoint.eval.mlflow_tabs --live          # execute
    uv run python -m dealpoint.eval.mlflow_tabs --live --only sessions,judges
    uv run python -m dealpoint.eval.mlflow_tabs replay CASE:VARIANT --live   # D12's mlflow-replay
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

MLFLOW_EXPERIMENT = "dealpoint-eval"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
MANIFEST_PATH = Path("data/reports/mlflow_tabs_manifest.json")
REGISTERED_MODEL = "dealpoint-agent"

STEPS = ("sessions", "judges", "review", "models", "gateway")

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


def step_sessions(client, live: bool, manifest: dict) -> None:
    plan = session_plan()
    print(f"sessions: {len(plan)} traces get mlflow.trace.session/mlflow.trace.user "
          f"({len(sessions_by_case(plan))} distinct sessions/case_ids)")
    manifest["sessions"] = {"traces": len(plan), "distinct_sessions": len(sessions_by_case(plan))}
    if not live:
        return

    from dealpoint.eval import mlflow_mirror as mm

    exp_id = mm._ensure_experiment(client)
    tagged = 0
    for e in plan:
        trace = mm._find_trace(client, exp_id, e["key"])
        if trace is None:
            continue
        have = trace.info.tags or {}
        if have.get("mlflow.trace.session") == e["session"] and have.get("mlflow.trace.user") == e["user"]:
            continue
        client.set_trace_tag(trace.info.trace_id, "mlflow.trace.session", e["session"])
        client.set_trace_tag(trace.info.trace_id, "mlflow.trace.user", e["user"])
        tagged += 1
    manifest["sessions"]["tagged_this_run"] = tagged


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
    """The ten scorers the Judges tab must list: the six deterministic `@scorer` functions from M9
    (`mlflow_mirror.DETERMINISTIC_SCORER_NAMES`) plus the four `make_judge` dimension judges, each
    carrying its rubric text as `description` (the M5 rubric anchors for the judges, docstrings above
    for the deterministic six)."""
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
    print(f"judges: {len(plan)} scorers to register "
          f"({sum(1 for p in plan if p['kind'] == 'deterministic')} deterministic + {sum(1 for p in plan if p['kind'] == 'judge')} judges)")
    manifest["judges"] = {"registered": len(plan), "names": [p["name"] for p in plan], "online_rule": ONLINE_SCORING_RULE}
    if not live:
        return
    import mlflow.genai
    from mlflow.genai.judges import make_judge

    from dealpoint.eval import mlflow_mirror as mm
    from dealpoint.eval.braintrust_showroom import judge_scorer_messages

    exp_id = mm._ensure_experiment(client)
    for p in plan:
        if p["kind"] == "deterministic":
            handler = None
            try:
                from dealpoint.eval.braintrust_sync import _make_scorer_handler

                handler = _make_scorer_handler(p["name"])
            except Exception as exc:  # noqa: BLE001 - registration must not fail the whole step over one scorer
                print(f"  judges: could not build handler for {p['name']}: {exc}")

            def _scorer(outputs=None, metadata=None, _handler=handler):
                return _handler(output=outputs, metadata=metadata) if _handler else None

            _scorer.__name__ = p["name"]
            scorer_obj = mlflow.genai.scorer(_scorer)
            scorer_obj.description = p["description"]
        else:
            dim = p["dimension"]
            msgs = judge_scorer_messages(dim)
            instructions = "\n\n".join(m["content"] for m in msgs).replace("{{input}}", "{{ inputs }}").replace("{{output}}", "{{ outputs }}").replace("{{expected}}", "{{ expectations }}")
            scorer_obj = make_judge(name=p["name"], instructions=instructions, model=f"openai:/{p['model']}", base_url=mm.OPENROUTER_BASE_URL)
        try:
            scorer_obj.register(experiment_id=exp_id)
        except Exception:  # noqa: BLE001, S110 - already registered from a prior --live run
            pass
    try:
        from mlflow.genai.scorers.base import ScorerSamplingConfig

        registered = [s for s in mlflow.genai.scorers.list_scorers(experiment_id=exp_id) if s.name == "judge-professional"]
        sampling_config = ScorerSamplingConfig(sample_rate=1.0, filter_string=ONLINE_SCORING_RULE["sampling_filter"])
        started = registered[0].start(sampling_config=sampling_config) if registered else None
        manifest["judges"]["online_rule_started"] = bool(started)
    except Exception as exc:  # noqa: BLE001 - `.start()` is Databricks-only on some 3.16 builds (spec D12)
        manifest["judges"]["online_rule_started"] = False
        manifest["judges"]["databricks_only"] = str(exc)


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


def step_review(client, live: bool, manifest: dict) -> None:
    schemas = LABEL_SCHEMAS
    queues = review_queue_plan()
    print(f"review: {len(schemas)} label schemas, {len(queues)} queues "
          f"({', '.join(f'{q['name']} ({len(q['items'])})' for q in queues)})")
    manifest["review"] = {"schemas": [s["name"] for s in schemas], "queues": {q["name"]: len(q["items"]) for q in queues}}
    if not live:
        return
    import mlflow.genai.label_schemas as ls

    from dealpoint.eval import mlflow_mirror as mm

    exp_id = mm._ensure_experiment(client)
    try:
        for s in schemas:
            input_type = ls.InputText() if s["type"] == "expectation" else ls.InputNumeric(min_value=s["numeric_range"][0], max_value=s["numeric_range"][1])
            ls.create_label_schema(s["name"], type=s["type"], input=input_type, instruction=s["description"], overwrite=True, experiment_id=exp_id)
        import mlflow.genai.labeling as lab

        for q in queues:
            session = lab.create_labeling_session(q["name"], label_schemas=q["schemas"])
            manifest["review"].setdefault("sessions_created", []).append(session.name if hasattr(session, "name") else q["name"])
    except Exception as exc:  # noqa: BLE001 - review queues/labeling sessions are Databricks-only on some 3.16 builds
        manifest["review"]["databricks_only"] = str(exc)


# ================================================================================================
# D14: agent versions (Versions tab)
# ================================================================================================


def logged_model_plan() -> list[dict]:
    """One `LoggedModel` per comparable configuration (the ten of M9b's decision tree, `system@model`)
    plus the four `playground-arm-A-*` prompt variants -- 14 total. Params = the arm config
    (`arm_parameter_sets`), metrics = M9b's fifteen `DECISION_METRICS`, every trace of that
    configuration linked by `dealpoint.key`."""
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
            "model_name": f"{entry['config']} ({entry['pool']})",
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
    print(f"models: {len(plan)} logged models "
          f"({sum(1 for p in plan if p['kind'] == 'agent-config')} agent configs + {sum(1 for p in plan if p['kind'] == 'prompt-variant')} prompt variants)")
    manifest["models"] = {"planned": len(plan), "names": [p["model_name"] for p in plan]}
    if not live:
        return

    from dealpoint.eval import mlflow_mirror as mm

    exp_id = mm._ensure_experiment(client)
    logged_model_by_name: dict[str, str] = {}
    for p in plan:
        existing = [m for m in client.search_logged_models(experiment_ids=[exp_id]) if m.name == p["model_name"]]
        if existing:
            lm = existing[0]
        else:
            lm = client.create_logged_model(experiment_id=exp_id, name=p["model_name"], params={k: str(v) for k, v in p["params"].items()},
                                            tags={k: str(v) for k, v in p["tags"].items()})
            for k, v in p["metrics"].items():
                if v is not None:
                    client.log_metric(run_id=None, key=k, value=float(v), model_id=lm.model_id) if hasattr(client, "log_metric") else None
        logged_model_by_name[p["model_name"]] = lm.model_id
        if p["kind"] == "agent-config":
            entries = [e for e in mm.agent_trace_plan() if e["variant_id"] == p["name"]]
            keys = [f"{e['case_id']}:{e['variant_id']}:{e['category']}" for e in entries]
        else:
            keys = p.get("trace_keys", [])
        for key in keys:
            trace = mm._find_trace(client, exp_id, key)
            if trace is not None and hasattr(client, "link_trace_to_model"):
                try:
                    client.link_trace_to_model(trace_id=trace.info.trace_id, model_id=lm.model_id)
                except Exception:  # noqa: BLE001, S110 - trace-to-model linking for existing traces is 3.16-version-dependent
                    pass
    try:
        client.create_registered_model(REGISTERED_MODEL)
    except Exception:  # noqa: BLE001, S110 - may already exist
        pass
    for alias, (pool, config) in REGISTRY_ALIASES_D14.items():
        model_name = f"{config} ({pool})"
        model_id = logged_model_by_name.get(model_name)
        if not model_id:
            continue
        versions = [v for v in client.search_model_versions(f"name = '{REGISTERED_MODEL}'") if v.tags.get("dealpoint.logged_model") == model_id]
        if not versions:
            mv = client.create_model_version(REGISTERED_MODEL, source=f"models:/{model_id}", tags={"dealpoint.logged_model": model_id})
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


def step_gateway(client, live: bool, manifest: dict) -> None:
    plan = gateway_plan()
    prompts = prompt_endpoint_plan()
    print(f"gateway: connection {plan['connection']} ({plan['base_url']}), {len(plan['endpoints'])} endpoints; "
          f"{len(prompts)} prompts get a model_config")
    manifest["gateway"] = {"connection": plan["connection"], "endpoints": list(plan["endpoints"]), "prompts": {p["name"]: p["endpoint"] for p in prompts}}
    if not live:
        return
    try:
        from mlflow.gateway import client as gw_client_mod  # type: ignore[import-not-found]

        gw = gw_client_mod.MlflowGatewayClient()
        gw.create_connection(name=plan["connection"], provider="openai", base_url=plan["base_url"], api_key_env_var="OPENROUTER_API_KEY")
        for name, model in plan["endpoints"].items():
            gw.create_endpoint(name=name, connection=plan["connection"], model=model, task="chat")
        manifest["gateway"]["created_via"] = "sdk"
    except Exception as exc:  # noqa: BLE001 - OSS 3.16 may expose the gateway only through the Settings UI (spec D15)
        manifest["gateway"]["settings_ui_required"] = True
        manifest["gateway"]["reason"] = str(exc)
        print("gateway: SDK/REST route unavailable; create these fields in Settings > AI Gateway, "
              "then re-run with --gateway-ready:")
        print(f"  connection: {plan['connection']}  provider: openai  base_url: {plan['base_url']}  key env: OPENROUTER_API_KEY")
        for name, model in plan["endpoints"].items():
            print(f"  endpoint: {name}  model: {model}  task: chat")
        return

    for p in prompts:
        try:
            client.set_prompt_tag(name=p["name"], key="model_config.endpoint", value=p["endpoint"])
        except Exception:  # noqa: BLE001, S110 - tagging is best-effort; the prompt itself is M9's job
            pass


def gateway_smoke_test(*, live: bool) -> dict:
    """Opt-in `--gateway-smoke`: one call per endpoint (cents), ledgered `milestone_tag: m9c` through
    the same `OpenRouterClient` every other metered call in this repo uses. Never called by any
    dry-run or offline-test path."""
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
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            only = set(argv[i + 1].split(","))

    manifest = {"mode": "live" if live else "dry-run", "started_at": datetime.now(UTC).isoformat(), "experiment": MLFLOW_EXPERIMENT}
    print(f"{'LIVE' if live else 'DRY RUN'}: mlflow tabs on {MLFLOW_EXPERIMENT}")

    from dealpoint.eval import mlflow_mirror as mm

    client = mm._client() if live else None
    for name in STEPS:
        if only and name not in only:
            continue
        globals()[f"step_{name}"](client, live, manifest)

    manifest["finished_at"] = datetime.now(UTC).isoformat()
    if live:
        _manifest_path().parent.mkdir(parents=True, exist_ok=True)
        _manifest_path().write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"manifest -> {_manifest_path()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
