"""M9: the MLflow mirror of the Braintrust showroom (`specs/milestones/m9.md`).

Same rows, second cockpit. Every MLflow object here is built from the stored rows and pure
functions M1-M8 already computed (`braintrust_sync`, `braintrust_showroom`, `braintrust_cockpit`);
this module never re-runs, re-tunes or re-judges anything, and never reads or writes Braintrust.

Dry run is the default and prints the plan; `--live` executes against `MLFLOW_TRACKING_URI`
(default `http://127.0.0.1:5000`). `--dry-run` always wins over `--live`, matching
`braintrust_sync.main`'s precedence. `mlflow` is imported lazily so the planning functions (and the
whole offline test suite) work without the `mlflow` extra installed.

    uv run python -m dealpoint.eval.mlflow_mirror                 # dry run
    uv run python -m dealpoint.eval.mlflow_mirror --live          # execute
    uv run python -m dealpoint.eval.mlflow_mirror --live --only datasets,prompts
    uv run python -m dealpoint.eval.mlflow_mirror --live --limit 20   # cap trace-heavy steps (tests)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MLFLOW_EXPERIMENT = "dealpoint-eval"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
MANIFEST_PATH = Path("data/reports/mlflow_manifest.json")
VIEWS_PATH = Path("data/reports/mlflow_views.json")
MLFLOW_DATA_DIR = Path("data/mlflow")
DISK_GUARD_BYTES = 200_000_000

STEPS = ("datasets", "prompts", "runs", "traces", "assessments", "scorers", "evaluations", "registry", "views")

JUDGE_DIMS = ("reasoning", "evidence", "trajectory", "professional")

# The four judges' models, keyed by DIMENSION (not judge family) -- braintrust_showroom.JUDGE_MODELS.
JUDGE_MODEL_FOR_DIM = {
    "reasoning": "mistralai/mistral-small-3.2-24b-instruct",
    "evidence": "mistralai/mistral-small-3.2-24b-instruct",
    "professional": "mistralai/mistral-small-3.2-24b-instruct",
    "trajectory": "bytedance-seed/seed-2.0-mini",
}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# retriever kind (retrieval_log_rows()'s `metadata.retriever`) -> the rag-* run it belongs to.
# Only the six dev-58 tournament configs and the LlamaIndex-native crosscheck ever appear here:
# retrieval_log_rows() filters to `query_type == "canonical"`, so the m7-synthetic pool (a different
# case set) never collides with m3-hybrid / m7-synthetic sharing the retriever kind "hybrid_rrf".
RETRIEVER_TO_RUN = {
    "dense": "rag-m3-dense",
    "bm25": "rag-m3-bm25",
    "hybrid_rrf": "rag-m3-hybrid",
    "hybrid_rrf_rerank": "rag-m3-hybrid-rerank",
    "multi_query_fusion": "rag-m3-fusion",
    "multi_query_fusion_rerank": "rag-m3-fusion-rerank",
    "li_native_bm25": "rag-m7-li-crosscheck",
}

DATASET_NAMES = (
    "maud-dealpoint-dev",
    "maud-dealpoint-test",
    "maud-dealpoint-counterfactual",
    "maud-dealpoint-judged_calibration",
    "maud-dealpoint-synthetic_query",
    "maud-dealpoint-review-set",
    "maud-dealpoint-playground-armA",
    "maud-dealpoint-judge-packets",
    "maud-dealpoint-retrieval-dev",
)

PROMPT_ALIASES = {
    "agent-base-system-prompt": "baseline",
    "arm-a-prompt-cite-first": "best",
    "judge-panel-rubric": "judge-panel",
    "judge-calibrated-rubric": "rubric",
    "agent-arm-d-skill-injection": "skill",
}


# --- manifest / ledger paths (test-overridable) ------------------------------------------------


def _manifest_path() -> Path:
    return MANIFEST_PATH


def _views_path() -> Path:
    return VIEWS_PATH


# --- axis -> dashboard verdict (reused verbatim so no number can drift) ------------------------


def _axis_to_dashboard_text() -> dict[str, str]:
    """`axis` (from `classify_experiment`) -> that question's verdict paragraph, taken verbatim from
    `braintrust_cockpit.DASHBOARDS` so a run description can never disagree with `docs/demo-tour.md`."""
    from dealpoint.eval.braintrust_cockpit import DASHBOARDS

    by_name = {name: text for name, text, _ in DASHBOARDS}
    return {
        "system": by_name.get("Which system?", ""),
        "model": by_name.get("Which model?", ""),
        "retrieval": by_name.get("Which retriever?", ""),
        "judge": by_name.get("Judges and the lawyer", ""),
        "crosscheck": by_name.get("DeepEval", ""),
        "prompt": by_name.get("Which prompt?", ""),
        "traces": by_name.get("DealPoint eval overview", ""),
        "other": by_name.get("DealPoint eval overview", ""),
    }


# --- runs ----------------------------------------------------------------------------------------


def mirror_run_plan() -> list[dict]:
    """The 33 MLflow runs: `braintrust_sync.experiment_plan()`'s 29 experiments (rag/agent/judge/
    deepeval/pareto) plus the four `playground-arm-A-<variant>` names `playground_prompts()` derives.
    One run per Braintrust experiment, by name (`mirror_run_plan()`'s own count is what both the dry-run
    test and the `runs` step assert against, so a data change fails loudly instead of silently agreeing
    with itself)."""
    from dealpoint.eval.braintrust_showroom import classify_experiment, playground_prompts
    from dealpoint.eval.braintrust_sync import experiment_plan

    verdicts = _axis_to_dashboard_text()
    names = [p["name"] for p in experiment_plan()]
    names += [f"playground-arm-A-{pr['slug'].removeprefix('arm-a-prompt-')}" for pr in playground_prompts()]

    plans = []
    for name in names:
        c = classify_experiment(name)
        tags = {k: str(v) for k, v in c.items() if k in ("axis", "varies", "holds", "arm", "loop", "retriever", "skill", "model", "cases", "milestone") and v is not None}
        tags["dealpoint.key"] = name
        plans.append({"name": name, "tags": tags, "axis": c["axis"], "description": verdicts.get(c["axis"], c["description"])})
    return plans


def _arm_params_by_arm() -> dict[str, dict]:
    from dealpoint.eval.braintrust_showroom import arm_parameter_sets

    return {s["arm"]: s for s in arm_parameter_sets()}


def _run_params_for(run: dict) -> dict:
    """Arm config + model, string-coerced (MLflow params are strings)."""
    from dealpoint.eval.braintrust_showroom import classify_experiment

    c = classify_experiment(run["name"])
    params: dict[str, str] = {}
    arm = c.get("arm")
    if arm:
        arm_cfg = _arm_params_by_arm().get(arm)
        if arm_cfg:
            params.update({"retriever": str(arm_cfg["retriever"]), "top_k": str(arm_cfg["top_k"]),
                           "max_tool_calls": str(arm_cfg["max_tool_calls"]), "agent_loop": str(arm_cfg["agent_loop"]), "skill": str(arm_cfg["skill"])})
        params["arm"] = arm
    if c.get("model"):
        params["model"] = c["model"]
    if c.get("retriever") and "retriever" not in params:
        params["retriever"] = c["retriever"]
    return params


def _numeric_mean_metrics(dicts: list[dict], prefix: str = "") -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, int | float) and v is not None:
                key = f"{prefix}{k}"
                sums[key] = sums.get(key, 0.0) + v
                counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums if counts[k]}


def _wall_percentile(mirrors: list[dict], p: float) -> float | None:
    vals = sorted(m["wall_s"] for m in mirrors if isinstance(m.get("wall_s"), int | float))
    if not vals:
        return None
    idx = min(len(vals) - 1, round(p * (len(vals) - 1)))
    return vals[idx]


def _run_metrics_for_agent_rows(entries: list[dict], ctx: dict) -> dict[str, float]:
    """Mirror + `scores` means for one run's (case, variant) rows -- the same values the Braintrust
    dashboards chart, from the same `mirror_for`/`metadata_mirror` pure function."""
    scores_dicts = []
    mirrors = []
    for e in entries:
        row = e.get("row") or {}
        if row.get("scores"):
            scores_dicts.append(row["scores"])
        mirror = mirror_for(e["case_id"], e["variant_id"], row, ctx, category=e["category"])
        mirrors.append(mirror)
    metrics = _numeric_mean_metrics(scores_dicts, prefix="obj/")
    metrics.update(_numeric_mean_metrics(mirrors))
    answered = sum(1 for m in mirrors if m.get("answered"))
    correct_answered = sum(1 for m in mirrors if m.get("correct_answered"))
    if answered:
        metrics["precision_when_answering"] = correct_answered / answered
    usd_total = sum(m["usd"] for m in mirrors if isinstance(m.get("usd"), int | float))
    correct_total = sum(1 for m in mirrors if m.get("correct_all"))
    if usd_total:
        metrics["correct_per_dollar"] = correct_total / usd_total
    p50 = _wall_percentile(mirrors, 0.5)
    p90 = _wall_percentile(mirrors, 0.9)
    if p50 is not None:
        metrics["wall_s_p50"] = p50
    if p90 is not None:
        metrics["wall_s_p90"] = p90
    metrics["safe_accuracy"] = metrics.get("safe", 0.0)
    metrics["misleading_rate"] = metrics.get("misleading", 0.0)
    metrics["silent_failure_rate"] = metrics.get("silent_failure", 0.0)
    metrics["cap_hit_rate"] = metrics.get("cap_hit", 0.0)
    metrics["fabrication_rate"] = metrics.get("fabrication", 0.0)
    metrics["usd_per_case"] = metrics.get("usd", 0.0)
    return metrics


def _run_metrics_for_retrieval_rows(rows: list[dict]) -> dict[str, float]:
    return _numeric_mean_metrics([r["metadata"] for r in rows])


def mirror_context() -> dict:
    from dealpoint.eval.braintrust_showroom import _mirror_context

    return _mirror_context()


def mirror_for(case_id: str, variant_id: str, row: dict, ctx: dict, category: str | None = None) -> dict:
    from dealpoint.eval.braintrust_showroom import mirror_for as _mirror_for

    return _mirror_for(case_id, variant_id, row, ctx, category=category)


# --- traces ----------------------------------------------------------------------------------------


def _variant_to_agent_run_name() -> dict[str, str]:
    from dealpoint.eval.braintrust_sync import _agent_experiment_plans

    return {f"{p['metadata']['arm']}@{p['metadata']['model']}": p["name"] for p in _agent_experiment_plans()}


def agent_trace_plan() -> list[dict]:
    """The 265 agent traces: 108 judged + 6 representative + 151 sweep, exactly
    `braintrust_showroom.step_logs`'s plan. Reused directly rather than re-derived, so a change to the
    judged subset, representative-case rule or sweep manifests is picked up automatically."""
    from dealpoint.eval.braintrust_cockpit import _judged_subset, _load_jsonl, _variant_results
    from dealpoint.eval.braintrust_showroom import agent_run_log_plan
    from dealpoint.eval.braintrust_sync import representative_cases

    subset = _judged_subset()
    plan: list[dict] = []
    for variant in subset.get("variants", []):
        rows = _variant_results(subset, variant["variant_id"])
        for case_id in subset.get("case_ids", list(rows)):
            plan.append({"case_id": case_id, "variant_id": variant["variant_id"], "category": "judged", "row": rows.get(case_id),
                         "run_name": f"judge-{variant['variant_id']}"})
    for sel in representative_cases().get("selections", []):
        if not sel.get("case_id"):
            continue
        rows = {r.get("case_id"): r for r in _load_jsonl(sel["results_path"])} if sel.get("results_path") else {}
        plan.append({"case_id": sel["case_id"], "variant_id": sel["variant_id"], "category": sel["category"], "row": rows.get(sel["case_id"]),
                     "run_name": None})
    variant_to_run = _variant_to_agent_run_name()
    already = {(p["case_id"], p["variant_id"]) for p in plan}
    for p in agent_run_log_plan(already):
        plan.append({**p, "run_name": variant_to_run.get(p["variant_id"])})
    return plan


def retrieval_trace_plan() -> list[dict]:
    """The 406 retrieval traces, `braintrust_showroom.retrieval_log_rows()` plus the run each belongs to."""
    from dealpoint.eval.braintrust_showroom import retrieval_log_rows

    out = []
    for r in retrieval_log_rows():
        out.append({**r, "run_name": RETRIEVER_TO_RUN.get(r["retriever"])})
    return out


def _prompt_variant_ledger_keys() -> list[tuple[str, str, str | None]]:
    """`(variant, case_id)` pairs for the 67 prompt-variant traces -- from the ledger keys on disk
    (`prompt:<variant>:<case_id>`), joined to `playground_rows()` for input/gold/metadata (see the
    build plan section 2: the Playground outputs and judge scores were never mirrored to disk).

    Reads a committed org ledger directly (never `org_report_path`/`org_slug`, which resolve the
    active org over the network) so this pure planning function stays offline even when a
    `BRAINTRUST_API_KEY` happens to be set in the environment."""
    import os

    override = os.environ.get("BRAINTRUST_LEDGER_FILE")
    if override and Path(override).exists():
        path = Path(override)
    else:
        candidates = sorted(Path("data/reports/orgs").glob("*/braintrust_score_ledger.jsonl")) if Path("data/reports/orgs").exists() else []
        if not candidates:
            return []
        path = candidates[0]
    out: dict[tuple[str, str], str | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("experiment") == "logs" and str(row.get("key", "")).startswith("prompt:"):
            _, variant, case_id = row["key"].split(":", 2)
            key = (variant, case_id)
            ts = row.get("ts")
            if key not in out or (ts and (out[key] is None or ts < out[key])):
                out[key] = ts
    return sorted((variant, case_id, ts) for (variant, case_id), ts in out.items())


def prompt_variant_trace_plan() -> list[dict]:
    """The 67 prompt-variant traces: ledger keys joined to `playground_rows()`. No output, no judge
    assessments -- neither exists on disk (build plan §2); `prompt_variant_outputs: "absent"` is
    recorded in the manifest and the tour names the gap. Each carries the ledger row's own `ts`
    (ISO timestamp) when present, so the trace gets a real start time instead of the run's."""
    from dealpoint.eval.braintrust_showroom import playground_rows

    rows_by_case = {r["id"]: r for r in playground_rows()}
    out = []
    for variant, case_id, ts in _prompt_variant_ledger_keys():
        pr = rows_by_case.get(case_id)
        out.append({"variant": variant, "case_id": case_id, "input": (pr or {}).get("input", case_id), "ts": ts,
                    "metadata": {"category": "prompt-variant", "prompt_variant": variant, "case_id": case_id,
                                 "question_id": (pr or {}).get("metadata", {}).get("question_id"),
                                 "gold_answer": (pr or {}).get("metadata", {}).get("gold_answer")},
                    "run_name": f"playground-arm-A-{variant}"})
    return out


# --- datasets --------------------------------------------------------------------------------------


def dataset_plan() -> dict[str, list[dict]]:
    """`{dataset_name: [{"inputs":..,"expectations":..,"tags":..}]}` for all 9 M9 datasets."""
    from dealpoint.eval.braintrust_showroom import judge_packet_rows, playground_rows
    from dealpoint.eval.braintrust_sync import dataset_rows, review_set
    from dealpoint.eval.cases import load_case_set

    out: dict[str, list[dict]] = {}
    for set_name, ds_name in (("dev", "maud-dealpoint-dev"), ("test", "maud-dealpoint-test"),
                              ("counterfactual", "maud-dealpoint-counterfactual"),
                              ("judged_calibration", "maud-dealpoint-judged_calibration"),
                              ("synthetic_query", "maud-dealpoint-synthetic_query")):
        out[ds_name] = [{"inputs": {"input": r["input"]}, "expectations": {"expected": r["expected"]} if r.get("expected") is not None else {},
                         "tags": {k: str(v) for k, v in (r.get("metadata") or {}).items() if not isinstance(v, dict)}}
                        for r in dataset_rows(set_name)]

    out["maud-dealpoint-review-set"] = [{"inputs": {"packet_text": it["packet_text"]}, "expectations": {},
                                         "tags": {"case_id": it["case_id"], "variant_id": it["variant_id"], "reasoning_type": it.get("reasoning_type") or "", "status": it.get("status") or ""}}
                                        for it in review_set()]

    out["maud-dealpoint-playground-armA"] = [{"inputs": {"input": r["input"]}, "expectations": {"expected": r["expected"]},
                                              "tags": {"case_id": r["id"], "arm": "A"}}
                                             for r in playground_rows()]

    out["maud-dealpoint-judge-packets"] = [{"inputs": {"input": r["input"]}, "expectations": {"expected": r["expected"]},
                                            "tags": {"packet_id": r["id"], "case_id": r["metadata"].get("case_id") or ""}}
                                           for r in judge_packet_rows()]

    dev_cases = load_case_set("dev")
    out["maud-dealpoint-retrieval-dev"] = [{"inputs": {"case_id": c["case_id"]},
                                            "expectations": {"gold_span_ids": [f"{s['start']}:{s['end']}" for s in c.get("gold_spans") or []]},
                                            "tags": {"agreement_id": c.get("agreement_id") or ""}}
                                           for c in dev_cases]
    return out


# --- prompts -----------------------------------------------------------------------------------------


def prompt_plan_m9() -> list[dict]:
    """The 8 registered prompts: the three canonical ones (`prompt_plan`), the four arm-A Playground
    variants (chat form), and the judge-panel rubric prompt."""
    from dealpoint.eval.braintrust_showroom import playground_prompts
    from dealpoint.eval.braintrust_sync import prompt_plan
    from dealpoint.eval.judge_run import _system_prompt

    out = [{"name": p["name"], "template": p["content"], "chat": False} for p in prompt_plan()]
    out += [{"name": pr["slug"], "template": pr["messages"], "chat": True} for pr in playground_prompts()]
    out.append({"name": "judge-panel-rubric", "template": [{"role": "system", "content": _system_prompt()}, {"role": "user", "content": "{{input}}"}], "chat": True})
    return out


# --- assessment plan (counts only; the live step recomputes from the same rows) ---------------------


def assessment_counts() -> dict[str, int]:
    """Exact planned assessment counts, derived from the same rows the `assessments` step reads --
    never hardcoded, so a data change fails the dry-run count test loudly."""
    from dealpoint.eval.braintrust_sync import _deepeval_scores_by_trace, _load_jsonl

    agent = agent_trace_plan()
    n_obj = sum(1 for e in agent if (e.get("row") or {}).get("scores"))
    obj_values = sum(len(e["row"]["scores"]) for e in agent if (e.get("row") or {}).get("scores"))

    judge_rows = _load_jsonl("data/eval/judge_scores.jsonl")
    n_llm_judge = sum(1 for r in judge_rows if r.get("ok", True) for d in JUDGE_DIMS if r.get(d) is not None)

    human_rows = _load_jsonl("data/eval/calibration/human_scores.jsonl")
    n_human = sum(1 for r in human_rows for d in JUDGE_DIMS if r.get(d) is not None)

    deepeval_by_trace = _deepeval_scores_by_trace()
    n_deepeval = sum(1 for v in deepeval_by_trace.values() for k in ("task_completion", "tool_correctness", "argument_correctness", "step_efficiency") if v.get(k) is not None)

    retrieval = retrieval_trace_plan()
    n_retrieval_code = sum(1 for r in retrieval if r["metadata"].get("scored_by_ours")) * 3
    n_li_code = sum(1 for r in retrieval if r["metadata"].get("li_hit_rate") is not None) * 2

    return {"obj_traces": n_obj, "obj_values": obj_values, "llm_judge": n_llm_judge, "human": n_human,
            "deepeval": n_deepeval, "retrieval_code": n_retrieval_code, "li_code": n_li_code}


# --- registry ------------------------------------------------------------------------------------


REGISTRY_ALIASES = {"champion": "D@gemini-3.1-flash-lite", "baseline": "A@haiku", "cost-floor": "D@qwen3.7-flash"}


def registry_plan() -> list[dict]:
    """One `dealpoint-agent` version per system@model with the run it reuses, and the three aliases."""
    plans = mirror_run_plan()
    by_variant: dict[str, dict] = {}
    from dealpoint.eval.braintrust_showroom import classify_experiment

    for p in plans:
        c = classify_experiment(p["name"])
        if c["axis"] not in ("system", "model"):
            continue
        arm, model = c.get("arm"), c.get("model")
        if not arm or not model:
            continue
        variant_id = f"{arm}@{model}"
        by_variant.setdefault(variant_id, p["name"])
    out = [{"variant_id": v, "run_name": name} for v, name in sorted(by_variant.items())]
    return out


# --- disk guard ------------------------------------------------------------------------------------


def mlflow_dir_bytes() -> int:
    total = 0
    if MLFLOW_DATA_DIR.exists():
        for p in MLFLOW_DATA_DIR.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


# --- mlflow client / step plumbing (lazy import) ---------------------------------------------------


def _client(tracking_uri: str | None = None):
    import mlflow
    import mlflow.genai
    from mlflow import MlflowClient

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient()


def _ensure_experiment(client, name: str = MLFLOW_EXPERIMENT) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    tracking_uri = client.tracking_uri
    artifact_location = None
    if tracking_uri.startswith("sqlite:///"):
        artifact_location = str(Path(tracking_uri.removeprefix("sqlite:///")).parent / "artifacts")
    return client.create_experiment(name, artifact_location=artifact_location)


def _find_run(client, exp_id: str, key: str):
    runs = client.search_runs([exp_id], filter_string=f"tags.`dealpoint.key` = '{key}'", max_results=1)
    return runs[0] if runs else None


def _find_trace(client, exp_id: str, key: str) -> Any:
    import mlflow
    import mlflow.genai

    traces: Any = mlflow.search_traces(locations=[exp_id], filter_string=f"tags.`dealpoint.key` = '{key}'", max_results=1, return_type="list")
    return traces[0] if traces else None


# --- steps -------------------------------------------------------------------------------------------


def step_datasets(client, live: bool, manifest: dict) -> None:
    plan = dataset_plan()
    counts = {name: len(rows) for name, rows in plan.items()}
    print(f"datasets: {len(plan)} datasets: {counts}")
    manifest["datasets"] = counts
    if not live:
        return
    import mlflow
    import mlflow.genai

    exp_id = _ensure_experiment(client)
    for name, rows in plan.items():
        ds = mlflow.genai.create_dataset(name=name, experiment_id=exp_id)
        records = [{"inputs": r["inputs"], "expectations": r["expectations"], "tags": r["tags"]} for r in rows]
        if records:
            ds.merge_records(records)


def step_prompts(client, live: bool, manifest: dict) -> None:
    plan = prompt_plan_m9()
    print(f"prompts: {len(plan)} prompts: {', '.join(p['name'] for p in plan)}")
    manifest["prompts"] = [p["name"] for p in plan]
    if not live:
        return
    import mlflow
    import mlflow.genai

    for p in plan:
        try:
            existing_versions = client.search_prompt_versions(p["name"])
        except Exception:  # noqa: BLE001 - a fresh prompt name has no versions yet, never fatal
            existing_versions = []
        latest = max(existing_versions, key=lambda v: v.version, default=None) if existing_versions else None
        if latest is not None and latest.template == p["template"]:
            version = latest.version
        else:
            pv = mlflow.genai.register_prompt(name=p["name"], template=p["template"])
            version = pv.version
        alias = PROMPT_ALIASES.get(p["name"])
        if alias:
            client.set_prompt_alias(name=p["name"], alias=alias, version=version)


def step_runs(client, live: bool, manifest: dict) -> None:
    plans = mirror_run_plan()
    print(f"runs: {len(plans)} runs (one per Braintrust experiment)")
    manifest["runs"] = {"planned": len(plans), "names": [p["name"] for p in plans]}
    if not live:
        return
    exp_id = _ensure_experiment(client)
    agent = agent_trace_plan()
    by_run: dict[str, list[dict]] = {}
    for e in agent:
        if e.get("run_name"):
            by_run.setdefault(e["run_name"], []).append(e)
    retrieval = retrieval_trace_plan()
    ret_by_run: dict[str, list[dict]] = {}
    for r in retrieval:
        if r.get("run_name"):
            ret_by_run.setdefault(r["run_name"], []).append(r)
    ctx = mirror_context()
    created = 0
    for p in plans:
        run = _find_run(client, exp_id, p["name"])
        if run is None:
            run = client.create_run(exp_id, tags=p["tags"])
            created += 1
        else:
            for k, v in p["tags"].items():
                client.set_tag(run.info.run_id, k, v)
        for k, v in _run_params_for(p).items():
            client.log_param(run.info.run_id, k, v)
        if p["name"] in by_run:
            metrics = _run_metrics_for_agent_rows(by_run[p["name"]], ctx)
        elif p["name"] in ret_by_run:
            metrics = _run_metrics_for_retrieval_rows(ret_by_run[p["name"]])
        else:
            metrics = {}
        for k, v in metrics.items():
            if v is not None:
                client.log_metric(run.info.run_id, k, float(v))
        client.set_tag(run.info.run_id, "mlflow.note.content", p["description"])
    manifest["runs"]["created_this_run"] = created


def _log_span_tree(client, exp_id: str, run_id: str | None, name: str, key: str, *, inputs, tags: dict, t0_ns: int, dur_ns: int, output=None) -> None:
    """A single-span trace (retrieval, prompt-variant): no per-step trajectory exists for these
    rows, so there is nothing to nest -- only the root span."""
    root = client.start_trace(name, inputs=inputs, tags={**tags, "dealpoint.key": key}, experiment_id=exp_id, start_time_ns=t0_ns, run_id=run_id)
    client.end_trace(root.trace_id, outputs=output, end_time_ns=t0_ns + dur_ns)


def _run_start_ns(client, run_id: str | None) -> int:
    """The run's own start time in ns -- the fallback timestamp for a trace whose row carries no
    timestamp of its own (retrieval rows; prompt-variant rows with no ledger `ts`)."""
    if not run_id:
        return 0
    run = client.get_run(run_id)
    return int(run.info.start_time or 0) * 1_000_000


def _span_type_for(node: dict) -> str:
    name = node.get("name", "")
    return {"agent": "AGENT", "search_agreement": "RETRIEVER", "scoring": "EVALUATOR", "final_answer": "TASK"}.get(name, "TOOL")


def _node_io(node: dict) -> tuple[Any, Any]:
    """(inputs, outputs) for one `log_hierarchy` node -- the same fields the Braintrust replay's
    `_emit_span_tree` logs as `input`/`output` on each span."""
    span = node.get("span")
    if span:
        return span.get("args"), span.get("retrieved_text")
    if "finding" in node:
        return None, node["finding"]
    if "provenance" in node:
        return None, node["provenance"]
    return None, None


def _emit_mlflow_span_tree(client, trace_id: str, parent_id: str, node: dict, cursor: list[int], t1_ns: int) -> None:
    """Recursively emit `node` (from `braintrust_sync.log_hierarchy`, the same function the
    Braintrust replay uses) as a nested MLflow span under `parent_id`, so the agent/search/tool/
    scoring nesting matches the Braintrust trace exactly. Spans are ordered in time within
    [start, t1_ns) via `cursor`, a mutable one-element list shared across the recursion -- the
    stored rows carry no per-step timestamps, so this keeps every child's span strictly within
    its parent's real [start, end) window rather than resorting to "now"."""
    inputs, outputs = _node_io(node)
    start = min(cursor[0], t1_ns - 1)
    span = client.start_span(node.get("name", "span"), trace_id=trace_id, parent_id=parent_id,
                             span_type=_span_type_for(node), inputs=inputs, start_time_ns=start)
    cursor[0] = min(start + 1_000_000, t1_ns)
    for child in node.get("children", []) or []:
        _emit_mlflow_span_tree(client, trace_id, span.span_id, child, cursor, t1_ns)
    end = max(start, cursor[0])
    client.end_span(trace_id, span.span_id, outputs=outputs, end_time_ns=end)


def step_traces(client, live: bool, manifest: dict, limit: int | None = None) -> None:
    agent = agent_trace_plan()
    retrieval = retrieval_trace_plan()
    prompt_variant = prompt_variant_trace_plan()
    total = len(agent) + len(retrieval) + len(prompt_variant)
    print(f"traces: {total} traces planned ({len(agent)} agent, {len(retrieval)} retrieval, {len(prompt_variant)} prompt-variant)")
    manifest["traces"] = {"planned": total, "agent": len(agent), "retrieval": len(retrieval), "prompt_variant": len(prompt_variant), "written_this_run": 0}
    if not live:
        return
    import mlflow
    import mlflow.genai

    from dealpoint.eval.braintrust_sync import log_hierarchy

    exp_id = _ensure_experiment(client)
    name_to_run_id: dict[str, str] = {}

    def _run_id_for(name: str | None) -> str | None:
        if not name:
            return None
        if name not in name_to_run_id:
            run = _find_run(client, exp_id, name)
            if run is None:
                run = client.create_run(exp_id, tags={"dealpoint.key": name})
            name_to_run_id[name] = run.info.run_id
        return name_to_run_id[name]

    ctx = mirror_context()
    written = 0
    for e in (agent[:limit] if limit else agent):
        key = f"{e['case_id']}:{e['variant_id']}:{e['category']}"
        if _find_trace(client, exp_id, key) is not None:
            continue
        row = e.get("row") or {}
        rec = row.get("record") or {}
        t0_ms = row.get("t_ms") or rec.get("t_ms") or 0
        wall_ms = (row.get("scores") or {}).get("wall_ms") or rec.get("wall_ms") or 0
        t0_ns = int(t0_ms) * 1_000_000
        t1_ns = t0_ns + max(int(wall_ms), 1) * 1_000_000
        mirror = mirror_for(e["case_id"], e["variant_id"], row, ctx, category=e["category"])
        tags = {"case_id": e["case_id"], "variant_id": e["variant_id"], "category": e["category"], **{k: str(v) for k, v in mirror.items() if v is not None and not isinstance(v, dict)}}
        try:
            from dealpoint.corpus.document import load_document
            from dealpoint.eval.cases import find_case, resolve_document_id

            case = find_case(e["case_id"]); doc = load_document(resolve_document_id(case))
        except (KeyError, FileNotFoundError):
            case, doc = {}, None
        hierarchy = log_hierarchy(row, case, doc, judge_dims=None)
        root = client.start_trace(f"{e['case_id']} | {e['variant_id']}", inputs={"case_id": e["case_id"]},
                                  tags={**tags, "dealpoint.key": key}, experiment_id=exp_id, start_time_ns=t0_ns,
                                  run_id=_run_id_for(e.get("run_name")))
        cursor = [t0_ns]
        for child in hierarchy.get("children", []):
            _emit_mlflow_span_tree(client, root.trace_id, root.span_id, child, cursor, t1_ns)
        client.end_trace(root.trace_id, outputs=row.get("finding"), end_time_ns=t1_ns)
        written += 1
        if written % 20 == 0:
            # Each agent trace is now many spans (log_hierarchy's full tree), not one; without a
            # periodic flush the async export queue (default size 1000) overflows on 738 traces
            # and silently discards traces past its capacity.
            mlflow.flush_trace_async_logging()
    mlflow.flush_trace_async_logging()
    for r in (retrieval[:limit] if limit else retrieval):
        key = f"{r['case_id']}:{r['retriever']}:retrieval"
        if _find_trace(client, exp_id, key) is not None:
            continue
        run_id = _run_id_for(r.get("run_name"))
        _log_span_tree(client, exp_id, run_id, f"{r['case_id']} | retrieval:{r['retriever']}", key,
                       inputs={"query": r["input"]}, tags={k: str(v) for k, v in r["metadata"].items() if v is not None and not isinstance(v, dict)},
                       t0_ns=_run_start_ns(client, run_id), dur_ns=1_000_000, output=r["output"])
        written += 1
        if written % 50 == 0:
            mlflow.flush_trace_async_logging()
    mlflow.flush_trace_async_logging()
    for pv in (prompt_variant[:limit] if limit else prompt_variant):
        key = f"{pv['case_id']}:{pv['variant']}:prompt-variant"
        if _find_trace(client, exp_id, key) is not None:
            continue
        run_id = _run_id_for(pv.get("run_name"))
        if pv.get("ts"):
            t0_ns = int(datetime.fromisoformat(pv["ts"]).timestamp() * 1000) * 1_000_000
        else:
            t0_ns = _run_start_ns(client, run_id)
        _log_span_tree(client, exp_id, run_id, f"{pv['case_id']} | prompt:{pv['variant']}", key,
                       inputs={"input": pv["input"]}, tags={k: str(v) for k, v in pv["metadata"].items() if v is not None},
                       t0_ns=t0_ns, dur_ns=1_000_000, output=None)
        written += 1
        if written % 50 == 0:
            mlflow.flush_trace_async_logging()
    mlflow.flush_trace_async_logging()
    manifest["traces"]["written_this_run"] = written


def _existing_assessment_names(trace) -> set[str]:
    return {a.name for a in (trace.info.assessments or [])} if trace is not None else set()


def step_assessments(client, live: bool, manifest: dict, limit: int | None = None) -> None:
    counts = assessment_counts()
    total = counts["obj_values"] + counts["llm_judge"] + counts["human"] + counts["deepeval"] + counts["retrieval_code"] + counts["li_code"]
    print(f"assessments: {total} planned ({counts})")
    manifest["assessments"] = {**counts, "written_this_run": 0}
    if not live:
        return
    import mlflow
    import mlflow.entities
    import mlflow.genai

    from dealpoint.eval.braintrust_sync import _deepeval_scores_by_trace, _load_jsonl

    exp_id = _ensure_experiment(client)
    written = 0

    def _feedback(trace_id: str, name: str, value, source_type: str, source_id: str, rationale: str | None = None, extra: dict | None = None) -> None:
        nonlocal written
        mlflow.log_feedback(trace_id=trace_id, name=name, value=value, rationale=rationale, metadata=extra or {},
                            source=mlflow.entities.AssessmentSource(source_type=source_type, source_id=source_id))
        written += 1

    agent = agent_trace_plan()
    for e in (agent[:limit] if limit else agent):
        key = f"{e['case_id']}:{e['variant_id']}:{e['category']}"
        trace = _find_trace(client, exp_id, key)
        if trace is None:
            continue
        have = _existing_assessment_names(trace)
        row = e.get("row") or {}
        for name, value in (row.get("scores") or {}).items():
            aname = f"obj/{name}"
            if aname in have or not isinstance(value, int | float | bool):
                continue
            _feedback(trace.info.trace_id, aname, float(bool(value)) if isinstance(value, bool) else float(value), "CODE", "dealpoint.eval.scorers")
        case = None
        try:
            from dealpoint.eval.cases import find_case

            case = find_case(e["case_id"])
        except Exception:  # noqa: BLE001 - unknown/redacted ids stay unannotated
            case = None
        if case and "gold_answer" not in have:
            mlflow.log_expectation(trace_id=trace.info.trace_id, name="gold_answer", value=case.get("gold_answer"))
            written += 1

    judge_rows = _load_jsonl("data/eval/judge_scores.jsonl")
    from dealpoint.eval.braintrust_sync import _judged_variant_id_for

    for e in (agent[:limit] if limit else agent):
        if e["category"] != "judged":
            continue
        key = f"{e['case_id']}:{e['variant_id']}:{e['category']}"
        trace = _find_trace(client, exp_id, key)
        if trace is None:
            continue
        have = _existing_assessment_names(trace)
        arm, _, model = e["variant_id"].partition("@")
        short = _judged_variant_id_for(arm, model) or e["variant_id"]
        for r in judge_rows:
            if r.get("case_id") != e["case_id"] or r.get("variant_id") != short or not r.get("ok", True):
                continue
            for d in JUDGE_DIMS:
                if r.get(d) is None:
                    continue
                aname = f"judge/{r['judge_family'].lower()}/{d}"
                if aname in have:
                    continue
                _feedback(trace.info.trace_id, aname, (r[d] - 1.0) / 4.0, "LLM_JUDGE", r.get("judge_model") or "unknown",
                         rationale=r.get("notes") or r.get("failure_detail"))

    human_rows = _load_jsonl("data/eval/calibration/human_scores.jsonl")
    for h in human_rows:
        from dealpoint.eval.braintrust_showroom import judge_packet_rows

        pid = h["packet_id"]
        meta = next((r["metadata"] for r in judge_packet_rows() if r["id"] == pid), None)
        if not meta:
            continue
        case_id, variant_id = meta.get("case_id"), meta.get("variant_id")
        matches = [e for e in agent if e["case_id"] == case_id and e["variant_id"] == variant_id and e["category"] == "judged"]
        if not matches:
            continue
        trace = _find_trace(client, exp_id, f"{case_id}:{variant_id}:judged")
        if trace is None:
            continue
        have = _existing_assessment_names(trace)
        for d in JUDGE_DIMS:
            if h.get(d) is None:
                continue
            aname = f"human/{d}"
            if aname in have:
                continue
            _feedback(trace.info.trace_id, aname, (h[d] - 1.0) / 4.0, "HUMAN", h.get("scorer") or "lawyer",
                     extra={"scored_at": h.get("scored_at") or ""})

    deepeval_by_trace = _deepeval_scores_by_trace()
    for (case_id, variant_id), metrics in deepeval_by_trace.items():
        matches = [e for e in agent if e["case_id"] == case_id and e["variant_id"] == variant_id]
        if not matches:
            continue
        entry = matches[0]
        trace = _find_trace(client, exp_id, f"{entry['case_id']}:{entry['variant_id']}:{entry['category']}")
        if trace is None:
            continue
        have = _existing_assessment_names(trace)
        for name, v in metrics.items():
            score = v.get("score") if isinstance(v, dict) else v
            aname = f"deepeval/{name}"
            if score is None or aname in have:
                continue
            _feedback(trace.info.trace_id, aname, float(score), "CODE", "deepeval")

    retrieval = retrieval_trace_plan()
    for r in (retrieval[:limit] if limit else retrieval):
        key = f"{r['case_id']}:{r['retriever']}:retrieval"
        trace = _find_trace(client, exp_id, key)
        if trace is None:
            continue
        have = _existing_assessment_names(trace)
        m = r["metadata"]
        if m.get("scored_by_ours"):
            for name in ("hit_at_5", "hit_at_10", "mrr"):
                aname = f"obj/{name}"
                if m.get(name) is None or aname in have:
                    continue
                _feedback(trace.info.trace_id, aname, float(m[name]), "CODE", "dealpoint.eval.scorers")
        if m.get("li_hit_rate") is not None and "li/hit_rate" not in have:
            _feedback(trace.info.trace_id, "li/hit_rate", float(m["li_hit_rate"]), "CODE", "llama_index")
        if m.get("li_mrr") is not None and "li/mrr" not in have:
            _feedback(trace.info.trace_id, "li/mrr", float(m["li_mrr"]), "CODE", "llama_index")

    manifest["assessments"]["written_this_run"] = written


DETERMINISTIC_SCORER_NAMES = ("grounded_accuracy", "answer_correct", "citation_gold_overlap", "citation_verbatim", "abstain_correct", "skill_adherence")


def build_deterministic_scorers() -> dict:
    """The six `dealpoint.eval.scorers` functions as in-process `@mlflow.genai.scorer`s -- they RUN
    here (Braintrust could not run a deterministic scorer server-side)."""
    import mlflow
    import mlflow.genai

    from dealpoint.eval.braintrust_sync import _make_scorer_handler

    out = {}
    for name in DETERMINISTIC_SCORER_NAMES:
        handler = _make_scorer_handler(name)

        def _scorer(outputs=None, metadata=None, _handler=handler):
            return _handler(output=outputs, metadata=metadata)

        _scorer.__name__ = name
        out[name] = mlflow.genai.scorer(_scorer)
    return out


def build_judges() -> dict:
    """The four dimension judges on OpenRouter, `make_judge`'d but never invoked here."""
    from mlflow.genai.judges import make_judge

    from dealpoint.eval.braintrust_showroom import judge_scorer_messages

    out = {}
    for dim in JUDGE_DIMS:
        msgs = judge_scorer_messages(dim)
        instructions = "\n\n".join(m["content"] for m in msgs).replace("{{input}}", "{{ inputs }}").replace("{{output}}", "{{ outputs }}").replace("{{expected}}", "{{ expectations }}")
        out[dim] = make_judge(name=f"judge-{dim}", instructions=instructions, model=f"openai:/{JUDGE_MODEL_FOR_DIM[dim]}", base_url=OPENROUTER_BASE_URL)
    return out


def step_scorers(client, live: bool, manifest: dict) -> None:
    print(f"scorers: {len(DETERMINISTIC_SCORER_NAMES)} in-process scorers + {len(JUDGE_DIMS)} OpenRouter judges (construction only)")
    manifest["scorers"] = {"deterministic": list(DETERMINISTIC_SCORER_NAMES), "judges": list(JUDGE_DIMS),
                           "note": "the six deterministic scorers run in-process here; Braintrust could not run them server-side"}
    if not live:
        return
    build_deterministic_scorers()
    build_judges()


def step_evaluations(client, live: bool, manifest: dict) -> None:
    from dealpoint.eval.braintrust_showroom import classify_experiment

    axes = sorted({classify_experiment(p["name"])["axis"] for p in mirror_run_plan()})
    print(f"evaluations: one mlflow.genai.evaluate run per axis pool: {axes}")
    manifest["evaluations"] = {"axes": axes, "written_this_run": 0}
    if not live:
        return
    import mlflow
    import mlflow.genai

    exp_id = _ensure_experiment(client)
    scorers = list(build_deterministic_scorers().values())
    written = 0
    import pandas as pd

    for axis in axes:
        names = [p["name"] for p in mirror_run_plan() if classify_experiment(p["name"])["axis"] == axis]
        frames = []
        for n in names:
            run = _find_run(client, exp_id, n)
            if run is None:
                continue
            try:
                frames.append(mlflow.search_traces(locations=[exp_id], run_id=run.info.run_id, return_type="pandas"))
            except Exception:  # noqa: BLE001, S112 - a run with no traces yet is skipped, never fatal
                continue
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            # A structural skip, not a swallowed failure: the `crosscheck` axis's one run
            # (deepeval-crosscheck) has no traces of its own -- DeepEval's assessments attach
            # onto the existing judged agent traces instead (`step_assessments`).
            print(f"  evaluations: axis {axis} has no traces of its own; skipped")
            continue
        mlflow.genai.evaluate(data=df, scorers=scorers)
        written += 1
    manifest["evaluations"]["written_this_run"] = written


def step_registry(client, live: bool, manifest: dict) -> None:
    plan = registry_plan()
    print(f"registry: dealpoint-agent, {len(plan)} versions, aliases {REGISTRY_ALIASES}")
    manifest["registry"] = {"versions": len(plan), "aliases": REGISTRY_ALIASES}
    if not live:
        return
    exp_id = _ensure_experiment(client)
    try:
        client.create_registered_model("dealpoint-agent")
    except Exception:  # noqa: BLE001, S110 - the model may already exist from a prior --live run
        pass
    verdicts = _axis_to_dashboard_text()
    from dealpoint.eval.braintrust_showroom import classify_experiment

    version_by_variant: dict[str, str] = {}
    for entry in plan:
        run = _find_run(client, exp_id, entry["run_name"])
        if run is None:
            continue
        existing = [v for v in client.search_model_versions("name = 'dealpoint-agent'") if v.tags.get("config_hash") == entry["run_name"]]
        if existing:
            version_by_variant[entry["variant_id"]] = existing[0].version
            continue
        mv = client.create_model_version("dealpoint-agent", source=f"runs:/{run.info.run_id}", run_id=run.info.run_id,
                                         tags={"config_hash": entry["run_name"], "variant_id": entry["variant_id"]},
                                         description=verdicts.get(classify_experiment(entry["run_name"])["axis"], ""))
        version_by_variant[entry["variant_id"]] = mv.version
    for alias, variant_id in REGISTRY_ALIASES.items():
        version = version_by_variant.get(variant_id)
        if version:
            client.set_registered_model_alias("dealpoint-agent", alias, version)


def step_views(client, live: bool, manifest: dict) -> None:
    from dealpoint.eval.braintrust_cockpit import DASHBOARDS, chart_catalogue

    catalogue = chart_catalogue()
    entries = []
    for name, text, chart_ids in DASHBOARDS:
        for cid in chart_ids:
            chart = catalogue.get(cid)
            if not chart:
                continue
            entries.append({"question": name, "tag_filter": " and ".join(chart.get("filters") or []), "metric": chart["measure"] if isinstance(chart["measure"], str) else chart.get("title"), "title": chart["title"]})
    print(f"views: no chart/view API in OSS MLflow 3.16; {len(entries)} run-comparison entries written to {_views_path()}")
    manifest["views"] = {"entries": len(entries)}
    if not live:
        return
    _views_path().parent.mkdir(parents=True, exist_ok=True)
    _views_path().write_text(json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8")
    exp_id = _ensure_experiment(client)
    overview_text = _axis_to_dashboard_text().get("traces", "")
    client.set_experiment_tag(exp_id, "mlflow.note.content", overview_text)


# --- main ----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    dry_run = "--live" not in argv or "--dry-run" in argv
    live = not dry_run
    only = None
    limit = None
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            only = set(argv[i + 1].split(","))
        if a == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])

    manifest = {"mode": "live" if live else "dry-run", "started_at": datetime.now(UTC).isoformat(), "experiment": MLFLOW_EXPERIMENT,
                "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI), "prompt_variant_outputs": "absent"}
    print(f"{'LIVE' if live else 'DRY RUN'}: mlflow mirror on {MLFLOW_EXPERIMENT} ({manifest['tracking_uri']})")

    client = _client() if live else None
    if live and mlflow_dir_bytes() > DISK_GUARD_BYTES:
        print(f"data/mlflow/ exceeds the {DISK_GUARD_BYTES} byte guard; refusing to write more", file=sys.stderr)
        return 2

    for name in STEPS:
        if only and name not in only:
            continue
        fn = globals()[f"step_{name}"]
        if name in ("traces", "assessments"):
            fn(client, live, manifest, limit=limit)
        else:
            fn(client, live, manifest)

    manifest["finished_at"] = datetime.now(UTC).isoformat()
    try:
        import mlflow

        manifest["mlflow_version"] = mlflow.__version__
    except ImportError:
        pass
    if live:
        _manifest_path().parent.mkdir(parents=True, exist_ok=True)
        _manifest_path().write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"manifest -> {_manifest_path()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
