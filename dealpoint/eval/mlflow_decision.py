"""M9b: the config decision, on MLflow's own strengths (`specs/milestones/m9b.md`).

Additive to M9 (`dealpoint.eval.mlflow_mirror`): reads the same stored rows through the same pure
functions (`mlflow_mirror.agent_trace_plan`, `mirror_for`, `braintrust_showroom.classify_experiment`,
`arm_parameter_sets`) and makes no model calls. Every metric this module logs is computed by
`mlflow_mirror._run_metrics_for_agent_rows` over the stored rows -- the Braintrust mirror's own
number, never re-derived.

Two comparable pools (M9's `comparable == 1` rule; no cross-pool comparison anywhere):

    judged-18   the same 18 judged cases, every system@model: A@haiku, D@haiku, D@glm,
                D@deepseek-v4-flash, D@qwen3.7-flash, D@gemini-3.1-flash-lite
    test-32     the four GLM systems (A/B/C/D) on the same 32 test cases

Each pool gets one MLflow parent run with a nested child run per configuration (15 metrics, 6
params), a Pareto-frontier artifact set, a registered-model version per child, and a zero-model-call
`mlflow.genai.evaluate(predict_fn=None)` row-level comparison.

Dry run is the default; `--live` executes against `MLFLOW_TRACKING_URI` (default
`http://127.0.0.1:5000`); `--dry-run` always wins over `--live`. `mlflow` and `pandas` are imported
lazily so the planning functions (and the whole offline test suite) work without the `mlflow` extra.

    uv run python -m dealpoint.eval.mlflow_decision                 # dry run
    uv run python -m dealpoint.eval.mlflow_decision --live          # execute
    uv run python -m dealpoint.eval.mlflow_decision --live --only tree,pareto
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

# `mlflow.genai.evaluate` opens one trace per row and exports it through a background thread pool;
# against a SQLite store the default SQLAlchemy pool (size 5, overflow 10) saturates under 10
# configurations' worth of evaluate() calls and each row then blocks for the full 30s connection
# timeout. Must be set before the tracking store's engine is first created for a given URI (module
# import time, ahead of any `_client()` call), never after.
os.environ.setdefault("MLFLOW_SQLALCHEMYSTORE_POOL_SIZE", "20")
os.environ.setdefault("MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW", "40")

MLFLOW_EXPERIMENT = "dealpoint-eval"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
MANIFEST_PATH = Path("data/reports/mlflow_decision_manifest.json")
VIEWS_PATH = Path("data/reports/mlflow_decision_views.json")
REGISTERED_MODEL = "dealpoint-agent"

STEPS = ("tree", "pareto", "views", "registry", "evaluate")
DECISION_TAG = "dealpoint.decision"  # "parent" | "config" | "evaluation"
POOLS = ("judged-18", "test-32")

# The 15 metrics D7 names, in this order. `config_metrics` must return exactly this key set.
DECISION_METRICS = (
    "safe_accuracy", "precision_when_answering", "net_accuracy", "correct_outcome_rate",
    "misleading_rate", "silent_failure_rate", "cap_hit_rate", "fabrication_rate",
    "verbatim_quote_rate", "usd_per_case", "wall_s_p50", "wall_s_p90",
    "tool_calls_per_case", "correct_per_dollar", "usd_per_correct",
)

REGISTRY_ALIASES = {
    "champion": ("judged-18", "D@gemini-3.1-flash-lite"),
    "baseline": ("judged-18", "A@haiku"),
    "cost-floor": ("judged-18", "D@qwen3.7-flash"),
    "safest": ("judged-18", "D@deepseek-v4-flash"),
}

# pareto_frontier minimises x, maximises y.
PARETO_PAIRS = (
    ("usd_per_case", "safe_accuracy"),
    ("wall_s_p50", "precision_when_answering"),
    ("usd_per_case", "net_accuracy"),
)

PARETO_RULE = (
    "Minimise x, maximise y (dealpoint.eval.pareto_report.pareto_frontier, the M6 rule, applied "
    "unchanged). A point is dominated iff another has x<=x' and y>=y' with at least one strict "
    "inequality. Exact ties on both axes keep the lexicographically-smallest config id."
)

# specs/milestones/m9b.md section 2 claims the judged-18 usd_per_case-vs-safe_accuracy frontier
# "names D@gemini and D@qwen". Applying the M6 rule unchanged to the measured numbers instead
# produces D@qwen3.7-flash and D@deepseek-v4-flash: D@deepseek-v4-flash dominates
# D@gemini-3.1-flash-lite on this axis (cheaper, $0.002358 vs $0.005134, and safer, 0.8889 vs
# 0.8333). D@gemini-3.1-flash-lite and D@qwen3.7-flash DO share a frontier -- on usd_per_case vs
# net_accuracy instead, the tie-breaker axis the deployment verdict actually turns on. The M6 rule
# is the authority; this difference is recorded here, in the tour and in the builder's report, and
# specs/milestones/m9b.md is never edited to match.
SPEC_DIFFERENCES_JUDGED18 = [
    {
        "axis": "usd_per_case vs safe_accuracy (judged-18)",
        "spec_claim": (
            "specs/milestones/m9b.md section 2 says this frontier 'names D@gemini and D@qwen "
            "on safe-vs-dollars for judged-18'."
        ),
        "measured_frontier": ["D@qwen3.7-flash", "D@deepseek-v4-flash"],
        "explanation": (
            "D@deepseek-v4-flash dominates D@gemini-3.1-flash-lite here: cheaper ($0.002358 vs "
            "$0.005134) and safer (0.8889 vs 0.8333). D@gemini-3.1-flash-lite and D@qwen3.7-flash "
            "do share a frontier, but on usd_per_case vs net_accuracy instead -- the tie-breaker "
            "axis the deployment verdict actually turns on."
        ),
    }
]

SCORING_DEFINITIONS_SENTENCE = (
    "Safe accuracy is the accuracy point (did not mislead: correct, correct abstention, or a "
    "silent failure); precision when answering is its check (of the cases it answered, how many "
    "were right); net accuracy, correct minus misleading, is the tie-breaker (docs/demo-tour.md "
    "stop 9)."
)

DECISION_SEARCH_SCOPE = "tags.`dealpoint.decision` = 'config'"
SEARCH_SAFE_AND_CHEAP = f"{DECISION_SEARCH_SCOPE} and metrics.safe_accuracy >= 0.8 and metrics.usd_per_case <= 0.003"
SEARCH_RIGHT_AND_FAST = f"{DECISION_SEARCH_SCOPE} and metrics.precision_when_answering >= 0.65 and metrics.wall_s_p50 <= 15"
SEARCH_NEVER_FABRICATE = f"{DECISION_SEARCH_SCOPE} and params.loop = 'agent' and metrics.fabrication_rate = 0"

STOP9_TWIN_CASES = ("contract_144__q05", "contract_39__redacted_q05")


def _manifest_path() -> Path:
    return MANIFEST_PATH


def _views_path() -> Path:
    return VIEWS_PATH


# --- pure planning: the two pools -----------------------------------------------------------------


def decision_pools() -> dict[str, dict[str, list[dict]]]:
    """`{pool_name: {config_label: [agent_trace_plan entries, each carrying its precomputed
    "mirror"]}}` for the two comparable pools (build plan section 0.1). `judged-18` is every
    `category == "judged"` comparable row; `test-32` unions `agent` and `judged` GLM comparable
    rows (`agent_run_log_plan` excludes (case, variant) pairs already planned as `judged`, so the
    raw `agent` category alone holds only 14 of D@glm's 32 rows)."""
    from dealpoint.eval import mlflow_mirror as mm

    ctx = mm.mirror_context()
    pools: dict[str, dict[str, list[dict]]] = {p: {} for p in POOLS}
    for e in mm.agent_trace_plan():
        row = e.get("row") or {}
        mirror = mm.mirror_for(e["case_id"], e["variant_id"], row, ctx, category=e["category"])
        if mirror.get("comparable") != 1:
            continue
        entry = {**e, "mirror": mirror}
        label = mirror["variant_label"]
        if e["category"] == "judged":
            pools["judged-18"].setdefault(label, []).append(entry)
        if e["category"] in ("agent", "judged") and mirror.get("model_label") == "glm":
            pools["test-32"].setdefault(label, []).append(entry)
    return pools


def config_metrics(entries: list[dict], ctx: dict) -> dict[str, float]:
    """The 15 `DECISION_METRICS`, every one of them a Braintrust-mirror number:
    `mlflow_mirror._run_metrics_for_agent_rows` computes safe accuracy, precision, net accuracy and
    every rate; `usd_per_correct` is the only new arithmetic (sum(usd)/sum(correct_all), never
    `1/correct_per_dollar` directly, so the same zero-divide guard applies)."""
    from dealpoint.eval import mlflow_mirror as mm

    m = mm._run_metrics_for_agent_rows(entries, ctx)
    metrics = {
        "safe_accuracy": m["safe_accuracy"],
        "precision_when_answering": m["precision_when_answering"],
        "net_accuracy": m["net_accuracy"],
        "correct_outcome_rate": m["correct_all"],
        "misleading_rate": m["misleading_rate"],
        "silent_failure_rate": m["silent_failure_rate"],
        "cap_hit_rate": m["cap_hit_rate"],
        "fabrication_rate": m["fabrication_rate"],
        "verbatim_quote_rate": m["cite_verbatim_answered"],
        "usd_per_case": m["usd_per_case"],
        "wall_s_p50": m["wall_s_p50"],
        "wall_s_p90": m["wall_s_p90"],
        "tool_calls_per_case": m["tool_calls"],
        "correct_per_dollar": m["correct_per_dollar"],
    }
    usd_total = sum(e["mirror"]["usd"] for e in entries if isinstance(e["mirror"].get("usd"), int | float))
    correct_total = sum(1 for e in entries if e["mirror"].get("correct_all"))
    if correct_total:
        metrics["usd_per_correct"] = usd_total / correct_total
    return metrics


# --- pure planning: the run tree ------------------------------------------------------------------


def _child_description(pool: str, config: str, metrics: dict[str, float], axis_text: str) -> str:
    lead = (
        f"{config} on {pool}: safe accuracy {metrics['safe_accuracy']:.1%}, precision when "
        f"answering {metrics['precision_when_answering']:.1%}, net accuracy "
        f"{metrics['net_accuracy']:.1%}, ${metrics['usd_per_case']:.4f}/case."
    )
    return f"{lead} {axis_text}".strip()


def decision_run_plan() -> list[dict]:
    """Two parent entries and one child entry per configuration (10 total), each child's `metrics`
    dict carrying exactly the 15 `DECISION_METRICS`."""
    from dealpoint.eval import mlflow_mirror as mm
    from dealpoint.eval.braintrust_showroom import classify_experiment

    pools = decision_pools()
    ctx = mm.mirror_context()
    variant_to_run_name = {r["variant_id"]: r["run_name"] for r in mm.registry_plan()}
    axis_text = mm._axis_to_dashboard_text()

    plans: list[dict] = []
    for pool in POOLS:
        configs = pools[pool]
        pool_axis = "model" if pool == "judged-18" else "system"
        parent_key = f"decision/{pool}"
        n_cases = "18" if pool == "judged-18" else "32"
        plans.append({
            "role": "parent", "pool": pool, "key": parent_key, "name": parent_key,
            "tags": {"dealpoint.key": parent_key, DECISION_TAG: "parent", "dealpoint.pool": pool,
                     "comparable": "1", "cases": n_cases, "n_configs": str(len(configs))},
            "description": f"{axis_text.get(pool_axis, '')} {SCORING_DEFINITIONS_SENTENCE}".strip(),
        })
        for config in sorted(configs):
            entries = configs[config]
            metrics = config_metrics(entries, ctx)
            mirror0 = entries[0]["mirror"]
            arm = config.split("@", 1)[0]
            run_name = variant_to_run_name.get(config)
            c = classify_experiment(run_name) if run_name else {}
            tags = {k: str(v) for k, v in c.items()
                    if k in ("axis", "varies", "holds", "arm", "loop", "retriever", "skill", "model", "cases") and v is not None}
            key = f"{parent_key}/{config}"
            tags["dealpoint.key"] = key
            tags[DECISION_TAG] = "config"
            tags["dealpoint.pool"] = pool
            params = {
                "arm": arm, "loop": str(c.get("loop", "")), "retriever": str(c.get("retriever", "")),
                "skill": str(c.get("skill", "")), "model": mirror0["model_label"], "system_label": mirror0["system_label"],
            }
            plans.append({
                "role": "config", "pool": pool, "config": config, "key": key, "name": key,
                "params": params, "tags": tags, "metrics": metrics,
                "description": _child_description(pool, config, metrics, axis_text.get(pool_axis, "")),
            })
    return plans


# --- pure planning: Pareto frontiers and the SVG artifact ------------------------------------------


def _frontier_block(x_key: str, y_key: str, metrics_by_config: dict[str, dict[str, float]]) -> dict:
    from dealpoint.eval.pareto_report import pareto_frontier

    points = [{"config": c, x_key: m.get(x_key), y_key: m.get(y_key)} for c, m in metrics_by_config.items()]
    result = pareto_frontier(points, x_key=x_key, y_key=y_key, id_key="config")
    pts_out, frontier_ids = [], []
    for r in result:
        c = r["config"]
        m = metrics_by_config[c]
        pts_out.append({"config": c, "x": m.get(x_key), "y": m.get(y_key), "frontier": r["frontier"],
                        "frontier_note": r["frontier_note"], "tied_with": r["tied_with"]})
        if r["frontier"]:
            frontier_ids.append(c)
    return {"x": x_key, "y": y_key, "points": pts_out, "frontier": frontier_ids}


def pareto_artifact(pool: str, metrics_by_config: dict[str, dict[str, float]]) -> dict:
    """`pareto.json`'s content for one pool: the M6 rule applied unchanged to the three axis
    pairs, plus the section 0.3 spec/data difference (`judged-18` only)."""
    frontiers = [_frontier_block(x, y, metrics_by_config) for x, y in PARETO_PAIRS]
    return {"pool": pool, "rule": PARETO_RULE, "frontiers": frontiers,
            "spec_differences": SPEC_DIFFERENCES_JUDGED18 if pool == "judged-18" else []}


def _panel_body(points: list[dict], *, x_key: str, y_key: str, title: str, y0: float, height: float,
                width: float = 640, margin: float = 60) -> str:
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    xs = [p["x"] for p in points if isinstance(p.get("x"), int | float)]
    ys = [p["y"] for p in points if isinstance(p.get("y"), int | float)]
    x_min, x_max = (min(xs), max(xs)) if xs else (0.0, 1.0)
    y_min, y_max = (min(ys + [0.0]), max(ys + [1.0])) if ys else (0.0, 1.0)
    if x_min == x_max:
        x_min, x_max = x_min - 0.001, x_max + 0.001
    if y_min == y_max:
        y_min, y_max = y_min - 0.05, y_max + 0.05

    def sx(x: float) -> float:
        return margin + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    lines = [f'<g transform="translate(0,{y0:.2f})">']
    lines.append(f'<text x="{width / 2:.2f}" y="20" font-size="13" text-anchor="middle">{escape(title)}</text>')
    lines.append(f'<line x1="{margin:.2f}" y1="{margin + plot_h:.2f}" x2="{margin + plot_w:.2f}" y2="{margin + plot_h:.2f}" stroke="black"/>')
    lines.append(f'<line x1="{margin:.2f}" y1="{margin:.2f}" x2="{margin:.2f}" y2="{margin + plot_h:.2f}" stroke="black"/>')
    lines.append(f'<text x="{margin:.2f}" y="{margin + plot_h + 15:.2f}" font-size="9">{escape(x_key)}: {x_min:.2f}..{x_max:.2f}</text>')
    lines.append(f'<text x="10" y="{margin:.2f}" font-size="9">{escape(y_key)}</text>')

    frontier_pts = sorted((p for p in points if p.get("frontier")), key=lambda p: p["x"])
    if len(frontier_pts) > 1:
        path = " ".join(f"{sx(p['x']):.2f},{sy(p['y']):.2f}" for p in frontier_pts)
        lines.append(f'<polyline points="{path}" fill="none" stroke="#1a7f37" stroke-width="1.5"/>')
    for p in sorted(points, key=lambda p: p["config"]):
        cx, cy = round(sx(p["x"]), 2), round(sy(p["y"]), 2)
        fill = "#1a7f37" if p.get("frontier") else "white"
        stroke = "#1a7f37" if p.get("frontier") else "#555555"
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        lines.append(f'<text x="{cx + 8:.2f}" y="{cy + 3:.2f}" font-size="9">{escape(p["config"])}</text>')
    lines.append("</g>")
    return "\n".join(lines)


def pareto_svg(points: list[dict], *, x_key: str, y_key: str, title: str) -> str:
    """A deterministic, stdlib-only scatter for one (x_key, y_key) pair: frontier points filled and
    joined by a polyline sorted by x, dominated points hollow, each point labelled with its config.
    No timestamp, no random id, coordinates rounded to 2 decimals -- two calls on the same points
    produce byte-identical output."""
    height = 420
    body = _panel_body(points, x_key=x_key, y_key=y_key, title=title, y0=0.0, height=height)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{height}" viewBox="0 0 640 {height}">\n'
        f'<rect x="0" y="0" width="640" height="{height}" fill="white"/>\n{body}\n</svg>\n'
    )


def _combined_pareto_svg(pool: str, frontiers: list[dict]) -> str:
    """One `pareto.svg` artifact per parent: one `<g>` panel per axis pair, stacked vertically."""
    panel_h = 420
    total_h = panel_h * len(frontiers)
    body = "\n".join(
        _panel_body(f["points"], x_key=f["x"], y_key=f["y"], title=f"{pool}: {f['x']} vs {f['y']}", y0=i * panel_h, height=panel_h)
        for i, f in enumerate(frontiers)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{total_h}" viewBox="0 0 640 {total_h}">\n'
        f'<rect x="0" y="0" width="640" height="{total_h}" fill="white"/>\n{body}\n</svg>\n'
    )


def stop9_verdict(path: Path = Path("docs/demo-tour.md")) -> str:
    """The verdict paragraph from `docs/demo-tour.md` stop 9, verbatim: `"The decision, written
    down: ..."` through the end of that paragraph. Read at build time, never pasted, so the two
    documents cannot drift."""
    text = path.read_text(encoding="utf-8")
    section = text[text.index("### Stop 9. What should we deploy?"):]
    start = section.index("The decision, written down:")
    end = section.index("\n\n", start)
    return section[start:end].strip()


# --- pure planning: row-level evaluation frame (D9) ------------------------------------------------


def evaluation_frame(entries: list[dict], ctx: dict) -> list[dict]:
    """One row per (case, variant) for `mlflow.genai.evaluate`. MLflow's evaluate() only ever
    delivers `inputs`/`outputs`/`expectations`/`trace`/`session` to a scorer -- a `metadata` column
    or a scorer parameter named `metadata` is never populated -- so every value a scorer needs
    (case_id for the six deterministic scorers' case lookup; safe/misleading/correct_all/
    silent_failure for the three pure scorers) travels inside `outputs`."""
    from dealpoint.eval import mlflow_mirror as mm

    rows = []
    for e in entries:
        row = e.get("row") or {}
        mirror = e.get("mirror") or mm.mirror_for(e["case_id"], e["variant_id"], row, ctx, category=e["category"])
        rows.append({
            "inputs": {"case_id": e["case_id"], "variant_id": e["variant_id"]},
            "outputs": {
                "case_id": e["case_id"], "finding": row.get("finding"), "record": row.get("record") or {},
                "safe": mirror.get("safe"), "misleading": mirror.get("misleading"),
                "correct_all": mirror.get("correct_all"), "silent_failure": mirror.get("silent_failure"),
            },
        })
    return rows


def deployable_row(outputs: dict) -> int:
    """`correct or silent` (never misleading) -- by `metadata_mirror`'s own definitions this is
    identical to `safe` (`1 - misleading`); pinned as an identity, not restated as an independent
    metric."""
    return int(bool(outputs.get("correct_all")) or bool(outputs.get("silent_failure")))


def build_evaluation_scorers() -> dict:
    """The six `dealpoint.eval.scorers` (reused from `mlflow_mirror.build_deterministic_scorers`,
    unchanged) plus the three pure row scorers `safe`, `misleading`, `deployable`."""
    import mlflow.genai

    from dealpoint.eval import mlflow_mirror as mm

    def safe(outputs: dict | None = None):
        assert outputs is not None
        return outputs["safe"]

    def misleading(outputs: dict | None = None):
        assert outputs is not None
        return outputs["misleading"]

    def deployable(outputs: dict | None = None):
        assert outputs is not None
        return deployable_row(outputs)

    for fn, name in ((safe, "safe"), (misleading, "misleading"), (deployable, "deployable")):
        fn.__name__ = name

    out = dict(mm.build_deterministic_scorers())
    out["safe"] = mlflow.genai.scorer(safe)
    out["misleading"] = mlflow.genai.scorer(misleading)
    out["deployable"] = mlflow.genai.scorer(deployable)
    return out


# --- mlflow client plumbing (lazy import; reuse mlflow_mirror's) -----------------------------------


def _client(tracking_uri: str | None = None):
    from dealpoint.eval import mlflow_mirror as mm

    return mm._client(tracking_uri)


def _ensure_experiment(client) -> str:
    from dealpoint.eval import mlflow_mirror as mm

    return mm._ensure_experiment(client, MLFLOW_EXPERIMENT)


def _find_run(client, exp_id: str, key: str):
    from dealpoint.eval import mlflow_mirror as mm

    return mm._find_run(client, exp_id, key)


# --- steps -------------------------------------------------------------------------------------------


def step_tree(client, live: bool, manifest: dict) -> None:
    plan = decision_run_plan()
    parents = [p for p in plan if p["role"] == "parent"]
    children = [p for p in plan if p["role"] == "config"]
    print(f"tree: {len(parents)} parents, {len(children)} children, 15 metrics per child")
    manifest["tree"] = {"parents": len(parents), "children": len(children), "created_this_run": 0}
    if not live:
        return
    exp_id = _ensure_experiment(client)
    created = 0
    parent_run_id_by_pool: dict[str, str] = {}
    for p in parents:
        run = _find_run(client, exp_id, p["key"])
        if run is None:
            run = client.create_run(exp_id, run_name=p["name"], tags=p["tags"])
            created += 1
        else:
            for k, v in p["tags"].items():
                client.set_tag(run.info.run_id, k, v)
        client.set_tag(run.info.run_id, "mlflow.note.content", p["description"])
        parent_run_id_by_pool[p["pool"]] = run.info.run_id

    for c in children:
        parent_run_id = parent_run_id_by_pool[c["pool"]]
        tags = {**c["tags"], "mlflow.parentRunId": parent_run_id}
        run = _find_run(client, exp_id, c["key"])
        if run is None:
            run = client.create_run(exp_id, run_name=c["name"], tags=tags)
            created += 1
        else:
            for k, v in tags.items():
                client.set_tag(run.info.run_id, k, v)
        for k, v in c["params"].items():
            client.log_param(run.info.run_id, k, v)
        for k, v in c["metrics"].items():
            client.log_metric(run.info.run_id, k, float(v))
        client.set_tag(run.info.run_id, "mlflow.note.content", c["description"])
    manifest["tree"]["created_this_run"] = created


def step_pareto(client, live: bool, manifest: dict) -> None:
    from dealpoint.eval import mlflow_mirror as mm

    pools = decision_pools()
    print("pareto: 2 parents x 3 artifacts (pareto.json, pareto.svg, decision.md)")
    ctx = mm.mirror_context()
    artifacts_by_pool = {}
    frontiers_by_pool = {}
    for pool in POOLS:
        metrics_by_config = {c: config_metrics(entries, ctx) for c, entries in pools[pool].items()}
        artifact = pareto_artifact(pool, metrics_by_config)
        artifacts_by_pool[pool] = artifact
        frontiers_by_pool[pool] = {f'{f["x"]}|{f["y"]}': f["frontier"] for f in artifact["frontiers"]}
    manifest["pareto"] = {"artifacts_per_parent": 3, "frontiers": frontiers_by_pool}
    if not live:
        return
    import tempfile

    exp_id = _ensure_experiment(client)
    decision_text = stop9_verdict()
    for pool in POOLS:
        run = _find_run(client, exp_id, f"decision/{pool}")
        if run is None:
            continue
        artifact = artifacts_by_pool[pool]
        svg = _combined_pareto_svg(pool, artifact["frontiers"])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "pareto.json").write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
            (tmp_path / "pareto.svg").write_text(svg, encoding="utf-8")
            (tmp_path / "decision.md").write_text(decision_text + "\n", encoding="utf-8")
            existing = {f.path for f in client.list_artifacts(run.info.run_id)}
            for name in ("pareto.json", "pareto.svg", "decision.md"):
                if name not in existing:
                    client.log_artifact(run.info.run_id, str(tmp_path / name))


def step_views(client, live: bool, manifest: dict) -> None:
    probed = ["create_chart", "set_experiment_view", "create_experiment_view", "set_chart"]
    chart_api_available = any(hasattr(client, name) for name in probed)
    entries = []
    for pool in POOLS:
        entries.append({"pool": pool, "kind": "parallel_coordinates", "params": ["loop", "retriever", "skill", "model"],
                        "metrics": ["safe_accuracy", "precision_when_answering", "usd_per_case", "wall_s_p50"],
                        "title": f"{pool}: parallel coordinates", "clicks": (
                            "Experiment > Chart view > + > Parallel coordinates; "
                            "params loop, retriever, skill, model; metrics safe_accuracy, "
                            "precision_when_answering, usd_per_case, wall_s_p50.")})
        entries.append({"pool": pool, "kind": "scatter", "x": "usd_per_case", "y": "safe_accuracy", "color": "model",
                        "title": f"{pool}: usd_per_case vs safe_accuracy",
                        "clicks": "Experiment > Chart view > + > Scatter; x usd_per_case; y safe_accuracy; colour by model."})
        entries.append({"pool": pool, "kind": "scatter", "x": "wall_s_p50", "y": "precision_when_answering",
                        "title": f"{pool}: wall_s_p50 vs precision_when_answering",
                        "clicks": "Experiment > Chart view > + > Scatter; x wall_s_p50; y precision_when_answering."})
        entries.append({"pool": pool, "kind": "bar", "metric": "correct_per_dollar",
                        "title": f"{pool}: correct_per_dollar by run",
                        "clicks": "Experiment > Chart view > + > Bar; metric correct_per_dollar; group by run."})
    print(f"views: no chart/view API in OSS MLflow 3.16 (probed={probed}, available={chart_api_available}); {len(entries)} saved-view entries written to {_views_path()}")
    manifest["views"] = {"entries": len(entries), "chart_api_available": chart_api_available, "probed": probed}
    if not live:
        return
    _views_path().parent.mkdir(parents=True, exist_ok=True)
    _views_path().write_text(json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8")


def step_registry(client, live: bool, manifest: dict) -> None:
    plan = decision_run_plan()
    children = [p for p in plan if p["role"] == "config"]
    print(f"registry: {REGISTERED_MODEL}, {len(children)} versions, aliases {REGISTRY_ALIASES}")
    manifest["registry"] = {"versions": len(children), "aliases": {a: f"{pool}/{c}" for a, (pool, c) in REGISTRY_ALIASES.items()}, "created_this_run": 0}
    if not live:
        return
    exp_id = _ensure_experiment(client)
    try:
        client.create_registered_model(REGISTERED_MODEL)
    except Exception:  # noqa: BLE001, S110 - the model may already exist from a prior --live run
        pass
    created = 0
    version_by_key: dict[str, str] = {}
    for child in children:
        key = child["key"]
        run = _find_run(client, exp_id, key)
        if run is None:
            continue
        existing = [v for v in client.search_model_versions(f"name = '{REGISTERED_MODEL}'") if v.tags.get("config_hash") == key]
        if existing:
            version_by_key[key] = existing[0].version
            continue
        mv = client.create_model_version(REGISTERED_MODEL, source=f"runs:/{run.info.run_id}", run_id=run.info.run_id,
                                         tags={"config_hash": key, "variant_id": child["config"], "pool": child["pool"]},
                                         description=child["description"])
        for k, v in child["metrics"].items():
            client.set_model_version_tag(REGISTERED_MODEL, mv.version, k, str(v))
        version_by_key[key] = mv.version
        created += 1

    for alias, (pool, config) in REGISTRY_ALIASES.items():
        key = f"decision/{pool}/{config}"
        version = version_by_key.get(key)
        if version is None:
            existing = [v for v in client.search_model_versions(f"name = '{REGISTERED_MODEL}'") if v.tags.get("config_hash") == key]
            version = existing[0].version if existing else None
        if version:
            client.set_registered_model_alias(REGISTERED_MODEL, alias, version)
    manifest["registry"]["created_this_run"] = created


def step_evaluate(client, live: bool, manifest: dict) -> None:
    pools = decision_pools()
    configs = [(pool, config) for pool in POOLS for config in sorted(pools[pool])]
    print(f"evaluate: {len(configs)} configurations x mlflow.genai.evaluate(predict_fn=None)")
    manifest["evaluate"] = {"configs": len(configs), "written_this_run": 0}
    if not live:
        return
    import mlflow
    import mlflow.genai
    import pandas as pd

    from dealpoint.eval import mlflow_mirror as mm

    exp_id = _ensure_experiment(client)
    ctx = mm.mirror_context()
    scorers = list(build_evaluation_scorers().values())
    written = 0
    for pool, config in configs:
        key = f"eval/{pool}/{config}"
        if _find_run(client, exp_id, key) is not None:
            continue
        frame = pd.DataFrame(evaluation_frame(pools[pool][config], ctx))
        with mlflow.start_run(run_name=key, experiment_id=exp_id,
                              tags={"dealpoint.key": key, DECISION_TAG: "evaluation", "dealpoint.pool": pool, "dealpoint.config": config}):
            mlflow.genai.evaluate(data=frame, scorers=scorers, predict_fn=None)
        written += 1
    manifest["evaluate"]["written_this_run"] = written


# --- main ----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from dealpoint.eval.mlflow_mirror import DISK_GUARD_BYTES, mlflow_dir_bytes

    argv = list(argv) if argv is not None else sys.argv[1:]
    dry_run = "--live" not in argv or "--dry-run" in argv
    live = not dry_run
    only = None
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            only = set(argv[i + 1].split(","))

    manifest = {"mode": "live" if live else "dry-run", "started_at": datetime.now(UTC).isoformat(), "experiment": MLFLOW_EXPERIMENT,
                "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)}
    print(f"{'LIVE' if live else 'DRY RUN'}: mlflow decision on {MLFLOW_EXPERIMENT} ({manifest['tracking_uri']})")

    client = _client() if live else None
    if live and mlflow_dir_bytes() > DISK_GUARD_BYTES:
        print(f"data/mlflow/ exceeds the {DISK_GUARD_BYTES} byte guard; refusing to write more", file=sys.stderr)
        return 2

    for name in STEPS:
        if only and name not in only:
            continue
        globals()[f"step_{name}"](client, live, manifest)

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
