"""gate_m9b: the config decision on MLflow's own strengths (specs/milestones/m9b.md).

Offline, no network: MLflow runs against a temporary SQLite tracking URI with a real client (no
fake), matching tests/test_mlflow_mirror.py's proven pattern. Makes no model calls.
"""

from __future__ import annotations

import json
import os

import pytest

mlflow = pytest.importorskip("mlflow")

pytestmark = pytest.mark.gate_m9b

from dealpoint.eval import mlflow_decision as d
from dealpoint.eval import mlflow_mirror as mm


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory):
    p = tmp_path_factory.mktemp("mlflow_decision")
    return f"sqlite:///{p}/m.db"


@pytest.fixture(scope="module")
def client(tracking_uri):
    return d._client(tracking_uri)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def live_manifest(tracking_uri, client, monkeypatch_module, tmp_path_factory):
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    monkeypatch_module.setattr(d, "VIEWS_PATH", tmp_path_factory.mktemp("views") / "mlflow_decision_views.json")
    manifest: dict = {}
    for name in d.STEPS:
        getattr(d, f"step_{name}")(client, True, manifest)
    return manifest


# --- pure planning: dry-run counts --------------------------------------------------------------


def test_decision_pools_counts():
    pools = d.decision_pools()
    assert set(pools) == {"judged-18", "test-32"}
    assert {k: len(v) for k, v in pools["judged-18"].items()} == {
        "A@haiku": 18, "D@haiku": 18, "D@glm": 18,
        "D@deepseek-v4-flash": 18, "D@qwen3.7-flash": 18, "D@gemini-3.1-flash-lite": 18,
    }
    assert {k: len(v) for k, v in pools["test-32"].items()} == {
        "A@glm": 32, "B@glm": 32, "C@glm": 32, "D@glm": 32,
    }


def test_decision_run_plan_shape():
    plan = d.decision_run_plan()
    parents = [p for p in plan if p["role"] == "parent"]
    children = [p for p in plan if p["role"] == "config"]
    assert len(parents) == 2
    assert len(children) == 10
    for c in children:
        assert set(c["metrics"]) == set(d.DECISION_METRICS)
        assert set(d.DECISION_METRICS).issubset(c["metrics"])
        assert len(c["metrics"]) == 15


def test_main_dry_run_by_default_writes_no_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(d, "VIEWS_PATH", tmp_path / "views.json")
    rc = d.main([])
    assert rc == 0
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "views.json").exists()


def test_dry_run_always_wins_over_live(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(d, "VIEWS_PATH", tmp_path / "views.json")
    rc = d.main(["--live", "--dry-run", "--only", "views"])
    assert rc == 0
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "views.json").exists()


# --- live: tree, artifacts, registry -------------------------------------------------------------


def test_live_creates_two_parents_and_ten_children(client, live_manifest):
    exp_id = d._ensure_experiment(client)
    parents = client.search_runs([exp_id], filter_string="tags.`dealpoint.decision` = 'parent'")
    children = client.search_runs([exp_id], filter_string="tags.`dealpoint.decision` = 'config'")
    assert len(parents) == 2
    assert len(children) == 10
    for c in children:
        assert set(c.data.metrics) == set(d.DECISION_METRICS)
        for p in ("arm", "loop", "retriever", "skill", "model", "system_label"):
            assert p in c.data.params


def test_live_children_nested_under_the_right_parent(client, live_manifest):
    exp_id = d._ensure_experiment(client)
    parents = {r.data.tags["dealpoint.pool"]: r.info.run_id for r in
               client.search_runs([exp_id], filter_string="tags.`dealpoint.decision` = 'parent'")}
    children = client.search_runs([exp_id], filter_string="tags.`dealpoint.decision` = 'config'")
    for c in children:
        pool = c.data.tags["dealpoint.pool"]
        assert c.data.tags.get("mlflow.parentRunId") == parents[pool]


def test_live_parents_have_three_artifacts(client, live_manifest):
    exp_id = d._ensure_experiment(client)
    parents = client.search_runs([exp_id], filter_string="tags.`dealpoint.decision` = 'parent'")
    for p in parents:
        names = {a.path for a in client.list_artifacts(p.info.run_id)}
        assert names == {"pareto.json", "pareto.svg", "decision.md"}


def test_live_registry_versions_and_aliases(client, live_manifest):
    versions = [v for v in client.search_model_versions(f"name = '{d.REGISTERED_MODEL}'") if str(v.tags.get("config_hash", "")).startswith("decision/")]
    assert len(versions) == 10
    for alias, (pool, config) in d.REGISTRY_ALIASES.items():
        mv = client.get_model_version_by_alias(d.REGISTERED_MODEL, alias)
        assert mv.tags.get("variant_id") == config
        assert mv.tags.get("pool") == pool


def test_live_second_run_is_idempotent(client, tracking_uri):
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    exp_id = d._ensure_experiment(client)
    before_runs = len(client.search_runs([exp_id], max_results=5000))
    before_versions = len(client.search_model_versions(f"name = '{d.REGISTERED_MODEL}'"))

    manifest: dict = {}
    for name in d.STEPS:
        getattr(d, f"step_{name}")(client, True, manifest)

    after_runs = len(client.search_runs([exp_id], max_results=5000))
    after_versions = len(client.search_model_versions(f"name = '{d.REGISTERED_MODEL}'"))
    assert after_runs == before_runs
    assert after_versions == before_versions
    assert manifest["tree"]["created_this_run"] == 0
    assert manifest["registry"]["created_this_run"] == 0
    assert manifest["evaluate"]["written_this_run"] == 0


# --- equality with the Braintrust mirror (spine checks) -------------------------------------------


def test_every_child_metric_equals_the_mirror_recomputation(client, live_manifest):
    exp_id = d._ensure_experiment(client)
    pools = d.decision_pools()
    ctx = mm.mirror_context()
    for pool in d.POOLS:
        for config, entries in pools[pool].items():
            key = f"decision/{pool}/{config}"
            run = d._find_run(client, exp_id, key)
            assert run is not None
            expected = d.config_metrics(entries, ctx)
            for k, v in expected.items():
                assert run.data.metrics[k] == pytest.approx(v, rel=1e-6), f"{key}:{k}"


def test_named_spine_metrics():
    pools = d.decision_pools()
    ctx = mm.mirror_context()
    m = d.config_metrics(pools["judged-18"]["D@gemini-3.1-flash-lite"], ctx)
    assert m["safe_accuracy"] == pytest.approx(15 / 18)

    m = d.config_metrics(pools["judged-18"]["A@haiku"], ctx)
    assert m["precision_when_answering"] == pytest.approx(5 / 11)

    m = d.config_metrics(pools["judged-18"]["D@qwen3.7-flash"], ctx)
    assert m["correct_per_dollar"] == pytest.approx(198.98, rel=1e-3)

    m_a = d.config_metrics(pools["test-32"]["A@glm"], ctx)
    m_c = d.config_metrics(pools["test-32"]["C@glm"], ctx)
    assert m_a["net_accuracy"] == pytest.approx(0.28125)
    assert m_c["net_accuracy"] == pytest.approx(0.25)
    assert m_a["net_accuracy"] > m_c["net_accuracy"]


# --- Pareto ---------------------------------------------------------------------------------------


def test_pareto_frontier_reproduces_m6_rule_on_m6_data():
    from dealpoint.eval.pareto_report import pareto_frontier

    with open("data/reports/pareto.json", encoding="utf-8") as f:
        report = json.load(f)
    points = [{"model": m, "usd_per_case": v["usd_per_case"], "grounded_accuracy": (v.get("grounded_accuracy") or {}).get("mean")}
              for m, v in report["models"].items()]
    result = pareto_frontier(points, x_key="usd_per_case", y_key="grounded_accuracy", id_key="model")
    frontier = [r["model"] for r in result if r["frontier"]]
    assert frontier == report["frontier"]["models"]
    assert frontier == ["qwen/qwen3.7-flash", "deepseek/deepseek-v4-flash"]


def test_judged18_pareto_frontiers_and_spec_difference():
    pools = d.decision_pools()
    ctx = mm.mirror_context()
    metrics_by_config = {c: d.config_metrics(entries, ctx) for c, entries in pools["judged-18"].items()}
    artifact = d.pareto_artifact("judged-18", metrics_by_config)
    by_pair = {f'{f["x"]}|{f["y"]}': f["frontier"] for f in artifact["frontiers"]}

    # specs/milestones/m9b.md section 2 claims the safe-vs-dollars frontier "names D@gemini and
    # D@qwen"; the M6 rule applied unchanged to the measured numbers instead produces D@qwen and
    # D@deepseek (D@deepseek dominates D@gemini: cheaper and safer). D@gemini and D@qwen do share
    # a frontier -- on usd_per_case vs net_accuracy, the tie-breaker axis. See spec_differences.
    assert set(by_pair["usd_per_case|safe_accuracy"]) == {"D@qwen3.7-flash", "D@deepseek-v4-flash"}
    assert {"D@gemini-3.1-flash-lite", "D@qwen3.7-flash"}.issubset(set(by_pair["usd_per_case|net_accuracy"]))

    assert artifact["spec_differences"]
    diff = artifact["spec_differences"][0]
    assert "D@gemini" in diff["spec_claim"] and "D@qwen" in diff["spec_claim"]
    assert diff["measured_frontier"] == ["D@qwen3.7-flash", "D@deepseek-v4-flash"]


def test_pareto_svg_deterministic_and_names_configs():
    points = [
        {"config": "D@qwen3.7-flash", "x": 0.001117, "y": 0.8333, "frontier": True},
        {"config": "D@gemini-3.1-flash-lite", "x": 0.005134, "y": 0.8333, "frontier": False},
    ]
    s1 = d.pareto_svg(points, x_key="usd_per_case", y_key="safe_accuracy", title="t")
    s2 = d.pareto_svg(points, x_key="usd_per_case", y_key="safe_accuracy", title="t")
    assert s1 == s2
    assert s1.startswith("<svg")
    assert s1.strip().endswith("</svg>")
    for p in points:
        assert p["config"] in s1


# --- searches ---------------------------------------------------------------------------------------


def _config_set(client, exp_id, filter_string):
    runs = client.search_runs([exp_id], filter_string=filter_string, max_results=1000)
    return {r.data.tags["dealpoint.key"].split("/")[-1] for r in runs}


def test_safe_and_cheap_search(client, live_manifest):
    exp_id = d._ensure_experiment(client)
    assert _config_set(client, exp_id, d.SEARCH_SAFE_AND_CHEAP) == {"D@deepseek-v4-flash", "D@glm", "D@qwen3.7-flash"}


def test_right_and_fast_search(client, live_manifest):
    exp_id = d._ensure_experiment(client)
    assert _config_set(client, exp_id, d.SEARCH_RIGHT_AND_FAST) == {"D@gemini-3.1-flash-lite", "A@glm"}


def test_never_fabricate_search(client, live_manifest):
    exp_id = d._ensure_experiment(client)
    assert _config_set(client, exp_id, d.SEARCH_NEVER_FABRICATE) == {"D@gemini-3.1-flash-lite"}


# --- D9: row-level evaluation at zero model calls --------------------------------------------------


def test_evaluation_aggregates_equal_child_metrics(client, live_manifest):
    exp_id = d._ensure_experiment(client)
    pools = d.decision_pools()
    for pool in d.POOLS:
        for config in pools[pool]:
            child = d._find_run(client, exp_id, f"decision/{pool}/{config}")
            eval_run = d._find_run(client, exp_id, f"eval/{pool}/{config}")
            assert child is not None
            assert eval_run is not None
            er = client.get_run(eval_run.info.run_id)
            assert er.data.metrics["safe/mean"] == pytest.approx(child.data.metrics["safe_accuracy"])
            assert er.data.metrics["misleading/mean"] == pytest.approx(child.data.metrics["misleading_rate"])


def test_deployable_identical_to_safe_on_every_row():
    pools = d.decision_pools()
    ctx = mm.mirror_context()
    for pool in d.POOLS:
        for entries in pools[pool].values():
            for row in d.evaluation_frame(entries, ctx):
                assert d.deployable_row(row["outputs"]) == row["outputs"]["safe"]


def test_judged18_evaluation_frames_contain_the_spine_and_twin_case():
    pools = d.decision_pools()
    ctx = mm.mirror_context()
    for config in ("D@gemini-3.1-flash-lite", "D@glm"):
        frame = d.evaluation_frame(pools["judged-18"][config], ctx)
        case_ids = {row["inputs"]["case_id"] for row in frame}
        for case_id in d.STOP9_TWIN_CASES:
            assert case_id in case_ids or config != "D@glm", f"{config} missing {case_id}"
    frame = d.evaluation_frame(pools["judged-18"]["D@glm"], ctx)
    case_ids = {row["inputs"]["case_id"] for row in frame}
    assert set(d.STOP9_TWIN_CASES).issubset(case_ids)


# --- zero network ------------------------------------------------------------------------------------


def test_zero_network_openrouter_never_constructed(tracking_uri, monkeypatch, tmp_path_factory):
    def _boom(*a, **k):
        raise AssertionError("OpenRouterClient must never be constructed in the offline suite")

    monkeypatch.setattr("dealpoint.llm.client.OpenRouterClient.__init__", _boom)
    monkeypatch.setattr("openai.OpenAI.__init__", _boom)

    def _judge_boom(*a, **k):
        raise AssertionError("mlflow_decision must never call mlflow.genai.judges.make_judge")

    monkeypatch.setattr("mlflow.genai.judges.make_judge", _judge_boom)

    fresh_uri = f"sqlite:///{tmp_path_factory.mktemp('zero_network')}/m.db"
    os.environ["MLFLOW_TRACKING_URI"] = fresh_uri
    fresh_client = d._client(fresh_uri)
    monkeypatch.setattr(d, "VIEWS_PATH", tmp_path_factory.mktemp("zero_network_views") / "views.json")
    manifest: dict = {}
    for name in d.STEPS:
        getattr(d, f"step_{name}")(fresh_client, True, manifest)
    assert manifest["tree"]["created_this_run"] == 12


# --- docs -----------------------------------------------------------------------------------------


def test_mlflow_tour_stop9_is_the_config_decision_showcase():
    from pathlib import Path

    text = Path("docs/mlflow-tour.md").read_text(encoding="utf-8")
    for phrase in ("decision/judged-18", "decision/test-32", "parallel coordinates", "pareto.json",
                   "champion", "baseline", "cost-floor", "safest", "deployable", "contract_144__q05"):
        assert phrase in text, phrase
    for search in (d.SEARCH_SAFE_AND_CHEAP, d.SEARCH_RIGHT_AND_FAST, d.SEARCH_NEVER_FABRICATE):
        assert search in text
    assert "not here" in text.lower() or "not here." in text.lower()


def test_stop9_verdict_is_verbatim_in_demo_tour():
    from pathlib import Path

    verdict = d.stop9_verdict()
    assert verdict
    assert "D@gemini" in verdict
    demo_text = Path("docs/demo-tour.md").read_text(encoding="utf-8")
    assert verdict in demo_text
