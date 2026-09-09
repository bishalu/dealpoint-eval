"""The eight Braintrust dashboards as MLflow saved chart views: every chart of every dashboard is a
BAR card in some view, over runs that all carry that chart's metric, with the snapshot's values."""

from __future__ import annotations

import json

import pytest

from dealpoint.eval import mlflow_views as mv
from dealpoint.eval.mlflow_dashboards import load_snapshot


@pytest.fixture(scope="module")
def plan():
    return mv.view_plan()


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot()


def test_every_snapshot_chart_is_a_card_in_a_view_on_its_dashboard_or_pointed_at(plan, snapshot):
    by_dash = {d["name"]: d for d in plan["dashboards"]}
    views = {v["slug"]: v for v in plan["views"]}
    for entry in snapshot:
        d = by_dash[entry["dashboard"]]
        in_view = any(entry["chart_key"] in views[s]["chart_keys"] for s in d["views"])
        pointed = any(entry["chart_key"] in e["chart_keys"] and e["view"] for e in d["elsewhere"])
        assert in_view or pointed, (entry["dashboard"], entry["chart_key"])
    assert sum(len(v["chart_keys"]) for v in plan["views"]) == len(snapshot) - sum(
        len(e["chart_keys"]) for d in plan["dashboards"] for e in d["elsewhere"])


def test_views_are_homogeneous_every_run_carries_every_card_metric(plan):
    """A run may lack a metric only where Braintrust itself has no value for that row (e.g. our
    gold-span scorer cannot see LlamaIndex's native bm25), so the empty row is faithful."""
    for v in plan["views"]:
        rs = plan["rowsets"][v["rowset"]]
        for run in rs["runs"]:
            missing = [cid for cid in v["chart_keys"]
                       if cid not in run["metrics"] and rs["metrics"][cid].get(run["run_name"]) is not None]
            assert not missing, (v["title"], run["run_name"], missing)


def test_values_equal_the_snapshot_percent_scaled_to_100(plan, snapshot):
    for entry in snapshot:
        scale = 100.0 if entry["unit"] == "percent" else 1.0
        rs = next(rs for rs in plan["rowsets"].values() if entry["chart_key"] in rs["metrics"]
                  and {str(r["label"]) for r in entry["rows"]} <= set(rs["labels"]))
        runs = {r["run_name"]: r for r in rs["runs"]}
        for row in entry["rows"]:
            if row["value"] is None:
                continue
            assert runs[str(row["label"])]["metrics"][entry["chart_key"]] == pytest.approx(row["value"] * scale, abs=1e-9)


def test_metric_keys_and_run_names_are_valid_for_mlflow(plan):
    for rs in plan["rowsets"].values():
        for run in rs["runs"]:
            assert run["run_name"]
            for k in run["metrics"]:
                assert mv.METRIC_KEY_RE.match(k), k


def test_saved_view_tag_is_the_ui_envelope_with_bar_cards_in_dashboard_order(plan):
    v = next(v for v in plan["views"] if v["dashboard"] == "Which system?" and v["rowset_name"] == "system")
    key, value = mv.saved_view_tag(v, plan["rowsets"][v["rowset"]], now_ms=1)
    assert key == "mlflow.sharedViewState." + v["slug"]
    envelope = json.loads(value)
    assert envelope["name"] == v["title"] and envelope["createdAt"] == 1
    state = json.loads(envelope["state"])
    assert state["searchFilter"] == "tags.`dealpoint.rowset` = 'system'"
    assert state["orderByKey"] == "metrics.`sys_safe`" and state["orderByAsc"] is False
    assert state["runsHiddenMode"] == "SHOW_ALL"
    cards = state["compareRunCharts"]
    assert [c["metricKey"] for c in cards] == v["chart_keys"]
    assert all(c["type"] == "BAR" and c["metricSectionId"] == state["compareRunSections"][0]["uuid"] for c in cards)
    assert cards[0]["displayName"].startswith("SAFE ACCURACY") and cards[0]["displayName"].endswith("(%)")
    assert len(value) <= 20000  # MAX_EXPERIMENT_TAG_VAL_LENGTH


def test_overview_keeps_the_accuracy_point_and_points_the_rest_home(plan):
    overview = plan["dashboards"][0]
    assert overview["name"] == "DealPoint eval overview"
    # the accuracy point, plus one view per overview-only chart (judge_evidence_pick has no home)
    assert 1 <= len(overview["views"]) <= 2
    v = next(v for v in plan["views"] if v["slug"] == overview["views"][0])
    assert v["chart_keys"][0] == "pareto_safe" and v["rowset_name"] == "system@model"
    assert overview["elsewhere"] and all(e["view"] for e in overview["elsewhere"])


def test_a_chart_on_two_dashboards_is_one_metric_on_one_run_set(plan):
    homes = [v for v in plan["views"] if "pareto_net" in v["chart_keys"]]
    assert len({v["rowset"] for v in homes}) == 1 and len(homes) >= 2


def test_note_names_every_dashboard_and_verdict(plan):
    note = mv.experiment_note(plan, base_url="http://x", experiment_id="7")
    for d in plan["dashboards"]:
        assert d["name"] in note and d["verdict"] in note
    assert "viewStateShareKey=" in note and len(note) <= 20000
