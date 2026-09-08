"""gate_m9c D16: the eight Braintrust dashboards, ported word for word, result for result
(specs/milestones/m9c.md). Offline: the local evaluator against a committed snapshot, never against a
live Braintrust project (that would need `--live` + network + credentials, run separately)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gate_m9c

from html import escape as _escape

from dealpoint.eval import mlflow_dashboards as d
from dealpoint.eval.braintrust_cockpit import DASHBOARDS, chart_catalogue


def test_all_rows_covers_the_738_traces():
    rows = d.all_rows()
    assert len(rows) == 738


def test_every_chart_in_the_catalogue_is_covered():
    snapshot = d.build_dashboard_snapshot()
    keys = {(e["dashboard"], e["chart_key"]) for e in snapshot}
    expected = {(name, cid) for name, _verdict, chart_ids in DASHBOARDS for cid in chart_ids}
    assert keys == expected


def test_locally_evaluable_charts_have_rows_and_the_prompt_charts_are_named_as_a_gap():
    snapshot = d.build_dashboard_snapshot()
    for entry in snapshot:
        if entry["chart_key"] in d.NOT_LOCALLY_EVALUABLE:
            assert entry["locally_evaluable"] is False
            assert entry["rows"] == []
        else:
            assert entry["locally_evaluable"] is True
            assert entry["rows"], f"{entry['chart_key']} has no rows"


def test_known_verified_numbers_match_the_tour():
    # docs/mlflow-tour.md stop 9: "DeepSeek is the safest (89%) and most precise (71%)".
    rows = dict(d.chart_values(chart_catalogue()["pareto_safe"], d.all_rows()))
    assert rows["D@deepseek-v4-flash"] == pytest.approx(8 / 9, abs=1e-9)
    rows = dict(d.chart_values(chart_catalogue()["pareto_precision"], d.all_rows()))
    assert rows["D@deepseek-v4-flash"] == pytest.approx(5 / 7, abs=1e-9)


def test_toplist_ordering_is_value_descending():
    snapshot = d.build_dashboard_snapshot()
    for entry in snapshot:
        values = [r["value"] for r in entry["rows"] if r["value"] is not None]
        assert values == sorted(values, reverse=True)


def test_local_evaluator_matches_the_committed_snapshot_within_1e_minus_9():
    """The committed `data/reports/braintrust_dashboard_values.json` was itself produced by
    `write_local_snapshot` (this environment has no live Braintrust network access, see the module
    docstring); this test at minimum guards against silent drift between the evaluator and the file
    a reviewer would diff against a real `--live` run."""
    committed = d.load_snapshot()
    fresh = d.build_dashboard_snapshot()
    assert d.snapshot_matches(committed, fresh) == []


def test_filter_parser_handles_and_or_parens():
    row_a = {"comparable": 1, "category": "agent"}
    row_b = {"comparable": 1, "category": "judged"}
    row_c = {"comparable": 0, "category": "judged"}
    expr = "metadata.comparable = 1 and (metadata.category = 'agent' or metadata.category = 'judged')"
    assert d.eval_filter(expr, row_a) is True
    assert d.eval_filter(expr, row_b) is True
    assert d.eval_filter(expr, row_c) is False


def test_measure_parser_avg_ratio_percentile():
    rows = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": None}]
    assert d.eval_measure("avg(metadata.x)", rows) == pytest.approx(2.0)
    ratio_rows = [{"a": 2.0, "b": 4.0}, {"a": 3.0, "b": 6.0}]
    assert d.eval_measure("sum(metadata.a) / sum(metadata.b)", ratio_rows) == pytest.approx(0.5)
    assert d.eval_measure("percentile(metadata.x, 0.5)", rows) == pytest.approx(2.0)


def test_dashboard_run_plan_has_eight_runs_with_verbatim_titles_and_verdicts():
    plans = d.dashboard_run_plan()
    assert len(plans) == 8
    catalogue = chart_catalogue()
    for (name, verdict, chart_ids), plan in zip(DASHBOARDS, plans, strict=True):
        assert plan["name"] == name
        assert plan["verdict"] == verdict
        assert plan["run_key"] == f"dashboard/{d._slug(name)}"
        assert _escape(name) in plan["html"]
        assert _escape(verdict) in plan["html"]
        for cid in chart_ids:
            assert _escape(catalogue[cid]["title"]) in plan["html"]


def test_render_html_is_self_contained_no_external_assets():
    plans = d.dashboard_run_plan()
    for plan in plans:
        assert "http://" not in plan["html"] and "https://" not in plan["html"]
        assert "<script" not in plan["html"]
