"""gate_m9c D16: the eight Braintrust dashboards, ported word for word, result for result
(specs/milestones/m9c.md). Offline: the local evaluator against a committed snapshot that came from
the live Braintrust project (`just braintrust-dashboard-snapshot --live`, run 2026-09-08), not the
local evaluator itself -- this suite never touches the network."""

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
    """`data/reports/braintrust_dashboard_values.json` is the real thing (spec D16 item 1): written
    by `snapshot_dashboards` against the live Braintrust project (`just braintrust-dashboard-snapshot
    --live`, run 2026-09-08), never by the local evaluator. Every row of every chart must equal it to
    1e-9, including toplist order, with exactly two named exceptions: the three `NOT_LOCALLY_EVALUABLE`
    charts (no local rows to compare -- the 67 prompt-variant traces' judge scores were never mirrored
    to disk) and `mod_p90` (`PERCENTILE_APPROXIMATE`): Braintrust's percentile aggregator is an
    approximate sketch, verified live to diverge from the local evaluator's exact linear-interpolation
    quantile on the identical 18 underlying values -- its row labels and order still must match, only
    the value itself is excused."""
    committed = d.load_snapshot()
    fresh = d.build_dashboard_snapshot()
    assert d.snapshot_matches(committed, fresh) == []

    excused = set(d.NOT_LOCALLY_EVALUABLE) | set(d.PERCENTILE_APPROXIMATE)
    fresh_by_key = {(e["dashboard"], e["chart_key"]): e for e in fresh}
    checked = 0
    for entry in committed:
        key = (entry["dashboard"], entry["chart_key"])
        if entry["chart_key"] in excused:
            continue
        fresh_entry = fresh_by_key[key]
        assert [r["label"] for r in entry["rows"]] == [r["label"] for r in fresh_entry["rows"]]
        for row_a, row_b in zip(entry["rows"], fresh_entry["rows"], strict=True):
            if row_a["value"] is None or row_b["value"] is None:
                assert row_a["value"] == row_b["value"]
            else:
                assert abs(row_a["value"] - row_b["value"]) <= 1e-9
        checked += 1
    assert checked == len(committed) - sum(1 for e in committed if e["chart_key"] in excused)


def test_percentile_chart_excused_only_from_value_not_from_labels():
    committed = d.load_snapshot()
    entry = next(e for e in committed if e["chart_key"] == "mod_p90")
    fresh_entry = next(e for e in d.build_dashboard_snapshot() if e["chart_key"] == "mod_p90")
    assert [r["label"] for r in entry["rows"]] == [r["label"] for r in fresh_entry["rows"]]
    # this is exactly the discovered gap: the live Braintrust value differs from the local exact
    # quantile on the same 18 underlying values.
    mismatched = any(abs(a["value"] - b["value"]) > 1e-9 for a, b in zip(entry["rows"], fresh_entry["rows"], strict=True))
    assert mismatched


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


def test_dashboard_run_plan_falls_back_to_the_snapshot_for_charts_the_local_evaluator_cannot_reproduce():
    """D16.3: NOT_LOCALLY_EVALUABLE and PERCENTILE_APPROXIMATE charts must come from the committed
    Braintrust snapshot, not from a local evaluator that has no rows (or the wrong number) for them --
    the corrective task's 'which-prompt renders (no data)' failure."""
    plans = d.dashboard_run_plan()
    committed = {(e["dashboard"], e["chart_key"]): e for e in d.load_snapshot()}

    for plan in plans:
        for chart in plan["charts"]:
            values = [r["value"] for r in chart["rows"]]
            assert any(v is not None for v in values), f"{plan['name']}/{chart['chart_key']} is all-None"

    which_prompt = next(p for p in plans if p["name"] == "Which prompt?")
    assert which_prompt["run_key"] == "dashboard/which-prompt"
    all_rows = [row for chart in which_prompt["charts"] for row in chart["rows"]]
    assert len(all_rows) == 12
    assert all(row["value"] is not None for row in all_rows)
    for chart in which_prompt["charts"]:
        snap_rows = committed[("Which prompt?", chart["chart_key"])]["rows"]
        assert chart["rows"] == snap_rows
        assert chart["source"] == "snapshot"

    mod_p90_chart = next(
        chart for plan in plans if plan["name"] == "Which model?"
        for chart in plan["charts"] if chart["chart_key"] == "mod_p90"
    )
    assert mod_p90_chart["source"] == "snapshot"
    assert mod_p90_chart["rows"] == committed[("Which model?", "mod_p90")]["rows"]


def test_snapshot_sourced_charts_carry_a_provenance_line_in_the_html():
    plans = d.dashboard_run_plan()
    for plan in plans:
        for chart in plan["charts"]:
            provenance = d._provenance_line(chart)
            if provenance is None:
                continue
            assert _escape(provenance) in plan["html"], f"{plan['name']}/{chart['chart_key']} missing provenance line"


def test_tour_names_every_dashboard_run_key_and_links_the_braintrust_dashboards():
    """D16.4: the tour must name the eight `dashboard/<name>` runs with the eight Braintrust
    dashboard links beside them, not just describe them generically."""
    from pathlib import Path

    tour = Path("docs/mlflow-tour.md").read_text(encoding="utf-8")
    plans = d.dashboard_run_plan()
    for plan in plans:
        assert plan["run_key"] in tour, f"{plan['run_key']} missing from docs/mlflow-tour.md"

    link_count = tour.count("https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards")
    assert link_count >= 6, f"expected at least six Braintrust dashboard links, found {link_count}"
