"""build_pareto_report tested on fake per-model rows (M6 spec DoD)."""

from __future__ import annotations

import pytest

from dealpoint.eval.pareto_report import (
    build_pareto_report,
    percentile,
    render_markdown,
    render_svg,
)

pytestmark = pytest.mark.gate_m6


def _row(case_id, grounded=True, usd=0.001, wall_ms=100, execution_failed=False, tool_calls=1):
    return {
        "case_id": case_id,
        "case_set": "test",
        "question_id": "q01",
        "finding": None if execution_failed else {"answer": "All Cash", "evidence": [], "rationale": "ok"},
        "record": {"status": "EXECUTION_FAILED" if execution_failed else "ANSWERED"},
        "scores": {
            "grounded_accuracy": None if execution_failed else grounded,
            "answer_correct": None if execution_failed else grounded,
            "citation_gold_overlap": None if execution_failed else grounded,
            "citation_verbatim": True,
            "fabrication": None if execution_failed else (not grounded),
            "abstain_correct": True,
            "skill_adherence": None if execution_failed else (0.8 if grounded else 0.4),
            "tool_calls": tool_calls,
            "usd": usd,
            "wall_ms": wall_ms,
            "input_tokens": 500,
            "output_tokens": 50,
            "cap_hit": False,
            "execution_failed": execution_failed,
            "gold_seen": True,
            "gold_first_rank": 1,
            "redacted_fabrication": None,
            "required_evidence_met": None,
        },
        "skill_rules": {
            "rules": {"1": "satisfied"},
            "n_applicable": 1,
            "n_satisfied": 1,
            "score": 1.0,
        },
    }


def test_percentile_known_list():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 90) == 50
    assert percentile([], 50) is None


def test_every_metric_present_per_model_with_n():
    entries = [
        {
            "model": "modelA",
            "family": "FamA",
            "rank": 1,
            "reused": False,
            "rows": [_row("c1", grounded=True), _row("c2", grounded=False)],
            "variant_id": "D\u0040modela",
        },
    ]
    report = build_pareto_report(entries)
    m = report["models"]["modelA"]
    for key in (
        "grounded_accuracy",
        "abstain_recall",
        "execution_failed",
        "cap_hit",
        "answer_correct",
        "citation_gold_overlap",
        "skill_adherence",
    ):
        assert key in m
    assert m["grounded_accuracy"]["n"] == 2
    assert m["latency_median_ms"] is not None
    assert m["latency_p90_ms"] is not None


def test_usd_per_case_is_realised_not_estimate():
    entries = [
        {
            "model": "modelA",
            "rows": [_row("c1", usd=0.01), _row("c2", usd=0.03)],
            "est_usd": 999.0,  # deliberately wrong; must not be used
        },
    ]
    report = build_pareto_report(entries)
    m = report["models"]["modelA"]
    assert m["usd_per_case"] == pytest.approx(0.02)
    assert m["est_usd"] == 999.0  # recorded separately, not conflated


def test_latency_percentiles_correct_on_known_list():
    rows = [_row(f"c{i}", wall_ms=v) for i, v in enumerate([100, 200, 300, 400, 500])]
    entries = [{"model": "modelA", "rows": rows}]
    report = build_pareto_report(entries)
    m = report["models"]["modelA"]
    assert m["latency_median_ms"] == 300
    assert m["latency_p90_ms"] == 500


def test_judged_quality_is_mean_of_judges_and_secondary():
    entries = [{"model": "modelA", "rows": [_row("c1")], "variant_id": "D\u0040modela"}]
    judge_rows = [
        {"variant_id": "D\u0040modela", "ok": True, "reasoning": 4, "evidence": 4, "trajectory": 3, "professional": 4},
        {"variant_id": "D\u0040modela", "ok": True, "reasoning": 2, "evidence": 4, "trajectory": 3, "professional": 4},
    ]
    report = build_pareto_report(entries, judge_rows=judge_rows)
    jq = report["models"]["modelA"]["judged_quality"]
    assert jq["secondary"] is True
    assert jq["reasoning"]["mean"] == pytest.approx(3.0)


def test_judged_quality_null_with_reason_when_no_judge_rows():
    entries = [{"model": "modelA", "rows": [_row("c1")], "variant_id": "D\u0040modela"}]
    report = build_pareto_report(entries, judge_rows=[])
    jq = report["models"]["modelA"]["judged_quality"]
    assert jq.get("reason")


def test_not_run_entries_survive_with_reasons():
    report = build_pareto_report([], not_run=[{"model": "anthropic/claude-opus-5", "reason": "not run (budget)"}])
    assert report["not_run"][0]["reason"] == "not run (budget)"


def test_all_execution_failed_model_appears_and_excluded_from_frontier():
    entries = [
        {"model": "healthy", "rows": [_row("c1", grounded=True, usd=0.01)]},
        {
            "model": "broken",
            "rows": [_row("c2", execution_failed=True, usd=0.01), _row("c3", execution_failed=True, usd=0.01)],
        },
    ]
    report = build_pareto_report(entries)
    broken = report["models"]["broken"]
    assert broken["grounded_accuracy"]["n"] == 0
    assert broken["grounded_accuracy"]["mean"] is None
    assert broken["frontier"] is False
    assert "healthy" not in report["frontier"]["models"] or "broken" not in report["frontier"]["models"]
    assert "broken" not in report["frontier"]["models"]


def test_spend_buckets_sum_to_realized():
    report = build_pareto_report(
        [],
        spend={"m6_probe_usd": 0.01, "m6_sweep_usd": 0.5, "m6_judge_usd": 0.02},
    )
    s = report["spend"]
    assert s["m6_realized_usd"] == pytest.approx(s["m6_probe_usd"] + s["m6_sweep_usd"] + s["m6_judge_usd"])


def test_render_markdown_first_sentence_carries_disclosures():
    report = build_pareto_report([{"model": "m", "rows": [_row("c1")]}])
    md = render_markdown(report)
    assert "budget-scaled" in md
    assert "objective" in md
    assert "not comparable" in md


def test_render_svg_one_marker_per_model_and_byte_identical():
    entries = [
        {"model": "modelA", "rows": [_row("c1", grounded=True, usd=0.01, wall_ms=100)]},
        {"model": "modelB", "rows": [_row("c2", grounded=True, usd=0.02, wall_ms=200)]},
    ]
    report = build_pareto_report(entries)
    svg1 = render_svg(report)
    svg2 = render_svg(report)
    assert svg1 == svg2
    assert svg1.count("<circle") == 2


# --- corrective-cycle: partial/abandoned sweep legs must never disappear ---


def test_build_pareto_report_surfaces_partial_runs_with_realized_usd_and_reason():
    partial_runs = [
        {
            "model": "openai/gpt-5.6-luna-pro",
            "n_cases_attempted": 21,
            "n_cases_completed": 0,
            "realized_usd": 0.28339,
            "reason": "stopped (stop-floor)",
        },
    ]
    report = build_pareto_report([], partial_runs=partial_runs)
    assert report["partial_runs"] == partial_runs
    assert report["partial_runs"][0]["realized_usd"] == pytest.approx(0.28339)
    assert report["partial_runs"][0]["reason"] == "stopped (stop-floor)"


def test_render_markdown_emits_partially_run_section_when_non_empty():
    partial_runs = [
        {
            "model": "openai/gpt-5.6-luna-pro",
            "n_cases_attempted": 21,
            "n_cases_completed": 0,
            "realized_usd": 0.28339,
            "reason": "stopped (stop-floor)",
        },
    ]
    report = build_pareto_report([{"model": "m", "rows": [_row("c1")]}], partial_runs=partial_runs)
    md = render_markdown(report)
    assert "## Partially run (metered, not completed)" in md
    assert "openai/gpt-5.6-luna-pro" in md
    assert "0.283390" in md or "0.28339" in md
    assert "stopped (stop-floor)" in md


def test_render_markdown_omits_partially_run_section_when_empty():
    report = build_pareto_report([{"model": "m", "rows": [_row("c1")]}], partial_runs=[])
    md = render_markdown(report)
    assert "## Partially run" not in md
