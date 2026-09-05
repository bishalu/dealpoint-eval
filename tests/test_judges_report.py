"""build_judges_report tested on fake judge rows (spec §9)."""

from __future__ import annotations

import pytest

from dealpoint.eval.judges_report import DIMENSIONS, build_judges_report, render_markdown

pytestmark = pytest.mark.gate_m5


def _judge_row(packet_id, case_id, variant_id, judge_model, judge_family, ok=True, **dims):
    base = {
        "packet_id": packet_id,
        "variant_id": variant_id,
        "case_id": case_id,
        "question_id": "q01",
        "judge_model": judge_model,
        "judge_family": judge_family,
        "rubric_version": "abc123def456",
        "ok": ok,
        "reasoning": dims.get("reasoning"),
        "evidence": dims.get("evidence"),
        "trajectory": dims.get("trajectory"),
        "professional": dims.get("professional"),
        "notes": "n",
        "failure_detail": None if ok else "boom",
    }
    return base


def _fake_rows():
    rows = []
    judges = [
        ("mistralai/mistral-small-3.2-24b-instruct", "Mistral"),
        ("nvidia/nemotron-3-super-120b-a12b", "NVIDIA"),
        ("bytedance-seed/seed-2.0-mini", "ByteDance"),
    ]
    cases = [("c1", "A@haiku"), ("c1", "D@haiku"), ("c2", "D@glm")]
    for case_id, variant_id in cases:
        for i, (jm, jf) in enumerate(judges):
            rows.append(
                _judge_row(
                    f"{case_id}-{variant_id}",
                    case_id,
                    variant_id,
                    jm,
                    jf,
                    reasoning=3 + i % 2,
                    evidence=4,
                    trajectory=3,
                    professional=4,
                )
            )
    return rows


def test_no_human_scores_reports_pending():
    rows = _fake_rows()
    report = build_judges_report(rows)
    assert report["human_calibration"] == "pending"
    assert report["pending_human_input"] == "score data/eval/calibration/form.md"


def test_per_dimension_pairwise_correlation_blocks_all_present():
    rows = _fake_rows()
    trace_scores = {("c1", "A@haiku"): True, ("c1", "D@haiku"): False, ("c2", "D@glm"): None}
    report = build_judges_report(rows, trace_scores=trace_scores)
    for dim in DIMENSIONS:
        assert dim in report["per_dimension"]
        assert dim in report["pairwise_judge_agreement"]
        assert dim in report["correlation_with_grounded_accuracy"]
    assert report["per_dimension"]["evidence"]["mean_of_judges"] == pytest.approx(4.0)


def test_failed_judge_reduces_n_rather_than_skewing_mean():
    rows = _fake_rows()
    # add a failed row for a new packet -- must not contribute to means
    rows.append(_judge_row("c3-D@haiku", "c3", "D@haiku", "mistralai/mistral-small-3.2-24b-instruct", "Mistral", ok=False))
    report = build_judges_report(rows)
    # n should not count the failed row
    assert report["n_judge_rows_failed"] == 1
    assert report["n_judge_rows_ok"] == len(rows) - 1


def test_computed_branch_appears_with_synthetic_human_scores():
    rows = _fake_rows()
    human_rows = [
        {
            "packet_id": "c1-A@haiku",
            "scorer": "tester",
            "scored_at": "2026-09-05T00:00:00+00:00",
            "reasoning": 4,
            "evidence": 4,
            "trajectory": 3,
            "professional": 4,
            "notes": "n",
        },
        {
            "packet_id": "c1-D@haiku",
            "scorer": "tester",
            "scored_at": "2026-09-05T00:00:00+00:00",
            "reasoning": 3,
            "evidence": 4,
            "trajectory": 3,
            "professional": 4,
            "notes": "n",
        },
    ]
    report = build_judges_report(rows, human_rows=human_rows)
    assert report["human_calibration"] != "pending"
    assert report["pending_human_input"] is None
    for dim in DIMENSIONS:
        assert "mean_of_judges_vs_human" in report["human_calibration"][dim]
        assert "per_judge_vs_human" in report["human_calibration"][dim]


def test_spend_block_present():
    rows = _fake_rows()
    report = build_judges_report(rows)
    assert "spend" in report
    assert "target_usd" in report["spend"]


def test_markdown_first_sentence_carries_secondary_budget_scaled_caveat():
    rows = _fake_rows()
    report = build_judges_report(rows)
    md = render_markdown(report)
    assert "secondary" in md.lower()
    assert "budget-scaled" in md.lower()
    assert "not objective truth" in md.lower() or "never" in md.lower()
