"""Report generator tested on fake results (spec DoD: "report generator tested
on fake results"), including the D-regresses honesty case (spec §7.1).
"""

from __future__ import annotations

import pytest

from dealpoint.eval.report import (
    ADJACENT_PAIRS,
    arm_metrics,
    build_report,
    config_diff,
    paired_flips,
    render_markdown,
    verdict,
)

pytestmark = pytest.mark.gate_m4


def _row(case_id, question_id="q01", grounded=True, skill_rules=None, usd=0.001, tool_calls=1):
    return {
        "case_id": case_id,
        "case_set": "test",
        "question_id": question_id,
        "finding": {"answer": "All Cash", "evidence": [], "rationale": "ok"},
        "record": {"status": "ANSWERED"},
        "scores": {
            "grounded_accuracy": grounded,
            "answer_correct": grounded,
            "citation_gold_overlap": grounded,
            "citation_verbatim": True,
            "fabrication": not grounded,
            "abstain_correct": True,
            "skill_adherence": 0.8 if grounded else 0.4,
            "tool_calls": tool_calls,
            "usd": usd,
            "wall_ms": 100,
            "input_tokens": 500,
            "output_tokens": 50,
            "cap_hit": False,
            "execution_failed": False,
            "gold_seen": True,
            "gold_first_rank": 1,
            "redacted_fabrication": None,
            "required_evidence_met": None,
        },
        "skill_rules": skill_rules
        or {
            "rules": {"1": "satisfied", "2": "n/a", "3": "n/a", "4": "n/a", "5": "satisfied", "6": "n/a", "7": "n/a", "8": "satisfied"},
            "n_applicable": 3,
            "n_satisfied": 3,
            "score": 1.0,
        },
    }


def test_config_diff_has_three_adjacent_pairs():
    diffs = config_diff()
    assert len(diffs) == 3
    assert {(d["from"], d["to"]) for d in diffs} == set(ADJACENT_PAIRS)
    for d in diffs:
        assert d["key"] in ("loop", "retriever", "skill")


def test_arm_metrics_computes_mean_and_n():
    rows = [_row("c1", grounded=True), _row("c2", grounded=False)]
    m = arm_metrics(rows)
    assert m["grounded_accuracy"]["n"] == 2
    assert m["grounded_accuracy"]["mean"] == pytest.approx(0.5)


def test_arm_metrics_includes_abstain_recall_and_false_abstain():
    """Corrective task #1: the two brief \u00a72.4 abstention aggregates must appear
    in arm_metrics' output (used for both `overall` and `per_question`)."""
    rows = [_row("c1", grounded=True), _row("c2", grounded=False)]
    m = arm_metrics(rows)
    assert "abstain_recall" in m
    assert "false_abstain" in m
    assert set(m["abstain_recall"].keys()) == {"mean", "n"}
    assert set(m["false_abstain"].keys()) == {"mean", "n"}


def test_paired_flips_detects_gained_and_lost():
    rows_a = [_row("c1", grounded=False), _row("c2", grounded=True), _row("c3", grounded=True)]
    rows_b = [_row("c1", grounded=True), _row("c2", grounded=False), _row("c3", grounded=True)]
    flips = paired_flips(rows_a, rows_b)
    assert flips["gained"] == ["c1"]
    assert flips["lost"] == ["c2"]
    assert flips["unchanged_correct"] == 1
    assert flips["unchanged_wrong"] == 0


def _rows5(grounded):
    return [_row(f"c{i}", grounded=grounded) for i in range(1, 6)]


def test_build_report_d_improves_over_c_positive_case():
    model = "z-ai/glm-5.3-flash"
    rows_by_arm_model = {
        (model, "A"): _rows5(False),
        (model, "B"): _rows5(False),
        (model, "C"): _rows5(False),
        (model, "D"): _rows5(True),
    }
    report = build_report(rows_by_arm_model)
    verdict_cd = report["arm_d_improves_over_c"][model]
    assert verdict_cd["improves"] is True
    assert "improves" in verdict_cd["sentence"]
    assert "NOT" not in verdict_cd["sentence"]


def test_build_report_d_regresses_honesty_rule():
    """Spec §7.1: if arm D does not improve grounded_accuracy over C, the report
    says so and shows adherence/fabrication/abstention/trajectory/efficiency
    deltas instead -- no wording may claim an improvement the numbers do not
    support."""
    model = "z-ai/glm-5.3-flash"
    rows_by_arm_model = {
        (model, "A"): _rows5(False),
        (model, "B"): _rows5(False),
        (model, "C"): _rows5(True),
        (model, "D"): _rows5(False),
    }
    report = build_report(rows_by_arm_model)
    v = report["arm_d_improves_over_c"][model]
    assert v["improves"] is False
    assert v["delta"] < 0
    assert "NOT" in v["sentence"]
    assert "improves" not in v["sentence"].split("NOT")[0].split("does")[-1]

    md = render_markdown(report)
    # No unqualified "arm D improves" claim in the rendered markdown.
    assert "does NOT improve" in md
    # The delta table (adherence/fabrication/abstention/trajectory/efficiency)
    # for C->D must be present.
    assert "C_to_D" in md
    assert "skill_adherence" in md
    assert "fabrication" in md
    assert "abstain_correct" in md
    assert "tool_calls" in md
    assert "usd" in md


def test_verdict_is_none_when_a_leg_is_missing():
    report = {"arms": {}}
    v = verdict(report, "some-model", "C", "D")
    assert v["improves"] is None
    assert "insufficient data" in v["sentence"]


def test_build_report_includes_per_question_tables():
    model = "z-ai/glm-5.3-flash"
    rows_by_arm_model = {
        (model, "A"): [_row("c1", question_id="q01"), _row("c2", question_id="q02")],
    }
    report = build_report(rows_by_arm_model)
    per_q = report["arms"][model]["A"]["per_question"]
    assert "q01" in per_q
    assert "q02" in per_q


def test_build_report_skill_adherence_detail_present():
    model = "z-ai/glm-5.3-flash"
    rows_by_arm_model = {
        (model, "D"): [_row("c1"), _row("c2")],
    }
    report = build_report(rows_by_arm_model)
    detail = report["skill_adherence_detail"][model]["D"]
    assert "1" in detail
    assert detail["1"]["n_applicable"] == 2
    assert detail["1"]["n_satisfied"] == 2


def test_build_report_budget_scaled_flag_true():
    report = build_report({})
    assert report["budget_scaled"] is True


def test_render_markdown_does_not_crash_on_empty_report():
    report = build_report({})
    md = render_markdown(report)
    assert isinstance(md, str)
    assert len(md) > 0


def test_render_markdown_has_per_question_section_naming_a_question_id():
    """Corrective task #4/#6: the per-question tables must actually render,
    naming at least one real question id."""
    model = "z-ai/glm-5.3-flash"
    rows_by_arm_model = {
        (model, "A"): [_row("c1", question_id="q01"), _row("c2", question_id="q02")],
    }
    report = build_report(rows_by_arm_model)
    md = render_markdown(report)
    assert "## Per-question metrics" in md
    assert "q01" in md
    assert "q02" in md


def test_verdict_not_comparable_when_denominators_mismatch():
    """Corrective task #3: n=9 vs n=2 with 1 common case must not be reported
    as an improvement or a regression -- the Haiku A->D 55.6% claim this
    fixes was computed from exactly this shape."""
    model = "anthropic/claude-haiku-4.5"
    rows_a = [_row(f"a{i}", grounded=(i % 2 == 0)) for i in range(9)]
    rows_b = [_row("a0", grounded=True), _row("b1", grounded=True)]
    report = {
        "arms": {
            model: {
                "A": {"overall": arm_metrics(rows_a)},
                "D": {"overall": arm_metrics(rows_b)},
            }
        }
    }
    v = verdict(report, model, "A", "D", rows_a=rows_a, rows_b=rows_b)
    assert v["improves"] is None
    assert "not comparable" in v["sentence"]
    assert "9" in v["sentence"]
    assert "2" in v["sentence"]
    assert "1" in v["sentence"]
