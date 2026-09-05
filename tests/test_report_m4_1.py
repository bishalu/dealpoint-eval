"""M4.1 report extensions (spec deliverable 4/5/6): failure_rate_table,
majority_baseline_note, harness_repair/version, and the report-consistency
gate over the live data/reports/four_arm.json (skips cleanly when absent,
same convention as tests/test_readme_results.py).
"""

from __future__ import annotations

import json

import pytest

from dealpoint.config import FOUR_ARM_JSON_PATH, M4_1_PROBES_JSON_PATH
from dealpoint.eval.report import (
    RESIDUAL_FAILURE_LEGS,
    RESIDUAL_FAILURE_THRESHOLD,
    build_report,
    failure_rate_table,
    majority_baseline_note,
    residual_failures,
)

pytestmark = pytest.mark.gate_m4


def _row(case_id, status="ANSWERED", question_id="q01"):
    return {
        "case_id": case_id,
        "case_set": "test",
        "question_id": question_id,
        "finding": {"answer": "All Cash", "evidence": [], "rationale": "ok"} if status == "ANSWERED" else None,
        "record": {"status": status, "failure_reason": ("api_error" if status == "EXECUTION_FAILED" else None),
                   "failure_detail": (f"ApiError: boom {case_id}" if status == "EXECUTION_FAILED" else None)},
        "scores": {
            "grounded_accuracy": status == "ANSWERED",
            "answer_correct": status == "ANSWERED",
            "citation_gold_overlap": status == "ANSWERED",
            "citation_verbatim": True,
            "fabrication": False,
            "abstain_correct": True,
            "skill_adherence": 0.5,
            "tool_calls": 1,
            "usd": 0.001,
            "wall_ms": 100,
            "input_tokens": 500,
            "output_tokens": 50,
            "cap_hit": status == "CAP_HIT",
            "execution_failed": status == "EXECUTION_FAILED",
            "gold_seen": True,
            "gold_first_rank": 1,
            "redacted_fabrication": None,
            "required_evidence_met": None,
        },
        "skill_rules": {"rules": {}, "n_applicable": 0, "n_satisfied": 0, "score": None},
    }


def test_failure_rate_table_computes_rates_and_delta():
    v1 = {
        ("m1", "A"): [_row("a1", "EXECUTION_FAILED"), _row("a2", "ANSWERED")],
    }
    v2 = {
        ("m1", "A"): [_row("a1", "ANSWERED"), _row("a2", "ANSWERED")],
    }
    table = failure_rate_table(v1, v2)
    assert len(table) == 1
    row = table[0]
    assert row["model"] == "m1"
    assert row["arm"] == "A"
    assert row["execution_failed_v1"] == pytest.approx(0.5)
    assert row["execution_failed_v2"] == pytest.approx(0.0)
    assert row["delta"] == pytest.approx(-0.5)
    assert row["n_cases_v1"] == 2
    assert row["n_cases_v2"] == 2


def test_failure_rate_table_handles_leg_present_in_only_one_version():
    v1 = {("m1", "A"): [_row("a1", "ANSWERED")]}
    v2 = {("m1", "B"): [_row("b1", "ANSWERED")]}
    table = failure_rate_table(v1, v2)
    keys = {(r["model"], r["arm"]) for r in table}
    assert keys == {("m1", "A"), ("m1", "B")}
    a_row = next(r for r in table if r["arm"] == "A")
    assert a_row["execution_failed_v2"] is None
    b_row = next(r for r in table if r["arm"] == "B")
    assert b_row["execution_failed_v1"] is None


def test_majority_baseline_note_mentions_both_numbers():
    sentence = majority_baseline_note(0.0, 0.461)
    assert "0.0%" in sentence
    assert "46.1%" in sentence
    assert "by construction" in sentence


def test_residual_failures_only_above_threshold_with_explanation():
    model, arm = RESIDUAL_FAILURE_LEGS[0]
    rows_high = [_row(f"c{i}", "EXECUTION_FAILED") for i in range(3)] + [_row("c_ok", "ANSWERED")]
    rows_low = [_row("d1", "EXECUTION_FAILED")] + [_row(f"d{i}", "ANSWERED") for i in range(9)]
    v2 = {(model, arm): rows_high, (RESIDUAL_FAILURE_LEGS[1]): rows_low}
    result = residual_failures(v2)
    key_high = f"{model}/{arm}"
    assert key_high in result
    assert result[key_high]["explanation"]
    assert "failure_detail" not in result[key_high]  # metadata name, not literal key
    assert result[key_high]["attribution"] in ("provider", "harness")
    key_low = f"{RESIDUAL_FAILURE_LEGS[1][0]}/{RESIDUAL_FAILURE_LEGS[1][1]}"
    assert key_low not in result  # 1/10 = 10% is not > threshold


def test_build_report_sets_version_and_majority_baseline_note_and_harness_repair():
    model = "z-ai/glm-5.3-flash"
    rows_by_arm_model = {
        (model, "A"): [_row("c1"), _row("c2")],
    }
    baseline_cases = [
        {"question_id": "q01", "gold_answer": "All Cash", "majority_answer": "All Stock"},
    ]
    full_test_cases = [
        {"question_id": "q01", "gold_answer": "All Cash", "majority_answer": "All Cash"},
        {"question_id": "q01", "gold_answer": "All Stock", "majority_answer": "All Cash"},
    ]
    report = build_report(
        rows_by_arm_model,
        baseline_cases=baseline_cases,
        full_test_baseline_cases=full_test_cases,
        version="v2 after harness repair",
    )
    assert report["version"] == "v2 after harness repair"
    assert "harness_repair" in report
    assert report["harness_repair"]["fixes"]
    note = report["majority_baseline_note"]
    assert note["subset_overall"] == pytest.approx(0.0)
    assert note["full_test_overall"] == pytest.approx(0.5)
    assert "by construction" in note["sentence"]
    assert "v1_artifacts" in report


def test_report_consistency_gate_over_live_four_arm_json():
    """DoD: if data/reports/four_arm.json exists, every (model, arm) in
    {GLM A-D, Haiku D} either has execution_failed.mean <= 0.10, or
    residual_failures[key]['explanation'] is a non-empty string naming a
    failure class. Skips cleanly when the file is absent; must NOT skip once
    it exists.
    """
    if not FOUR_ARM_JSON_PATH.exists():
        pytest.skip("data/reports/four_arm.json not generated yet")
    report = json.loads(FOUR_ARM_JSON_PATH.read_text(encoding="utf-8"))
    if report.get("version") != "v2 after harness repair":
        pytest.skip("four_arm.json predates the v2 re-run")

    residuals = report.get("residual_failures") or {}
    arms = report.get("arms") or {}
    for model, arm in RESIDUAL_FAILURE_LEGS:
        model_arms = arms.get(model) or {}
        data = model_arms.get(arm)
        if not data:
            continue
        mean = (data["overall"].get("execution_failed") or {}).get("mean")
        if mean is None or mean <= RESIDUAL_FAILURE_THRESHOLD:
            continue
        key = f"{model}/{arm}"
        assert key in residuals, f"{key} exceeds 10% EXECUTION_FAILED with no residual_failures entry"
        explanation = residuals[key].get("explanation")
        assert explanation and isinstance(explanation, str) and len(explanation) > 0


def test_m4_1_probes_after_round_meets_threshold_or_has_provider_note():
    if not M4_1_PROBES_JSON_PATH.exists():
        pytest.skip("data/reports/m4_1_probes.json not generated yet")
    payload = json.loads(M4_1_PROBES_JSON_PATH.read_text(encoding="utf-8"))
    after_rounds = [r for r in payload.get("rounds", []) if r.get("round", "").startswith("after")]
    assert after_rounds, "m4_1_probes.json has no 'after' round"
    last_after = after_rounds[-1]
    for leg in last_after.get("legs", []):
        rate = leg.get("failure_rate")
        has_note = bool(leg.get("provider_attribution"))
        assert (rate is not None and rate <= 0.10) or has_note, (
            f"probe leg {leg.get('model')}/{leg.get('arm')} exceeds 10% with no provider_attribution note"
        )
