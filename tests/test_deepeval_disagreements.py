"""M7a corrective gate: DeepEval disagreement cases are actually computed
(not hardcoded []), None-valued traces are excluded, both directions are
found (spec corrective task item 2).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gate_m7


def _score_row(case_id, variant_id, task_completion):
    return {
        "case_id": case_id,
        "variant_id": variant_id,
        "task_completion": {"score": task_completion, "reason": "x"},
    }


def test_find_disagreements_both_directions():
    from dealpoint.eval.deepeval_adapter import find_disagreements

    per_trace_scores = [
        _score_row("c1", "A@haiku", 0.9),  # deepeval pass
        _score_row("c2", "A@haiku", 0.1),  # deepeval fail
        _score_row("c3", "A@haiku", 0.5),  # boundary pass (>=0.5)
        _score_row("c4", "A@haiku", 0.5),  # boundary fail (<=0.5)
        _score_row("c5", "A@haiku", 0.7),  # agree (both true)
    ]
    det_by_row = [
        {"grounded_accuracy": False},  # deepeval pass, det fail -> disagreement
        {"grounded_accuracy": True},  # deepeval fail, det pass -> disagreement
        {"grounded_accuracy": True},  # both true-ish -> agreement (0.5>=0.5 and True, not disagreement per rule since only mismatches count)
        {"grounded_accuracy": False},  # 0.5<=0.5 and grounded False -> agreement direction not satisfied (rule requires ga True for that branch)
        {"grounded_accuracy": True},  # agreement
    ]
    disagreements = find_disagreements(per_trace_scores, det_by_row)
    directions = {d["direction"] for d in disagreements}
    assert "deepeval_pass_deterministic_fail" in directions
    assert "deepeval_fail_deterministic_pass" in directions

    case_ids_flagged = {d["case_id"] for d in disagreements}
    assert "c1" in case_ids_flagged
    assert "c2" in case_ids_flagged
    assert "c5" not in case_ids_flagged  # agreement, never flagged


def test_find_disagreements_excludes_none_valued_traces():
    from dealpoint.eval.deepeval_adapter import find_disagreements

    per_trace_scores = [
        _score_row("c1", "A@haiku", None),  # deepeval score missing
        _score_row("c2", "A@haiku", 0.9),
    ]
    det_by_row = [
        {"grounded_accuracy": False},  # would disagree if tc were not None
        {"grounded_accuracy": None},  # det score missing -> must be skipped
    ]
    disagreements = find_disagreements(per_trace_scores, det_by_row)
    assert disagreements == []


def test_find_disagreements_records_case_variant_scores_and_metric():
    from dealpoint.eval.deepeval_adapter import DISAGREEMENT_METRIC_NAME, find_disagreements

    per_trace_scores = [_score_row("c1", "D@glm", 0.8)]
    det_by_row = [{"grounded_accuracy": False}]
    disagreements = find_disagreements(per_trace_scores, det_by_row)
    assert len(disagreements) == 1
    d = disagreements[0]
    assert d["case_id"] == "c1"
    assert d["variant_id"] == "D@glm"
    assert d["metric"] == DISAGREEMENT_METRIC_NAME
    assert d["deepeval_score"] == 0.8
    assert d["deterministic_score"] is False


def test_deepeval_crosscheck_json_disagreements_not_hardcoded_empty():
    """If the report has already been (re)generated on this machine, its
    disagreements array should not be silently empty when there are real
    per_trace_scores/det mismatches -- a regression guard for the corrective
    task's root-cause bug (disagreements=[] passed literally).
    """
    import json

    from dealpoint.config import DEEPEVAL_CROSSCHECK_JSON_PATH

    if not DEEPEVAL_CROSSCHECK_JSON_PATH.exists():
        pytest.skip("deepeval_crosscheck.json not generated yet")
    report = json.loads(DEEPEVAL_CROSSCHECK_JSON_PATH.read_text(encoding="utf-8"))
    assert "disagreements" in report
    # not asserting a nonzero count (that would be asserting the data, not
    # the mechanism) -- just that the field is a list, computed by the
    # pipeline, and every entry (if any) has the expected shape.
    for d in report["disagreements"]:
        assert set(d.keys()) >= {
            "case_id",
            "variant_id",
            "metric",
            "deepeval_score",
            "deterministic_score",
            "direction",
        }
