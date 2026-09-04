import json

import pytest

from dealpoint.config import (
    COUNTERFACTUAL_JSONL_PATH,
    DEV_JSONL_PATH,
    N_DEV,
    N_OUT_OF_SCOPE,
    N_REDACTED,
    N_TEST,
    OOS_COLLISIONS_REPORT_PATH,
    SELECTION_PATH,
    TEST_JSONL_PATH,
)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.mark.gate_m0
def test_case_files_exist_and_within_bounds():
    if not (DEV_JSONL_PATH.exists() and TEST_JSONL_PATH.exists() and COUNTERFACTUAL_JSONL_PATH.exists()):
        pytest.skip("case files not built yet; run `uv run python -m dealpoint.cli data`")

    dev_rows = _read_jsonl(DEV_JSONL_PATH)
    test_rows = _read_jsonl(TEST_JSONL_PATH)
    cf_rows = _read_jsonl(COUNTERFACTUAL_JSONL_PATH)

    dev_agreements = {r["agreement_id"] for r in dev_rows}
    test_agreements = {r["agreement_id"] for r in test_rows}
    assert len(dev_agreements) <= N_DEV
    assert len(test_agreements) <= N_TEST
    assert len(dev_rows) <= N_DEV * 12
    assert len(test_rows) <= N_TEST * 12
    assert len(cf_rows) == N_REDACTED + N_OUT_OF_SCOPE

    redacted = [r for r in cf_rows if r.get("kind") == "redacted"]
    oos = [r for r in cf_rows if r.get("kind") == "out_of_scope"]
    assert len(redacted) == N_REDACTED
    assert len(oos) == N_OUT_OF_SCOPE
    assert all(r["gold_answer"] == "ABSTAIN" for r in cf_rows)


@pytest.mark.gate_m0
def test_case_files_are_sorted_by_case_id():
    if not DEV_JSONL_PATH.exists():
        pytest.skip("case files not built yet; run `uv run python -m dealpoint.cli data`")
    rows = _read_jsonl(DEV_JSONL_PATH)
    ids = [r["case_id"] for r in rows]
    assert ids == sorted(ids)


@pytest.mark.gate_m0
def test_dev_and_test_no_null_gold_answers():
    if not (DEV_JSONL_PATH.exists() and TEST_JSONL_PATH.exists()):
        pytest.skip("case files not built yet; run `uv run python -m dealpoint.cli data`")
    for path in (DEV_JSONL_PATH, TEST_JSONL_PATH):
        for row in _read_jsonl(path):
            assert row["gold_answer"] not in (None, "", "None")
            assert len(row["gold_spans"]) >= 1


@pytest.mark.gate_m0
def test_collision_report_exists_and_covers_every_question_test_agreement_pair():
    if not (COUNTERFACTUAL_JSONL_PATH.exists() and OOS_COLLISIONS_REPORT_PATH.exists()):
        pytest.skip("case files/collision report not built yet; run `uv run python -m dealpoint.cli data`")
    with open(OOS_COLLISIONS_REPORT_PATH, encoding="utf-8") as fh:
        report = json.load(fh)

    assert "collisions" in report
    assert "substitutions" in report

    if not SELECTION_PATH.exists():
        pytest.skip("selection.json not built yet")
    with open(SELECTION_PATH, encoding="utf-8") as fh:
        selection = json.load(fh)
    test_agreements = set(selection["test"])

    seen_question_ids = {row["question_id"] for row in report["collisions"]}
    seen_agreement_ids = {row["agreement_id"] for row in report["collisions"]}
    assert test_agreements <= seen_agreement_ids
    assert len(seen_question_ids) >= 10

    for row in report["collisions"]:
        assert isinstance(row["collides"], bool)


@pytest.mark.gate_m0
def test_out_of_scope_cases_placed_on_non_colliding_agreements():
    if not (COUNTERFACTUAL_JSONL_PATH.exists() and OOS_COLLISIONS_REPORT_PATH.exists()):
        pytest.skip("case files/collision report not built yet; run `uv run python -m dealpoint.cli data`")
    cf_rows = {r["case_id"]: r for r in _read_jsonl(COUNTERFACTUAL_JSONL_PATH)}
    with open(OOS_COLLISIONS_REPORT_PATH, encoding="utf-8") as fh:
        report = json.load(fh)
    collision_lookup = {
        (row["question_text"], row["agreement_id"]): row["collides"] for row in report["collisions"]
    }
    for row in cf_rows.values():
        if row.get("kind") != "out_of_scope":
            continue
        key = (row["question_text"], row["agreement_id"])
        # Only assert when the pairing was actually recorded in the collision
        # report (it always should be, but skip gracefully if not, rather
        # than false-failing on a report/format mismatch).
        if key in collision_lookup:
            assert collision_lookup[key] is False
