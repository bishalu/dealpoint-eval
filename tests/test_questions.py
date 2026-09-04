import csv

import pytest

from dealpoint.config import CSV_PATHS
from dealpoint.data.questions import OUT_OF_SCOPE_QUESTIONS, QUESTION_SPEC

csv.field_size_limit(10**9)


def _load_main_rows():
    rows = []
    for path in CSV_PATHS.values():
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows.extend(row for row in reader if row["data_type"] == "main")
    return rows


def test_exactly_twelve_questions():
    assert len(QUESTION_SPEC) == 12
    assert [q.id for q in QUESTION_SPEC] == [f"q{i:02d}" for i in range(1, 13)]


def test_q10_has_two_spaces_verbatim():
    q10 = next(q for q in QUESTION_SPEC if q.id == "q10")
    assert q10.maud_question == "Fiduciary exception:  Board determination standard-Answer (no-shop)"
    assert "  " in q10.maud_question


def test_out_of_scope_questions_count():
    assert len(OUT_OF_SCOPE_QUESTIONS) == 10


@pytest.mark.gate_m0
def test_every_question_and_option_string_exists_in_csv(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    rows = _load_main_rows()
    all_questions = {r["question"] for r in rows}
    for q in QUESTION_SPEC:
        assert q.maud_question in all_questions, f"{q.id}: question text not found in CSV"
        sub = [
            r for r in rows if r["question"] == q.maud_question and r["subquestion"] == "<NONE>"
        ]
        text_types = {r["text_type"] for r in sub}
        assert q.text_type in text_types, f"{q.id}: text_type {q.text_type!r} not found"
        answers_in_data = {r["answer"] for r in sub if r["answer"] not in ("", "None")}
        for option in q.options:
            assert option in answers_in_data, f"{q.id}: option {option!r} not found in CSV data"
