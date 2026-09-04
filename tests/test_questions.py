import csv

import pytest

from dealpoint.config import CSV_PATHS, OOS_COLLISION_WINDOW
from dealpoint.data.questions import (
    OUT_OF_SCOPE_QUESTION_SPEC,
    OUT_OF_SCOPE_QUESTIONS,
    OUT_OF_SCOPE_SPARES,
    QUESTION_SPEC,
    question_collides,
)

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


def test_out_of_scope_questions_match_spec_fixed_list_verbatim():
    # specs/milestones/m0_1.md Problem B, questions 1-10, verbatim.
    expected = (
        (
            "What percentage of the Target's employees have executed invention-assignment "
            "agreements?"
        ),
        "What is the Target's current cyber-insurance deductible?",
        "Which ERP system will the combined company use after closing?",
        (
            "What annualized cost synergies are expected during the first twelve months "
            "after closing?"
        ),
        (
            "What percentage of the Target's revenue comes from customer contracts "
            "containing change-of-control termination rights?"
        ),
        "What is the Target's accrued employee PTO liability as of signing?",
        "What is the weighted-average remaining term of the Target's office leases?",
        (
            "What is the expected post-closing retention-bonus pool for non-executive "
            "employees?"
        ),
        (
            "What percentage of the Target's source code has been reviewed for "
            "open-source license compliance?"
        ),
        "Which jurisdiction governs the Target's ten largest customer contracts?",
    )
    assert OUT_OF_SCOPE_QUESTIONS == expected


def test_every_out_of_scope_question_has_at_least_one_key_term():
    for q in OUT_OF_SCOPE_QUESTION_SPEC:
        assert len(q.key_terms) >= 1
    for q in OUT_OF_SCOPE_SPARES:
        assert len(q.key_terms) >= 1


def test_out_of_scope_ids_are_unique():
    ids = [q.id for q in (*OUT_OF_SCOPE_QUESTION_SPEC, *OUT_OF_SCOPE_SPARES)]
    assert len(ids) == len(set(ids))


def test_question_collides_requires_word_boundaries():
    # "pto" as a bare substring must not match inside "Lipton"/"raptor"/etc.
    canonical = "Lipton and raptor and Hampton and laptops and symptoms, no real match here."
    assert not question_collides(canonical, ("pto",), OOS_COLLISION_WINDOW)
    canonical_real = "The employee accrued PTO liability is material to this deal."
    assert question_collides(canonical_real, ("pto",), OOS_COLLISION_WINDOW)


def test_question_collides_requires_all_terms_within_window():
    canonical = "cyber " + ("x" * 500) + " deductible"
    assert not question_collides(canonical, ("cyber", "deductible"), 400)
    canonical_close = "cyber " + ("x" * 50) + " deductible"
    assert question_collides(canonical_close, ("cyber", "deductible"), 400)


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
