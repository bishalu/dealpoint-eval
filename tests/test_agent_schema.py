import pytest
from pydantic import ValidationError

from dealpoint.agent.schema import Evidence, Finding, finding_json_schema, validate_finding_json
from dealpoint.data.questions import QUESTION_BY_ID

pytestmark = pytest.mark.gate_m1


def test_finding_valid_answered():
    f = Finding(
        answer="All Cash",
        evidence=[Evidence(section_ref="2.1", quote="cash consideration")],
        rationale="Short rationale.",
    )
    assert f.answer == "All Cash"
    assert len(f.evidence) == 1


def test_finding_abstain_requires_empty_evidence():
    f = Finding(answer="ABSTAIN", evidence=[], rationale="Not addressed.")
    assert f.evidence == []

    with pytest.raises(ValidationError):
        Finding(
            answer="ABSTAIN",
            evidence=[Evidence(section_ref="2.1", quote="x")],
            rationale="Should fail.",
        )


def test_finding_non_abstain_requires_1_to_3_evidence():
    with pytest.raises(ValidationError):
        Finding(answer="All Cash", evidence=[], rationale="No evidence given.")

    four_items = [Evidence(section_ref="1.1", quote=f"q{i}") for i in range(4)]
    with pytest.raises(ValidationError):
        Finding(answer="All Cash", evidence=four_items, rationale="Too many.")

    three_items = [Evidence(section_ref="1.1", quote=f"q{i}") for i in range(3)]
    f = Finding(answer="All Cash", evidence=three_items, rationale="Fine.")
    assert len(f.evidence) == 3


def test_finding_rationale_word_limit():
    long_rationale = " ".join(["word"] * 81)
    with pytest.raises(ValidationError):
        Finding(
            answer="All Cash",
            evidence=[Evidence(section_ref="1.1", quote="x")],
            rationale=long_rationale,
        )

    ok_rationale = " ".join(["word"] * 80)
    f = Finding(
        answer="All Cash",
        evidence=[Evidence(section_ref="1.1", quote="x")],
        rationale=ok_rationale,
    )
    assert len(f.rationale.split()) == 80


def test_finding_json_schema_enum_from_real_question_options():
    q = QUESTION_BY_ID["q01"]
    schema = finding_json_schema(q)
    enum = schema["json_schema"]["schema"]["properties"]["answer"]["enum"]
    assert set(enum) == set(q.options) | {"ABSTAIN"}
    assert schema["json_schema"]["strict"] is True
    assert schema["json_schema"]["schema"]["additionalProperties"] is False


def test_validate_finding_json_valid():
    q = QUESTION_BY_ID["q01"]
    raw = (
        '{"answer": "All Cash", "evidence": [{"section_ref": "2.1", "quote": "cash"}], '
        '"rationale": "Because."}'
    )
    finding, err = validate_finding_json(raw, q)
    assert err is None
    assert finding is not None
    assert finding.answer == "All Cash"


def test_validate_finding_json_bad_answer_not_in_options():
    q = QUESTION_BY_ID["q01"]
    raw = (
        '{"answer": "Not An Option", "evidence": [{"section_ref": "2.1", "quote": "x"}], '
        '"rationale": "x"}'
    )
    finding, err = validate_finding_json(raw, q)
    assert finding is None
    assert err is not None


def test_validate_finding_json_unparseable():
    q = QUESTION_BY_ID["q01"]
    finding, err = validate_finding_json("not json at all", q)
    assert finding is None
    assert err is not None
    assert "invalid JSON" in err
