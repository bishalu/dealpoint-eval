"""One-call-four-dims parsing with a fake judge (spec DoD check 3)."""

from __future__ import annotations

import pytest

from dealpoint.eval.judge_run import judge_one
from dealpoint.eval.judge_slate import parse_judge_json
from dealpoint.llm.client import FakeClient, ScriptedTurn

pytestmark = pytest.mark.gate_m5


def _judge_kwargs(**overrides):
    base = {
        "judge_model": "mistralai/mistral-small-3.2-24b-instruct",
        "judge_family": "Mistral",
        "packet_id": "abc123def456",
        "variant_id": "D\u0040haiku",
        "case_id": "contract_1__q01",
        "question_id": "q01",
    }
    base.update(overrides)
    return base


def test_clean_json_parses_to_four_ints_and_notes():
    payload, error = parse_judge_json(
        '{"reasoning": 4, "evidence": 3, "trajectory": 4, "professional": 5, "notes": "ok"}'
    )
    assert error is None
    assert payload == {"reasoning": 4, "evidence": 3, "trajectory": 4, "professional": 5, "notes": "ok"}


def test_fenced_json_block_parses():
    raw = '```json\n{"reasoning": 2, "evidence": 2, "trajectory": 2, "professional": 2, "notes": "n"}\n```'
    payload, error = parse_judge_json(raw)
    assert error is None
    assert payload is not None
    assert payload["reasoning"] == 2


def test_prose_wrapped_json_parses():
    raw = 'Sure, here is my answer: {"reasoning": 3, "evidence": 3, "trajectory": 3, "professional": 3, "notes": "n"} Hope this helps.'
    payload, error = parse_judge_json(raw)
    assert error is None
    assert payload is not None
    assert payload["evidence"] == 3


def test_missing_dimension_fails():
    payload, error = parse_judge_json('{"reasoning": 4, "evidence": 3, "trajectory": 4, "notes": "n"}')
    assert payload is None
    assert error is not None
    assert "professional" in error


@pytest.mark.parametrize("bad_value", [0, 6])
def test_out_of_range_fails(bad_value):
    raw = f'{{"reasoning": {bad_value}, "evidence": 3, "trajectory": 3, "professional": 3, "notes": "n"}}'
    payload, _error = parse_judge_json(raw)
    assert payload is None


def test_non_integer_fails():
    raw = '{"reasoning": 3.5, "evidence": 3, "trajectory": 3, "professional": 3, "notes": "n"}'
    payload, _error = parse_judge_json(raw)
    assert payload is None

    raw2 = '{"reasoning": "4", "evidence": 3, "trajectory": 3, "professional": 3, "notes": "n"}'
    payload2, _error2 = parse_judge_json(raw2)
    assert payload2 is None


def test_judge_one_success_single_call():
    fake = FakeClient(
        script=[
            ScriptedTurn(
                content='{"reasoning": 4, "evidence": 3, "trajectory": 4, "professional": 4, "notes": "ok"}'
            )
        ]
    )
    row = judge_one(fake, "packet text", **_judge_kwargs())
    assert row["ok"] is True
    assert row["reasoning"] == 4
    assert row["evidence"] == 3
    assert row["trajectory"] == 4
    assert row["professional"] == 4
    assert row["notes"] == "ok"
    assert len(fake.calls) == 1


def test_judge_one_retries_exactly_once_then_gives_up():
    fake = FakeClient(
        script=[
            ScriptedTurn(content="not json at all"),
            ScriptedTurn(content="still not json"),
        ]
    )
    row = judge_one(fake, "packet text", **_judge_kwargs())
    assert row["ok"] is False
    assert row["reasoning"] is None
    assert row["evidence"] is None
    assert row["trajectory"] is None
    assert row["professional"] is None
    assert row["failure_detail"] is not None
    assert len(fake.calls) == 2


def test_judge_one_recovers_on_retry():
    fake = FakeClient(
        script=[
            ScriptedTurn(content="garbage"),
            ScriptedTurn(
                content='{"reasoning": 2, "evidence": 2, "trajectory": 2, "professional": 2, "notes": "n"}'
            ),
        ]
    )
    row = judge_one(fake, "packet text", **_judge_kwargs())
    assert row["ok"] is True
    assert row["reasoning"] == 2
    assert len(fake.calls) == 2


def test_rubric_version_comes_from_our_code_never_the_models_output():
    from dealpoint.eval.rubric import rubric_version

    fake = FakeClient(
        script=[
            ScriptedTurn(
                content='{"reasoning": 4, "evidence": 3, "trajectory": 4, "professional": 4, '
                '"notes": "n", "rubric_version": "fakefakefake"}'
            )
        ]
    )
    row = judge_one(fake, "packet text", **_judge_kwargs())
    assert row["rubric_version"] == rubric_version()
    assert row["rubric_version"] != "fakefakefake"


def test_notes_truncated_to_500_chars():
    long_notes = "x" * 900
    fake = FakeClient(
        script=[
            ScriptedTurn(
                content='{"reasoning": 3, "evidence": 3, "trajectory": 3, "professional": 3, '
                f'"notes": "{long_notes}"}}'
            )
        ]
    )
    row = judge_one(fake, "packet text", **_judge_kwargs())
    assert len(row["notes"]) == 500


def test_request_has_no_tools_temperature_zero_json_object_format():
    fake = FakeClient(
        script=[
            ScriptedTurn(
                content='{"reasoning": 4, "evidence": 3, "trajectory": 4, "professional": 4, "notes": "n"}'
            )
        ]
    )
    judge_one(fake, "packet text", **_judge_kwargs())
    call = fake.calls[0]
    assert call["tools"] is None
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
