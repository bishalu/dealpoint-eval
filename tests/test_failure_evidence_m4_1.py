"""M4.1 deliverable 1/2/3: failure_detail/raw_final_text captured on every
failure, tolerant JSON extraction, and option normalisation (spec deliverable
2/3). All offline, `gate_m4`.
"""

from __future__ import annotations

import json

import pytest

from dealpoint.agent.loop import run_agent
from dealpoint.agent.pipeline import run_pipeline
from dealpoint.agent.schema import extract_json_object, match_option, validate_finding_json
from dealpoint.config import FAILURE_DETAIL_MAX_CHARS, RAW_FINAL_TEXT_MAX_CHARS
from dealpoint.corpus.chunks import Chunk
from dealpoint.corpus.document import Document
from dealpoint.data.questions import QUESTION_BY_ID
from dealpoint.data.sections import Section
from dealpoint.llm.client import ApiError, FakeClient, ScriptedTurn

pytestmark = pytest.mark.gate_m4

Q01 = QUESTION_BY_ID["q01"]
Q05 = QUESTION_BY_ID["q05"]
Q10 = QUESTION_BY_ID["q10"]


def _make_doc() -> Document:
    text = (
        "Article I THE MERGER 1.1. Certain Definitions. "
        '"Material Adverse Effect" means a materially adverse change to the Company. '
        "6.3. Merger Consideration. Each share shall be converted into the right to "
        "receive cash consideration."
    )
    sections = [
        Section(ref="ARTICLE I", title="THE MERGER", start=0, end=text.index("1.1.")),
        Section(ref="1.1", title="Certain Definitions", start=text.index("1.1."), end=text.index("6.3.")),
        Section(ref="6.3", title="Merger Consideration", start=text.index("6.3."), end=len(text)),
    ]
    return Document(
        document_id="doc_ev",
        agreement_id="doc_ev",
        text=text,
        sections=sections,
        body_start=0,
        body_end=len(text),
    )


class StubRetriever:
    def search(self, agreement_id, query, k=5):
        doc = _make_doc()
        return [
            Chunk(
                agreement_id=agreement_id,
                chunk_id=f"{agreement_id}:0-20",
                section_ref="6.3",
                start=0,
                end=20,
                text=doc.text[0:20],
            )
        ]


# --- failure_detail / raw_final_text on arm B (run_agent) -------------------


def test_arm_b_api_error_captures_failure_detail_truncated():
    doc = _make_doc()
    retriever = StubRetriever()
    long_msg = "upstream 429 rate limited " + ("x" * 5000)
    client = FakeClient(script=[ScriptedTurn(error=ApiError(long_msg)) for _ in range(10)])
    finding, record = run_agent({"case_id": "doc_ev__q01"}, doc, retriever, client, Q01, "fake/model")

    assert finding is None
    assert record.status == "EXECUTION_FAILED"
    assert record.failure_reason == "api_error"
    assert record.failure_detail is not None
    assert record.failure_detail.startswith("ApiError: ")
    assert "429" in record.failure_detail
    assert len(record.failure_detail) <= FAILURE_DETAIL_MAX_CHARS


def test_arm_b_schema_invalid_twice_captures_raw_final_text_truncated():
    doc = _make_doc()
    retriever = StubRetriever()
    long_broken = "Here is my answer: {broken" + ("z" * 5000)
    client = FakeClient(
        script=[
            ScriptedTurn(content=None),
            ScriptedTurn(content=long_broken),
            ScriptedTurn(content=long_broken),
        ]
    )
    finding, record = run_agent({"case_id": "doc_ev__q01"}, doc, retriever, client, Q01, "fake/model")

    assert finding is None
    assert record.status == "EXECUTION_FAILED"
    assert record.failure_reason == "schema_invalid_after_retry"
    assert record.raw_final_text is not None
    assert record.raw_final_text.startswith("Here is my answer:")
    assert len(record.raw_final_text) <= RAW_FINAL_TEXT_MAX_CHARS


def test_arm_b_finish_reasons_populated_on_normal_run():
    doc = _make_doc()
    retriever = StubRetriever()
    valid = json.dumps(
        {
            "answer": "All Cash",
            "evidence": [{"section_ref": "6.3", "quote": "cash consideration"}],
            "rationale": "ok",
        }
    )
    from dealpoint.llm.client import ScriptedToolCall

    client = FakeClient(
        script=[
            ScriptedTurn(
                tool_calls=[ScriptedToolCall(name="search_agreement", arguments={"query": "x"})]
            ),
            ScriptedTurn(content=None),
            ScriptedTurn(content=valid),
        ]
    )
    finding, record = run_agent({"case_id": "doc_ev__q01"}, doc, retriever, client, Q01, "fake/model")
    assert finding is not None
    assert record.status == "ANSWERED"
    assert record.finish_reasons == ["tool_calls", "stop", "stop"]


# --- failure_detail / raw_final_text on arm A (run_pipeline) ----------------


def test_arm_a_api_error_captures_failure_detail_truncated():
    doc = _make_doc()
    retriever = StubRetriever()
    long_msg = "upstream 429 rate limited " + ("x" * 5000)
    client = FakeClient(script=[ScriptedTurn(error=ApiError(long_msg)) for _ in range(10)])
    finding, record = run_pipeline({"case_id": "doc_ev__q01"}, doc, retriever, client, Q01, "fake/model")

    assert finding is None
    assert record.status == "EXECUTION_FAILED"
    assert record.failure_detail is not None
    assert record.failure_detail.startswith("ApiError: ")
    assert "429" in record.failure_detail
    assert len(record.failure_detail) <= FAILURE_DETAIL_MAX_CHARS


def test_arm_a_schema_invalid_twice_captures_raw_final_text_truncated():
    doc = _make_doc()
    retriever = StubRetriever()
    long_broken = "Here is my answer: {broken" + ("z" * 5000)
    client = FakeClient(script=[ScriptedTurn(content=long_broken), ScriptedTurn(content=long_broken)])
    finding, record = run_pipeline({"case_id": "doc_ev__q01"}, doc, retriever, client, Q01, "fake/model")

    assert finding is None
    assert record.status == "EXECUTION_FAILED"
    assert record.failure_reason == "schema_invalid_after_retry"
    assert record.raw_final_text is not None
    assert record.raw_final_text.startswith("Here is my answer:")
    assert len(record.raw_final_text) <= RAW_FINAL_TEXT_MAX_CHARS


# --- runner's catch-all tool_error carries failure_detail -------------------


def test_runner_tool_error_row_carries_failure_detail(tmp_path, dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    from dealpoint.eval.run import run_eval_set

    bogus_case = {
        "case_id": "nonexistent_doc__q01",
        "agreement_id": "this_document_does_not_exist_at_all",
        "question_id": "q01",
        "case_set": "test",
    }
    summary = run_eval_set(
        case_set="test",
        arm="A",
        model="fake/model",
        case_rows=[bogus_case],
        fake=True,
        out_dir=tmp_path,
    )
    with open(summary["results_path"], encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 1
    record = rows[0]["record"]
    assert record["status"] == "EXECUTION_FAILED"
    assert record["failure_reason"] == "tool_error"
    assert record["failure_detail"]
    assert len(record["failure_detail"]) > 0


# --- extract_json_object -----------------------------------------------


def test_extract_json_object_bare():
    assert extract_json_object('{"a": 1}') == '{"a": 1}'


def test_extract_json_object_fenced_json():
    raw = 'Sure!\n```json\n{"a": 1}\n```\n'
    assert extract_json_object(raw) == '{"a": 1}'


def test_extract_json_object_fenced_bare():
    raw = "```\n{\"a\": 1}\n```"
    assert extract_json_object(raw) == '{"a": 1}'


def test_extract_json_object_prose_before_and_after():
    raw = 'Sure! {"a": 1} Let me know if you need anything else.'
    assert extract_json_object(raw) == '{"a": 1}'


def test_extract_json_object_brace_inside_string_value():
    raw = 'prefix {"a": "value with } inside"} suffix'
    extracted = extract_json_object(raw)
    assert extracted is not None
    assert json.loads(extracted) == {"a": "value with } inside"}


def test_extract_json_object_two_objects_first_wins():
    raw = '{"a": 1} then {"b": 2}'
    assert extract_json_object(raw) == '{"a": 1}'


def test_extract_json_object_truncated_returns_none():
    assert extract_json_object('{"a": "broken') is None


def test_extract_json_object_none_and_empty():
    assert extract_json_object(None) is None
    assert extract_json_object("") is None
    assert extract_json_object("   ") is None


# --- match_option ------------------------------------------------------


def test_match_option_exact():
    assert match_option("All Cash", ("All Cash", "All Stock")) == "All Cash"


def test_match_option_different_case():
    assert match_option("all cash", ("All Cash", "All Stock")) == "All Cash"


def test_match_option_curly_quotes():
    assert match_option("\u201cAll Cash\u201d", ("All Cash", "All Stock")) == "All Cash"


def test_match_option_whitespace_and_wrapping_quote():
    assert match_option('  "All Cash"  ', ("All Cash", "All Stock")) == "All Cash"


def test_match_option_abstain_case_insensitive():
    assert match_option("abstain", ("All Cash",)) == "ABSTAIN"


def test_match_option_non_option_returns_none():
    assert match_option("All cash or stock", ("All Cash", "All Stock")) is None


def test_validate_finding_json_rewrites_answer_to_exact_option_string_q01():
    raw = (
        '{"answer": "all cash", "evidence": [{"section_ref": "2.1", "quote": "cash"}], '
        '"rationale": "Because."}'
    )
    finding, err = validate_finding_json(raw, Q01)
    assert err is None
    assert finding is not None
    assert finding.answer == "All Cash"
    assert finding.answer in Q01.options


def test_validate_finding_json_rewrites_answer_with_curly_quotes_q10():
    # q10 options include quoted phrases like '"Inconsistent" with fiduciary duties'
    target = Q10.options[0]
    mangled = target.replace('"', "\u201c", 1).replace('"', "\u201d", 1)
    raw = json.dumps(
        {
            "answer": mangled,
            "evidence": [{"section_ref": "2.1", "quote": "x"}],
            "rationale": "Because.",
        }
    )
    finding, err = validate_finding_json(raw, Q10)
    assert err is None
    assert finding is not None
    assert finding.answer == target


def test_validate_finding_json_empty_options_out_of_scope_unaffected():
    class FakeQ:
        options = ()

    raw = '{"answer": "ABSTAIN", "evidence": [], "rationale": "n/a"}'
    finding, err = validate_finding_json(raw, FakeQ())
    assert err is None
    assert finding is not None
    assert finding.answer == "ABSTAIN"

    raw2 = (
        '{"answer": "42%", "evidence": [{"section_ref": "1.1", "quote": "x"}], '
        '"rationale": "n/a"}'
    )
    finding2, err2 = validate_finding_json(raw2, FakeQ())
    assert finding2 is None
    assert err2 is not None
