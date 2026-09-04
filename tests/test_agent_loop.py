import json

import pytest

from dealpoint.agent.loop import run_agent
from dealpoint.config import MAX_TOOL_CALLS
from dealpoint.corpus.chunks import Chunk
from dealpoint.corpus.document import Document
from dealpoint.data.questions import QUESTION_BY_ID
from dealpoint.data.sections import Section
from dealpoint.llm.client import ApiError, FakeClient, ScriptedToolCall, ScriptedTurn

pytestmark = pytest.mark.gate_m1

Q01 = QUESTION_BY_ID["q01"]  # All Cash / All Stock / ... direct question
Q09 = QUESTION_BY_ID["q09"]  # Yes / No


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
        document_id="doc_loop",
        agreement_id="doc_loop",
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


def _valid_answer_json(answer: str) -> str:
    return json.dumps(
        {
            "answer": answer,
            "evidence": [{"section_ref": "6.3", "quote": "cash consideration"}],
            "rationale": "The agreement specifies cash consideration.",
        }
    )


def _abstain_json() -> str:
    return json.dumps({"answer": "ABSTAIN", "evidence": [], "rationale": "Not addressed."})


def test_loop_terminates_normally_with_tool_calls_then_answer():
    doc = _make_doc()
    retriever = StubRetriever()
    client = FakeClient(
        script=[
            ScriptedTurn(
                tool_calls=[ScriptedToolCall(name="search_agreement", arguments={"query": "consideration"})]
            ),
            ScriptedTurn(tool_calls=[ScriptedToolCall(name="get_section", arguments={"section_ref": "6.3"})]),
            ScriptedTurn(content=None),
            ScriptedTurn(content=_valid_answer_json("All Cash")),
        ]
    )
    finding, record = run_agent({"case_id": "doc_loop__q01"}, doc, retriever, client, Q01, "fake/model")
    assert record.status == "ANSWERED"
    assert record.failure_reason is None
    assert finding is not None
    assert finding.answer == "All Cash"
    assert len(record.trajectory) == 2
    assert record.trajectory[0].tool == "search_agreement"
    assert record.trajectory[1].tool == "get_section"


def test_loop_cap_hit_when_ninth_tool_call_requested():
    doc = _make_doc()
    retriever = StubRetriever()
    nine_calls = [
        ScriptedToolCall(name="search_agreement", arguments={"query": f"q{i}"}) for i in range(9)
    ]
    client = FakeClient(script=[ScriptedTurn(tool_calls=nine_calls)])
    finding, record = run_agent({"case_id": "doc_loop__q01"}, doc, retriever, client, Q01, "fake/model")
    assert finding is None
    assert record.status == "CAP_HIT"
    assert record.failure_reason == "cap_hit"
    assert len(record.trajectory) == MAX_TOOL_CALLS
    assert record.usage.tool_calls == MAX_TOOL_CALLS


def test_loop_schema_invalid_then_retry_succeeds():
    doc = _make_doc()
    retriever = StubRetriever()
    client = FakeClient(
        script=[
            ScriptedTurn(content=None),
            ScriptedTurn(content="not valid json at all"),
            ScriptedTurn(content=_valid_answer_json("All Cash")),
        ]
    )
    finding, record = run_agent({"case_id": "doc_loop__q01"}, doc, retriever, client, Q01, "fake/model")
    assert record.status == "ANSWERED"
    assert finding is not None
    assert finding.answer == "All Cash"


def test_loop_schema_invalid_twice_is_execution_failed():
    doc = _make_doc()
    retriever = StubRetriever()
    client = FakeClient(
        script=[
            ScriptedTurn(content=None),
            ScriptedTurn(content="not valid json"),
            ScriptedTurn(content="still not valid json"),
        ]
    )
    finding, record = run_agent({"case_id": "doc_loop__q01"}, doc, retriever, client, Q01, "fake/model")
    assert finding is None
    assert record.status == "EXECUTION_FAILED"
    assert record.failure_reason == "schema_invalid_after_retry"


def test_loop_abstain_path():
    doc = _make_doc()
    retriever = StubRetriever()
    client = FakeClient(script=[ScriptedTurn(content=None), ScriptedTurn(content=_abstain_json())])
    finding, record = run_agent({"case_id": "doc_loop__q01"}, doc, retriever, client, Q01, "fake/model")
    assert finding is not None
    assert finding.answer == "ABSTAIN"
    assert finding.evidence == []
    assert record.status == "ABSTAINED"


def test_loop_api_error_retried_then_execution_failed():
    doc = _make_doc()
    retriever = StubRetriever()
    client = FakeClient(script=[ScriptedTurn(error=ApiError("boom")) for _ in range(10)])
    finding, record = run_agent({"case_id": "doc_loop__q01"}, doc, retriever, client, Q01, "fake/model")
    assert finding is None
    assert record.status == "EXECUTION_FAILED"
    assert record.failure_reason == "api_error"


def test_loop_correct_no_is_answered_not_abstained():
    doc = _make_doc()
    retriever = StubRetriever()
    no_json = json.dumps(
        {
            "answer": "No",
            "evidence": [{"section_ref": "6.3", "quote": "cash consideration"}],
            "rationale": "The tail period does not require public disclosure.",
        }
    )
    client = FakeClient(script=[ScriptedTurn(content=None), ScriptedTurn(content=no_json)])
    finding, record = run_agent({"case_id": "doc_loop__q09"}, doc, retriever, client, Q09, "fake/model")
    assert finding is not None
    assert finding.answer == "No"
    assert record.status == "ANSWERED"
