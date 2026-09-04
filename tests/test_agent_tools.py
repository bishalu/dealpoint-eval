import pytest

from dealpoint.agent.tools import get_section, lookup_defined_term, search_agreement, tool_schemas
from dealpoint.config import MAX_TOOL_RESULT_CHARS
from dealpoint.corpus.chunks import Chunk
from dealpoint.corpus.document import Document
from dealpoint.data.sections import Section

pytestmark = pytest.mark.gate_m1


class StubRetriever:
    """A plain-list-backed stub -- no Qdrant, no ONNX."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]:
        return [c for c in self._chunks if c.agreement_id == agreement_id][:k]


def _make_doc() -> Document:
    text = (
        "Article I THE MERGER 6.1. Foo. Some text about representations. "
        '6.3. Baz. "Material Adverse Effect" means a materially adverse change '
        "to the Company."
    )
    sections = [
        Section(ref="ARTICLE I", title="THE MERGER", start=0, end=text.index("6.1.")),
        Section(ref="6.1", title="Foo", start=text.index("6.1."), end=text.index("6.3. Baz")),
        Section(ref="6.3", title="Baz", start=text.index("6.3. Baz"), end=len(text)),
    ]
    return Document(
        document_id="doc_x",
        agreement_id="doc_x",
        text=text,
        sections=sections,
        body_start=0,
        body_end=len(text),
    )


def test_search_agreement_returns_section_ref_and_offsets():
    doc = _make_doc()
    chunk = Chunk(
        agreement_id="doc_x",
        chunk_id="doc_x:0-10",
        section_ref="6.1",
        start=0,
        end=10,
        text=doc.text[0:10],
    )
    retriever = StubRetriever([chunk])
    result = search_agreement(doc, retriever, "foo", k=5)
    assert "section 6.1" in result.text
    assert "chars 0-10" in result.text
    assert result.chunk_ids == ["doc_x:0-10"]
    assert result.char_ranges == [(0, 10)]


def test_search_agreement_miss_returns_message_not_exception():
    doc = _make_doc()
    retriever = StubRetriever([])
    result = search_agreement(doc, retriever, "nothing matches", k=5)
    assert "no results" in result.text
    assert result.chunk_ids == []


def test_get_section_hit():
    doc = _make_doc()
    result = get_section(doc, "6.3")
    assert "section 6.3" in result.text
    assert "Material Adverse Effect" in result.text
    assert len(result.char_ranges) == 1


def test_get_section_miss_returns_message():
    doc = _make_doc()
    result = get_section(doc, "99.99")
    assert 'no section matching "99.99"' in result.text


def test_lookup_defined_term_hit():
    doc = _make_doc()
    result = lookup_defined_term(doc, "Material Adverse Effect")
    assert "Material Adverse Effect" in result.text
    assert "section 6.3" in result.text


def test_lookup_defined_term_miss_returns_message():
    doc = _make_doc()
    result = lookup_defined_term(doc, "Nonexistent Term")
    assert 'no definition found for "Nonexistent Term"' in result.text


def test_tool_result_truncation_preserves_section_ref():
    long_text = "x " * (MAX_TOOL_RESULT_CHARS + 500)
    section = Section(ref="9.9", title="Long", start=0, end=len(long_text))
    doc = Document(
        document_id="doc_long",
        agreement_id="doc_long",
        text=long_text,
        sections=[section],
        body_start=0,
        body_end=len(long_text),
    )
    result = get_section(doc, "9.9")
    assert "section 9.9" in result.text
    assert "[truncated]" in result.text
    # header + truncated body should be much shorter than the raw section
    assert len(result.text) < len(long_text)


def test_search_agreement_never_returns_another_documents_chunks():
    doc = _make_doc()
    other_chunk = Chunk(
        agreement_id="other_doc",
        chunk_id="other_doc:0-5",
        section_ref="1.1",
        start=0,
        end=5,
        text="hello",
    )
    retriever = StubRetriever([other_chunk])
    result = search_agreement(doc, retriever, "anything", k=5)
    assert "no results" in result.text


def test_tool_schemas_search_agreement_exposes_only_query_and_k():
    schemas = tool_schemas()
    search_schema = next(s for s in schemas if s["function"]["name"] == "search_agreement")
    props = search_schema["function"]["parameters"]["properties"]
    assert set(props.keys()) == {"query", "k"}
