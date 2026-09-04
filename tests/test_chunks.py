import pytest

from dealpoint.corpus.chunks import chunk_document, chunk_version
from dealpoint.corpus.document import load_document

pytestmark = pytest.mark.gate_m1


@pytest.mark.gate_m1
def test_chunks_never_cross_section_boundary(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    try:
        doc = load_document("contract_0")
    except FileNotFoundError:
        pytest.skip("derived data not built; run `just data`")
    chunks = chunk_document(doc)
    by_ref = {}
    for section in doc.sections:
        by_ref[section.ref] = (section.start, section.end)
    for chunk in chunks:
        if not chunk.section_ref:
            continue
        section_start, section_end = by_ref[chunk.section_ref]
        assert chunk.start >= section_start
        assert chunk.end <= section_end


@pytest.mark.gate_m1
def test_chunk_offsets_round_trip(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    try:
        doc = load_document("contract_0")
    except FileNotFoundError:
        pytest.skip("derived data not built; run `just data`")
    chunks = chunk_document(doc)
    assert len(chunks) > 0
    for chunk in chunks:
        assert doc.text[chunk.start : chunk.end] == chunk.text


@pytest.mark.gate_m1
def test_chunks_deterministic(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    try:
        doc = load_document("contract_0")
    except FileNotFoundError:
        pytest.skip("derived data not built; run `just data`")
    first = [c.chunk_id for c in chunk_document(doc)]
    second = [c.chunk_id for c in chunk_document(doc)]
    assert first == second


@pytest.mark.gate_m1
def test_chunk_agreement_id_is_document_id_for_redacted_variant(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    try:
        doc = load_document("contract_103__redacted_q09")
    except FileNotFoundError:
        pytest.skip("derived data not built; run `just data`")
    chunks = chunk_document(doc)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.agreement_id == "contract_103__redacted_q09"


def test_chunk_version_stable_and_changes_with_params(monkeypatch):
    v1 = chunk_version()
    v2 = chunk_version()
    assert v1 == v2

    import dealpoint.corpus.chunks as chunks_module
    from dealpoint import config

    monkeypatch.setattr(config, "CHUNK_TARGET_CHARS", config.CHUNK_TARGET_CHARS + 1)
    monkeypatch.setattr(chunks_module, "CHUNK_TARGET_CHARS", config.CHUNK_TARGET_CHARS)
    v3 = chunks_module.chunk_version()
    assert v3 != v1


def test_chunks_respect_overlap_within_a_long_section():
    from dealpoint.corpus.document import Document
    from dealpoint.data.sections import Section

    text = "word " * 2000  # ~10000 chars, well above CHUNK_TARGET_CHARS
    section = Section(ref="1.1", title="Long", start=0, end=len(text))
    doc = Document(
        document_id="synthetic_long",
        agreement_id="synthetic_long",
        text=text,
        sections=[section],
        body_start=0,
        body_end=len(text),
    )
    import itertools

    chunks = chunk_document(doc)
    assert len(chunks) > 1
    for a, b in itertools.pairwise(chunks):
        # consecutive sub-chunks of the same section must overlap or be
        # contiguous, never leave a gap
        assert b.start <= a.end
