import pytest

from dealpoint.corpus.document import Document, defined_term, get_section, load_document
from dealpoint.data.sections import Section

pytestmark = pytest.mark.gate_m1


def _make_synthetic_doc() -> Document:
    text = (
        'Article I THE MERGER 6.1. Foo. Some text. '
        '6.2. Bar. More text. '
        '6.3. Baz. "Material Adverse Effect" means with respect to the Company '
        'any change that is materially adverse, provided that none of the following '
        'shall be deemed to constitute a Material Adverse Effect: (a) changes in general '
        'economic conditions. '
        '"Company Material Adverse Effect" shall mean the same thing stated again for '
        'testing purposes. '
        '"Knowledge" means the actual knowledge of the executive officers, in each case '
        'after due inquiry. '
        '6.3(b). Sub-clause text here. '
        'Article VI OTHER PROVISIONS 6.4. Quux. Final text.'
    )
    sections = [
        Section(ref="ARTICLE I", title="THE MERGER", start=0, end=text.index("6.1.")),
        Section(ref="6.1", title="Foo", start=text.index("6.1."), end=text.index("6.2.")),
        Section(ref="6.2", title="Bar", start=text.index("6.2."), end=text.index("6.3. Baz")),
        Section(
            ref="6.3",
            title="Baz",
            start=text.index("6.3. Baz"),
            end=text.index("Article VI OTHER"),
        ),
        Section(
            ref="ARTICLE VI",
            title="OTHER PROVISIONS",
            start=text.index("Article VI OTHER"),
            end=len(text),
        ),
    ]
    return Document(
        document_id="synthetic_0",
        agreement_id="synthetic_0",
        text=text,
        sections=sections,
        body_start=0,
        body_end=len(text),
    )


# --- get_section ---------------------------------------------------------


def _ref(doc, section_ref: str) -> str:
    section = get_section(doc, section_ref)
    assert section is not None
    return section.ref


def test_get_section_bare_dotted_ref():
    doc = _make_synthetic_doc()
    assert _ref(doc, "6.3") == "6.3"


def test_get_section_section_keyword():
    doc = _make_synthetic_doc()
    assert _ref(doc, "Section 6.3") == "6.3"


def test_get_section_section_symbol():
    doc = _make_synthetic_doc()
    assert _ref(doc, "\u00a7 6.3") == "6.3"


def test_get_section_zero_padded():
    doc = _make_synthetic_doc()
    assert _ref(doc, "Section 6.03") == "6.3"


def test_get_section_with_subclause_paren():
    doc = _make_synthetic_doc()
    assert _ref(doc, "6.3(b)") == "6.3"


def test_get_section_with_multiple_parens():
    doc = _make_synthetic_doc()
    assert _ref(doc, "Section 6.3(b)(ii)") == "6.3"


def test_get_section_article_roman_vs_arabic():
    doc = _make_synthetic_doc()
    assert _ref(doc, "Article VI") == "ARTICLE VI"
    assert _ref(doc, "ARTICLE 6") == "ARTICLE VI"


def test_get_section_unknown_returns_none():
    doc = _make_synthetic_doc()
    assert get_section(doc, "99.99") is None
    assert get_section(doc, "not a ref at all") is None


# --- defined_term ----------------------------------------------------------


def test_defined_term_finds_bare_form_first():
    doc = _make_synthetic_doc()
    dt = defined_term(doc, "Material Adverse Effect")
    assert dt is not None
    assert dt.term_as_written == "Material Adverse Effect"
    assert doc.text[dt.start : dt.start + len(dt.term_as_written) + 1].startswith(
        '"Material Adverse Effect'
    )


def test_defined_term_shall_mean_and_has_the_meaning():
    text = (
        '"Knowledge" shall mean actual knowledge. Separately, "Superior Proposal" has '
        "the meaning set forth in Section 1."
    )
    doc = Document(
        document_id="synthetic_1",
        agreement_id="synthetic_1",
        text=text,
        sections=[],
        body_start=0,
        body_end=len(text),
    )
    dt_knowledge = defined_term(doc, "Knowledge")
    assert dt_knowledge is not None
    assert dt_knowledge.term_as_written == "Knowledge"

    dt_superior = defined_term(doc, "Superior Proposal")
    assert dt_superior is not None
    assert dt_superior.term_as_written == "Superior Proposal"


def test_defined_term_prefixed_company_form():
    text = '"Company Material Adverse Effect" means a materially adverse change.'
    doc = Document(
        document_id="synthetic_2",
        agreement_id="synthetic_2",
        text=text,
        sections=[],
        body_start=0,
        body_end=len(text),
    )
    dt = defined_term(doc, "Material Adverse Effect")
    assert dt is not None
    assert dt.term_as_written == "Company Material Adverse Effect"


def test_defined_term_offsets_round_trip():
    doc = _make_synthetic_doc()
    dt = defined_term(doc, "Knowledge")
    assert dt is not None
    assert doc.text[dt.start : dt.end] == dt.text


def test_defined_term_unknown_returns_none():
    doc = _make_synthetic_doc()
    assert defined_term(doc, "Nonexistent Defined Term") is None


def test_defined_term_prefers_bare_over_company_over_parent():
    text = (
        '"Parent Material Adverse Effect" means the parent version. '
        '"Company Material Adverse Effect" means the company version. '
        '"Material Adverse Effect" means the bare version.'
    )
    doc = Document(
        document_id="synthetic_3",
        agreement_id="synthetic_3",
        text=text,
        sections=[],
        body_start=0,
        body_end=len(text),
    )
    dt = defined_term(doc, "Material Adverse Effect")
    assert dt is not None
    assert dt.term_as_written == "Material Adverse Effect"
    assert "bare version" in dt.text


def test_defined_term_company_only_when_no_bare_form():
    text = (
        '"Parent Material Adverse Effect" means the parent version. '
        '"Company Material Adverse Effect" means the company version.'
    )
    doc = Document(
        document_id="synthetic_4",
        agreement_id="synthetic_4",
        text=text,
        sections=[],
        body_start=0,
        body_end=len(text),
    )
    dt = defined_term(doc, "Material Adverse Effect")
    assert dt is not None
    assert dt.term_as_written == "Company Material Adverse Effect"


# --- real selected agreements ----------------------------------------------


@pytest.mark.gate_m1
def test_material_adverse_effect_resolves_on_real_agreement_bare_form(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    try:
        doc = load_document("contract_0")
    except FileNotFoundError:
        pytest.skip("derived data not built; run `just data`")
    dt = defined_term(doc, "Material Adverse Effect")
    assert dt is not None
    assert "material adverse effect" in dt.term_as_written.lower()
    assert doc.text[dt.start : dt.end] == dt.text


@pytest.mark.gate_m1
def test_material_adverse_effect_resolves_on_real_agreement_company_only_form(dataset_available):
    # contract_1: measured during planning to define only "Company Material
    # Adverse Effect", never the bare term -- gates the prefix-tolerance path.
    if not dataset_available:
        pytest.skip("dataset not present")
    try:
        doc = load_document("contract_1")
    except FileNotFoundError:
        pytest.skip("derived data not built; run `just data`")
    dt = defined_term(doc, "Material Adverse Effect")
    assert dt is not None
    assert dt.term_as_written.lower().startswith("company")
    assert doc.text[dt.start : dt.end] == dt.text


@pytest.mark.gate_m1
def test_redacted_variant_loads_without_body_bounds(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    try:
        doc = load_document("contract_103__redacted_q09")
    except FileNotFoundError:
        pytest.skip("derived data not built; run `just data`")
    assert doc.body_start is None
    assert doc.body_end is None
    assert doc.agreement_id == "contract_103"
    assert len(doc.text) > 0
