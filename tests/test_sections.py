import pytest

from dealpoint.config import CONTRACTS_DIR, MIN_PARSER_COVERAGE, N_CONTRACTS
from dealpoint.data.canonical import canonicalise
from dealpoint.data.sections import parse_sections, parser_coverage


def test_parse_sections_empty_string():
    assert parse_sections("") == []


def test_parse_sections_no_headings():
    assert parse_sections("just some prose with no headings at all.") == []


def test_parser_coverage_empty_sections_is_zero():
    assert parser_coverage("anything", []) == 0.0


def test_toc_run_is_skipped():
    # A dense run of 12 "N.N" matches within 200 chars of each other (TOC),
    # followed by a real multi-article heading run further away, should not
    # have the TOC entries counted as sections. Three ARTICLE headings keep
    # this above the module's bare-heading fallback threshold.
    toc = " ".join(f"{i}.1 Title{i}" for i in range(1, 13))
    body = (
        "ARTICLE I TITLE " + "x" * 500 + " 1.1. Real Heading. " + "y" * 300
        + " ARTICLE II TITLE " + "x" * 300 + " 2.1. Another Heading. " + "y" * 300
        + " ARTICLE III TITLE " + "x" * 300 + " 3.1. Third Heading. " + "y" * 300
    )
    canonical = canonicalise(toc + " " + body)
    sections = parse_sections(canonical)
    # The TOC-only "N.1" fragments should not all appear as separate sections;
    # real section detection should find the ARTICLE I / 1.1 pair in the body.
    refs = [s.ref for s in sections]
    assert "ARTICLE I" in refs
    assert "1.1" in refs


def test_cross_reference_is_not_a_new_heading():
    # "Section 6.3(b)" appearing in prose after section 1.1, without 1.2..6.2
    # in between, must not be accepted as a heading (non-advancing). Three
    # ARTICLE headings keep this above the bare-heading fallback threshold.
    canonical = canonicalise(
        "ARTICLE I TITLE 1.1. First Section. Some text referencing Section 6.3(b) here. "
        "1.2. Second Section. More text. "
        "ARTICLE II TITLE 2.1. Another Section. More text. "
        "ARTICLE III TITLE 3.1. Yet Another Section. More text."
    )
    sections = parse_sections(canonical)
    refs = [s.ref for s in sections]
    assert "6.3" not in refs
    assert "1.1" in refs
    assert "1.2" in refs


@pytest.mark.gate_m0
def test_all_selected_agreements_meet_min_parser_coverage(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    import json

    from dealpoint.config import SELECTION_PATH

    if not SELECTION_PATH.exists():
        pytest.skip("selection.json not built yet; run `uv run python -m dealpoint.cli data`")
    with open(SELECTION_PATH, encoding="utf-8") as fh:
        selection = json.load(fh)
    for agreement_id in selection["selected"]:
        path = CONTRACTS_DIR / f"{agreement_id}.txt"
        with open(path, encoding="utf-8-sig", newline=None) as fh:
            raw = fh.read()
        canonical = canonicalise(raw)
        sections = parse_sections(canonical)
        cov = parser_coverage(canonical, sections)
        assert cov >= MIN_PARSER_COVERAGE, f"{agreement_id}: coverage {cov} < {MIN_PARSER_COVERAGE}"


def test_n_contracts_constant_matches_dataset(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    files = list(CONTRACTS_DIR.glob("contract_*.txt"))
    assert len(files) == N_CONTRACTS
