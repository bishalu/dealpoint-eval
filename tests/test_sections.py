import json

import pytest

from dealpoint.config import (
    CONTRACTS_DIR,
    MIN_ELIGIBLE_COUNT,
    MIN_STRUCTURAL_COVERAGE,
    N_CONTRACTS,
    PARSER_REPORT_PATH,
)
from dealpoint.data.canonical import canonicalise
from dealpoint.data.sections import (
    PARSER_VERSION,
    StaleDerivedDataError,
    check_parser_version,
    compute_body_bounds,
    compute_structural_metrics,
    gold_span_section_rate,
    parse_sections,
)


def test_parse_sections_empty_string():
    assert parse_sections("") == []


def test_parse_sections_no_headings():
    assert parse_sections("just some prose with no headings at all.") == []


def test_compute_structural_metrics_empty_sections_is_zero():
    metrics = compute_structural_metrics("anything", [])
    assert metrics.structural_coverage == 0.0
    assert metrics.n_sections == 0
    assert metrics.giant_single_section is False


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


def test_section_1_01_zero_padded_heading_form():
    canonical = canonicalise(
        "ARTICLE I DEFINITIONS Section 1.01. First Term. Some definition text here that is "
        "long enough. Section 1.02. Second Term. More definition text long enough too. "
        "ARTICLE II COVENANTS Section 2.01. A Covenant. Covenant text here that is plenty long. "
        "ARTICLE III MISC Section 3.01. Misc Term. Misc text that is also long enough here."
    )
    sections = parse_sections(canonical)
    refs = [s.ref for s in sections]
    assert "1.1" in refs  # int() on zero-padded "01" == 1
    assert "1.2" in refs


def test_zero_padded_advance_1_01_to_1_02_to_1_03():
    canonical = canonicalise(
        "ARTICLE I DEFINITIONS Section 1.01. First. Text here that is long enough to matter. "
        "Section 1.02. Second. More text here that is long enough to matter too really. "
        "Section 1.03. Third. Even more text here that is long enough to matter as well. "
        "ARTICLE II COVENANTS Section 2.01. A Covenant. Long enough covenant text goes here. "
        "ARTICLE III MISC Section 3.01. Misc. Long enough misc text goes here as well too."
    )
    sections = parse_sections(canonical)
    refs = [s.ref for s in sections]
    assert "1.1" in refs
    assert "1.2" in refs
    assert "1.3" in refs


def test_article_i_with_trailing_period_mixed_case():
    canonical = canonicalise(
        "Article I. Definitions. 1.1 First Term. Definition text long enough to matter here. "
        "1.2 Second Term. More definition text long enough to matter here too really. "
        "Article II. Covenants. 2.1 A Covenant. Covenant text that is long enough to matter. "
        "Article III. Misc. 3.1 Misc Term. Misc text that is long enough to matter as well."
    )
    sections = parse_sections(canonical)
    refs = [s.ref for s in sections]
    assert any(r.upper().replace(".", "") == "ARTICLE I" for r in refs)
    assert "1.1" in refs


def test_section_keyword_any_case_and_section_symbol():
    canonical = canonicalise(
        "ARTICLE I TITLE section 1.1 First. Text long enough to matter goes right here now. "
        "SECTION 1.2 Second. More text long enough to matter goes right here again now. "
        "\u00a7 1.3 Third. Even more text long enough to matter goes right here once more. "
        "ARTICLE II TITLE 2.1 Another. Text long enough to matter goes right here as well. "
        "ARTICLE III TITLE 3.1 Yet Another. Text long enough to matter goes right here too."
    )
    sections = parse_sections(canonical)
    refs = [s.ref for s in sections]
    assert "1.1" in refs
    assert "1.2" in refs
    assert "1.3" in refs


def _make_synthetic_document(n_sections: int, giant_index: int | None, section_len: int) -> str:
    """A synthetic document with `n_sections` sections in one article, each `section_len`
    chars, except `giant_index` (if given) which is made much longer."""
    parts = ["ARTICLE I TITLE "]
    for i in range(1, n_sections + 1):
        length = section_len * 20 if i == giant_index else section_len
        parts.append(f"1.{i} Heading{i}. " + ("z" * length) + " ")
    parts.append("IN WITNESS WHEREOF the parties have executed this agreement.")
    return "".join(parts)


def test_structural_coverage_on_synthetic_document_with_known_bounds():
    doc = _make_synthetic_document(n_sections=4, giant_index=None, section_len=100)
    canonical = canonicalise(doc)
    sections = parse_sections(canonical)
    metrics = compute_structural_metrics(canonical, sections)
    # Every char of the body is inside some advancing section (they tile the
    # body contiguously by construction), so coverage should be (near) 1.0.
    assert metrics.structural_coverage == pytest.approx(1.0, abs=1e-6)
    assert metrics.n_sections == 4
    assert metrics.giant_single_section is False


def test_giant_single_section_true_above_40_percent_false_at_39_percent():
    # One section > 40% of body -> giant.
    doc_giant = _make_synthetic_document(n_sections=4, giant_index=1, section_len=100)
    canonical_giant = canonicalise(doc_giant)
    sections_giant = parse_sections(canonical_giant)
    metrics_giant = compute_structural_metrics(canonical_giant, sections_giant)
    assert metrics_giant.giant_single_section is True

    # Construct a document where the largest section is just under 40% of body.
    # 3 small sections of ~100 chars each + 1 section padded to `large` chars
    # of filler; `large=210` is measured (see plan) to land the biggest
    # section's share of body comfortably under 40%, `large=300` (used
    # above) comfortably over it.
    small = 100
    large = 210
    doc = (
        "ARTICLE I TITLE "
        f"1.1 Heading1. {'z' * small} "
        f"1.2 Heading2. {'z' * large} "
        f"1.3 Heading3. {'z' * small} "
        f"1.4 Heading4. {'z' * small} "
        "IN WITNESS WHEREOF the parties have executed this agreement."
    )
    canonical = canonicalise(doc)
    sections = parse_sections(canonical)
    metrics = compute_structural_metrics(canonical, sections)
    assert metrics.giant_single_section is False


def test_structural_coverage_penalises_gap_between_toc_and_real_headings():
    """Regression test for the degenerate-coverage bug (M0.1 review finding).

    An earlier version of `compute_body_bounds` defined `body_start` as the
    first *accepted* section's start (`sections[0].start`). Since sections
    tile contiguously from their own first heading to the end of the
    document, that made `covered == body_len` identically -- structural
    coverage could only ever be 1.0 (or 0.0 via the empty-sections early
    exit), regardless of how much of the document was actually unparsed.

    The correct `body_start` is the post-TOC heading offset
    (`_true_body_start`, independent of the accept-chain's anchor). This
    document reproduces the real-world gap seen on ~58/152 MAUD contracts: a
    TOC-like dense run, immediately followed by a stray heading-like match
    that is NOT article 1 (so it becomes the post-TOC body_start but is
    rejected as the chain anchor), then a large stretch of ordinary prose
    with no recognised heading at all, and only then the real ARTICLE I /
    1.1 numbering chain the parser actually anchors on. That stretch is real
    body text and must count as uncovered.

    The existing `test_structural_coverage_on_synthetic_document_with_known_
    bounds` test tiles the body by construction (coverage == 1.0 always) and
    cannot detect this bug; this test can and does fail against the broken
    implementation.
    """
    toc = " ".join(f"{i}.1 Title{i}" for i in range(1, 13))
    stray = " ARTICLE 5 SOME STRAY HEADING-LIKE TEXT HERE. "
    gap = "unrecognised filler prose that contains no heading whatsoever here. " * 20
    body = (
        "ARTICLE I TITLE " + "x" * 300 + " 1.1. Real Heading. " + "y" * 300
        + " ARTICLE II TITLE " + "x" * 300 + " 2.1. Another Heading. " + "y" * 300
        + " ARTICLE III TITLE " + "x" * 300 + " 3.1. Third Heading. " + "y" * 300
    )
    canonical = canonicalise(toc + " " + stray + " " + gap + " " + body)
    sections = parse_sections(canonical)
    metrics = compute_structural_metrics(canonical, sections)
    assert 0.0 < metrics.structural_coverage < 1.0


def test_gold_span_section_rate_excludes_giant_and_out_of_body():
    doc = _make_synthetic_document(n_sections=4, giant_index=2, section_len=100)
    canonical = canonicalise(doc)
    sections = parse_sections(canonical)
    body = compute_body_bounds(canonical, sections)
    metrics = compute_structural_metrics(canonical, sections)
    assert metrics.giant_single_section is True

    giant_section = next(
        s for s in sections if (min(s.end, body[1]) - max(s.start, body[0])) == metrics.max_section_chars
    )
    non_giant_section = next(s for s in sections if s is not giant_section)

    inside_non_giant_mid = (non_giant_section.start + non_giant_section.end) // 2
    inside_giant_mid = (giant_section.start + giant_section.end) // 2
    outside_body_mid = body[1] + 50  # past the signature block

    gold_ranges = [
        (inside_non_giant_mid - 5, inside_non_giant_mid + 5),  # in scope, non-giant
        (inside_giant_mid - 5, inside_giant_mid + 5),  # in a giant -> excluded
        (outside_body_mid - 5, outside_body_mid + 5),  # outside body -> excluded
    ]
    rate = gold_span_section_rate(sections, body, gold_ranges)
    assert rate == pytest.approx(1 / 3, abs=1e-6)


def test_gold_span_section_rate_no_ranges_is_none():
    assert gold_span_section_rate([], (0, 0), []) is None


def test_parser_version_is_short_hex_string():
    assert isinstance(PARSER_VERSION, str)
    assert len(PARSER_VERSION) == 12


def test_stale_state_guard_fires_on_mismatched_version(tmp_path):
    fake_path = tmp_path / "contract_999.json"
    with pytest.raises(StaleDerivedDataError) as excinfo:
        check_parser_version(fake_path, "deadbeefcafe")
    assert "just data" in str(excinfo.value)
    assert str(fake_path) in str(excinfo.value)


def test_stale_state_guard_does_not_fire_on_matching_version(tmp_path):
    fake_path = tmp_path / "contract_999.json"
    check_parser_version(fake_path, PARSER_VERSION)  # must not raise


@pytest.mark.gate_m0
def test_all_selected_agreements_meet_min_structural_coverage(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
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
        metrics = compute_structural_metrics(canonical, sections)
        assert metrics.structural_coverage >= MIN_STRUCTURAL_COVERAGE, (
            f"{agreement_id}: coverage {metrics.structural_coverage} < {MIN_STRUCTURAL_COVERAGE}"
        )
        assert not metrics.giant_single_section, f"{agreement_id}: giant_single_section"


@pytest.mark.gate_m0
def test_n_contracts_constant_matches_dataset(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    files = list(CONTRACTS_DIR.glob("contract_*.txt"))
    assert len(files) == N_CONTRACTS


@pytest.mark.gate_m0
def test_parser_report_eligible_floor_and_gold_span_rate(dataset_available):
    """See specs/milestones/m0_1.md Definition of Done: eligible >= 100 (a
    floor, not the target).

    Corpus `gold_span_section_rate` is report-only (amended 2026-09-04 after
    reviewer escalation in session a534bc40): no numeric threshold on it, in
    the spec or in any test -- a threshold chosen after measurement is
    exactly the tuning §A.3 forbids. This test only asserts the field is
    present and well-formed; the full "equals a fresh recomputation" check
    lives in tests/test_reports.py::test_corpus_gold_span_section_rate_reported.
    """
    if not dataset_available:
        pytest.skip("dataset not present")
    if not PARSER_REPORT_PATH.exists():
        pytest.skip("parser report not built yet; run `uv run python -m dealpoint.cli data`")
    with open(PARSER_REPORT_PATH, encoding="utf-8") as fh:
        report = json.load(fh)

    assert report["eligible_count"] >= MIN_ELIGIBLE_COUNT

    rate = report["corpus_gold_span_section_rate"]
    assert rate is not None
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0
