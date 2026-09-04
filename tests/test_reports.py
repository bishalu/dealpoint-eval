import json

import pytest

from dealpoint.config import MIN_ELIGIBLE_COUNT, PARSER_REPORT_PATH, PARSER_VERSION_TXT_PATH
from dealpoint.data.reports import recompute_corpus_gold_span_section_rate_from_disk
from dealpoint.data.sections import PARSER_VERSION


@pytest.mark.gate_m0
def test_parser_report_exists_and_parses(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    if not PARSER_REPORT_PATH.exists():
        pytest.skip("parser report not built yet; run `uv run python -m dealpoint.cli data`")
    with open(PARSER_REPORT_PATH, encoding="utf-8") as fh:
        report = json.load(fh)
    assert "eligible_count" in report
    assert "per_agreement" in report
    assert "structural_coverage_distribution" in report
    assert "giant_section_failures" in report
    assert "corpus_gold_span_section_rate" in report


@pytest.mark.gate_m0
def test_eligible_count_floor(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    if not PARSER_REPORT_PATH.exists():
        pytest.skip("parser report not built yet; run `uv run python -m dealpoint.cli data`")
    with open(PARSER_REPORT_PATH, encoding="utf-8") as fh:
        report = json.load(fh)
    # Floor, not the target (specs/milestones/m0_1.md §A.6): the target
    # "~120" is reported, not gated.
    assert report["eligible_count"] >= MIN_ELIGIBLE_COUNT


@pytest.mark.gate_m0
def test_corpus_gold_span_section_rate_reported(dataset_available):
    """See specs/milestones/m0_1.md Definition of Done (amended 2026-09-04,
    after reviewer escalation in session a534bc40): corpus
    `gold_span_section_rate` is report-only and MUST NOT be gated by any
    numeric threshold -- a threshold chosen after measurement is exactly the
    tuning §A.3 forbids. This test instead asserts the field is present, is
    a float in [0, 1], and equals a fresh, independent recomputation from
    the committed derived sections + alignment files (tolerance 1e-9) -- see
    `recompute_corpus_gold_span_section_rate_from_disk` in
    dealpoint/data/reports.py. The report also carries the miss breakdown by
    cause and the arithmetic ceiling, asserted below, so a reader can see
    *why* the rate is what it is without a pass/fail bar being attached to
    it.
    """
    if not dataset_available:
        pytest.skip("dataset not present")
    if not PARSER_REPORT_PATH.exists():
        pytest.skip("parser report not built yet; run `uv run python -m dealpoint.cli data`")
    with open(PARSER_REPORT_PATH, encoding="utf-8") as fh:
        report = json.load(fh)

    rate = report["corpus_gold_span_section_rate"]
    assert rate is not None
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0

    recomputed = recompute_corpus_gold_span_section_rate_from_disk()
    assert recomputed is not None
    assert rate == pytest.approx(recomputed, abs=1e-9)

    breakdown = report["corpus_gold_span_miss_breakdown"]
    for cause in ("before_body_start", "unrecognised_gap", "giant_section", "after_signature_block"):
        assert cause in breakdown
        assert isinstance(breakdown[cause], int)
        assert breakdown[cause] >= 0

    ceiling = report["gold_span_section_rate_ceiling"]
    assert ceiling is not None
    assert 0.0 <= ceiling <= 1.0
    # The ceiling can only ever be >= the rate itself (it is the rate plus
    # whatever share of misses are, in principle, reclaimable).
    assert ceiling >= rate - 1e-9


@pytest.mark.gate_m0
def test_parser_version_txt_matches_module_constant(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    if not PARSER_VERSION_TXT_PATH.exists():
        pytest.skip("parser_version.txt not built yet; run `uv run python -m dealpoint.cli data`")
    content = PARSER_VERSION_TXT_PATH.read_text(encoding="utf-8").strip()
    assert content == PARSER_VERSION
