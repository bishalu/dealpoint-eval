"""Scorer unit tests, incl. the three synthetic traps (spec §2.5)."""

from __future__ import annotations

import json

import pytest

from dealpoint.agent.schema import Evidence, ExecutionRecord, Finding, TrajectoryStep, Usage
from dealpoint.config import COUNTERFACTUAL_JSONL_PATH, DEV_JSONL_PATH, TEST_JSONL_PATH
from dealpoint.corpus.document import Document
from dealpoint.data.sections import Section
from dealpoint.eval.scorers import (
    abstain_correct,
    answer_correct,
    citation_gold_overlap,
    citation_verbatim,
    fabrication,
    gold_first_rank,
    gold_seen,
    grounded_accuracy,
    majority_baseline,
    overlap_chars,
    parse_required_evidence,
    redacted_fabrication,
    required_evidence_met,
    score_case,
    summarise,
)

pytestmark = pytest.mark.gate_m2

CANONICAL_TEXT = (
    "This is the recitals. The Company shall pay the Merger Consideration in cash "
    "to each holder of Company Common Stock as set forth herein. "
    '"Material Adverse Effect" means any change that would reasonably be expected '
    "to have a materially adverse effect on the Company and its subsidiaries taken "
    "as a whole. Nothing further follows in this synthetic document."
)


def _record(
    status: str = "ANSWERED",
    trajectory: list[TrajectoryStep] | None = None,
    tool_calls: int = 0,
) -> ExecutionRecord:
    return ExecutionRecord(
        status=status,  # type: ignore[arg-type]
        failure_reason=None,
        trajectory=trajectory or [],
        usage=Usage(input_tokens=10, output_tokens=10, cost_usd=0.01, tool_calls=tool_calls),
        wall_ms=100,
        model="anthropic/claude-haiku-4.5",
        arm="B",
        case_id="synthetic__q00",
        index_version="test",
        chunk_version="test",
    )


def _dev_case(**overrides) -> dict:
    case = {
        "agreement_id": "synthetic",
        "case_id": "synthetic__q00",
        "case_set": "dev",
        "gold_answer": "All Cash",
        "gold_spans": [{"start": 60, "end": 130}],
        "majority_answer": "All Cash",
        "question_id": "q01",
        "required_evidence": None,
        "span_type": "Type of Consideration",
    }
    case.update(overrides)
    return case


# --- overlap_chars / gold_ranges -------------------------------------------


def test_overlap_chars_basic():
    assert overlap_chars((0, 10), (5, 15)) == 5
    assert overlap_chars((0, 10), (10, 20)) == 0
    assert overlap_chars((0, 10), (20, 30)) == 0


# --- geometry boundary: 49 vs 50 chars ---------------------------------


def test_gold_overlap_boundary_49_false_50_true():
    # gold span is 100 chars wide: [0, 100)
    case = _dev_case(gold_spans=[{"start": 0, "end": 100}])

    # a trajectory char_range overlapping by exactly 49 chars: [51, 100) has 49
    step_49 = TrajectoryStep(tool="search_agreement", args={}, chunk_ids=[], char_ranges=[(51, 100)])
    record_49 = _record(trajectory=[step_49])
    assert gold_seen(case, None, record_49, CANONICAL_TEXT) is False

    # exactly 50 chars: [50, 100)
    step_50 = TrajectoryStep(tool="search_agreement", args={}, chunk_ids=[], char_ranges=[(50, 100)])
    record_50 = _record(trajectory=[step_50])
    assert gold_seen(case, None, record_50, CANONICAL_TEXT) is True


def test_gold_seen_none_when_no_gold_spans():
    case = _dev_case(gold_spans=[])
    record = _record(trajectory=[])
    assert gold_seen(case, None, record, CANONICAL_TEXT) is None


# --- gold_first_rank ---------------------------------------------------


def test_gold_first_rank_ordering():
    case = _dev_case(gold_spans=[{"start": 200, "end": 260}])
    # first search step returns 3 ranges; only the 2nd (index 2, 1-based) overlaps gold
    step = TrajectoryStep(
        tool="search_agreement",
        args={"query": "x"},
        chunk_ids=["a", "b", "c"],
        char_ranges=[(0, 10), (210, 260), (300, 310)],
    )
    record = _record(trajectory=[step])
    assert gold_first_rank(case, None, record, CANONICAL_TEXT) == 2


def test_gold_first_rank_none_when_no_search_step():
    case = _dev_case(gold_spans=[{"start": 200, "end": 260}])
    step = TrajectoryStep(tool="lookup_defined_term", args={}, char_ranges=[(210, 260)])
    record = _record(trajectory=[step])
    assert gold_first_rank(case, None, record, CANONICAL_TEXT) is None


def test_gold_first_rank_none_when_no_hit():
    case = _dev_case(gold_spans=[{"start": 200, "end": 260}])
    step = TrajectoryStep(
        tool="search_agreement", args={}, chunk_ids=["a"], char_ranges=[(0, 10)]
    )
    record = _record(trajectory=[step])
    assert gold_first_rank(case, None, record, CANONICAL_TEXT) is None


# --- answer_correct: None on CAP_HIT / EXECUTION_FAILED ----------------


def test_answer_correct_none_on_cap_hit_and_execution_failed():
    case = _dev_case()
    assert answer_correct(case, None, _record(status="CAP_HIT"), CANONICAL_TEXT) is None
    assert answer_correct(case, None, _record(status="EXECUTION_FAILED"), CANONICAL_TEXT) is None


def test_answer_correct_true_false():
    case = _dev_case(gold_answer="All Cash")
    finding = Finding(answer="All Cash", evidence=[Evidence(section_ref="", quote="x")], rationale="ok")
    assert answer_correct(case, finding, _record(status="ANSWERED"), CANONICAL_TEXT) is True
    finding2 = Finding(answer="All Stock", evidence=[Evidence(section_ref="", quote="x")], rationale="ok")
    assert answer_correct(case, finding2, _record(status="ANSWERED"), CANONICAL_TEXT) is False


# --- abstain_correct -----------------------------------------------------


def test_abstain_correct_counterfactual_vs_dev():
    counterfactual_case = _dev_case(case_set="counterfactual")
    dev_case = _dev_case(case_set="dev")
    assert abstain_correct(counterfactual_case, None, _record(status="ABSTAINED"), CANONICAL_TEXT) is True
    assert abstain_correct(dev_case, None, _record(status="ABSTAINED"), CANONICAL_TEXT) is False
    assert abstain_correct(dev_case, None, _record(status="ANSWERED"), CANONICAL_TEXT) is True
    assert (
        abstain_correct(counterfactual_case, None, _record(status="ANSWERED"), CANONICAL_TEXT) is False
    )


# --- citation_gold_overlap: quote occurs twice, only 2nd occurrence overlaps


def test_citation_gold_overlap_second_occurrence_only():
    # A quote occurring twice; gold overlaps only the second occurrence, so
    # citation_gold_overlap must be True (best-overlapping occurrence wins).
    long_quote = "B" * 60
    text2 = long_quote + ("z" * 200) + long_quote + ("w" * 200)
    occ1 = text2.index(long_quote)
    occ2 = text2.index(long_quote, occ1 + 1)
    case2 = _dev_case(gold_spans=[{"start": occ2, "end": occ2 + 60}])
    finding2 = Finding(
        answer="All Cash", evidence=[Evidence(section_ref="", quote=long_quote)], rationale="ok"
    )
    assert citation_gold_overlap(case2, finding2, _record(status="ANSWERED"), text2) is True
    # sanity: gold does NOT overlap the first occurrence
    case_first_only = _dev_case(gold_spans=[{"start": 0, "end": 5}])
    assert (
        citation_gold_overlap(case_first_only, finding2, _record(status="ANSWERED"), text2) is False
    )


# --- required_evidence_met: rule 1 (direct tool call) and rule 2 (coverage)


def _synthetic_doc() -> Document:
    text = CANONICAL_TEXT
    sections = [Section(ref="1", title="Definitions", start=0, end=len(text))]
    return Document(
        document_id="synthetic",
        agreement_id="synthetic",
        text=text,
        sections=sections,
        body_start=0,
        body_end=len(text),
    )


def test_required_evidence_met_rule1_direct_tool_call():
    case = _dev_case(required_evidence="defined term Material Adverse Effect (or Company MAE)")
    step = TrajectoryStep(tool="lookup_defined_term", args={"term": "Material Adverse Effect"})
    record = _record(trajectory=[step])
    assert required_evidence_met(case, None, record, CANONICAL_TEXT, doc=None) is True


def test_required_evidence_met_rule2_by_coverage():
    doc = _synthetic_doc()
    case = _dev_case(required_evidence="defined term Material Adverse Effect (or Company MAE)")
    from dealpoint.corpus.document import defined_term

    dt = defined_term(doc, "Material Adverse Effect")
    assert dt is not None
    step = TrajectoryStep(
        tool="search_agreement", args={"query": "x"}, char_ranges=[(dt.start, dt.end)]
    )
    record = _record(trajectory=[step])
    assert required_evidence_met(case, None, record, doc.text, doc=doc) is True


def test_required_evidence_met_none_when_no_required_evidence():
    case = _dev_case(required_evidence=None)
    record = _record(trajectory=[])
    assert required_evidence_met(case, None, record, CANONICAL_TEXT, doc=None) is None


def test_required_evidence_met_false_when_neither_rule_satisfied():
    case = _dev_case(required_evidence="defined term Material Adverse Effect (or Company MAE)")
    record = _record(trajectory=[])
    assert required_evidence_met(case, None, record, CANONICAL_TEXT, doc=None) is False


def test_parse_required_evidence_five_real_values():
    assert parse_required_evidence(
        'defined term "Knowledge" (or "Knowledge of the Company")'
    ) == ("Knowledge", "Knowledge of the Company")
    assert parse_required_evidence(
        "defined term Material Adverse Effect (or Company MAE)"
    ) == ("Material Adverse Effect", "Company MAE")
    assert parse_required_evidence("defined term Superior Proposal") == ("Superior Proposal",)
    assert parse_required_evidence("defined term Intervening Event") == ("Intervening Event",)
    assert parse_required_evidence("defined term Material Adverse Effect") == (
        "Material Adverse Effect",
    )
    assert parse_required_evidence(None) == ()


# --- the three synthetic traps (spec §2.5) ------------------------------


def test_trap_1_fabricated_quote_with_correct_answer_is_not_grounded():
    case = _dev_case(gold_answer="All Cash", gold_spans=[{"start": 60, "end": 130}])
    finding = Finding(
        answer="All Cash",
        evidence=[Evidence(section_ref="1.1", quote="this quote was never in the document")],
        rationale="ok",
    )
    record = _record(status="ANSWERED")
    assert citation_verbatim(case, finding, record, CANONICAL_TEXT) is False
    assert fabrication(case, finding, record, CANONICAL_TEXT) is True
    assert citation_gold_overlap(case, finding, record, CANONICAL_TEXT) is False
    assert grounded_accuracy(case, finding, record, CANONICAL_TEXT) is False
    assert answer_correct(case, finding, record, CANONICAL_TEXT) is True


def test_trap_2_redacted_case_invented_quote():
    redacted_text = "This redacted document has had the operative clause removed entirely."
    case = _dev_case(
        case_id="synthetic__redacted_q08",
        case_set="counterfactual",
        kind="redacted",
        gold_answer="ABSTAIN",
        gold_spans=[],
    )
    finding = Finding(
        answer="Yes",
        evidence=[Evidence(section_ref="1.1", quote="an invented quote not present here")],
        rationale="ok",
    )
    record = _record(status="ANSWERED")
    assert redacted_fabrication(case, finding, record, redacted_text) is True
    assert abstain_correct(case, finding, record, redacted_text) is False

    # same record on a non-redacted case -> redacted_fabrication is None
    non_redacted_case = _dev_case(case_set="dev")
    assert redacted_fabrication(non_redacted_case, finding, record, redacted_text) is None


def test_trap_3_correct_no_is_not_abstention():
    case = _dev_case(
        case_id="synthetic__q11",
        gold_answer="No",
        gold_spans=[{"start": 200, "end": 280}],
    )
    text = "x" * 200 + "Company litigation carve-out language appears here in this clause." + "y" * 200
    quote = "Company litigation carve-out language appears here in this clause."
    finding = Finding(
        answer="No",
        evidence=[Evidence(section_ref="1.1", quote=quote)],
        rationale="ok",
    )
    record = _record(status="ANSWERED")
    assert answer_correct(case, finding, record, text) is True
    assert abstain_correct(case, finding, record, text) is True
    assert citation_gold_overlap(case, finding, record, text) is True
    assert grounded_accuracy(case, finding, record, text) is True

    rows = [
        {
            "case_set": "dev",
            "record": {"status": "ANSWERED"},
            "scores": score_case(case, finding, record, None),
        }
    ]
    summary = summarise(rows)
    assert summary["false_abstain"]["n"] == 1
    assert summary["false_abstain"]["mean"] == 0.0


# --- score_case: every key present, None included -----------------------


def test_score_case_has_every_key():
    from dealpoint.eval.scorers import SCORE_FIELD_NAMES

    case = _dev_case()
    finding = Finding(answer="All Cash", evidence=[Evidence(section_ref="", quote="x")], rationale="ok")
    record = _record(status="ANSWERED")
    scores = score_case(case, finding, record, None)
    assert set(scores.keys()) == set(SCORE_FIELD_NAMES)
    # M4: skill_adherence is now a real deterministic score for every arm
    # (spec deliverable 2), computed from an applicable/satisfied rule set
    # over the trajectory -- never None when at least one rule applies
    # (rule 1, ">= 1 search_agreement step", is always applicable).
    assert scores["skill_adherence"] is not None


# --- majority_baseline matches a fresh recount from data/eval -----------


def _load_jsonl(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def test_majority_baseline_matches_data_eval_recount(dataset_available):
    if not (DEV_JSONL_PATH.exists() and TEST_JSONL_PATH.exists()):
        pytest.skip("dev.jsonl/test.jsonl not present")
    cases = _load_jsonl(DEV_JSONL_PATH) + _load_jsonl(TEST_JSONL_PATH)
    baseline = majority_baseline(cases)

    # independent recount
    per_question: dict[str, list[bool]] = {}
    overall: list[bool] = []
    for case in cases:
        if case.get("majority_answer") is None:
            continue
        correct = case["majority_answer"] == case["gold_answer"]
        per_question.setdefault(case["question_id"], []).append(correct)
        overall.append(correct)

    for qid, values in per_question.items():
        assert baseline[qid] == pytest.approx(sum(values) / len(values))
    assert baseline["overall"] == pytest.approx(sum(overall) / len(overall))


def test_majority_baseline_skips_out_of_scope_null_majority():
    cases = [
        {"question_id": "oos00", "majority_answer": None, "gold_answer": "ABSTAIN"},
        {"question_id": "q01", "majority_answer": "All Cash", "gold_answer": "All Cash"},
    ]
    baseline = majority_baseline(cases)
    assert "oos00" not in baseline
    assert baseline["q01"] == 1.0
    assert baseline["overall"] == 1.0


def test_counterfactual_jsonl_present_for_completeness(dataset_available):
    if not COUNTERFACTUAL_JSONL_PATH.exists():
        pytest.skip("counterfactual.jsonl not present")
    rows = _load_jsonl(COUNTERFACTUAL_JSONL_PATH)
    assert len(rows) > 0
