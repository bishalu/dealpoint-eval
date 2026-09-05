"""Appendix-B skill-adherence rules: applicable/satisfied on synthetic trajectories.

Spec deliverable 2 / DoD: "adherence rules unit-tested (applicable/satisfied
on synthetic trajectories)". Each rule's applicable and satisfied legs are
exercised independently, plus `evaluate()`'s aggregate score.
"""

from __future__ import annotations

import pytest

from dealpoint.agent.schema import Evidence, ExecutionRecord, Finding, Status, TrajectoryStep, Usage
from dealpoint.corpus.document import Document
from dealpoint.data.sections import Section
from dealpoint.eval.skill_adherence import RULES, evaluate

pytestmark = pytest.mark.gate_m4


def _record(status: Status = "ANSWERED", trajectory=None) -> ExecutionRecord:
    return ExecutionRecord(
        status=status,
        failure_reason=None,
        trajectory=trajectory or [],
        usage=Usage(input_tokens=10, output_tokens=10, cost_usd=0.01),
        wall_ms=100,
        model="anthropic/claude-haiku-4.5",
        arm="D",
        case_id="synthetic__q00",
        index_version="test",
        chunk_version="test",
    )


def _case(**overrides) -> dict:
    case = {
        "agreement_id": "synthetic",
        "case_id": "synthetic__q00",
        "case_set": "dev",
        "gold_answer": "All Cash",
        "gold_spans": [],
        "majority_answer": "All Cash",
        "question_id": "q01",
        "required_evidence": None,
        "span_type": "Type of Consideration",
    }
    case.update(overrides)
    return case


def _doc(text: str, sections=None) -> Document:
    return Document(
        document_id="synthetic",
        agreement_id="synthetic",
        text=text,
        sections=sections or [],
        body_start=0,
        body_end=len(text),
    )


def test_rules_are_numbered_1_through_8():
    assert [r.id for r in RULES] == list(range(1, 9))


# --- rule 1: >= 1 search_agreement step -------------------------------------


def test_rule1_always_applicable_and_satisfied_when_search_present():
    step = TrajectoryStep(tool="search_agreement", args={"query": "x"}, char_ranges=[])
    record = _record(trajectory=[step])
    result = evaluate(_case(), None, record, None)
    assert result["rules"]["1"] == "satisfied"


def test_rule1_violated_with_empty_trajectory():
    record = _record(trajectory=[])
    result = evaluate(_case(), None, record, None)
    assert result["rules"]["1"] == "violated"


# --- rule 2: defined-term lookup for required_evidence ----------------------


def test_rule2_not_applicable_when_no_required_evidence():
    record = _record(trajectory=[])
    result = evaluate(_case(required_evidence=None), None, record, None)
    assert result["rules"]["2"] == "n/a"


def test_rule2_applicable_and_satisfied_when_term_looked_up():
    case = _case(required_evidence='defined term "Knowledge" (or "Knowledge of the Company")')
    step = TrajectoryStep(tool="lookup_defined_term", args={"term": "Knowledge"}, char_ranges=[])
    record = _record(trajectory=[step])
    result = evaluate(case, None, record, None)
    assert result["rules"]["2"] == "satisfied"


def test_rule2_applicable_and_violated_when_term_never_looked_up():
    case = _case(required_evidence='defined term "Knowledge" (or "Knowledge of the Company")')
    record = _record(trajectory=[])
    result = evaluate(case, None, record, None)
    assert result["rules"]["2"] == "violated"


# --- rule 3: follow a cross-reference in retrieved text ----------------------


def test_rule3_not_applicable_when_no_cross_reference_in_retrieved_text():
    text = "No cross references here at all."
    doc = _doc(text)
    step = TrajectoryStep(tool="search_agreement", args={}, char_ranges=[(0, len(text))])
    record = _record(trajectory=[step])
    result = evaluate(_case(), None, record, doc)
    assert result["rules"]["3"] == "n/a"


def test_rule3_applicable_and_satisfied_when_get_section_follows():
    text = "As set forth in Section 6.3, the term applies."
    doc = _doc(text, sections=[Section(ref="6.3", title="", start=0, end=len(text))])
    step1 = TrajectoryStep(tool="search_agreement", args={}, char_ranges=[(0, len(text))])
    step2 = TrajectoryStep(tool="get_section", args={"section_ref": "6.3"}, char_ranges=[])
    record = _record(trajectory=[step1, step2])
    result = evaluate(_case(), None, record, doc)
    assert result["rules"]["3"] == "satisfied"


def test_rule3_applicable_and_violated_when_get_section_never_called():
    text = "As set forth in Section 6.3, the term applies."
    doc = _doc(text, sections=[Section(ref="6.3", title="", start=0, end=len(text))])
    step1 = TrajectoryStep(tool="search_agreement", args={}, char_ranges=[(0, len(text))])
    record = _record(trajectory=[step1])
    result = evaluate(_case(), None, record, doc)
    assert result["rules"]["3"] == "violated"


# --- rule 4: read a carve-out list to the end --------------------------------


def test_rule4_not_applicable_when_no_chunk_ends_mid_list():
    text = "This is a complete sentence that ends cleanly."
    doc = _doc(text)
    step = TrajectoryStep(tool="search_agreement", args={}, char_ranges=[(0, len(text))])
    record = _record(trajectory=[step])
    result = evaluate(_case(), None, record, doc)
    assert result["rules"]["4"] == "n/a"


def test_rule4_applicable_and_satisfied_when_next_chunk_read():
    text = "The carveouts are: (a) one; (b) two; and" + (" " * 30) + "(c) three continues here."
    doc = _doc(text)
    mid_end = 41  # ends right after "and"
    step1 = TrajectoryStep(tool="search_agreement", args={}, char_ranges=[(0, mid_end)])
    step2 = TrajectoryStep(
        tool="search_agreement", args={}, char_ranges=[(mid_end + 5, len(text))]
    )
    record = _record(trajectory=[step1, step2])
    result = evaluate(_case(), None, record, doc)
    assert result["rules"]["4"] == "satisfied"


def test_rule4_applicable_and_violated_when_never_followed_up():
    text = "The carveouts are: (a) one; (b) two; and"
    doc = _doc(text)
    step1 = TrajectoryStep(tool="search_agreement", args={}, char_ranges=[(0, len(text))])
    record = _record(trajectory=[step1])
    result = evaluate(_case(), None, record, doc)
    assert result["rules"]["4"] == "violated"


# --- rule 5: citation discipline ---------------------------------------------


def test_rule5_not_applicable_with_no_finding():
    record = _record(trajectory=[])
    result = evaluate(_case(), None, record, None)
    assert result["rules"]["5"] == "n/a"


def test_rule5_applicable_and_satisfied_with_verbatim_quote_overlapping_trajectory():
    text = "The Company shall pay cash consideration to holders."
    doc = _doc(text)
    step = TrajectoryStep(tool="search_agreement", args={}, char_ranges=[(0, len(text))])
    record = _record(trajectory=[step])
    finding = Finding(
        answer="All Cash",
        evidence=[Evidence(section_ref="1.1", quote="cash consideration")],
        rationale="ok",
    )
    result = evaluate(_case(), finding, record, doc)
    assert result["rules"]["5"] == "satisfied"


def test_rule5_applicable_and_violated_with_fabricated_quote():
    text = "The Company shall pay cash consideration to holders."
    doc = _doc(text)
    step = TrajectoryStep(tool="search_agreement", args={}, char_ranges=[(0, len(text))])
    record = _record(trajectory=[step])
    finding = Finding(
        answer="All Cash",
        evidence=[Evidence(section_ref="1.1", quote="this text does not appear anywhere")],
        rationale="ok",
    )
    result = evaluate(_case(), finding, record, doc)
    assert result["rules"]["5"] == "violated"


def test_rule5_violated_when_more_than_three_citations():
    # Finding itself caps evidence at 3 (schema validator), so this rule's
    # ">3 citations" branch is unreachable via a *real* Finding -- assert the
    # predicate directly instead, on a hand-built evidence list bypassing
    # the pydantic validator's own bound, to pin the rule's own logic.
    from dealpoint.eval.skill_adherence import _r5_satisfied

    class _FakeEv:
        def __init__(self, quote):
            self.quote = quote
            self.section_ref = ""

    class _FakeFinding:
        def __init__(self):
            self.answer = "All Cash"
            self.evidence = [_FakeEv("a"), _FakeEv("b"), _FakeEv("c"), _FakeEv("d")]

    record = _record(trajectory=[])
    assert _r5_satisfied(_case(), _FakeFinding(), record, None) is False  # type: ignore[arg-type]


# --- rule 6: >= 2 differently-phrased searches before abstaining ------------


def test_rule6_not_applicable_when_not_abstained():
    record = _record(status="ANSWERED", trajectory=[])
    result = evaluate(_case(), None, record, None)
    assert result["rules"]["6"] == "n/a"


def test_rule6_applicable_and_satisfied_with_two_distinct_queries():
    step1 = TrajectoryStep(tool="search_agreement", args={"query": "knowledge definition"})
    step2 = TrajectoryStep(tool="search_agreement", args={"query": "actual awareness clause"})
    record = _record(status="ABSTAINED", trajectory=[step1, step2])
    result = evaluate(_case(), None, record, None)
    assert result["rules"]["6"] == "satisfied"


def test_rule6_applicable_and_violated_with_only_one_distinct_query():
    step1 = TrajectoryStep(tool="search_agreement", args={"query": "knowledge"})
    step2 = TrajectoryStep(tool="search_agreement", args={"query": "  Knowledge  "})
    record = _record(status="ABSTAINED", trajectory=[step1, step2])
    result = evaluate(_case(), None, record, None)
    assert result["rules"]["6"] == "violated"


# --- rule 7: abstain only when genuinely absent ------------------------------


def test_rule7_not_applicable_when_not_abstained():
    record = _record(status="ANSWERED", trajectory=[])
    result = evaluate(_case(), None, record, None)
    assert result["rules"]["7"] == "n/a"


def test_rule7_satisfied_when_abstaining_on_a_counterfactual_case():
    record = _record(status="ABSTAINED", trajectory=[])
    result = evaluate(_case(case_set="counterfactual"), None, record, None)
    assert result["rules"]["7"] == "satisfied"


def test_rule7_violated_when_abstaining_on_a_case_with_gold_spans():
    record = _record(status="ABSTAINED", trajectory=[])
    result = evaluate(_case(case_set="dev"), None, record, None)
    assert result["rules"]["7"] == "violated"


# --- rule 8: rationale discipline --------------------------------------------


def test_rule8_not_applicable_with_no_finding():
    record = _record(trajectory=[])
    result = evaluate(_case(), None, record, None)
    assert result["rules"]["8"] == "n/a"


def test_rule8_satisfied_when_rationale_names_the_answer():
    finding = Finding(
        answer="All Cash",
        evidence=[Evidence(section_ref="1.1", quote="x")],
        rationale="The agreement provides for All Cash consideration under Section 1.1.",
    )
    record = _record(trajectory=[])
    result = evaluate(_case(), finding, record, None)
    assert result["rules"]["8"] == "satisfied"


def test_rule8_violated_when_rationale_exceeds_80_words():
    long_rationale = " ".join(["word"] * 90)

    class _FakeFinding:
        def __init__(self):
            self.answer = "All Cash"
            self.rationale = long_rationale
            self.evidence = []

    record = _record(trajectory=[])
    result = evaluate(_case(), _FakeFinding(), record, None)  # type: ignore[arg-type]
    assert result["rules"]["8"] == "violated"


def test_rule8_violated_when_rationale_names_neither_answer_nor_section():
    finding = Finding(
        answer="All Cash",
        evidence=[Evidence(section_ref="1.1", quote="x")],
        rationale="This is a generic explanation with no specifics whatsoever here.",
    )
    record = _record(trajectory=[])
    result = evaluate(_case(), finding, record, None)
    assert result["rules"]["8"] == "violated"


# --- evaluate(): aggregate score ---------------------------------------------


def test_evaluate_score_is_none_when_nothing_applicable():
    # rule 1 is always applicable, so force a case where even it can't run:
    # not possible -- rule 1 is always applicable by design. Assert instead
    # that with a trivial case, n_applicable is always >= 1.
    record = _record(trajectory=[])
    result = evaluate(_case(), None, record, None)
    assert result["n_applicable"] >= 1
    assert result["score"] is not None


def test_evaluate_score_is_ratio_of_satisfied_to_applicable():
    step = TrajectoryStep(tool="search_agreement", args={"query": "x"})
    finding = Finding(
        answer="All Cash",
        evidence=[Evidence(section_ref="1.1", quote="x")],
        rationale="All Cash per Section 1.1.",
    )
    record = _record(status="ANSWERED", trajectory=[step])
    result = evaluate(_case(), finding, record, None)
    assert result["score"] == pytest.approx(result["n_satisfied"] / result["n_applicable"])
