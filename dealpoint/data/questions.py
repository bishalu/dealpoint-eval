"""Frozen 12-question spec (Appendix A of the product brief) plus out-of-scope questions.

All ``maud_question`` and ``option`` strings are copied character-for-character
from the MAUD CSVs, including the two spaces in q10's question string. This is
guarded by a test that checks every string against the real CSV data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionSpec:
    id: str  # "q01".."q12"
    maud_question: str
    text_type: str
    category: str
    gloss: str  # one-line plain English, shown to the model in ALL arms
    options: tuple[str, ...]  # exact answer strings, majority first
    canonical_query: str  # for the M3 retrieval tournament
    reasoning_type: str
    required_evidence: str | None


QUESTION_SPEC: tuple[QuestionSpec, ...] = (
    QuestionSpec(
        id="q01",
        maud_question="Type of Consideration-Answer",
        text_type="Type of Consideration",
        category="General Information",
        gloss="What form of consideration do target stockholders receive in the merger?",
        options=(
            "All Cash",
            "All Stock",
            "Mixed Cash/Stock",
            "Mixed Cash/Stock: Election",
        ),
        canonical_query=(
            "merger consideration per share cash stock conversion of company common stock"
        ),
        reasoning_type="direct",
        required_evidence=None,
    ),
    QuestionSpec(
        id="q02",
        maud_question="Liability standard for no-shop breach by Target Non-D&O Representatives",
        text_type="No-Shop",
        category="Deal Protection and Related Provisions",
        gloss=(
            "What standard governs the target's liability for a no-shop breach committed by its "
            "non-officer/director representatives?"
        ),
        options=("Strict liability", "Reasonable standard"),
        canonical_query=(
            "no solicitation breach by representatives deemed breach by the company"
        ),
        reasoning_type="direct",
        required_evidence=None,
    ),
    QuestionSpec(
        id="q03",
        maud_question="Initial matching rights period (COR)-Answer",
        text_type="Agreement provides for matching rights in connection with COR",
        category="Deal Protection and Related Provisions",
        gloss=(
            "How long does the acquirer have to match a competing offer before the target board "
            "may change its recommendation?"
        ),
        options=(
            "4 business days",
            "5 business days",
            "3 business days",
            "4 calendar days",
            "3 calendar days",
            "2 business days or less",
            "Greater than 5 business days",
        ),
        canonical_query=(
            "change of recommendation notice period parent match negotiate business days"
        ),
        reasoning_type="numeric",
        required_evidence=None,
    ),
    QuestionSpec(
        id="q04",
        maud_question="Ordinary course efforts standard-Answer",
        text_type="Ordinary course covenant",
        category="Operating and Efforts Covenant",
        gloss="What efforts standard governs the target's ordinary-course-of-business covenant?",
        options=(
            "Flat covenant (no efforts standard)",
            "Commercially reasonable efforts",
            "Reasonable best efforts",
        ),
        canonical_query=(
            "conduct of business prior to closing ordinary course consistent with past "
            "practice efforts"
        ),
        reasoning_type="structured",
        required_evidence=None,
    ),
    QuestionSpec(
        id="q05",
        maud_question="Knowledge Definition-Answer",
        text_type="Knowledge Definition",
        category="Knowledge",
        gloss="Does the agreement's definition of Knowledge include constructive knowledge?",
        options=("Constructive knowledge", "Actual knowledge"),
        canonical_query=(
            "definition of knowledge of the company actual knowledge after due inquiry"
        ),
        reasoning_type="defined-term",
        required_evidence='defined term "Knowledge" (or "Knowledge of the Company")',
    ),
    QuestionSpec(
        id="q06",
        maud_question="FLS (MAE) Standard-Answer",
        text_type="MAE Definition",
        category="Material Adverse Effect",
        gloss=(
            "What forward-looking standard does the Material Adverse Effect definition use "
            "(e.g. 'would' vs 'could' reasonably be expected to)?"
        ),
        options=(
            '"Would" (reasonably) be expected to',
            "No",
            '"Would"',
            "Other forward-looking standard",
            '"Could" (reasonably) be expected to',
        ),
        canonical_query=(
            "material adverse effect definition would reasonably be expected to have"
        ),
        reasoning_type="defined-term",
        required_evidence="defined term Material Adverse Effect (or Company MAE)",
    ),
    QuestionSpec(
        id="q07",
        maud_question="Definition includes asset deals-Answer",
        text_type="Superior Offer Definition",
        category="Deal Protection and Related Provisions",
        gloss=(
            "What percentage-of-assets threshold does the Superior Proposal definition require "
            "for an asset-deal alternative?"
        ),
        options=(
            'Greater than 50% but not "all or substantially all"',
            "50%",
            '"All or substantially all"',
            "Less than 50%",
        ),
        canonical_query=(
            "superior proposal definition assets of the company percentage consolidated"
        ),
        reasoning_type="defined-term",
        required_evidence="defined term Superior Proposal",
    ),
    QuestionSpec(
        id="q08",
        maud_question="Definition contains knowledge requirement - answer",
        text_type="Intervening Event Definition",
        category="Deal Protection and Related Provisions",
        gloss=(
            "Does the Intervening Event definition require the event to be unknown, or its "
            "consequences unknown, as of signing?"
        ),
        options=(
            "Known, but consequences unknown or not reasonably foreseeable, at signing",
            "Not known and not reasonably foreseeable at signing",
            "Known, but consequences unknown, at signing",
            "Not known at signing",
        ),
        canonical_query=(
            "intervening event definition not known to the board reasonably foreseeable as "
            "of the date"
        ),
        reasoning_type="defined-term",
        required_evidence="defined term Intervening Event",
    ),
    QuestionSpec(
        id="q09",
        maud_question="Acquisition Proposal required to be publicly disclosed-Answer (Y/N)",
        text_type="Tail Period & Acquisition Proposal Details",
        category="Deal Protection and Related Provisions",
        gloss=(
            "During the tail period, must an acquisition proposal have been publicly disclosed "
            "or made known to trigger the termination fee?"
        ),
        options=("Yes", "No"),
        canonical_query=(
            "termination fee tail acquisition proposal publicly disclosed made known within "
            "twelve months"
        ),
        reasoning_type="cross-ref",
        required_evidence=None,
    ),
    QuestionSpec(
        id="q10",
        maud_question="Fiduciary exception:  Board determination standard-Answer (no-shop)",
        text_type="Fiduciary exception:  Board determination (no-shop)",
        category="Deal Protection and Related Provisions",
        gloss=(
            "What standard must the target board find its fiduciary duties require before "
            "taking action on an unsolicited proposal under the no-shop's fiduciary exception?"
        ),
        options=(
            '"Inconsistent" with fiduciary duties',
            '"Reasonably likely/expected to be inconsistent" with fiduciary duties',
            "Other specified standard",
            '"Reasonably likely/expected violation" of fiduciary duties',
            '"Reasonably likely/expected breach" of fiduciary duties',
            '"Required to comply" with fiduciary duties',
            '"Breach" of fiduciary duties',
            '"Violation" of fiduciary duties',
        ),
        canonical_query=(
            "board determines in good faith failure to take action would be inconsistent "
            "with fiduciary duties unsolicited proposal"
        ),
        reasoning_type="cross-ref",
        required_evidence=None,
    ),
    QuestionSpec(
        id="q11",
        maud_question="Target stockholder proceedings-Answer (Y/N)",
        text_type="MAE Definition",
        category="Material Adverse Effect",
        gloss=(
            "Does the Material Adverse Effect definition carve out stockholder litigation "
            "arising from the merger agreement?"
        ),
        options=("Yes", "No"),
        canonical_query=(
            "material adverse effect carve-out stockholder litigation arising from the "
            "merger agreement"
        ),
        reasoning_type="carve-out",
        required_evidence="defined term Material Adverse Effect",
    ),
    QuestionSpec(
        id="q12",
        maud_question=(
            "Negative Interim Covenant includes carveout for pandemic responses-Answer (Y/N)"
        ),
        text_type="Negative interim operating covenant",
        category="Operating and Efforts Covenant",
        gloss=(
            "Does the negative interim operating covenant carve out actions required by "
            "COVID-19/pandemic measures?"
        ),
        options=("No", "Yes"),
        canonical_query=(
            "interim operating covenants except as required by COVID-19 pandemic measures"
        ),
        reasoning_type="carve-out",
        required_evidence=None,
    ),
)

QUESTION_BY_ID: dict[str, QuestionSpec] = {q.id: q for q in QUESTION_SPEC}


@dataclass(frozen=True)
class OutOfScopeQuestion:
    """A question no merger agreement answers, with hand-written collision-check terms.

    `key_terms` is used by the deterministic collision check in `cases.py`:
    an agreement's canonical text "collides" with this question if every
    term in `key_terms` appears (word-boundary, case-insensitive) within a
    single `OOS_COLLISION_WINDOW`-char window (specs/milestones/m0_1.md
    Problem B).
    """

    id: str  # "oos00".."oos09", or "oos_spare_*"
    text: str  # verbatim from specs/milestones/m0_1.md Problem B
    key_terms: tuple[str, ...]


# 10 hand-written questions no merger agreement answers (specs/milestones/
# m0_1.md Problem B); used to build the out-of-scope counterfactual cases
# (expected gold_answer = "ABSTAIN"). Key-term lists are measured to have
# zero (or, for #6/oos05, low and false-positive-only) collisions across all
# 152 MAUD contracts -- see the M0.1 plan §0.5 for the measured counts.
OUT_OF_SCOPE_QUESTION_SPEC: tuple[OutOfScopeQuestion, ...] = (
    OutOfScopeQuestion(
        id="oos00",
        text=(
            "What percentage of the Target's employees have executed invention-assignment "
            "agreements?"
        ),
        key_terms=("invention", "assignment", "percentage"),
    ),
    OutOfScopeQuestion(
        id="oos01",
        text="What is the Target's current cyber-insurance deductible?",
        key_terms=("cyber", "deductible"),
    ),
    OutOfScopeQuestion(
        id="oos02",
        text="Which ERP system will the combined company use after closing?",
        key_terms=("erp",),
    ),
    OutOfScopeQuestion(
        id="oos03",
        text=(
            "What annualized cost synergies are expected during the first twelve months "
            "after closing?"
        ),
        key_terms=("synergies", "annualized"),
    ),
    OutOfScopeQuestion(
        id="oos04",
        text=(
            "What percentage of the Target's revenue comes from customer contracts "
            "containing change-of-control termination rights?"
        ),
        key_terms=("change of control", "customer contracts", "percentage"),
    ),
    OutOfScopeQuestion(
        id="oos05",
        text="What is the Target's accrued employee PTO liability as of signing?",
        key_terms=("accrued", "vacation"),
    ),
    OutOfScopeQuestion(
        id="oos06",
        text="What is the weighted-average remaining term of the Target's office leases?",
        key_terms=("weighted average", "leases"),
    ),
    OutOfScopeQuestion(
        id="oos07",
        text=(
            "What is the expected post-closing retention-bonus pool for non-executive "
            "employees?"
        ),
        key_terms=("retention", "bonus", "pool"),
    ),
    OutOfScopeQuestion(
        id="oos08",
        text=(
            "What percentage of the Target's source code has been reviewed for "
            "open-source license compliance?"
        ),
        key_terms=("open source", "percentage"),
    ),
    OutOfScopeQuestion(
        id="oos09",
        text="Which jurisdiction governs the Target's ten largest customer contracts?",
        key_terms=("governed by the laws", "customer contracts"),
    ),
)

# Spares, tried in order if a question collides on every selected test
# agreement (specs/milestones/m0_1.md Problem B). Measured: none of the 10
# collide on every agreement, so these are expected to sit idle.
OUT_OF_SCOPE_SPARES: tuple[OutOfScopeQuestion, ...] = (
    OutOfScopeQuestion(
        id="oos_spare_a",
        text=(
            "What is the Target's average customer contract renewal rate over the last "
            "three fiscal years?"
        ),
        key_terms=("renewal rate",),
    ),
    OutOfScopeQuestion(
        id="oos_spare_b",
        text="How many of the Target's patents expire within five years of closing?",
        key_terms=("patents", "expire"),
    ),
    OutOfScopeQuestion(
        id="oos_spare_c",
        text=(
            "What is the Target's headcount attrition rate for the twelve months before "
            "signing?"
        ),
        key_terms=("attrition",),
    ),
)

OUT_OF_SCOPE_QUESTION_BY_ID: dict[str, OutOfScopeQuestion] = {
    q.id: q for q in (*OUT_OF_SCOPE_QUESTION_SPEC, *OUT_OF_SCOPE_SPARES)
}

# Name exporting just the question texts, in order, for callers that only
# need the strings (cases.py, tests).
OUT_OF_SCOPE_QUESTIONS: tuple[str, ...] = tuple(q.text for q in OUT_OF_SCOPE_QUESTION_SPEC)


def _term_offsets(canonical: str, term: str) -> list[int]:
    """Word-boundary, case-insensitive match start offsets for `term` in `canonical`.

    Word boundaries are mandatory (specs/milestones/m0_1.md Problem B, plan
    section 0.5): a bare substring match for "pto" collides on 41/152
    contracts via "Lipton", "raptor", "Hampton", "laptops", "symptoms"; the
    same check with \\b anchors collides on 1.
    """
    pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    return [m.start() for m in pattern.finditer(canonical)]


def term_match_offsets(canonical: str, key_terms: tuple[str, ...]) -> dict[str, list[int]]:
    """All word-boundary match offsets per term, for the collision report's diagnostic detail."""
    return {term: _term_offsets(canonical, term) for term in key_terms}


def question_collides(canonical: str, key_terms: tuple[str, ...], window: int) -> bool:
    """True iff some `window`-char span of `canonical` contains every term in `key_terms`.

    Implementation: collect all term match offsets, merge into one sorted
    list tagged by which term matched, then slide a window over them and
    report a hit as soon as every distinct term has an occurrence inside the
    current window (specs/milestones/m0_1.md Problem B: "if an agreement's
    text plausibly establishes the asked fact (all key terms co-occur within
    400 characters), do not use that pairing").
    """
    if not key_terms:
        return False
    tagged: list[tuple[int, int]] = []  # (offset, term_index)
    for term_idx, term in enumerate(key_terms):
        for offset in _term_offsets(canonical, term):
            tagged.append((offset, term_idx))
    if len(tagged) < len(key_terms):
        return False
    tagged.sort()
    n_terms = len(key_terms)
    left = 0
    counts = [0] * n_terms
    distinct = 0
    for right in range(len(tagged)):
        right_term = tagged[right][1]
        if counts[right_term] == 0:
            distinct += 1
        counts[right_term] += 1
        while tagged[right][0] - tagged[left][0] >= window:
            left_term = tagged[left][1]
            counts[left_term] -= 1
            if counts[left_term] == 0:
                distinct -= 1
            left += 1
        if distinct == n_terms:
            return True
    return False
