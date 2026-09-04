"""Seeded selection of 20 agreements for the dev/test eval sets.

Eligibility (spec §3.7):
  - `parser_coverage >= MIN_PARSER_COVERAGE`
  - >= 10 of the 12 questions have a non-null gold answer *and* a usable
    aligned span (>=1 aligned fragment, `covered_fraction >= MIN_FRAGMENT_COVER`)

Then a seeded greedy pass repeatedly picks the eligible agreement that adds
the most unseen (question, answer-option) pairs, ties broken by lowest
`contract_name` for determinism. The 20 chosen agreements are split 5 dev /
15 test, stratified to preserve the q01 (Type of Consideration) answer mix.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass

from dealpoint.config import (
    CONTRACTS_DIR,
    MIN_FRAGMENT_COVER,
    MIN_PARSER_COVERAGE,
    N_AGREEMENTS,
    N_DEV,
    SEED,
)
from dealpoint.data.align import AlignResult, align_span
from dealpoint.data.canonical import canonicalise
from dealpoint.data.labels import Label, is_null_answer
from dealpoint.data.questions import QUESTION_SPEC
from dealpoint.data.sections import Section, parse_sections, parser_coverage

_CONTRACT_ID_RE = re.compile(r"(contract_\d+)")


@dataclass(frozen=True)
class AgreementData:
    agreement_id: str
    canonical: str
    sections: list[Section]
    parser_coverage: float


@dataclass(frozen=True)
class SelectionResult:
    selected: list[str]  # 20 chosen, in selection order
    dev: list[str]  # 5
    test: list[str]  # 15
    rejected: dict[str, str]  # agreement_id -> reason


def list_all_agreement_ids() -> list[str]:
    files = glob.glob(str(CONTRACTS_DIR / "*.txt"))
    ids = [m.group(1) for m in (_CONTRACT_ID_RE.search(f) for f in files) if m]
    return sorted(ids, key=lambda s: int(s.split("_")[1]))


def load_agreement(agreement_id: str) -> AgreementData:
    path = CONTRACTS_DIR / f"{agreement_id}.txt"
    with open(path, encoding="utf-8-sig", newline=None) as fh:
        raw = fh.read()
    canonical = canonicalise(raw)
    sections = parse_sections(canonical)
    coverage = parser_coverage(canonical, sections)
    return AgreementData(agreement_id, canonical, sections, coverage)


def compute_question_alignments(
    agreement: AgreementData, labels: dict[tuple[str, str, str], Label]
) -> dict[str, tuple[Label, AlignResult]]:
    """Return {question_id: (label, align_result)} for every non-null-answer question.

    Unlike `_usable_questions`, this does not filter by the MIN_FRAGMENT_COVER
    threshold -- it is the raw material for both eligibility checks and the
    global alignment-rate gates, which need to see every non-null (contract,
    question) pair regardless of whether it ends up usable.

    Alignment is cached per (contract, text_type): spec notes q06/q11 share
    byte-identical MAE-definition text for every contract, so caching on
    text_type halves the work and is provably correct (0/2870 (contract,
    text_type) keys have >1 distinct gold text).
    """
    results: dict[str, tuple[Label, AlignResult]] = {}
    align_cache: dict[str, AlignResult] = {}
    for q in QUESTION_SPEC:
        key = (agreement.agreement_id, q.maud_question, "<NONE>")
        lbl = labels.get(key)
        if lbl is None or is_null_answer(lbl.answer):
            continue
        if lbl.text_type not in align_cache:
            align_cache[lbl.text_type] = align_span(agreement.canonical, lbl.text)
        results[q.id] = (lbl, align_cache[lbl.text_type])
    return results


def _usable_questions(
    agreement: AgreementData, labels: dict[tuple[str, str, str], Label]
) -> dict[str, tuple[Label, AlignResult]]:
    """Return {question_id: (label, align_result)} for questions with a usable gold span."""
    all_aligned = compute_question_alignments(agreement, labels)
    return {
        qid: (lbl, result)
        for qid, (lbl, result) in all_aligned.items()
        if result.n_aligned >= 1 and result.covered_fraction >= MIN_FRAGMENT_COVER
    }


def select_agreements(
    labels: dict[tuple[str, str, str], Label],
    agreement_ids: list[str] | None = None,
) -> tuple[SelectionResult, dict[str, AgreementData], dict[str, dict[str, tuple[Label, AlignResult]]]]:
    """Run the full eligibility + greedy selection + dev/test split pipeline.

    Returns (SelectionResult, agreement_data_by_id, usable_questions_by_id) so
    callers (cases.py) can reuse the already-computed canonical text, section
    map and alignment results without recomputation.
    """
    if agreement_ids is None:
        agreement_ids = list_all_agreement_ids()

    rejected: dict[str, str] = {}
    eligible: list[str] = []
    agreement_data: dict[str, AgreementData] = {}
    usable_by_id: dict[str, dict[str, tuple[Label, AlignResult]]] = {}

    for agreement_id in agreement_ids:
        data = load_agreement(agreement_id)
        agreement_data[agreement_id] = data
        if data.parser_coverage < MIN_PARSER_COVERAGE:
            rejected[agreement_id] = (
                f"parser_coverage {data.parser_coverage:.3f} < {MIN_PARSER_COVERAGE}"
            )
            continue
        usable = _usable_questions(data, labels)
        usable_by_id[agreement_id] = usable
        if len(usable) < 10:
            rejected[agreement_id] = f"only {len(usable)}/12 questions have a usable gold span (< 10)"
            continue
        eligible.append(agreement_id)

    # SEED is not needed for randomness here (the greedy pass and tie-break
    # are fully deterministic), but is asserted present so the selection
    # process documents that it is seeded end-to-end per spec.
    assert SEED == 42

    seen_pairs: set[tuple[str, str | None]] = set()
    remaining = set(eligible)
    selected: list[str] = []

    while remaining and len(selected) < N_AGREEMENTS:
        best_id = None
        best_gain = -1
        # Deterministic tie-break: lowest contract_name (numeric order via sort key).
        for agreement_id in sorted(remaining, key=lambda s: int(s.split("_")[1])):
            usable = usable_by_id[agreement_id]
            pairs: set[tuple[str, str | None]] = {
                (qid, lbl.answer) for qid, (lbl, _res) in usable.items()
            }
            gain = len(pairs - seen_pairs)
            if gain > best_gain:
                best_gain = gain
                best_id = agreement_id
        assert best_id is not None
        selected.append(best_id)
        remaining.discard(best_id)
        usable = usable_by_id[best_id]
        seen_pairs.update((qid, lbl.answer) for qid, (lbl, _res) in usable.items())

    for agreement_id in remaining:
        rejected[agreement_id] = "eligible but not selected (greedy diversity pass stopped at 20)"

    # Stratified dev/test split preserving the q01 (Type of Consideration) mix.
    def q01_answer(agreement_id: str) -> str | None:
        usable = usable_by_id[agreement_id]
        if "q01" in usable:
            return usable["q01"][0].answer
        return None

    by_answer: dict[str | None, list[str]] = {}
    for agreement_id in selected:
        by_answer.setdefault(q01_answer(agreement_id), []).append(agreement_id)

    dev: list[str] = []
    test: list[str] = []
    dev_target = N_DEV
    total = len(selected)
    # Deterministic order over strata: sort by answer string (None last).
    strata_keys = sorted(by_answer.keys(), key=lambda a: (a is None, a or ""))
    remaining_dev_slots = dev_target
    for i, key in enumerate(strata_keys):
        group = by_answer[key]
        # Proportional share of dev slots for this stratum, rounded, but never
        # exceeding what's left; last stratum takes the remainder.
        if i == len(strata_keys) - 1:
            n_dev_here = remaining_dev_slots
        else:
            n_dev_here = round(dev_target * len(group) / total)
            n_dev_here = min(n_dev_here, remaining_dev_slots, len(group))
        remaining_dev_slots -= n_dev_here
        group_sorted = sorted(group, key=lambda s: int(s.split("_")[1]))
        dev.extend(group_sorted[:n_dev_here])
        test.extend(group_sorted[n_dev_here:])

    dev.sort(key=lambda s: int(s.split("_")[1]))
    test.sort(key=lambda s: int(s.split("_")[1]))

    result = SelectionResult(selected=selected, dev=dev, test=test, rejected=rejected)
    return result, agreement_data, usable_by_id
