"""Gates 1-3 from spec §6: global fragment- and case-level alignment rates.

Computed once over all 12 questions x all 152 contracts (not just the 20
selected agreements), matching the CLI's own summary computation.
"""

from __future__ import annotations

import pytest

from dealpoint.data.labels import load_labels
from dealpoint.data.select import (
    compute_question_alignments,
    list_all_agreement_ids,
    load_agreement,
)

_MIN_FRAGMENT_RATE = 0.95
_MIN_CASE_ANY_RATE = 0.95
_MIN_CASE_ALL_RATE = 0.90


def _compute_rates():
    labels = load_labels()
    total_frag = 0
    total_frag_aligned = 0
    total_cases = 0
    total_any = 0
    total_all = 0

    for agreement_id in list_all_agreement_ids():
        data = load_agreement(agreement_id)
        aligned = compute_question_alignments(data, labels)
        for _label, result in aligned.values():
            if result.n_fragments == 0:
                continue
            total_frag += result.n_fragments
            total_frag_aligned += result.n_aligned
            total_cases += 1
            total_any += int(result.n_aligned >= 1)
            total_all += int(result.n_aligned == result.n_fragments)

    return {
        "fragment_rate": total_frag_aligned / total_frag,
        "case_any_rate": total_any / total_cases,
        "case_all_rate": total_all / total_cases,
    }


@pytest.mark.gate_m0
def test_fragment_level_alignment_rate_gate(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    rates = _compute_rates()
    assert rates["fragment_rate"] >= _MIN_FRAGMENT_RATE


@pytest.mark.gate_m0
def test_case_any_fragment_aligned_rate_gate(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    rates = _compute_rates()
    assert rates["case_any_rate"] >= _MIN_CASE_ANY_RATE


@pytest.mark.gate_m0
def test_case_all_fragments_aligned_rate_gate(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    rates = _compute_rates()
    assert rates["case_all_rate"] >= _MIN_CASE_ALL_RATE
