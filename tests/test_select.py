import pytest

from dealpoint.config import MIN_PARSER_COVERAGE, N_AGREEMENTS, N_DEV, N_TEST
from dealpoint.data.labels import load_labels
from dealpoint.data.select import select_agreements


@pytest.mark.gate_m0
def test_select_agreements_counts_and_coverage(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    labels = load_labels()
    result, agreement_data, _usable_by_id = select_agreements(labels)

    assert len(result.selected) == N_AGREEMENTS
    assert len(result.dev) == N_DEV
    assert len(result.test) == N_TEST
    assert set(result.dev).isdisjoint(result.test)
    assert set(result.dev) | set(result.test) == set(result.selected)

    for agreement_id in result.selected:
        assert agreement_data[agreement_id].parser_coverage >= MIN_PARSER_COVERAGE


def test_select_agreements_is_deterministic(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    labels = load_labels()
    result1, _, _ = select_agreements(labels)
    result2, _, _ = select_agreements(labels)
    assert result1.selected == result2.selected
    assert result1.dev == result2.dev
    assert result1.test == result2.test
