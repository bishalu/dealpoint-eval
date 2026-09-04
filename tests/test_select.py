import json

import pytest

from dealpoint.config import (
    GIANT_SECTION_FRACTION,
    MIN_STRUCTURAL_COVERAGE,
    N_AGREEMENTS,
    N_DEV,
    N_TEST,
)
from dealpoint.data.labels import load_labels
from dealpoint.data.sections import StaleDerivedDataError, assert_sections_dir_not_stale
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
        data = agreement_data[agreement_id]
        assert data.structural_coverage >= MIN_STRUCTURAL_COVERAGE
        assert not data.giant_single_section


def test_select_agreements_is_deterministic(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    labels = load_labels()
    result1, _, _ = select_agreements(labels)
    result2, _, _ = select_agreements(labels)
    assert result1.selected == result2.selected
    assert result1.dev == result2.dev
    assert result1.test == result2.test


@pytest.mark.gate_m0
def test_stale_state_guard_fires_for_select_and_cases(tmp_path):
    """A derived sections file with a mismatched parser_version must be refused.

    Points `assert_sections_dir_not_stale` (the same function `select.py` and
    `cases.py` call) at a temp directory holding a single stale file, rather
    than corrupting a real file under data/derived/ (specs/milestones/m0_1.md
    §A.5: "A gate_m0 test asserts the guard fires on a mismatched file").
    """
    stale_dir = tmp_path / "sections"
    stale_dir.mkdir()
    stale_path = stale_dir / "contract_0.json"
    stale_path.write_text(json.dumps({"parser_version": "deadbeefcafe"}), encoding="utf-8")

    with pytest.raises(StaleDerivedDataError) as excinfo:
        assert_sections_dir_not_stale(stale_dir)
    assert "just data" in str(excinfo.value)


def test_no_stale_state_guard_on_missing_dir(tmp_path):
    # A directory that doesn't exist yet (e.g. before the first `data` run)
    # must not be treated as stale.
    assert_sections_dir_not_stale(tmp_path / "does_not_exist")


@pytest.mark.gate_m0
def test_eligibility_requires_not_giant(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    labels = load_labels()
    result, agreement_data, _usable_by_id = select_agreements(labels)
    for agreement_id, reason in result.rejected.items():
        if "giant_single_section" in reason:
            data = agreement_data[agreement_id]
            assert data.giant_single_section
            assert data.max_section_chars > GIANT_SECTION_FRACTION * (data.body_end - data.body_start)
