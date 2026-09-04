import pytest

from dealpoint.config import N_CONTRACTS, N_MAIN_ROWS
from dealpoint.data.labels import is_null_answer, load_labels


def test_is_null_answer_variants():
    assert is_null_answer(None)
    assert is_null_answer("")
    assert is_null_answer("None")
    assert is_null_answer("<NONE>")
    assert is_null_answer("nan")
    assert not is_null_answer("Yes")
    assert not is_null_answer("All Cash")


@pytest.mark.gate_m0
def test_label_integrity(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    labels = load_labels()
    assert len(labels) == N_MAIN_ROWS
    contracts = {lbl.contract_name for lbl in labels.values()}
    assert len(contracts) == N_CONTRACTS
    # keys are unique by construction (dict), but assert no collisions were
    # silently overwritten by checking the loader's own invariant held.
    for key, lbl in labels.items():
        assert key == (lbl.contract_name, lbl.question, lbl.subquestion)
