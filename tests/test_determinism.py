"""Gate 7: rebuilding the case files into a temp dir must be byte-identical
to the committed files under data/eval/.
"""

from __future__ import annotations

import filecmp

import pytest

from dealpoint.config import (
    COUNTERFACTUAL_JSONL_PATH,
    DEV_JSONL_PATH,
    EXCLUDED_PATH,
    TEST_JSONL_PATH,
)
from dealpoint.data import cases as cases_module
from dealpoint.data.labels import load_labels
from dealpoint.data.select import select_agreements


@pytest.mark.gate_m0
def test_rebuild_matches_committed_case_files(tmp_path, monkeypatch, dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    if not (DEV_JSONL_PATH.exists() and TEST_JSONL_PATH.exists() and COUNTERFACTUAL_JSONL_PATH.exists()):
        pytest.skip("committed case files not present yet")

    tmp_dev = tmp_path / "dev.jsonl"
    tmp_test = tmp_path / "test.jsonl"
    tmp_cf = tmp_path / "counterfactual.jsonl"
    tmp_excluded = tmp_path / "excluded.json"

    monkeypatch.setattr(cases_module, "DEV_JSONL_PATH", tmp_dev)
    monkeypatch.setattr(cases_module, "TEST_JSONL_PATH", tmp_test)
    monkeypatch.setattr(cases_module, "COUNTERFACTUAL_JSONL_PATH", tmp_cf)
    monkeypatch.setattr(cases_module, "EXCLUDED_PATH", tmp_excluded)

    labels = load_labels()
    selection, agreement_data, usable_by_id = select_agreements(labels)
    dev_rows, test_rows, excluded_rows = cases_module.build_dev_test_cases(
        labels, selection, agreement_data, usable_by_id
    )
    redacted_rows, _docs = cases_module.build_redacted_cases(test_rows, agreement_data)
    oos_rows = cases_module.build_out_of_scope_cases(selection.test)
    counterfactual_rows = redacted_rows + oos_rows

    cases_module.write_case_files(dev_rows, test_rows, counterfactual_rows, excluded_rows)

    assert filecmp.cmp(tmp_dev, DEV_JSONL_PATH, shallow=False)
    assert filecmp.cmp(tmp_test, TEST_JSONL_PATH, shallow=False)
    assert filecmp.cmp(tmp_cf, COUNTERFACTUAL_JSONL_PATH, shallow=False)
    assert filecmp.cmp(tmp_excluded, EXCLUDED_PATH, shallow=False)
