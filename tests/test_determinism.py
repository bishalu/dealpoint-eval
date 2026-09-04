"""Gate 7: rebuilding the case files into a temp dir must be byte-identical
to the committed files under data/eval/.
"""

from __future__ import annotations

import filecmp

import pytest

from dealpoint.config import (
    COUNTERFACTUAL_JSONL_PATH,
    DATASET_VERSION_TXT_PATH,
    DEV_JSONL_PATH,
    EXCLUDED_PATH,
    OOS_COLLISIONS_REPORT_PATH,
    PARSER_REPORT_PATH,
    PARSER_VERSION_TXT_PATH,
    TEST_JSONL_PATH,
)
from dealpoint.data import cases as cases_module
from dealpoint.data import reports as reports_module
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
    oos_rows, _collision_rows, _substitutions = cases_module.build_out_of_scope_cases(
        selection.test, agreement_data
    )
    counterfactual_rows = redacted_rows + oos_rows

    cases_module.write_case_files(dev_rows, test_rows, counterfactual_rows, excluded_rows)

    assert filecmp.cmp(tmp_dev, DEV_JSONL_PATH, shallow=False)
    assert filecmp.cmp(tmp_test, TEST_JSONL_PATH, shallow=False)
    assert filecmp.cmp(tmp_cf, COUNTERFACTUAL_JSONL_PATH, shallow=False)
    assert filecmp.cmp(tmp_excluded, EXCLUDED_PATH, shallow=False)


@pytest.mark.gate_m0
def test_rebuild_matches_committed_report_artifacts(tmp_path, monkeypatch, dataset_available):
    """The M0.1 report artifacts must also rebuild byte-identical (plan section 7:
    it must now also cover the committed reports)."""
    if not dataset_available:
        pytest.skip("dataset not present")
    if not (PARSER_REPORT_PATH.exists() and PARSER_VERSION_TXT_PATH.exists() and DATASET_VERSION_TXT_PATH.exists()):
        pytest.skip("committed report artifacts not present yet")

    tmp_report = tmp_path / "m0_1_parser_report.json"
    tmp_collisions = tmp_path / "m0_1_oos_collisions.json"
    tmp_parser_version = tmp_path / "parser_version.txt"
    tmp_dataset_version = tmp_path / "dataset_version.txt"

    monkeypatch.setattr(reports_module, "PARSER_REPORT_PATH", tmp_report)
    monkeypatch.setattr(cases_module, "OOS_COLLISIONS_REPORT_PATH", tmp_collisions)
    monkeypatch.setattr(reports_module, "PARSER_VERSION_TXT_PATH", tmp_parser_version)
    monkeypatch.setattr(reports_module, "DATASET_VERSION_TXT_PATH", tmp_dataset_version)

    labels = load_labels()
    selection, agreement_data, usable_by_id = select_agreements(labels)
    dev_rows, test_rows, excluded_rows = cases_module.build_dev_test_cases(
        labels, selection, agreement_data, usable_by_id
    )
    _redacted_rows, _docs = cases_module.build_redacted_cases(test_rows, agreement_data)
    _oos_rows, collision_rows, substitutions = cases_module.build_out_of_scope_cases(
        selection.test, agreement_data
    )
    del excluded_rows, dev_rows, test_rows  # not needed for this comparison

    cases_module.write_collision_report(collision_rows, substitutions)

    report = reports_module.build_parser_report(labels, selection, agreement_data)
    reports_module.write_parser_report(report)
    reports_module.write_parser_version_txt()
    # dataset_version.txt hashes the committed dev/test/counterfactual.jsonl,
    # which are unaffected by this test's monkeypatching, so hashing the
    # already-committed files here is equivalent to a full rebuild's value.
    reports_module.write_dataset_version_txt()

    assert filecmp.cmp(tmp_report, PARSER_REPORT_PATH, shallow=False)
    assert filecmp.cmp(tmp_collisions, OOS_COLLISIONS_REPORT_PATH, shallow=False)
    assert filecmp.cmp(tmp_parser_version, PARSER_VERSION_TXT_PATH, shallow=False)
    assert filecmp.cmp(tmp_dataset_version, DATASET_VERSION_TXT_PATH, shallow=False)
