"""The frozen discriminative subset regenerates byte-for-byte (spec deliverable 4)."""

from __future__ import annotations

import filecmp
import json

import pytest

from dealpoint.config import TEST_SUBSET_V1_PATH
from dealpoint.eval.subset import generate_subset, load_subset_case_ids, write_subset

pytestmark = pytest.mark.gate_m4


def test_committed_subset_exists():
    assert TEST_SUBSET_V1_PATH.exists()


def test_regeneration_is_byte_identical(tmp_path, dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    tmp_out = tmp_path / "test_subset_v1.json"
    write_subset(path=tmp_out)
    assert filecmp.cmp(tmp_out, TEST_SUBSET_V1_PATH, shallow=False)


def test_subset_has_32_cases():
    payload = json.loads(TEST_SUBSET_V1_PATH.read_text(encoding="utf-8"))
    assert payload["n_cases"] == 32
    assert len(payload["case_ids"]) == 32


def test_subset_composition_24_test_6_redacted_2_oos():
    payload = json.loads(TEST_SUBSET_V1_PATH.read_text(encoding="utf-8"))
    case_ids = payload["case_ids"]
    redacted = [c for c in case_ids if "__redacted_" in c]
    oos = [c for c in case_ids if "__oos" in c]
    test_cases = [c for c in case_ids if c not in redacted and c not in oos]
    assert len(test_cases) == 24
    assert len(redacted) == 6
    assert len(oos) == 2


def test_every_question_appears_exactly_twice_in_test_portion():
    payload = json.loads(TEST_SUBSET_V1_PATH.read_text(encoding="utf-8"))
    case_ids = payload["case_ids"]
    test_cases = [c for c in case_ids if "__redacted_" not in c and "__oos" not in c]
    from collections import Counter

    counts = Counter(c.split("__")[1] for c in test_cases)
    for qid in payload["question_priority"]:
        assert counts[qid] == 2, f"{qid}: {counts[qid]}"


def test_redacted_cases_span_at_least_four_questions():
    payload = json.loads(TEST_SUBSET_V1_PATH.read_text(encoding="utf-8"))
    redacted = [c for c in payload["case_ids"] if "__redacted_" in c]
    questions = {c.split("redacted_")[1] for c in redacted}
    assert len(questions) >= 4


def test_tranche_1_has_all_twelve_questions():
    payload = json.loads(TEST_SUBSET_V1_PATH.read_text(encoding="utf-8"))
    tranche_1 = payload["tranche_1"]
    test_cases = [c for c in tranche_1 if "__redacted_" not in c and "__oos" not in c]
    questions = {c.split("__")[1] for c in test_cases}
    assert questions == set(payload["question_priority"])
    assert len(test_cases) == 12


def test_tranche_1_and_2_together_equal_case_ids():
    payload = json.loads(TEST_SUBSET_V1_PATH.read_text(encoding="utf-8"))
    assert payload["tranche_1"] + payload["tranche_2"] == payload["case_ids"]
    assert len(payload["tranche_1"]) == 18
    assert len(payload["tranche_2"]) == 14


def test_no_case_id_appears_twice():
    payload = json.loads(TEST_SUBSET_V1_PATH.read_text(encoding="utf-8"))
    case_ids = payload["case_ids"]
    assert len(set(case_ids)) == len(case_ids)


def test_seed_and_rule_are_present_in_the_file():
    payload = json.loads(TEST_SUBSET_V1_PATH.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert isinstance(payload["rule"], str) and len(payload["rule"]) > 0


def test_load_subset_case_ids_helpers():
    all_ids = load_subset_case_ids("test_subset_v1")
    t1 = load_subset_case_ids("test_subset_v1", tranche=1)
    t2 = load_subset_case_ids("test_subset_v1", tranche=2)
    assert all_ids == t1 + t2


def test_generate_subset_is_pure_and_deterministic(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    a = generate_subset()
    b = generate_subset()
    assert a == b
