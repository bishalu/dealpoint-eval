"""The M5 judged subset regenerates byte-for-byte (spec deliverable 4)."""

from __future__ import annotations

import filecmp
import json

import pytest

from dealpoint.config import JUDGED_SUBSET_PATH
from dealpoint.eval.subset import write_judged_subset

pytestmark = pytest.mark.gate_m5


def test_committed_judged_subset_exists():
    assert JUDGED_SUBSET_PATH.exists()


def test_regeneration_is_byte_identical(tmp_path, dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    tmp_out = tmp_path / "judged_subset.json"
    write_judged_subset(path=tmp_out)
    assert filecmp.cmp(tmp_out, JUDGED_SUBSET_PATH, shallow=False)


def test_judged_subset_has_18_cases():
    payload = json.loads(JUDGED_SUBSET_PATH.read_text(encoding="utf-8"))
    assert payload["n_cases"] == 18
    assert len(payload["case_ids"]) == 18


def test_every_question_has_at_least_one_case():
    payload = json.loads(JUDGED_SUBSET_PATH.read_text(encoding="utf-8"))
    case_ids = payload["case_ids"]
    question_ids = set()
    for cid in case_ids:
        # case_id shape: contract_N__[redacted_|oos]qNN or contract_N__oosNN
        suffix = cid.split("__", 1)[1]
        suffix = suffix.removeprefix("redacted_")
        question_ids.add(suffix)
    for i in range(1, 13):
        qid = f"q{i:02d}"
        assert qid in question_ids, f"question {qid} missing from judged subset"


def test_at_least_one_redacted_and_one_out_of_scope_present():
    payload = json.loads(JUDGED_SUBSET_PATH.read_text(encoding="utf-8"))
    case_ids = payload["case_ids"]
    assert any("__redacted_" in cid for cid in case_ids)
    assert any("__oos" in cid for cid in case_ids)


def test_every_case_id_in_test_subset_v1():
    from dealpoint.eval.subset import load_subset_case_ids

    payload = json.loads(JUDGED_SUBSET_PATH.read_text(encoding="utf-8"))
    universe = set(load_subset_case_ids("test_subset_v1", tranche=1))
    for cid in payload["case_ids"]:
        assert cid in universe


def test_seed_and_rule_recorded():
    payload = json.loads(JUDGED_SUBSET_PATH.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert isinstance(payload["rule"], str)
    assert len(payload["rule"]) > 50


def test_subset_hash_matches_recomputation():
    import hashlib

    payload = json.loads(JUDGED_SUBSET_PATH.read_text(encoding="utf-8"))
    recomputed = hashlib.sha256(
        json.dumps(payload["case_ids"], sort_keys=False).encode("utf-8")
    ).hexdigest()[:12]
    assert recomputed == payload["subset_hash"]


def test_ranking_is_a_permutation_with_no_duplicates():
    payload = json.loads(JUDGED_SUBSET_PATH.read_text(encoding="utf-8"))
    rank = payload["rank"]
    values = sorted(rank.values())
    assert values == list(range(len(rank)))
    assert set(rank.keys()) == set(payload["case_ids"])


def test_three_variants_recorded():
    payload = json.loads(JUDGED_SUBSET_PATH.read_text(encoding="utf-8"))
    variants = payload["variants"]
    assert len(variants) == 3
    variant_ids = {v["variant_id"] for v in variants}
    assert variant_ids == {"A\u0040haiku", "D\u0040haiku", "D\u0040glm"}
