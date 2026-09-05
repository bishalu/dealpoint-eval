"""Arm configs differ from their predecessor by exactly one key (spec deliverable 3)."""

from __future__ import annotations

import pytest

from dealpoint.config import ARM_C_RETRIEVER, ARM_ORDER, ARMS, DENSE_RETRIEVER

pytestmark = pytest.mark.gate_m4

EXPECTED_DIFF_KEY = {
    ("A", "B"): "loop",
    ("B", "C"): "retriever",
    ("C", "D"): "skill",
}


def test_every_arm_has_exactly_three_keys():
    for arm_id, cfg in ARMS.items():
        assert set(cfg.keys()) == {"loop", "retriever", "skill"}, arm_id


def test_arm_order_is_a_b_c_d():
    assert ARM_ORDER == ("A", "B", "C", "D")


@pytest.mark.parametrize("pair", [("A", "B"), ("B", "C"), ("C", "D")])
def test_adjacent_arms_differ_by_exactly_one_key(pair):
    a_id, b_id = pair
    a, b = ARMS[a_id], ARMS[b_id]
    diffs = [k for k in a if a[k] != b[k]]
    assert len(diffs) == 1, f"{a_id}->{b_id} differs in {diffs}, expected exactly one key"
    assert diffs[0] == EXPECTED_DIFF_KEY[pair]


def test_arm_a_and_b_share_dense_retriever():
    assert ARMS["A"]["retriever"] == DENSE_RETRIEVER
    assert ARMS["B"]["retriever"] == DENSE_RETRIEVER


def test_arm_c_and_d_share_frozen_arm_c_retriever():
    assert ARMS["C"]["retriever"] == ARM_C_RETRIEVER
    assert ARMS["D"]["retriever"] == ARM_C_RETRIEVER


def test_only_arm_d_has_skill():
    assert ARMS["A"]["skill"] is False
    assert ARMS["B"]["skill"] is False
    assert ARMS["C"]["skill"] is False
    assert ARMS["D"]["skill"] is True


def test_only_arm_a_uses_the_pipeline_loop():
    assert ARMS["A"]["loop"] == "pipeline"
    for arm_id in ("B", "C", "D"):
        assert ARMS[arm_id]["loop"] == "agent"
