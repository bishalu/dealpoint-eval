"""M4.1: frozen-artefact hash assertions (spec §3.F). Pinned digests were
computed from the tree at the start of this build, before any change; if a
digest moves later, the benchmark itself changed and must be reverted.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from dealpoint.agent.prompts import question_spec_block, system_prompt
from dealpoint.agent.skill import skill_block, skill_version
from dealpoint.config import ARM_ORDER, ARMS, MAX_TOOL_CALLS, TEST_SUBSET_V1_PATH
from dealpoint.data.questions import QUESTION_SPEC
from dealpoint.eval.scorers import SCORE_FIELD_NAMES

pytestmark = pytest.mark.gate_m4

SUBSET_HASH = "be2e96433b14"
SUBSET_SHA256 = "37b9e61a55f96c00f23b192f826c86241151581b2383b5563665af06255bd19c"
SKILL_VERSION = "f8d255cc169b"
SYSTEM_PROMPT_SHA256 = "3dc7ffe99e23f80b08e69db81e64038f48aef342e96019687afdea07fc757da7"
QUESTION_SPEC_BLOCK_SHA256 = "28d5ff68bac899b41fde95fc7d6425850165f5b0efc4a5960ab628991801a04c"
SKILL_BLOCK_SHA256 = "7329e34b7f2ff61860a54bebd6203be8a4a1f0a95296b6f69cb3012c7275eb7a"
ARMS_SHA256 = "4ec8596c81bee2dddd2a1b901e24af55a84d4da7c192cdfa4817c8e1f1a74f4e"

EXPECTED_SCORE_FIELD_NAMES = (
    "gold_seen",
    "gold_first_rank",
    "answer_correct",
    "citation_verbatim",
    "citation_gold_overlap",
    "fabrication",
    "grounded_accuracy",
    "abstain_correct",
    "redacted_fabrication",
    "required_evidence_met",
    "tool_calls",
    "cap_hit",
    "execution_failed",
    "input_tokens",
    "output_tokens",
    "usd",
    "wall_ms",
    "skill_adherence",
)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def test_frozen_subset_hash_and_sha256_unchanged():
    payload = json.loads(TEST_SUBSET_V1_PATH.read_text(encoding="utf-8"))
    assert payload["subset_hash"] == SUBSET_HASH
    assert hashlib.sha256(TEST_SUBSET_V1_PATH.read_bytes()).hexdigest() == SUBSET_SHA256


def test_skill_version_unchanged():
    assert skill_version() == SKILL_VERSION


def test_system_prompt_unchanged():
    assert _sha256(system_prompt()) == SYSTEM_PROMPT_SHA256


def test_question_spec_blocks_unchanged():
    concat = "".join(question_spec_block(q) for q in QUESTION_SPEC)
    assert _sha256(concat) == QUESTION_SPEC_BLOCK_SHA256


def test_skill_blocks_unchanged():
    concat = "".join(skill_block(q.id) for q in QUESTION_SPEC)
    assert _sha256(concat) == SKILL_BLOCK_SHA256


def test_arms_config_unchanged():
    assert _sha256(json.dumps(ARMS, sort_keys=True)) == ARMS_SHA256


def test_arm_order_unchanged():
    assert ARM_ORDER == ("A", "B", "C", "D")


def test_score_field_names_unchanged():
    assert SCORE_FIELD_NAMES == EXPECTED_SCORE_FIELD_NAMES


def test_max_tool_calls_unchanged():
    assert MAX_TOOL_CALLS == 8
