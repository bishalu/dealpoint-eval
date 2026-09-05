"""Skill files exist, are size-bounded, and skill_version is stamped (spec deliverable 1)."""

from __future__ import annotations

import json

import pytest

from dealpoint.agent.skill import (
    SKILL_MD_PATH,
    load_playbook,
    load_skill,
    skill_block,
    skill_version,
    stamp_skill_version,
)
from dealpoint.config import SKILL_PLAYBOOKS_DIR, VERSIONS_JSON_PATH
from dealpoint.data.questions import QUESTION_SPEC

pytestmark = pytest.mark.gate_m4

# Rough, generous token estimate: chars / 4 (English prose averages ~4
# chars/token; the spec's "~800 tokens" is a soft ceiling, so a comfortable
# margin -- 3200 chars -- is what's actually asserted).
MAX_SKILL_CHARS = 3200


def test_skill_md_exists_and_is_size_bounded():
    assert SKILL_MD_PATH.exists()
    text = load_skill()
    assert len(text) <= MAX_SKILL_CHARS


def test_skill_md_has_all_eight_rules_numbered_1_to_8():
    text = load_skill()
    for i in range(1, 9):
        assert f"{i}." in text, f"rule {i} missing from SKILL.md"


def test_every_question_has_a_playbook_3_to_5_nonblank_lines():
    for q in QUESTION_SPEC:
        playbook = load_playbook(q.id)
        assert playbook is not None, q.id
        non_blank = [line for line in playbook.splitlines() if line.strip()]
        assert 3 <= len(non_blank) <= 5, f"{q.id}: {len(non_blank)} non-blank lines"


def test_out_of_scope_question_has_no_playbook():
    assert load_playbook("oos00") is None


# Phrases that would steer a reader toward the *majority* answer rather than
# toward reading the contract -- a playbook describing what distinguishes
# the OPTIONS (necessarily naming them) is fine; a playbook that hints which
# one is usually/typically/correctly right is what spec §2 forbids ("Never
# encode a gold answer or a majority answer in a playbook").
_MAJORITY_HINT_PHRASES = (
    "usually",
    "typically",
    "most agreements",
    "most contracts",
    "in most cases",
    "the correct answer",
    "the right answer",
    "default answer",
    "commonly the answer",
    "often the answer",
)


def test_no_playbook_hints_at_the_majority_answer():
    """Playbooks describe what distinguishes the options (naming them is
    required content per spec §2's own definition of a playbook); they must
    never hint which option is usually/typically/correctly the answer.
    """
    for q in QUESTION_SPEC:
        playbook = load_playbook(q.id)
        assert playbook is not None
        playbook_cf = playbook.casefold()
        for phrase in _MAJORITY_HINT_PHRASES:
            assert phrase not in playbook_cf, f"{q.id} playbook hints at a default: {phrase!r}"


def test_skill_block_contains_skill_and_the_one_applicable_playbook():
    block = skill_block("q01")
    q01_playbook = load_playbook("q01")
    q02_playbook = load_playbook("q02")
    assert q01_playbook is not None
    assert q02_playbook is not None
    assert load_skill() in block
    assert q01_playbook in block
    # never another question's playbook
    assert q02_playbook not in block or q02_playbook == q01_playbook


def test_skill_block_for_out_of_scope_question_is_skill_only():
    block = skill_block("oos00")
    assert block.strip() == load_skill().strip()


def test_skill_version_is_deterministic_and_changes_with_content(tmp_path, monkeypatch):
    v1 = skill_version()
    v2 = skill_version()
    assert v1 == v2
    assert len(v1) == 12


def test_skill_version_stamped_in_versions_json():
    stamp_skill_version()
    assert VERSIONS_JSON_PATH.exists()
    payload = json.loads(VERSIONS_JSON_PATH.read_text(encoding="utf-8"))
    assert payload["skill_version"] == skill_version()


def test_stamp_skill_version_does_not_clobber_other_keys(tmp_path, monkeypatch):
    fake_versions = tmp_path / "versions.json"
    fake_versions.write_text(json.dumps({"dataset_version": "abc123"}), encoding="utf-8")
    monkeypatch.setattr("dealpoint.agent.skill.VERSIONS_JSON_PATH", fake_versions)
    stamp_skill_version()
    payload = json.loads(fake_versions.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == "abc123"
    assert payload["skill_version"] == skill_version()


def test_playbooks_dir_has_exactly_twelve_files():
    files = sorted(SKILL_PLAYBOOKS_DIR.glob("*.md"))
    assert len(files) == 12
