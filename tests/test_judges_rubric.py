"""Rubric frozen first, hashed, stamped before any judge call (spec DoD check 2)."""

from __future__ import annotations

import json

import pytest

from dealpoint.config import RUBRICS_PATH, VERSIONS_JSON_PATH
from dealpoint.eval.rubric import rubric_text, rubric_version, stamp_rubric_version

pytestmark = pytest.mark.gate_m5

DIMENSIONS = ("reasoning", "evidence", "trajectory", "professional")


def test_rubrics_md_exists_with_four_dimensions_and_five_levels_each():
    text = rubric_text()
    for dim in DIMENSIONS:
        assert f"## Dimension: {dim}" in text
    for i in range(1, 6):
        assert text.count(f"\n{i}. ") >= 4  # at least one per dimension


def test_rubric_version_is_12_hex_and_stable():
    v1 = rubric_version()
    v2 = rubric_version()
    assert v1 == v2
    assert len(v1) == 12
    int(v1, 16)  # hex


def test_stamp_rubric_version_preserves_other_keys(tmp_path, monkeypatch):
    fake_versions = tmp_path / "versions.json"
    fake_versions.write_text(json.dumps({"dataset_version": "abc123"}), encoding="utf-8")
    monkeypatch.setattr("dealpoint.eval.rubric.VERSIONS_JSON_PATH", fake_versions)
    stamp_rubric_version()
    payload = json.loads(fake_versions.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == "abc123"
    assert payload["rubric_version"] == rubric_version()


def test_committed_versions_json_carries_matching_rubric_version():
    if not VERSIONS_JSON_PATH.exists():
        pytest.skip("versions.json not built yet")
    payload = json.loads(VERSIONS_JSON_PATH.read_text(encoding="utf-8"))
    if "rubric_version" not in payload:
        pytest.skip("rubric not stamped yet")
    assert payload["rubric_version"] == rubric_version()


def test_every_row_of_judge_scores_carries_the_frozen_rubric_version():
    from dealpoint.config import JUDGE_SCORES_PATH

    if not JUDGE_SCORES_PATH.exists():
        pytest.skip("no judge scores written yet")
    with open(JUDGE_SCORES_PATH, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        pytest.skip("judge_scores.jsonl is empty")
    for row in rows:
        assert row["rubric_version"] == rubric_version()


def test_rubric_file_mentions_no_vendor_model_or_arm_name():
    text = rubric_text().lower()
    for banned in (
        "anthropic", "claude", "haiku", "openai", "gpt", "google", "gemini",
        "deepseek", "qwen", "mistral", "nvidia", "bytedance", "z-ai", "glm",
        "arm a", "arm b", "arm c", "arm d",
    ):
        assert banned not in text


def test_rubrics_path_points_to_repo_root_eval_not_dealpoint_eval():
    assert "eval/judges/rubrics.md" in str(RUBRICS_PATH).replace("\\", "/")
    assert "dealpoint/eval" not in str(RUBRICS_PATH).replace("\\", "/")
