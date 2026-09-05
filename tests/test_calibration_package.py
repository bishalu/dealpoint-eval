"""Calibration package: identical rubric, schema, validation, idempotency (spec §7)."""

from __future__ import annotations

import json

import pytest

from dealpoint.eval.calibration import (
    FORM_MD_PATH,
    HUMAN_SCORES_SCHEMA,
    HUMAN_SCORES_SCHEMA_PATH,
    PACKETS_JSONL_PATH,
    VARIANT_KEY_PATH,
    validate_human_scores,
)
from dealpoint.eval.rubric import rubric_text

pytestmark = pytest.mark.gate_m5


def _dataset_and_index_available() -> bool:
    from dealpoint.config import CONTRACTS_DIR

    return CONTRACTS_DIR.exists()


def test_form_md_embeds_rubric_byte_identically():
    if not FORM_MD_PATH.exists():
        pytest.skip("calibration package not built yet; run `just calibration`")
    form_text = FORM_MD_PATH.read_text(encoding="utf-8")
    assert rubric_text() in form_text


def test_human_scores_schema_file_matches_documented_keys():
    if not HUMAN_SCORES_SCHEMA_PATH.exists():
        pytest.skip("calibration package not built yet; run `just calibration`")
    payload = json.loads(HUMAN_SCORES_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert set(payload.keys()) == set(HUMAN_SCORES_SCHEMA.keys())


def test_validate_human_scores_accepts_good_row():
    known = {"abc123"}
    rows = [
        {
            "packet_id": "abc123",
            "scorer": "jane",
            "scored_at": "2026-09-05T00:00:00+00:00",
            "reasoning": 3,
            "evidence": 4,
            "trajectory": 3,
            "professional": 4,
            "notes": "ok",
        }
    ]
    valid, errors = validate_human_scores(rows, known)
    assert len(valid) == 1
    assert errors == []


def test_validate_human_scores_rejects_unknown_packet_id():
    rows = [
        {
            "packet_id": "unknown",
            "scorer": "jane",
            "scored_at": "t",
            "reasoning": 3,
            "evidence": 3,
            "trajectory": 3,
            "professional": 3,
        }
    ]
    valid, errors = validate_human_scores(rows, {"known"})
    assert valid == []
    assert len(errors) == 1


def test_validate_human_scores_rejects_out_of_range_and_non_integer():
    known = {"abc123"}
    rows = [
        {
            "packet_id": "abc123",
            "scorer": "jane",
            "scored_at": "t",
            "reasoning": 6,
            "evidence": 3,
            "trajectory": 3,
            "professional": 3,
        },
        {
            "packet_id": "abc123",
            "scorer": "jane",
            "scored_at": "t",
            "reasoning": "3",
            "evidence": 3,
            "trajectory": 3,
            "professional": 3,
        },
    ]
    valid, errors = validate_human_scores(rows, known)
    assert valid == []
    assert len(errors) == 2


def test_packets_jsonl_line_count_matches_judged_traces():
    if not PACKETS_JSONL_PATH.exists():
        pytest.skip("calibration package not built yet; run `just calibration`")
    with open(PACKETS_JSONL_PATH, encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]
    from dealpoint.eval.subset import load_subset_payload

    judged_subset = load_subset_payload("judged_subset")
    assert len(lines) == judged_subset["n_traces"]


def test_variant_key_covers_every_packet_and_is_not_in_packets_md():
    from dealpoint.eval.calibration import PACKETS_MD_PATH
    from dealpoint.eval.subset import load_subset_payload

    if not VARIANT_KEY_PATH.exists() or not PACKETS_MD_PATH.exists():
        pytest.skip("calibration package not built yet; run `just calibration`")
    variant_key = json.loads(VARIANT_KEY_PATH.read_text(encoding="utf-8"))
    packets_md = PACKETS_MD_PATH.read_text(encoding="utf-8")
    judged_subset = load_subset_payload("judged_subset")

    assert len(variant_key) == judged_subset["n_traces"]

    # A bare arm letter ('A'..'D') is not distinctive enough to search for in
    # prose text, so the identity check confirms no MODEL id string (which
    # IS distinctive) appears anywhere in packets.md.
    for entry in variant_key.values():
        assert entry["model"] not in packets_md


def test_just_calibration_run_twice_is_byte_identical(tmp_path, dataset_available, monkeypatch):
    if not dataset_available:
        pytest.skip("dataset not present")
    from dealpoint.config import JUDGED_SUBSET_PATH

    if not JUDGED_SUBSET_PATH.exists():
        pytest.skip("judged_subset.json not built yet")

    import dealpoint.config as cfg
    import dealpoint.eval.calibration as calib_mod

    out_dir = tmp_path / "calibration"
    monkeypatch.setattr(cfg, "CALIBRATION_DIR", out_dir)
    monkeypatch.setattr(calib_mod, "CALIBRATION_DIR", out_dir)
    monkeypatch.setattr(calib_mod, "PACKETS_JSONL_PATH", out_dir / "packets.jsonl")
    monkeypatch.setattr(calib_mod, "PACKETS_MD_PATH", out_dir / "packets.md")
    monkeypatch.setattr(calib_mod, "FORM_MD_PATH", out_dir / "form.md")
    monkeypatch.setattr(calib_mod, "HUMAN_SCORES_PATH", out_dir / "human_scores.jsonl")
    monkeypatch.setattr(calib_mod, "HUMAN_SCORES_SCHEMA_PATH", out_dir / "human_scores.schema.json")
    monkeypatch.setattr(calib_mod, "VARIANT_KEY_PATH", out_dir / "variant_key.json")

    json_out1 = tmp_path / "judges1.json"
    json_out2 = tmp_path / "judges2.json"
    monkeypatch.setattr(cfg, "JUDGES_JSON_PATH", json_out1)
    monkeypatch.setattr(calib_mod, "JUDGES_JSON_PATH", json_out1)

    calib_mod.main([])
    first = (out_dir / "packets.jsonl").read_bytes()
    first_form = (out_dir / "form.md").read_bytes()

    monkeypatch.setattr(cfg, "JUDGES_JSON_PATH", json_out2)
    monkeypatch.setattr(calib_mod, "JUDGES_JSON_PATH", json_out2)
    calib_mod.main([])
    second = (out_dir / "packets.jsonl").read_bytes()
    second_form = (out_dir / "form.md").read_bytes()

    assert first == second
    assert first_form == second_form
