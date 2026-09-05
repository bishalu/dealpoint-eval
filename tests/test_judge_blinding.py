"""Blinding strips identities (spec DoD check 1 / gate_m5)."""

from __future__ import annotations

import json

import pytest

from dealpoint.corpus.document import Document
from dealpoint.data.sections import Section
from dealpoint.eval.blinding import (
    NO_GOLD_SPAN_NOTE,
    PACKET_ALLOWLIST,
    build_packet,
    render_packet_text,
    scrub_text,
)

pytestmark = pytest.mark.gate_m5

FORBIDDEN_SUBSTRINGS = (
    "anthropic",
    "claude",
    "haiku",
    "z-ai/glm-5.3-flash",
    '"arm"',
    '"model"',
)


def _doc(text: str | None = None, sections: list[Section] | None = None) -> Document:
    text = text if text is not None else "Section 1. Knowledge means actual knowledge. " * 5
    sections = sections or [Section(ref="1", title="Definitions", start=0, end=len(text))]
    return Document(
        document_id="synthetic",
        agreement_id="synthetic",
        text=text,
        sections=sections,
        body_start=0,
        body_end=len(text),
    )


def _row(**overrides) -> dict:
    base = {
        "arm": "D",
        "model": "anthropic/claude-haiku-4.5",
        "case_id": "contract_1__q01",
        "question_id": "q01",
        "git_sha7": "abcdef1",
        "index_version": "idxver",
        "chunk_version": "chunkver",
        "finding": {
            "answer": "All Cash",
            "evidence": [{"section_ref": "1", "quote": "Knowledge means actual knowledge."}],
            "rationale": "The section says so.",
        },
        "record": {
            "status": "ANSWERED",
            "failure_detail": None,
            "raw_final_text": None,
            "trajectory": [
                {
                    "tool": "search_agreement",
                    "args": {"query": "knowledge"},
                    "chunk_ids": ["synthetic:0-20"],
                    "char_ranges": [[0, 20]],
                }
            ],
        },
        "scores": {"grounded_accuracy": True, "skill_adherence": 0.5},
        "skill_rules": {"rules": {"1": "satisfied"}},
    }
    base.update(overrides)
    return base


def _case(**overrides) -> dict:
    base = {
        "case_id": "contract_1__q01",
        "question_id": "q01",
        "gold_spans": [{"start": 0, "end": 20}],
        "gold_answer": "All Cash",
        "category": "General Information",
        "reasoning_type": "direct",
        "required_evidence": None,
    }
    base.update(overrides)
    return base


def test_packet_top_level_keys_match_allowlist_exactly():
    doc = _doc()
    packet = build_packet(_row(), _case(), doc, variant_id="D\u0040haiku")
    assert set(packet.keys()) == set(PACKET_ALLOWLIST)


def test_packet_json_never_contains_arm_or_model_identity():
    doc = _doc()
    packet = build_packet(_row(), _case(), doc, variant_id="D\u0040haiku")
    text = json.dumps(packet)
    for forbidden in ("anthropic/claude-haiku-4.5", "z-ai/glm-5.3-flash"):
        assert forbidden not in text
    # explicit arm/model keys never present at all
    assert "arm" not in packet
    assert "model" not in packet


def test_hostile_rationale_is_scrubbed():
    row = _row(
        finding={
            "answer": "All Cash",
            "evidence": [],
            "rationale": "I am Claude Haiku running arm D for Anthropic",
        }
    )
    doc = _doc()
    packet = build_packet(row, _case(), doc, variant_id="D\u0040haiku")
    rationale = packet["finding"]["rationale"]
    assert "Claude" not in rationale
    assert "Haiku" not in rationale
    assert "Anthropic" not in rationale
    assert "[REDACTED-MODEL]" in rationale
    assert "[REDACTED-ARM]" in rationale


def test_scrub_text_direct():
    assert scrub_text("arm D") == "[REDACTED-ARM]"
    # the vendor list (spec §4.3) includes every candidate/judge family name,
    # since any of them could appear in a model's own free-text output
    assert scrub_text("Mistral") == "[REDACTED-MODEL]"
    assert scrub_text("GLM-5.3") == "[REDACTED-MODEL]-5.3"


def test_no_circularity_fields_in_packet():
    doc = _doc()
    packet = build_packet(_row(), _case(), doc, variant_id="D\u0040haiku")
    text = json.dumps(packet)
    for forbidden in ("grounded_accuracy", "skill_adherence", "skill_rules", '"scores"'):
        assert forbidden not in text


def test_no_run_identity_or_harness_internals_in_packet():
    doc = _doc()
    packet = build_packet(_row(), _case(), doc, variant_id="D\u0040haiku")
    text = json.dumps(packet)
    for forbidden in (
        "case_id",
        "agreement_id",
        "question_id",
        "gold_answer",
        "git_sha7",
        "index_version",
        "chunk_version",
        "failure_detail",
        "raw_final_text",
    ):
        assert forbidden not in text


def test_retrieved_text_truncated_to_600_chars_even_for_a_57k_range():
    big_text = "x" * 60000
    sections = [Section(ref="99", title="", start=0, end=len(big_text))]
    doc = _doc(text=big_text, sections=sections)
    row = _row(
        record={
            "status": "ANSWERED",
            "failure_detail": None,
            "raw_final_text": None,
            "trajectory": [
                {
                    "tool": "get_section",
                    # section_ref does not match any real section -> falls back to char_ranges
                    "args": {"section_ref": "does-not-exist"},
                    "chunk_ids": [],
                    "char_ranges": [[0, 57000]],
                }
            ],
        }
    )
    packet = build_packet(row, _case(), doc, variant_id="D\u0040haiku")
    step = packet["trajectory"][0]
    assert step["retrieved_text"] is not None
    for item in step["retrieved_text"]:
        assert len(item) <= 600


def test_lookup_defined_term_with_no_match_yields_null_not_crash():
    doc = _doc()
    row = _row(
        record={
            "status": "ANSWERED",
            "failure_detail": None,
            "raw_final_text": None,
            "trajectory": [
                {
                    "tool": "lookup_defined_term",
                    "args": {"term": "Nonexistent Term Xyz"},
                    "chunk_ids": [],
                    "char_ranges": [],
                }
            ],
        }
    )
    packet = build_packet(row, _case(), doc, variant_id="D\u0040haiku")
    step = packet["trajectory"][0]
    assert step["retrieved_text"] is None


def test_counterfactual_case_gets_null_gold_span_and_neutral_note():
    doc = _doc()
    case = _case(gold_spans=[])
    packet = build_packet(_row(), case, doc, variant_id="D\u0040haiku")
    assert packet["gold_span"] is None
    assert packet["gold_span_note"] == NO_GOLD_SPAN_NOTE
    note_lower = packet["gold_span_note"].lower()
    for banned in ("absent", "redacted", "abstain"):
        assert banned not in note_lower


def test_no_finding_row_renders_as_null_finding_and_status_kept():
    doc = _doc()
    row = _row(finding=None, record={"status": "CAP_HIT", "failure_detail": None, "raw_final_text": None, "trajectory": []})
    packet = build_packet(row, _case(), doc, variant_id="D\u0040haiku")
    assert packet["finding"] is None
    assert packet["status"] == "CAP_HIT"
    text = render_packet_text(packet)
    assert "no finding" in text.lower()


def test_render_packet_text_is_the_single_rendering_function():
    doc = _doc()
    packet = build_packet(_row(), _case(), doc, variant_id="D\u0040haiku")
    text1 = render_packet_text(packet)
    text2 = render_packet_text(packet)
    assert text1 == text2
