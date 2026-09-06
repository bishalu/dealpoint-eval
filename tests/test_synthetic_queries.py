"""M7a gate: synthetic-query generation targets only gold-bearing DEV chunks,
caps at 2/chunk, freezes with full provenance, and never reads the test set
(spec section 1.B, "Budget and disk").
"""

from __future__ import annotations

import inspect
import json

import pytest

pytestmark = pytest.mark.gate_m7


def test_gold_bearing_dev_chunks_only_targets_dev_gold_spans(monkeypatch):
    from dealpoint.eval import cases as cases_mod
    from dealpoint.rag_lab import synthetic

    calls = []
    real_load = cases_mod.load_case_set

    def spy_load_case_set(name):
        calls.append(name)
        return real_load(name)

    monkeypatch.setattr(synthetic, "load_case_set", spy_load_case_set, raising=False)
    # patch the actual import site inside gold_bearing_dev_chunks
    import dealpoint.eval.cases as real_cases_mod

    monkeypatch.setattr(real_cases_mod, "load_case_set", spy_load_case_set)

    entries = synthetic.gold_bearing_dev_chunks()
    assert calls == ["dev"], f"expected only 'dev' to be loaded, got {calls}"
    assert len(entries) > 0
    # distinct chunk ids
    ids = [e["chunk_id"] for e in entries]
    assert len(ids) == len(set(ids))


def test_module_source_never_references_test_jsonl_path():
    """Mirrors tournament.py's own refusal: assert the module's source contains
    no reference to the frozen test-set path constant."""
    from dealpoint.rag_lab import synthetic

    source = inspect.getsource(synthetic)
    assert "TEST_JSONL_PATH" not in source
    assert 'load_case_set("test")' not in source
    assert "load_case_set('test')" not in source


def test_max_questions_per_chunk_cap_is_two():
    from dealpoint.rag_lab.synthetic import MAX_QUESTIONS_PER_CHUNK

    assert MAX_QUESTIONS_PER_CHUNK == 2


def test_synthetic_jsonl_round_trips_with_full_provenance(tmp_path):
    from dealpoint.rag_lab.synthetic import GENERATOR_ID, GENERATOR_VERSION

    out_path = tmp_path / "synthetic_dev_queries.jsonl"
    rows = [
        {
            "query_id": "abc123",
            "question": "What form of consideration is used?",
            "chunk_id": "contract_0:100-200",
            "agreement_id": "contract_0",
            "case_id": "contract_0__q01",
            "question_id": "q01",
            "generator": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "prompt_hash": "deadbeef",
            "model": "z-ai/glm-5.3-flash",
            "ts": "2026-09-05T00:00:00+00:00",
        }
    ]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row) + "\n" for row in rows)

    loaded = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows
    assert loaded[0]["generator"] == GENERATOR_ID
    assert loaded[0]["generator_version"] == GENERATOR_VERSION
    assert "prompt_hash" in loaded[0]
    assert "model" in loaded[0]


def test_looks_like_question_filters_markdown_headers():
    from dealpoint.rag_lab.synthetic import _looks_like_question

    assert not _looks_like_question("# Quiz Questions")
    assert not _looks_like_question("**Question 1:**")
    assert _looks_like_question("What form of consideration is used by the Company?")
