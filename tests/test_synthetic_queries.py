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


def test_frozen_synthetic_dev_queries_file_matches_provenance():
    import hashlib

    from dealpoint.config import LI_RAG_EVAL_JSON_PATH, SYNTHETIC_DEV_QUERIES_PATH
    from dealpoint.eval.cases import load_case_set

    if not SYNTHETIC_DEV_QUERIES_PATH.exists():
        pytest.skip("synthetic_dev_queries.jsonl not frozen yet")

    rows = [
        json.loads(line)
        for line in SYNTHETIC_DEV_QUERIES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "frozen synthetic query file is empty"

    dev_chunk_ids: set[str] = set()
    for case in load_case_set("dev"):
        from dealpoint.corpus.chunks import chunk_document
        from dealpoint.corpus.document import load_document
        from dealpoint.eval.cases import resolve_document_id

        doc_id = resolve_document_id(case)
        dev_chunk_ids.update(c.chunk_id for c in chunk_document(load_document(doc_id)))

    counts: dict[str, int] = {}
    for row in rows:
        assert row["chunk_id"] in dev_chunk_ids, f"{row['chunk_id']} is not a dev chunk"
        for key in ("generator", "generator_version", "prompt_hash", "model"):
            assert row.get(key), f"row missing {key}"
        counts[row["chunk_id"]] = counts.get(row["chunk_id"], 0) + 1
    assert all(n <= 2 for n in counts.values()), "more than 2 questions for some chunk"

    file_hash = hashlib.sha256(SYNTHETIC_DEV_QUERIES_PATH.read_bytes()).hexdigest()
    if LI_RAG_EVAL_JSON_PATH.exists():
        report = json.loads(LI_RAG_EVAL_JSON_PATH.read_text(encoding="utf-8"))
        synth = report.get("synthetic") or {}
        if synth.get("file_sha256"):
            assert synth["file_sha256"] == file_hash


def test_li_rag_eval_covers_all_frozen_configs_plus_native():
    from dealpoint.config import LI_RAG_EVAL_JSON_PATH

    if not LI_RAG_EVAL_JSON_PATH.exists():
        pytest.skip("li_rag_eval.json not generated yet; run `just li-rag-eval`")
    report = json.loads(LI_RAG_EVAL_JSON_PATH.read_text(encoding="utf-8"))
    retrievers = report["retrievers"]
    expected = {
        "dense",
        "bm25",
        "hybrid_rrf",
        "hybrid_rrf_rerank",
        "multi_query_fusion",
        "multi_query_fusion_rerank",
        "li_native_bm25",
    }
    assert expected <= set(retrievers.keys())
