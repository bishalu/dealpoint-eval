"""M7a gate: ranking-order agreement, disagreement classification, version
assertion, and pure report rendering (spec section 3.2/3.4).
"""

from __future__ import annotations

import json

import pytest

from dealpoint.corpus.chunks import Chunk
from dealpoint.corpus.retrievers import RetrieverConfig
from dealpoint.eval.agreement import spearman

pytestmark = pytest.mark.gate_m7


def test_ranking_order_agreement_reuses_agreement_spearman():
    from dealpoint.rag_lab.report import build_report

    xs = [0.5, 0.6, 0.9, 0.9]
    ys = [0.4, 0.65, 0.85, 0.95]
    rho_direct = spearman(xs, ys)

    per_config = {
        "dense": {"li": {"hit_rate": 0.5, "mrr": 0.4, "n": 10}, "obj": {"hit_at_5": 0.4, "hit_at_10": 0.4, "mrr": 0.4, "n": 10}, "disagreements": []},
        "bm25": {"li": {"hit_rate": 0.6, "mrr": 0.5, "n": 10}, "obj": {"hit_at_5": 0.65, "hit_at_10": 0.6, "mrr": 0.5, "n": 10}, "disagreements": []},
        "hybrid_rrf": {"li": {"hit_rate": 0.9, "mrr": 0.7, "n": 10}, "obj": {"hit_at_5": 0.85, "hit_at_10": 0.8, "mrr": 0.7, "n": 10}, "disagreements": []},
        "hybrid_rrf_rerank": {"li": {"hit_rate": 0.9, "mrr": 0.8, "n": 10}, "obj": {"hit_at_5": 0.95, "hit_at_10": 0.9, "mrr": 0.8, "n": 10}, "disagreements": []},
    }
    report = build_report(
        per_config,
        versions={"chunk_version": "abc", "index_version": "def"},
        n_cases=10,
        dataset_version="v1",
        framework_versions={},
    )
    got = report["ranking_agreement"]["hit_rate_vs_hit_at_5"]["spearman"]
    assert got == pytest.approx(rho_direct)


def _synthetic_case_and_doc():
    from typing import ClassVar

    from dealpoint.data.sections import Section

    class FakeDoc:
        text = "A" * 500
        sections: ClassVar = [Section(ref="1", title="", start=0, end=500)]

    doc = FakeDoc()
    chunks = [
        Chunk(agreement_id="doc_x", chunk_id="doc_x:0-100", section_ref="1", start=0, end=100, text="a" * 100),
        Chunk(agreement_id="doc_x", chunk_id="doc_x:100-200", section_ref="1", start=100, end=200, text="a" * 100),
    ]
    return doc, chunks


def test_classify_disagreement_reranking():
    from dealpoint.rag_lab.evaluate import classify_disagreement

    config = RetrieverConfig(name="x", kind="hybrid_rrf", rerank_model="fake-model")
    _doc, chunks = _synthetic_case_and_doc()
    cause = classify_disagreement(
        config=config,
        expected_ids=["doc_x:0-100"],
        retrieved_ids=["doc_x:0-100"],
        chunks=chunks,
        golds=[(0, 100)],
        rerank_would_hit=True,
        current_hit=False,
    )
    assert cause == "reranking"


def test_classify_disagreement_partial_overlap():
    from dealpoint.rag_lab.evaluate import classify_disagreement

    config = RetrieverConfig(name="x", kind="dense")
    _doc, chunks = _synthetic_case_and_doc()
    # gold span [95, 145): overlap with chunk doc_x:0-100 is [95,100) = 5 chars (< 50)
    cause = classify_disagreement(
        config=config,
        expected_ids=["doc_x:100-200"],
        retrieved_ids=["doc_x:0-100"],
        chunks=chunks,
        golds=[(95, 145)],
    )
    assert cause == "partial_overlap"


def test_classify_disagreement_duplicate_relevant_chunks():
    from dealpoint.rag_lab.evaluate import classify_disagreement

    config = RetrieverConfig(name="x", kind="dense")
    _doc, chunks = _synthetic_case_and_doc()
    cause = classify_disagreement(
        config=config,
        expected_ids=["doc_x:0-100", "doc_x:100-200"],
        retrieved_ids=["doc_x:100-200"],
        chunks=chunks,
        golds=[(0, 100), (100, 200)],
    )
    assert cause == "duplicate_relevant_chunks"


def test_classify_disagreement_section_boundary():
    from dealpoint.corpus.chunks import Chunk as C
    from dealpoint.rag_lab.evaluate import classify_disagreement

    config = RetrieverConfig(name="x", kind="dense")
    chunks = [
        C(agreement_id="doc_x", chunk_id="doc_x:0-60", section_ref="1", start=0, end=60, text="a" * 60),
        C(agreement_id="doc_x", chunk_id="doc_x:60-120", section_ref="2", start=60, end=120, text="a" * 60),
        C(agreement_id="doc_x", chunk_id="doc_x:200-260", section_ref="3", start=200, end=260, text="a" * 60),
    ]
    # single gold span straddles both section refs 1 and 2 (20 chars overlap
    # each -- below MIN_GOLD_OVERLAP_CHARS, so it is not itself an expected_id
    # source); the RETRIEVED top chunk has zero overlap so partial_overlap
    # never fires, and expected_ids has only one entry so duplicate_relevant
    # never fires either -- isolating the section_boundary rule.
    cause = classify_disagreement(
        config=config,
        expected_ids=["doc_x:0-60"],
        retrieved_ids=["doc_x:200-260"],
        chunks=chunks,
        golds=[(40, 80)],
    )
    assert cause == "section_boundary"


def test_classify_disagreement_chunk_identity_default():
    from dealpoint.rag_lab.evaluate import classify_disagreement

    config = RetrieverConfig(name="x", kind="dense")
    _doc, chunks = _synthetic_case_and_doc()
    cause = classify_disagreement(
        config=config,
        expected_ids=["doc_x:0-100"],
        retrieved_ids=["doc_x:100-200"],
        chunks=chunks,
        golds=[(0, 100)],
    )
    assert cause == "chunk_identity"


def test_assert_versions_match_raises_on_mismatch(tmp_path, monkeypatch):
    from dealpoint.rag_lab.evaluate import assert_versions_match

    bad_versions = tmp_path / "versions.json"
    bad_versions.write_text(
        json.dumps({"chunk_version": "WRONG", "index_version": "WRONG"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError):
        assert_versions_match(versions_path=bad_versions)
