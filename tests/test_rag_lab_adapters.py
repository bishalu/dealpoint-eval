"""M7a gate: LlamaIndex adapters -- gold-bearing chunk ids, node mapping,
adapter ordering (spec section 3.1/3.2). Mapping tests run unconditionally;
only the true LlamaIndex-object tests are guarded by importorskip.
"""

from __future__ import annotations

import pytest

from dealpoint.corpus.chunks import Chunk

pytestmark = pytest.mark.gate_m7


def _case(gold_spans):
    return {"case_id": "x__q01", "gold_spans": gold_spans}


def _chunk(cid, start, end, text="x"):
    return Chunk(agreement_id="doc_x", chunk_id=cid, section_ref="1", start=start, end=end, text=text)


def test_gold_bearing_chunk_ids_agrees_with_tournament_first_hit_rank():
    from dealpoint.eval.tournament import _first_hit_rank
    from dealpoint.rag_lab.evaluate import gold_bearing_chunk_ids

    # exactly 50 chars overlap -> hit
    chunk_hit = _chunk("doc_x:0-50", 0, 50)
    case = _case([{"start": 0, "end": 50}])
    ids = gold_bearing_chunk_ids(case, [chunk_hit])
    assert ids == ["doc_x:0-50"]
    assert _first_hit_rank([chunk_hit], [(0, 50)]) == 1

    # 49 chars overlap -> not a hit
    chunk_near = _chunk("doc_x:0-49", 0, 49)
    case_49 = _case([{"start": 0, "end": 50}])
    ids_49 = gold_bearing_chunk_ids(case_49, [chunk_near])
    assert ids_49 == []
    assert _first_hit_rank([chunk_near], [(0, 50)]) is None


def test_gold_bearing_chunk_ids_empty_when_no_gold_spans():
    from dealpoint.rag_lab.evaluate import gold_bearing_chunk_ids

    case = _case([])
    chunks = [_chunk("doc_x:0-50", 0, 50)]
    assert gold_bearing_chunk_ids(case, chunks) == []


def test_chunk_to_node_preserves_id_and_metadata():
    li_core = pytest.importorskip("llama_index.core")  # noqa: F841
    from dealpoint.rag_lab.adapters import chunk_to_node

    chunk = _chunk("doc_x:10-60", 10, 60, text="hello world")
    node = chunk_to_node(chunk)
    assert node.id_ == "doc_x:10-60"
    assert node.text == "hello world"
    assert node.metadata["start"] == 10
    assert node.metadata["end"] == 60
    assert node.metadata["section_ref"] == "1"
    assert node.metadata["agreement_id"] == "doc_x"


class StubRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, agreement_id, query, k=5):
        return self._chunks[:k]


def test_adapter_returns_nodes_in_order_with_nonincreasing_scores():
    pytest.importorskip("llama_index.core")
    from dealpoint.rag_lab.adapters import ProjectRetrieverAdapter

    chunks = [_chunk(f"doc_x:{i}-{i+5}", i, i + 5) for i in range(0, 25, 5)]
    adapter = ProjectRetrieverAdapter(StubRetriever(chunks), "doc_x", 5)
    results = adapter.retrieve("some query")

    assert [r.node.id_ for r in results] == [c.chunk_id for c in chunks]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0
