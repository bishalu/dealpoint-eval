"""M3 gate: retriever candidates conform to the Protocol; RRF/rerank ordering;
`data/eval/tournament_queries.json` (spec §5.1).

Fully offline: no test here constructs a real `TextCrossEncoder`, opens
`data/index/`, or hits the network. Dense retrieval uses an in-memory Qdrant
client with an injected fake embedder; BM25 uses an injected
`chunks_provider`; the reranker is always given a fake `scorer`.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from dealpoint.config import ARM_C_RETRIEVER, TOURNAMENT_QUERIES_PATH
from dealpoint.corpus.chunks import Chunk
from dealpoint.corpus.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRRFRetriever,
    MultiQueryFusionRetriever,
    RerankRetriever,
    arm_c_index_version,
    index_version,
    reciprocal_rank_fusion,
)
from dealpoint.data.questions import QUESTION_BY_ID

pytestmark = pytest.mark.gate_m3


def _fake_embed(text: str, dims: int = 8) -> list[float]:
    """Deterministic text -> normalised token-hash-count vector, no model."""
    vec = [0.0] * dims
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dims] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


class FakeEmbedder:
    def query_embed(self, texts):
        return [_fake_embed(t) for t in texts]

    def passage_embed(self, texts, batch_size=64):
        return [_fake_embed(t) for t in texts]


def _make_corpus() -> list[Chunk]:
    """~12 hand-made chunks over 2 fake document ids, distinctive vocabulary,
    long enough (>= MIN_GOLD_OVERLAP_CHARS=50 chars) for full-chunk gold spans."""
    doc_a_texts = [
        "apple banana orchard fruit harvest season begins early autumn",
        "cherry date fig grove seasonal picking crew arrives each year",
        "grape honeydew melon vineyard summer irrigation schedule set",
        "kiwi lemon lime citrus grove watered daily during dry months",
        "mango nectarine papaya tropical greenhouse climate controlled",
        "orange peach pear autumn harvest festival held every October",
    ]
    doc_b_texts = [
        "rocket satellite orbit launch pad crew final countdown begins",
        "telescope nebula galaxy observatory night sky survey program",
        "asteroid comet meteor shower visible from northern hemisphere",
        "spaceship engine thruster fuel tank pressure nominal readings",
        "astronaut spacesuit oxygen tank pressure checked before walk",
        "planet moon eclipse solar system alignment rare event tonight",
    ]
    chunks: list[Chunk] = []
    for doc_id, texts in (("doc_a", doc_a_texts), ("doc_b", doc_b_texts)):
        offset = 0
        for text in texts:
            start = offset
            end = start + len(text)
            chunks.append(
                Chunk(
                    agreement_id=doc_id,
                    chunk_id=f"{doc_id}:{start}-{end}",
                    section_ref="",
                    start=start,
                    end=end,
                    text=text,
                )
            )
            offset = end + 1
    return chunks


def _build_dense_retriever(chunks: list[Chunk]) -> DenseRetriever:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="test_m3",
        vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(
            id=i,
            vector=_fake_embed(c.text),
            payload={
                "agreement_id": c.agreement_id,
                "chunk_id": c.chunk_id,
                "section_ref": c.section_ref,
                "start": c.start,
                "end": c.end,
                "text": c.text,
            },
        )
        for i, c in enumerate(chunks)
    ]
    client.upsert(collection_name="test_m3", points=points)
    return DenseRetriever(client=client, collection="test_m3", embedder=FakeEmbedder())


def _chunks_provider(chunks: list[Chunk]):
    def provider(agreement_id: str) -> list[Chunk]:
        return [c for c in chunks if c.agreement_id == agreement_id]

    return provider


class RecordingRetriever:
    """A plain-list-backed stub that records every `search` call."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, str, int]] = []

    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]:
        self.calls.append((agreement_id, query, k))
        return [c for c in self._chunks if c.agreement_id == agreement_id][:k]


# --- Protocol conformance -----------------------------------------------


def _assert_protocol_conformance(retriever, doc_id: str, other_doc_id: str) -> None:
    result1 = retriever.search(doc_id, "harvest season", k=3)
    assert isinstance(result1, list)
    assert len(result1) <= 3
    assert all(isinstance(c, Chunk) for c in result1)
    assert all(c.agreement_id == doc_id for c in result1)

    result2 = retriever.search(doc_id, "harvest season", k=3)
    assert [c.chunk_id for c in result1] == [c.chunk_id for c in result2]

    other = retriever.search(other_doc_id, "orbit launch", k=3)
    assert all(c.agreement_id == other_doc_id for c in other)


def test_dense_retriever_protocol_conformance():
    chunks = _make_corpus()
    dense = _build_dense_retriever(chunks)
    _assert_protocol_conformance(dense, "doc_a", "doc_b")


def test_bm25_retriever_protocol_conformance():
    chunks = _make_corpus()
    bm25 = BM25Retriever(chunks_provider=_chunks_provider(chunks))
    _assert_protocol_conformance(bm25, "doc_a", "doc_b")


def test_hybrid_rrf_protocol_conformance():
    chunks = _make_corpus()
    dense = _build_dense_retriever(chunks)
    bm25 = BM25Retriever(chunks_provider=_chunks_provider(chunks))
    hybrid = HybridRRFRetriever(dense, bm25)
    _assert_protocol_conformance(hybrid, "doc_a", "doc_b")


def test_rerank_retriever_protocol_conformance():
    chunks = _make_corpus()
    dense = _build_dense_retriever(chunks)

    def fake_scorer(query: str, texts: list[str]) -> list[float]:
        return [float(len(t) % 11) for t in texts]

    rerank = RerankRetriever(dense, fetch_k=6, scorer=fake_scorer)
    _assert_protocol_conformance(rerank, "doc_a", "doc_b")


def test_multi_query_fusion_protocol_conformance():
    chunks = _make_corpus()
    dense = _build_dense_retriever(chunks)
    variants = {"harvest season": ["fruit picking", "autumn crop"]}
    multi = MultiQueryFusionRetriever(dense, variants)
    _assert_protocol_conformance(multi, "doc_a", "doc_b")


# --- RRF ordering ---------------------------------------------------------


def _mkchunk(chunk_id: str) -> Chunk:
    return Chunk(agreement_id="doc", chunk_id=chunk_id, section_ref="", start=0, end=1, text="x")


def test_rrf_ordering_hand_computed():
    c1, c2, c3 = _mkchunk("c1"), _mkchunk("c2"), _mkchunk("c3")
    list_a = [c1, c2, c3]
    list_b = [c3, c1]
    fused = reciprocal_rank_fusion([list_a, list_b], rrf_k=60)

    score_c1 = 1 / 61 + 1 / 62
    score_c2 = 1 / 62
    score_c3 = 1 / 63 + 1 / 61
    assert score_c1 > score_c3 > score_c2

    assert [c.chunk_id for c in fused] == ["c1", "c3", "c2"]


def test_rrf_tiebreak_by_chunk_id():
    chunk_z = _mkchunk("chunk_z")
    chunk_a = _mkchunk("chunk_a")
    # Both rank 1 in their own single-item list -> identical RRF score.
    fused = reciprocal_rank_fusion([[chunk_z], [chunk_a]], rrf_k=60)
    assert [c.chunk_id for c in fused] == ["chunk_a", "chunk_z"]


# --- Rerank ordering -------------------------------------------------------


def test_rerank_ordering_and_fetch_k_and_tiebreak():
    chunks = [_mkchunk(f"c{i}") for i in range(5)]
    base = RecordingRetriever(
        [
            Chunk(agreement_id="doc_a", chunk_id=c.chunk_id, section_ref="", start=i, end=i + 1, text="x")
            for i, c in enumerate(chunks)
        ]
    )
    # Fixed permutation: c3 highest, then c1, then a tie between c0 and c2 (broken by chunk_id), then c4.
    score_by_id = {"c0": 1.0, "c1": 3.0, "c2": 1.0, "c3": 4.0, "c4": 0.0}

    def fake_scorer(query: str, texts: list[str]) -> list[float]:
        # texts correspond 1:1 to whatever the base returned, in order.
        return [score_by_id[cid] for cid in [c.chunk_id for c in base._chunks[: len(texts)]]]

    retriever = RerankRetriever(base, fetch_k=5, scorer=fake_scorer)
    result = retriever.search("doc_a", "q", k=10)

    assert base.calls == [("doc_a", "q", 5)]  # base called with fetch_k, not caller's k=10
    assert [c.chunk_id for c in result] == ["c3", "c1", "c0", "c2", "c4"]


# --- BM25 clamping ----------------------------------------------------------


def test_bm25_clamping_k_larger_than_corpus():
    chunks = _make_corpus()
    bm25 = BM25Retriever(chunks_provider=_chunks_provider(chunks))
    result = bm25.search("doc_a", "harvest", k=1000)
    assert len(result) == 6  # doc_a has 6 chunks; no raise despite k=1000


def test_bm25_blank_query_returns_empty():
    chunks = _make_corpus()
    bm25 = BM25Retriever(chunks_provider=_chunks_provider(chunks))
    assert bm25.search("doc_a", "   ", k=5) == []


# --- Multi-query fusion -----------------------------------------------------


def test_multi_query_fusion_calls_base_for_every_variant():
    chunks = _make_corpus()
    base = RecordingRetriever(chunks)
    variants = {"harvest season": ["fruit picking", "autumn crop"]}
    multi = MultiQueryFusionRetriever(base, variants)

    multi.search("doc_a", "harvest season", k=3)
    queries_called = [call[1] for call in base.calls]
    assert queries_called == ["harvest season", "fruit picking", "autumn crop"]

    base.calls.clear()
    multi.search("doc_a", "an unknown query never in the table", k=3)
    assert len(base.calls) == 1
    assert base.calls[0][1] == "an unknown query never in the table"


# --- index_version stability -------------------------------------------------


def test_index_version_stability():
    assert index_version() == "e2b4a2b97561"
    assert index_version() == index_version()  # stable across calls
    arm_c_val = index_version(retriever_config=ARM_C_RETRIEVER)
    assert arm_c_val != index_version()
    assert arm_c_val == arm_c_index_version()
    assert arm_c_index_version() == arm_c_index_version()  # stable


# --- tournament_queries.json --------------------------------------------------


def test_tournament_queries_json():
    assert TOURNAMENT_QUERIES_PATH.exists()
    with open(TOURNAMENT_QUERIES_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)

    queries = payload["queries"]
    assert set(queries.keys()) == set(QUESTION_BY_ID.keys())
    assert len(queries) == 12

    for qid, entry in queries.items():
        assert entry["canonical"] == QUESTION_BY_ID[qid].canonical_query
        variants = entry["variants"]
        assert len(variants) >= 1
        assert len(variants) == len(set(variants))


# --- the one real-model test: needs_network, never needs_model, never in
# the offline gate --------------------------------------------------------


@pytest.mark.gate_m3
@pytest.mark.needs_network
def test_real_cross_encoder_scores_relevant_above_irrelevant():
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        from dealpoint.config import RERANK_MODEL

        ce = TextCrossEncoder(RERANK_MODEL)
        query = "material adverse effect definition carve-out stockholder litigation"
        docs = [
            (
                '"Material Adverse Effect" means any change that would reasonably be '
                "expected to have a material adverse effect on the Company, excluding any "
                "stockholder litigation arising from this Agreement."
            ),
            "The weather in the region has been unusually dry this quarter.",
        ]
        scores = list(ce.rerank(query, docs))
    except Exception as exc:  # noqa: BLE001 - any failure (offline, OOM, etc.) skips cleanly
        pytest.skip(f"real cross-encoder unavailable: {exc}")

    assert scores[0] > scores[1]
