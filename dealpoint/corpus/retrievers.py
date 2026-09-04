"""Retriever candidates behind the `Retriever` Protocol (spec §3.3, M3 tournament).

`qdrant_client` / `fastembed` / `bm25s` are imported lazily inside methods and
constructors, never at module top level, so the rest of the corpus/agent
packages -- and the whole offline test gate -- keep working when the
`retrieval` optional dependency group is not installed. Every expensive
dependency (the Qdrant client, the fastembed embedder, the cross-encoder
scorer) is constructor-injectable with a `None` default: when an argument is
supplied nothing is imported and nothing is downloaded, which is what lets
`gate_m3` build every candidate on a synthetic corpus with no network access.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from dealpoint.config import (
    EMBEDDING_MODEL,
    HYBRID_FETCH_K,
    INDEX_DIR,
    QDRANT_COLLECTION,
    RERANK_FETCH_K,
    RERANK_MODEL,
    RETRIEVER_DEFAULT_K,
    RRF_K,
)
from dealpoint.corpus.chunks import Chunk, chunk_document, chunk_version


class Retriever(Protocol):
    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]: ...


# (query, doc_texts) -> scores, higher = better
Scorer = Callable[[str, list[str]], list[float]]


def index_version(
    collection: str = QDRANT_COLLECTION,
    distance: str = "Cosine",
    default_k: int = RETRIEVER_DEFAULT_K,
    retriever_config: dict | None = None,
) -> str:
    """Short hash of (chunk_version, embedding model, retriever config). Pure, no I/O.

    With no arguments this reproduces the M1 value byte-for-byte
    (`e2b4a2b97561`), which is embedded in committed result filenames and
    asserted literally elsewhere in the suite -- never change that default
    payload shape. When `retriever_config` is given it replaces the default
    dense config in the hashed payload, which is how arm C's frozen
    retrieval identity is derived (brief §2.3: index_version = hash of
    chunking params, embedding model, retriever config).
    """
    retriever_payload = (
        retriever_config
        if retriever_config is not None
        else {
            "collection": collection,
            "distance": distance,
            "default_k": default_k,
        }
    )
    payload = {
        "chunk_version": chunk_version(),
        "embedding_model": EMBEDDING_MODEL,
        "retriever_config": retriever_payload,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def arm_c_index_version() -> str:
    """`index_version` for arm C's frozen retriever config (spec deliverable 3)."""
    from dealpoint.config import ARM_C_RETRIEVER

    return index_version(retriever_config=ARM_C_RETRIEVER)


class DenseRetriever:
    """Dense retrieval over Qdrant local (on-disk) mode with fastembed embeddings.

    Every search is scoped to one document via a payload filter on
    `agreement_id` (which, per `chunks.py`, holds the *document* id -- a
    search can never return another document's chunks).
    """

    def __init__(self, client=None, collection: str = QDRANT_COLLECTION, embedder=None) -> None:
        from qdrant_client import QdrantClient

        self._collection = collection
        self._client = client if client is not None else QdrantClient(path=str(INDEX_DIR))
        self._embedder = embedder if embedder is not None else self._build_default_embedder()

    @staticmethod
    def _build_default_embedder():
        from fastembed import TextEmbedding

        return TextEmbedding(EMBEDDING_MODEL)

    def _embed_query(self, text: str) -> list[float]:
        vec = next(iter(self._embedder.query_embed([text])))
        return list(vec)

    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]:
        from qdrant_client import models

        vector = self._embed_query(query)
        result = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="agreement_id", match=models.MatchValue(value=agreement_id)
                    )
                ]
            ),
            limit=k,
            with_payload=True,
        )
        chunks: list[Chunk] = []
        for point in result.points:
            payload = point.payload or {}
            chunks.append(
                Chunk(
                    agreement_id=payload["agreement_id"],
                    chunk_id=payload["chunk_id"],
                    section_ref=payload["section_ref"],
                    start=payload["start"],
                    end=payload["end"],
                    text=payload["text"],
                )
            )
        return chunks


class LazyRetriever:
    """Wraps `DenseRetriever`, deferring import + construction to the first
    `.search()` call.

    This is what lets `--fake` CLI/test runs that never actually call
    `search_agreement` avoid requiring network access or a built index --
    `DenseRetriever.__init__` loads the fastembed ONNX model and opens the
    on-disk Qdrant collection, either of which can fail offline.
    """

    def __init__(self, collection: str = QDRANT_COLLECTION) -> None:
        self._collection = collection
        self._inner: DenseRetriever | None = None

    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]:
        if self._inner is None:
            self._inner = DenseRetriever(collection=self._collection)
        return self._inner.search(agreement_id, query, k=k)


# --- M3: reciprocal rank fusion, a pure function -----------------------------


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Chunk]], rrf_k: int = RRF_K
) -> list[Chunk]:
    """Fuse ranked lists by sum of 1/(rrf_k + rank), rank 1-based. Deterministic.

    Score a chunk by `sum(1/(rrf_k + rank))` over every list it appears in,
    then sort by `(-score, chunk_id)`. The `chunk_id` tiebreak is mandatory
    -- without it the output order depends on dict insertion order. No I/O,
    no imports beyond `Chunk`.
    """
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            chunk_by_id.setdefault(chunk.chunk_id, chunk)
    ordered_ids = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [chunk_by_id[cid] for cid in ordered_ids]


# --- M3: BM25 ---------------------------------------------------------------


def _default_bm25_chunks_provider(agreement_id: str) -> list[Chunk]:
    from dealpoint.corpus.document import load_document

    return chunk_document(load_document(agreement_id))


class BM25Retriever:
    """Lexical retrieval via `bm25s`, over the same fixed M1 chunks the dense
    index holds. Never re-chunks: the default `chunks_provider` calls
    `chunk_document(load_document(doc_id))`, the identical function the
    dense index was built from.
    """

    def __init__(self, chunks_provider: Callable[[str], list[Chunk]] | None = None) -> None:
        self._chunks_provider = chunks_provider or _default_bm25_chunks_provider
        self._cache: dict[str, tuple[object, list[Chunk]]] = {}

    def _get_index(self, agreement_id: str) -> tuple[object, list[Chunk]]:
        cached = self._cache.get(agreement_id)
        if cached is not None:
            return cached
        import bm25s

        chunks = self._chunks_provider(agreement_id)
        texts = [c.text for c in chunks]
        tokenized = bm25s.tokenize(
            texts, stopwords="en", stemmer=None, show_progress=False  # type: ignore[arg-type]
        )
        index = bm25s.BM25()
        index.index(tokenized, show_progress=False)
        self._cache[agreement_id] = (index, chunks)
        return index, chunks

    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]:
        import bm25s

        index, chunks = self._get_index(agreement_id)
        if not chunks:
            return []
        query_tokens = bm25s.tokenize(
            [query], stopwords="en", stemmer=None, show_progress=False  # type: ignore[arg-type]
        )
        if not query_tokens.ids[0]:  # type: ignore[union-attr]
            # Empty/all-stopword query: return no results rather than an
            # arbitrary rank-0 chunk (bm25s returns all-zero scores here).
            return []
        k_clamped = min(k, len(chunks))
        if k_clamped <= 0:
            return []
        idx, _scores = index.retrieve(  # type: ignore[attr-defined]
            query_tokens, k=k_clamped, show_progress=False
        )
        return [chunks[i] for i in idx[0]]


# --- M3: hybrid RRF ----------------------------------------------------------


class HybridRRFRetriever:
    """Dense + sparse legs fused in Python with `reciprocal_rank_fusion`.

    Fusion happens in Python, not in Qdrant -- the spec allows either
    ("Qdrant sparse+dense RRF, or fusion in Python"), and Python fusion
    needs no re-index, no sparse vectors in the collection, and no change to
    `data/index/`.
    """

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        fetch_k: int = HYBRID_FETCH_K,
        rrf_k: int = RRF_K,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._fetch_k = fetch_k
        self._rrf_k = rrf_k

    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]:
        dense_hits = self._dense.search(agreement_id, query, k=self._fetch_k)
        sparse_hits = self._sparse.search(agreement_id, query, k=self._fetch_k)
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], rrf_k=self._rrf_k)
        return fused[:k]


# --- M3: cross-encoder rerank ------------------------------------------------


class RerankRetriever:
    """Rescore `base`'s top-`fetch_k` with a cross-encoder scorer.

    The default scorer lazily constructs one shared
    `TextCrossEncoder(RERANK_MODEL)` on first use (built at most once per
    process/instance -- construction + weights cost ~1.3 GB RSS) and wraps
    `list(ce.rerank(query, texts))`. Every `gate_m3` test must inject a fake
    `scorer` instead.
    """

    def __init__(
        self,
        base: Retriever,
        model: str = RERANK_MODEL,
        fetch_k: int = RERANK_FETCH_K,
        scorer: Scorer | None = None,
    ) -> None:
        self._base = base
        self._model = model
        self._fetch_k = fetch_k
        self._scorer = scorer
        self._lazy_scorer: Scorer | None = None

    def _score(self, query: str, texts: list[str]) -> list[float]:
        if self._scorer is not None:
            return self._scorer(query, texts)
        if self._lazy_scorer is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            ce = TextCrossEncoder(self._model)

            def _real_scorer(q: str, docs: list[str]) -> list[float]:
                return list(ce.rerank(q, docs))

            self._lazy_scorer = _real_scorer
        return self._lazy_scorer(query, texts)

    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]:
        candidates = self._base.search(agreement_id, query, k=self._fetch_k)
        if not candidates:
            return []
        texts = [c.text for c in candidates]
        scores = self._score(query, texts)
        ordered = sorted(zip(candidates, scores, strict=True), key=lambda pair: (-pair[1], pair[0].chunk_id))
        return [chunk for chunk, _score in ordered][:k]


# --- M3: multi-query fusion ---------------------------------------------------


class MultiQueryFusionRetriever:
    """RRF-fuse `base.search` over the incoming query plus frozen alternate
    variants, keyed by query text (the Protocol only passes a query string,
    not a question id).

    A query with no entry in `variants_by_query` falls back to a single
    `base.search` call -- never an error.
    """

    def __init__(
        self,
        base: Retriever,
        variants_by_query: dict[str, list[str]],
        fetch_k: int = HYBRID_FETCH_K,
        rrf_k: int = RRF_K,
    ) -> None:
        self._base = base
        self._variants_by_query = variants_by_query
        self._fetch_k = fetch_k
        self._rrf_k = rrf_k

    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]:
        queries = [query, *self._variants_by_query.get(query, [])]
        ranked_lists = [self._base.search(agreement_id, q, k=self._fetch_k) for q in queries]
        fused = reciprocal_rank_fusion(ranked_lists, rrf_k=self._rrf_k)
        return fused[:k]


# --- M3: frozen-config plumbing ----------------------------------------------


@dataclass(frozen=True)
class RetrieverConfig:
    name: str  # "dense" | "bm25" | "hybrid_rrf" | "hybrid_rrf_rerank" | ...
    kind: str  # "dense" | "bm25" | "hybrid_rrf" -- which leg(s) to build
    fetch_k: int = HYBRID_FETCH_K
    rrf_k: int = RRF_K
    rerank_model: str | None = None
    multi_query: bool = False


def build_retriever(
    config: RetrieverConfig,
    *,
    dense: Retriever,
    sparse: Retriever,
    variants_by_query: dict[str, list[str]] | None = None,
    scorer: Scorer | None = None,
) -> Retriever:
    """Build one candidate from a frozen `RetrieverConfig`.

    Never constructs a `DenseRetriever` itself -- Qdrant local mode allows
    exactly one client per path, so the caller passes the shared `dense`/
    `sparse` instances in. Composition order: leg (dense/bm25/hybrid_rrf) ->
    optional multi-query fusion -> optional cross-encoder rerank.
    """
    leg: Retriever
    if config.kind == "dense":
        leg = dense
    elif config.kind == "bm25":
        leg = sparse
    elif config.kind == "hybrid_rrf":
        leg = HybridRRFRetriever(dense, sparse, fetch_k=config.fetch_k, rrf_k=config.rrf_k)
    else:
        raise ValueError(f"unknown RetrieverConfig.kind {config.kind!r}")

    if config.multi_query:
        leg = MultiQueryFusionRetriever(
            leg, variants_by_query or {}, fetch_k=config.fetch_k, rrf_k=config.rrf_k
        )

    if config.rerank_model:
        leg = RerankRetriever(
            leg, model=config.rerank_model, fetch_k=config.fetch_k, scorer=scorer
        )

    return leg
