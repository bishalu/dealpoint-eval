"""Retriever interface + a Qdrant-local-mode dense retriever (spec §3.3).

`qdrant_client` / `fastembed` are imported lazily inside `DenseRetriever` so
that the rest of the corpus/agent packages -- and the whole offline test
gate -- keep working when the `retrieval` optional dependency group is not
installed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from dealpoint.config import EMBEDDING_MODEL, INDEX_DIR, QDRANT_COLLECTION, RETRIEVER_DEFAULT_K
from dealpoint.corpus.chunks import Chunk, chunk_version


class Retriever(Protocol):
    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]: ...


def index_version(
    collection: str = QDRANT_COLLECTION,
    distance: str = "Cosine",
    default_k: int = RETRIEVER_DEFAULT_K,
) -> str:
    """Short hash of (chunk_version, embedding model, retriever config). Pure, no I/O."""
    payload = {
        "chunk_version": chunk_version(),
        "embedding_model": EMBEDDING_MODEL,
        "retriever_config": {
            "collection": collection,
            "distance": distance,
            "default_k": default_k,
        },
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


class DenseRetriever:
    """Dense retrieval over Qdrant local (on-disk) mode with fastembed embeddings.

    Every search is scoped to one document via a payload filter on
    `agreement_id` (which, per `chunks.py`, holds the *document* id -- a
    search can never return another document's chunks).
    """

    def __init__(self, client=None, collection: str = QDRANT_COLLECTION) -> None:
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient

        self._collection = collection
        self._client = client if client is not None else QdrantClient(path=str(INDEX_DIR))
        self._embedder = TextEmbedding(EMBEDDING_MODEL)

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
