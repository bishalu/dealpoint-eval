"""Wrap the frozen M3 retriever candidates behind LlamaIndex's `BaseRetriever`.

Never rebuilds a candidate: `build_li_retriever` calls
`dealpoint.corpus.retrievers.build_retriever` (the exact function the M3
tournament and the four-arm sweep use) and wraps the *result* in an adapter.
Qdrant local mode allows exactly one client per path -- callers must build
`DenseRetriever()`/`BM25Retriever()` once and pass them in, never construct a
second one here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dealpoint.corpus.chunks import Chunk
from dealpoint.corpus.retrievers import (
    Retriever,
    RetrieverConfig,
    Scorer,
    build_retriever,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


def chunk_to_node(chunk: Chunk) -> TextNode:
    """`Chunk` -> LlamaIndex `TextNode`, preserving `chunk_id` as the node id.

    `id_=chunk.chunk_id` is what lets `expected_ids` (project gold-bearing
    chunk ids) be compared directly against LlamaIndex's own retrieved node
    ids -- no id translation layer, no second identity scheme.
    """
    from llama_index.core.schema import TextNode

    return TextNode(
        id_=chunk.chunk_id,
        text=chunk.text,
        metadata={
            "agreement_id": chunk.agreement_id,
            "section_ref": chunk.section_ref,
            "start": chunk.start,
            "end": chunk.end,
        },
    )


def _base_retriever_cls() -> type:
    """`llama_index.core.retrievers.BaseRetriever`, imported lazily.

    A small factory rather than a module-level import so
    `import dealpoint.rag_lab.adapters` succeeds with LlamaIndex absent --
    the class object itself is only touched inside `ProjectRetrieverAdapter`
    construction, never at import time.
    """
    from llama_index.core.retrievers import BaseRetriever

    return BaseRetriever


def _make_adapter_class() -> type:
    """Build the `ProjectRetrieverAdapter` class the first time it is needed.

    LlamaIndex's `BaseRetriever` must be subclassed, which means the class
    statement itself needs the real base class available -- so the class is
    defined inside this factory (called lazily) rather than at module scope.
    """
    base_cls = _base_retriever_cls()

    class _ProjectRetrieverAdapter(base_cls):  # type: ignore[misc, valid-type]
        """Adapts a project `Retriever` (search-only, ranked list, no scores)
        into LlamaIndex's `BaseRetriever` interface.

        The project retrievers return a *ranked list* of `Chunk`, never a
        similarity score, so `_retrieve` assigns each node a reciprocal-rank
        score (`1.0 / rank`, rank starting at 1) rather than inventing a
        similarity number the retriever never computed. This preserves the
        retriever's own order honestly: LlamaIndex's `RetrieverEvaluator`
        only needs the ORDER of `node_id`s, not a calibrated score.
        """

        def __init__(self, retriever: Retriever, agreement_id: str, k: int) -> None:
            self._retriever = retriever
            self._agreement_id = agreement_id
            self._k = k
            super().__init__()

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            from llama_index.core.schema import NodeWithScore

            chunks = self._retriever.search(self._agreement_id, query_bundle.query_str, k=self._k)
            return [
                NodeWithScore(node=chunk_to_node(chunk), score=1.0 / rank)
                for rank, chunk in enumerate(chunks, start=1)
            ]

    return _ProjectRetrieverAdapter


_ADAPTER_CLASS_CACHE: dict[str, type] = {}


def _adapter_class() -> type:
    cached = _ADAPTER_CLASS_CACHE.get("cls")
    if cached is None:
        cached = _make_adapter_class()
        _ADAPTER_CLASS_CACHE["cls"] = cached
    return cached


class ProjectRetrieverAdapter:
    """Public constructor facade: returns a real `BaseRetriever` subclass instance.

    Kept as a plain callable-looking class (rather than exporting the
    dynamically-built subclass directly) so `isinstance`/typing at call
    sites stays simple; `type(instance)` is the real LlamaIndex subclass.
    """

    def __new__(cls, retriever: Retriever, agreement_id: str, k: int) -> Any:
        adapter_cls = _adapter_class()
        return adapter_cls(retriever, agreement_id, k)


def build_li_retriever(
    config: RetrieverConfig,
    *,
    dense: Retriever,
    sparse: Retriever,
    variants_by_query: dict[str, list[str]] | None = None,
    scorer: Scorer | None = None,
    agreement_id: str,
    k: int,
) -> BaseRetriever:
    """Build one frozen M3 candidate via `build_retriever`, then wrap it.

    Never re-derives retrieval logic: composition (dense/bm25/hybrid-rrf,
    optional multi-query fusion, optional rerank) is entirely
    `dealpoint.corpus.retrievers.build_retriever`'s job.
    """
    inner = build_retriever(
        config,
        dense=dense,
        sparse=sparse,
        variants_by_query=variants_by_query,
        scorer=scorer,
    )
    return ProjectRetrieverAdapter(inner, agreement_id, k)
