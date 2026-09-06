"""LlamaIndex RAG lab (M7a): retriever composition + framework-native evaluation.

**Framework boundary, enforced by review (specs/milestones/m7a.md section 1):**
LlamaIndex owns retriever composition where useful, framework-native RAG
evaluation, and the synthetic-query robustness study. It does **not** own the
parser, canonical sections, MAUD offsets, the agent loop, the tools, the
answer schema, or the canonical scorers. MAUD gold-span overlap
(`dealpoint.eval.scorers.overlap_chars` against `MIN_GOLD_OVERLAP_CHARS`)
remains benchmark truth: LlamaIndex node ids or qrels never replace it and
never drive selection of the frozen M3 winner.

`llama_index` is imported lazily inside functions/methods, never at module
top level, so the offline test suite and pyright stay clean when the
optional `rag-lab` dependency group is not installed (same convention this
repo already applies to `openai`/`qdrant`/`braintrust`).
"""

from __future__ import annotations
