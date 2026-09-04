"""The three agent tools: pure functions over a `Document` + a `Retriever`.

No LLM, no global state. Each tool returns a `ToolResult` whose rendered
`text` is what the model sees (section ref + canonical offsets on every
block, truncated to `MAX_TOOL_RESULT_CHARS` with the ref preserved) plus
`chunk_ids`/`char_ranges` for the execution record's trajectory. A miss
returns an honest message, never an exception and never a silent empty
string — the agent needs to be able to tell "absent" from "broken".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dealpoint.config import MAX_TOOL_RESULT_CHARS, RETRIEVER_DEFAULT_K
from dealpoint.corpus.chunks import Chunk
from dealpoint.corpus.document import Document
from dealpoint.corpus.document import defined_term as _defined_term
from dealpoint.corpus.document import get_section as _get_section
from dealpoint.corpus.retrievers import Retriever


@dataclass(frozen=True)
class ToolResult:
    text: str
    chunk_ids: list[str] = field(default_factory=list)
    char_ranges: list[tuple[int, int]] = field(default_factory=list)


def _render_block(section_ref: str, start: int, end: int, text: str) -> str:
    header = f"[section {section_ref or '(none)'} | chars {start}-{end}]"
    body = text
    truncated = len(body) > MAX_TOOL_RESULT_CHARS
    if truncated:
        body = body[:MAX_TOOL_RESULT_CHARS]
    suffix = " \u2026[truncated]" if truncated else ""
    return f"{header} {body}{suffix}"


def search_agreement(
    doc: Document, retriever: Retriever, query: str, k: int = RETRIEVER_DEFAULT_K
) -> ToolResult:
    """Find candidate chunks in `doc` for `query`. Callable repeatedly."""
    chunks: list[Chunk] = retriever.search(doc.document_id, query, k=k)
    if not chunks:
        return ToolResult(text=f'no results for query "{query}" in this agreement')
    blocks = [_render_block(c.section_ref, c.start, c.end, c.text) for c in chunks]
    return ToolResult(
        text="\n\n".join(blocks),
        chunk_ids=[c.chunk_id for c in chunks],
        char_ranges=[(c.start, c.end) for c in chunks],
    )


def get_section(doc: Document, section_ref: str) -> ToolResult:
    """Follow a cross-reference (e.g. "as set forth in Section 6.3(b)")."""
    section = _get_section(doc, section_ref)
    if section is None:
        return ToolResult(text=f'no section matching "{section_ref}" in this agreement')
    text = doc.text[section.start : section.end]
    block = _render_block(section.ref, section.start, section.end, text)
    return ToolResult(text=block, char_ranges=[(section.start, section.end)])


def lookup_defined_term(doc: Document, term: str) -> ToolResult:
    """Return the `"Term" means ...` block for `term`, fuzzy on casing/quotes/prefix."""
    dt = _defined_term(doc, term)
    if dt is None:
        return ToolResult(text=f'no definition found for "{term}" in this agreement')
    block = _render_block(dt.section_ref, dt.start, dt.end, dt.text)
    return ToolResult(text=block, char_ranges=[(dt.start, dt.end)])


def tool_schemas() -> list[dict]:
    """OpenAI-format tool definitions for the three tools.

    `search_agreement` exposes only `query` and `k` — the document is bound
    by the harness, never chosen by the model (brief §1.3).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_agreement",
                "description": (
                    "Find candidate passages in the current agreement matching a query. "
                    "Callable repeatedly with different phrasing."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query": {"type": "string", "description": "search query"},
                        "k": {
                            "type": "integer",
                            "description": "number of results to return",
                            "default": RETRIEVER_DEFAULT_K,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_section",
                "description": (
                    "Look up a section of the current agreement by its reference "
                    '(e.g. "6.3", "Section 6.3(b)", "Article VI"), to follow a '
                    "cross-reference."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "section_ref": {"type": "string", "description": "section reference"},
                    },
                    "required": ["section_ref"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_defined_term",
                "description": (
                    'Return the `"Term" means ...` definition block for a defined term '
                    'in the current agreement (e.g. "Material Adverse Effect").'
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "term": {"type": "string", "description": "defined term to look up"},
                    },
                    "required": ["term"],
                },
            },
        },
    ]
