"""Section-bounded retrieval chunks (spec §3.4).

Chunks never cross a section boundary; a section longer than
`CHUNK_TARGET_CHARS` is sub-split with `CHUNK_OVERLAP_CHARS` overlap,
preferring a whitespace boundary near the target split point so chunks do
not split mid-word. Front matter (text before the first recognised section)
is chunked under `section_ref = ""`, so a gold span in the recitals stays
retrievable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from dealpoint.config import CHUNK_ALGO_VERSION, CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS
from dealpoint.corpus.document import Document
from dealpoint.data.sections import PARSER_VERSION

_LOOKBACK_WINDOW = 80


@dataclass(frozen=True)
class Chunk:
    # For a redacted variant this holds the DOCUMENT id (e.g.
    # "contract_103__redacted_q09"), not the base agreement id -- that is
    # the retrieval scope key that keeps a redacted variant's chunks from
    # colliding with its original's chunks in the shared index.
    agreement_id: str
    chunk_id: str
    section_ref: str
    start: int
    end: int
    text: str


def _split_boundary(text: str, target_end: int) -> int:
    """Nearest whitespace boundary at or before `target_end`, within a small lookback."""
    if target_end >= len(text):
        return len(text)
    window_start = max(0, target_end - _LOOKBACK_WINDOW)
    idx = text.rfind(" ", window_start, target_end)
    if idx == -1:
        return target_end
    return idx + 1


def _split_region(text: str, region_start: int, region_end: int) -> list[tuple[int, int]]:
    """Split [region_start, region_end) of `text` into sub-chunks of ~CHUNK_TARGET_CHARS."""
    region_len = region_end - region_start
    if region_len <= 0:
        return []
    if region_len <= CHUNK_TARGET_CHARS:
        return [(region_start, region_end)]

    stride = CHUNK_TARGET_CHARS - CHUNK_OVERLAP_CHARS
    ranges: list[tuple[int, int]] = []
    cursor = region_start
    while cursor < region_end:
        raw_end = min(cursor + CHUNK_TARGET_CHARS, region_end)
        if raw_end < region_end:
            end = _split_boundary(text, raw_end)
            if end <= cursor:
                end = raw_end
        else:
            end = raw_end
        ranges.append((cursor, end))
        if end >= region_end:
            break
        cursor = max(cursor + stride, end - CHUNK_OVERLAP_CHARS)
        # guard against a degenerate non-advancing cursor
        if ranges and cursor <= ranges[-1][0]:
            cursor = end
    return ranges


def chunk_document(doc: Document) -> list[Chunk]:
    """Section-bounded retrieval chunks for `doc`. Deterministic."""
    text = doc.text
    chunks: list[Chunk] = []

    sections_sorted = sorted(doc.sections, key=lambda s: s.start)

    # Front matter: everything before the first section's start.
    front_end = sections_sorted[0].start if sections_sorted else len(text)
    for start, end in _split_region(text, 0, front_end):
        if end <= start:
            continue
        chunks.append(
            Chunk(
                agreement_id=doc.document_id,
                chunk_id=f"{doc.document_id}:{start}-{end}",
                section_ref="",
                start=start,
                end=end,
                text=text[start:end],
            )
        )

    for section in sections_sorted:
        region_start = max(section.start, 0)
        region_end = min(section.end, len(text))
        for start, end in _split_region(text, region_start, region_end):
            if end <= start:
                continue
            chunks.append(
                Chunk(
                    agreement_id=doc.document_id,
                    chunk_id=f"{doc.document_id}:{start}-{end}",
                    section_ref=section.ref,
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )

    return chunks


def chunk_version() -> str:
    """Short hash of the chunking parameters + parser version. Pure, no I/O."""
    payload = {
        "chunk_target_chars": CHUNK_TARGET_CHARS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
        "chunk_algo_version": CHUNK_ALGO_VERSION,
        "parser_version": PARSER_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
