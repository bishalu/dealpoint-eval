"""Gold span alignment: locate MAUD gold excerpt text inside canonical contract text.

MAUD gold spans are not verbatim substrings of the contract (spec brief §5, C3/C4):

- 77.3% of spans contain ``(Page N)`` markers that are not present in the
  contract text and must be stripped before alignment (C3).
- 36.6% of spans are multi-excerpt, joined by a blank line (``\\n\\n``) in the
  raw CSV field, *in addition to* the ``<omitted>`` marker used elsewhere
  (C4). Both must be split on.

Fragments are aligned independently, exact match first, falling back to
`rapidfuzz.fuzz.partial_ratio_alignment` at a cutoff of 90.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from dealpoint.config import FUZZY_CUTOFF, MIN_FRAGMENT_CHARS
from dealpoint.data.canonical import canonicalise

_PAGE_MARKER_RE = re.compile(r"\(Page\s+\d+\)")
_OMITTED_MARKER = "<omitted>"


def fragments(raw_span: str) -> list[str]:
    """Split a raw gold span into canonicalised, min-length fragments.

    Order of operations:
      1. split on the literal ``<omitted>`` and on blank lines (``\\n\\n``)
      2. strip ``(Page N)`` markers from each part
      3. canonicalise each part
      4. drop parts shorter than MIN_FRAGMENT_CHARS
    """
    parts = re.split(rf"{re.escape(_OMITTED_MARKER)}|\n\s*\n", raw_span)
    out: list[str] = []
    for part in parts:
        part = _PAGE_MARKER_RE.sub(" ", part)
        canon_part = canonicalise(part)
        if len(canon_part) >= MIN_FRAGMENT_CHARS:
            out.append(canon_part)
    return out


@dataclass(frozen=True)
class AlignResult:
    ranges: list[tuple[int, int]]  # merged, sorted
    n_fragments: int
    n_aligned: int
    modes: list[str]  # "exact" | "fuzzy" per aligned fragment
    covered_fraction: float  # aligned fragment chars / total fragment chars


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def align_span(canonical: str, raw_span: str) -> AlignResult:
    frags = fragments(raw_span)
    n_fragments = len(frags)
    total_frag_chars = sum(len(f) for f in frags)

    aligned_ranges: list[tuple[int, int]] = []
    modes: list[str] = []
    aligned_chars = 0
    n_aligned = 0

    for frag in frags:
        idx = canonical.find(frag)
        if idx != -1:
            aligned_ranges.append((idx, idx + len(frag)))
            modes.append("exact")
            aligned_chars += len(frag)
            n_aligned += 1
            continue
        result = fuzz.partial_ratio_alignment(frag, canonical, score_cutoff=FUZZY_CUTOFF)
        if result is not None:
            aligned_ranges.append((result.dest_start, result.dest_end))
            modes.append("fuzzy")
            aligned_chars += len(frag)
            n_aligned += 1

    covered_fraction = (aligned_chars / total_frag_chars) if total_frag_chars else 0.0

    return AlignResult(
        ranges=_merge_ranges(aligned_ranges),
        n_fragments=n_fragments,
        n_aligned=n_aligned,
        modes=modes,
        covered_fraction=covered_fraction,
    )
