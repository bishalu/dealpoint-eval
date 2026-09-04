"""Inline section-heading parser over canonical text.

Canonicalisation (see ``canonical.py``) collapses all whitespace including
newlines, so line-anchored heading regexes (``^\\s*Section\\s+\\d+\\.\\d+``)
match nothing (spec C5). Headings must be found *inline*, which drags in two
false-positive classes that this module handles deterministically:

- **Table of contents**: a dense run of heading-like matches at the top of
  the document. Detected by a sliding window: if 10 consecutive matches all
  sit within 200 chars of each other, they are TOC. The body starts at the
  first match after that run. A TOC run requires >= 10 matches to be called
  a TOC at all -- some contracts have none.
- **Cross-references** (``Section 6.3(b)`` in prose): accepted as a real
  heading only if its (article, section) tuple *advances* monotonically --
  same article with section == last + 1, or a new, strictly higher article.
  Everything else is a cross-reference and is skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dealpoint.config import MAX_SECTION_CHARS

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

# Article headings are only reliably distinguishable from cross-references
# (e.g. "as set forth in Article II", mixed case) by their all-caps
# rendering in the contract's own heading style ("ARTICLE II"); a
# case-sensitive match separates real headings from prose references far
# better than a case-insensitive one (measured: median parser_coverage rises
# from ~0.03 to ~0.71 across the 152 contracts).
_ARTICLE_RE = re.compile(r"\bARTICLE\s+([IVXLCDM]+|\d{1,2})\b")
# Section headings appear either as "Section 6.3" (case-insensitive keyword,
# also used in cross-reference prose -- filtered by the monotonic-advance
# rule below) or as a bare "6.3. Title" heading token followed by whitespace.
_SECTION_RE = re.compile(
    r"\bSection\s+(\d{1,2})\.(\d{1,3})\b|\b(\d{1,2})\.(\d{1,3})\.(?=\s)",
    re.IGNORECASE,
)
# Fallback for the minority of contracts that number top-level parts as
# bare "1. DEFINITIONS" instead of "ARTICLE I" (used only when fewer than 3
# ARTICLE-style headings are found at all).
_TOPLEVEL_RE = re.compile(r"(?<![\d.])(\d{1,2})\.\s(?=[A-Z])")
_MIN_ARTICLE_HEADINGS = 3

_TOC_GAP_THRESHOLD = 200
_TOC_MIN_RUN = 10

_TITLE_MAX_CHARS = 120
_TITLE_END_RE = re.compile(r"[.](?:\s|$)")


def _roman_to_int(token: str) -> int:
    token = token.upper()
    total = 0
    prev = 0
    for ch in reversed(token):
        value = _ROMAN_VALUES.get(ch, 0)
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    kind: str  # "article" | "section"
    article: int
    section: int | None  # None for article candidates
    raw_ref: str


def _find_candidates(canonical: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    article_candidates: list[_Candidate] = []
    for m in _ARTICLE_RE.finditer(canonical):
        token = m.group(1)
        value = _roman_to_int(token) if not token.isdigit() else int(token)
        article_candidates.append(
            _Candidate(m.start(), m.end(), "article", value, None, f"ARTICLE {token}")
        )
    if len(article_candidates) < _MIN_ARTICLE_HEADINGS:
        # This contract numbers its top level as bare "1. TITLE" instead of
        # "ARTICLE I"; fall back to that pattern for the top-level candidate.
        article_candidates = [
            _Candidate(m.start(), m.end(), "article", int(m.group(1)), None, m.group(1))
            for m in _TOPLEVEL_RE.finditer(canonical)
        ]
    candidates.extend(article_candidates)
    for m in _SECTION_RE.finditer(canonical):
        if m.group(1):
            article, section = int(m.group(1)), int(m.group(2))
        else:
            article, section = int(m.group(3)), int(m.group(4))
        candidates.append(
            _Candidate(m.start(), m.end(), "section", article, section, f"{article}.{section}")
        )
    candidates.sort(key=lambda c: c.start)
    return candidates


def _find_body_start_index(candidates: list[_Candidate]) -> int:
    """Return the index of the first candidate past any TOC run."""
    n = len(candidates)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and candidates[j + 1].start - candidates[j].start <= _TOC_GAP_THRESHOLD:
            j += 1
        run_len = j - i + 1
        if run_len >= _TOC_MIN_RUN:
            i = j + 1
            continue
        return i
    return i


def _accept(candidates: list[_Candidate]) -> list[_Candidate]:
    """Keep only candidates whose (article, section) tuple advances monotonically."""
    accepted: list[_Candidate] = []
    cur_article = 0
    cur_section = 0
    for cand in candidates:
        if cand.kind == "article":
            if cand.article > cur_article:
                cur_article = cand.article
                cur_section = 0
                accepted.append(cand)
        else:
            assert cand.section is not None
            if cand.article == cur_article and cand.section == cur_section + 1:
                cur_section = cand.section
                accepted.append(cand)
    return accepted


def _extract_title(canonical: str, heading_end: int) -> str:
    window = canonical[heading_end : heading_end + _TITLE_MAX_CHARS + 1]
    window = window.lstrip()
    m = _TITLE_END_RE.search(window)
    if m:
        title = window[: m.start()]
    else:
        title = window[:_TITLE_MAX_CHARS]
    return title.strip()[:_TITLE_MAX_CHARS]


@dataclass(frozen=True)
class Section:
    ref: str  # "6.3" or "ARTICLE VI"
    title: str  # text up to the first sentence end, <= 120 chars, may be ""
    start: int  # inclusive, canonical offsets
    end: int  # exclusive, = next section's start


def parse_sections(canonical: str) -> list[Section]:
    """Parse an inline, TOC/cross-reference-aware section map from canonical text."""
    candidates = _find_candidates(canonical)
    if not candidates:
        return []
    body_start_idx = _find_body_start_index(candidates)
    accepted = _accept(candidates[body_start_idx:])
    if not accepted:
        return []
    sections: list[Section] = []
    for idx, cand in enumerate(accepted):
        end = accepted[idx + 1].start if idx + 1 < len(accepted) else len(canonical)
        title = _extract_title(canonical, cand.end)
        sections.append(Section(ref=cand.raw_ref, title=title, start=cand.start, end=end))
    return sections


def parser_coverage(canonical: str, sections: list[Section]) -> float:
    """Fraction of body characters lying in a section of length <= MAX_SECTION_CHARS.

    Body = the span from the first section's start to the last section's end
    (i.e. the front matter / recitals before the first heading is excluded,
    matching the fact that TOC and preamble are never "covered" by a heading).
    """
    if not sections:
        return 0.0
    body_start = sections[0].start
    body_end = sections[-1].end
    body_len = body_end - body_start
    if body_len <= 0:
        return 0.0
    covered = sum(
        (s.end - s.start) for s in sections if (s.end - s.start) <= MAX_SECTION_CHARS
    )
    return covered / body_len
