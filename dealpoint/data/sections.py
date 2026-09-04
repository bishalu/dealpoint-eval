"""Inline section-heading parser over canonical text (M0.1 rewrite).

Canonicalisation (see ``canonical.py``) collapses all whitespace including
newlines, so line-anchored heading regexes (``^\\s*Section\\s+\\d+\\.\\d+``)
match nothing (spec C5). Headings must be found *inline*, which drags in two
false-positive classes that this module handles deterministically:

- **Table of contents / defined-terms index / exhibit list**: a dense run of
  heading-like matches near the top of the document. Detected by a sliding
  window: if >= 10 consecutive matches all sit within 200 chars of each
  other, the run is TOC-like. Real MAUD contracts often have *several* such
  runs before the operative text (a table of contents, then an index of
  defined terms, then a list of exhibits); the body starts after the LAST
  TOC-like run that begins within the first ``TOC_SCAN_FRACTION`` of the
  document -- stopping at the first run (the M0 behaviour) anchors the
  numbering chain on TOC junk and silently collapses the whole document into
  one giant trailing section (see specs/milestones/m0_1.md Problem A).
- **Cross-references** (``Section 6.3(b)`` in prose): accepted as a real
  heading only if its (article, section) tuple *advances* -- same article
  with section increasing by up to ``SECTION_ADVANCE_WINDOW``, or a new
  article that is exactly one more than the current one (or a section that
  opens its own new article, i.e. ``N.1`` where article == current + 1).
  Everything else is a cross-reference and is skipped.

A *canonical legal section* is whatever this heading structure delimits, at
any length -- there is no length ceiling. Retrieval sub-chunking is a later
milestone's concern (spec §A.2) and must not influence the section map.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from dealpoint.config import (
    GIANT_SECTION_FRACTION,
    SECTION_ADVANCE_WINDOW,
    TOC_SCAN_FRACTION,
)

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

# ARTICLE headings: roman or arabic, optional trailing period, any case
# (spec §A.1). The any-case match is only safe because body-start detection
# and chain anchoring below carry the load that a case-sensitive match used
# to improvise in the M0 parser (that version's comment recorded raising
# median parser_coverage from ~0.03 to ~0.71 via case-sensitivity alone --
# the fix here is a better body/anchor model, not case sensitivity).
_ARTICLE_RE = re.compile(r"\bARTICLE\s+([IVXLCDM]+|\d{1,2})\.?\b", re.IGNORECASE)

# "Section N.N", "SECTION N.N", zero-padded "Section 1.01", "§ N.N" -- the
# minor number is parsed with `int()`, which already treats "01" == 1, so
# zero-padded numbering advances correctly with no special-casing.
_SECTION_KEYWORD_RE = re.compile(r"\b(?:Section|§)\s*(\d{1,2})\.(\d{1,3})\b", re.IGNORECASE)
# Bare "6.3 Title" / "6.03. Title" heading tokens (not preceded by another
# digit/period, followed by whitespace then an uppercase title character).
_SECTION_BARE_RE = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,3})\.?(?=\s+[A-Z])")

# Fallback for the minority of contracts that number top-level parts as bare
# "1. DEFINITIONS" instead of "ARTICLE I" (used only when fewer than 3
# ARTICLE-style headings are found at all).
_TOPLEVEL_RE = re.compile(r"(?<![\d.])(\d{1,2})\.\s(?=[A-Z])")
_MIN_ARTICLE_HEADINGS = 3

# Article numbers above this are dates ("Article 2020") or dollar amounts,
# not headings.
_ARTICLE_MAX_VALUE = 40

_TOC_GAP_THRESHOLD = 200
_TOC_MIN_RUN = 10

_WITNESS_RE = re.compile(r"IN WITNESS WHEREOF", re.IGNORECASE)

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


# --- parser_version / stale-state guard (spec §A.5) ------------------------


def _compute_parser_version() -> str:
    """Short hash of this module's own source, so derived data can detect staleness.

    Reading ``__file__`` at import time would fail inside a zipped/frozen
    install; this project always runs from a source checkout, but the read
    is wrapped so an unreadable source falls back to a constant rather than
    raising at import time -- a failure here would break every import of the
    package.
    """
    try:
        source = Path(__file__).read_bytes()
    except OSError:
        return "unreadable000"
    return hashlib.sha256(source).hexdigest()[:12]


PARSER_VERSION = _compute_parser_version()


class StaleDerivedDataError(RuntimeError):
    """Raised when derived data on disk was built by a different parser_version."""


def check_parser_version(path: Path | str, found_version: str | None) -> None:
    """Raise ``StaleDerivedDataError`` if `found_version` != the current PARSER_VERSION.

    Pure and filesystem-free: `path` is only used for the error message, so
    tests can pass a synthetic path with no file behind it.
    """
    if found_version != PARSER_VERSION:
        raise StaleDerivedDataError(
            f"{path} was built by parser_version {found_version!r}, current is "
            f"{PARSER_VERSION!r}. Run `just data` to rebuild."
        )


def assert_sections_dir_not_stale(sections_dir: Path) -> None:
    """Refuse to proceed if any derived sections file in `sections_dir` is stale.

    Missing files, or a missing directory, are not an error -- this only
    fires when a stale (or version-less legacy) derived file is actually
    present on disk (spec §A.5).
    """
    if not sections_dir.is_dir():
        return
    for path in sorted(sections_dir.glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        check_parser_version(path, payload.get("parser_version"))


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    kind: str  # "article" | "section"
    article: int
    section: int | None  # None for article candidates
    raw_ref: str


def _find_candidates(canonical: str) -> list[_Candidate]:
    raw: list[_Candidate] = []
    article_candidates: list[_Candidate] = []
    for m in _ARTICLE_RE.finditer(canonical):
        token = m.group(1)
        value = int(token) if token.isdigit() else _roman_to_int(token)
        if value == 0 or value > _ARTICLE_MAX_VALUE:
            continue
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
    raw.extend(article_candidates)
    for pattern in (_SECTION_KEYWORD_RE, _SECTION_BARE_RE):
        for m in pattern.finditer(canonical):
            article, section = int(m.group(1)), int(m.group(2))
            raw.append(
                _Candidate(m.start(), m.end(), "section", article, section, f"{article}.{section}")
            )
    # Sort by offset; when an article and a section candidate start at the
    # same offset, prefer the article (e.g. "ARTICLE 1" also loosely
    # matching a bare top-level fallback token).
    raw.sort(key=lambda c: (c.start, 0 if c.kind == "article" else 1))
    deduped: list[_Candidate] = []
    last_start = -1
    for cand in raw:
        if cand.start == last_start:
            continue
        deduped.append(cand)
        last_start = cand.start
    deduped.sort(key=lambda c: c.start)
    # A heading can be matched by more than one regex (e.g. "Section 1.01."
    # matches both the "Section N.N" keyword form and the bare "N.N " form,
    # a few chars apart). Collapse near-duplicates -- same (kind, article,
    # section) within a short span -- to one candidate (the earliest match),
    # so a single real heading cannot masquerade as two closely spaced
    # candidates and inflate a TOC-run's apparent density.
    collapsed: list[_Candidate] = []
    for cand in deduped:
        if (
            collapsed
            and collapsed[-1].kind == cand.kind
            and collapsed[-1].article == cand.article
            and collapsed[-1].section == cand.section
            and cand.start - collapsed[-1].start <= _TOC_GAP_THRESHOLD // 10
        ):
            continue
        collapsed.append(cand)
    return collapsed


def _find_body_start_index(candidates: list[_Candidate], doc_len: int) -> int:
    """Index after the LAST TOC-like run that begins within the first TOC_SCAN_FRACTION.

    Unlike the M0 parser (which stopped at the *first* dense run), this scans
    every dense run in the document and only treats a run as TOC-like if it
    both looks dense (>= 10 matches within 200 chars of each other) and
    starts near the top of the document. Real contracts commonly have
    several such runs in sequence (table of contents, defined-terms index,
    exhibit list) before the operative numbering begins.
    """
    n = len(candidates)
    body_start = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and candidates[j + 1].start - candidates[j].start <= _TOC_GAP_THRESHOLD:
            j += 1
        # A real ARTICLE heading can sit close enough to the tail of a TOC
        # run to be pulled into it by proximity alone (the TOC's last entry
        # and the document's real "ARTICLE I" are often only a line apart).
        # Trim trailing "article" candidates off the run before judging or
        # consuming it, so the real first heading is never swallowed as if
        # it were TOC content.
        trimmed_j = j
        while trimmed_j > i and candidates[trimmed_j].kind == "article":
            trimmed_j -= 1
        run_len = trimmed_j - i + 1
        if run_len >= _TOC_MIN_RUN and candidates[i].start < TOC_SCAN_FRACTION * doc_len:
            body_start = trimmed_j + 1
        i = j + 1
    return body_start


def _find_anchor_index(candidates: list[_Candidate], body_start_idx: int) -> int:
    """From body_start_idx, the first ARTICLE 1/I or section 1.1/1.01 candidate.

    Anchoring the numbering chain at the document's actual "Article 1"/"1.1"
    start (rather than whatever the TOC filter happened to leave behind)
    prevents the chain from dying early on stray matches between the TOC and
    the real body.
    """
    for idx in range(body_start_idx, len(candidates)):
        cand = candidates[idx]
        if cand.kind == "article" and cand.article == 1:
            return idx
        if cand.kind == "section" and cand.article == 1 and cand.section == 1:
            return idx
    return body_start_idx


def _accept(candidates: list[_Candidate]) -> list[_Candidate]:
    """Keep only candidates whose (article, section) tuple advances.

    - an article is accepted if it is the first candidate, or exactly one
      more than the current article
    - a section is accepted if it is in the same article and its number is
      > current_section and <= current_section + SECTION_ADVANCE_WINDOW, or
      if it opens a new article one higher than the current one with
      section == 1 (a section numbered "N.1" that starts its own article
      without an explicit "ARTICLE N" heading immediately before it)
    - everything else is a cross-reference and is skipped
    """
    accepted: list[_Candidate] = []
    cur_article: int | None = None
    cur_section = 0
    for cand in candidates:
        if cand.kind == "article":
            if cur_article is None or cand.article == cur_article + 1:
                cur_article = cand.article
                cur_section = 0
                accepted.append(cand)
            continue
        assert cand.section is not None
        if cur_article is None:
            cur_article = cand.article
            cur_section = cand.section
            accepted.append(cand)
        elif cand.article == cur_article and cur_section < cand.section <= cur_section + SECTION_ADVANCE_WINDOW:
            cur_section = cand.section
            accepted.append(cand)
        elif cand.article == cur_article + 1 and cand.section == 1:
            cur_article = cand.article
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
    """Parse an inline, TOC-aware, chain-anchored section map from canonical text."""
    if not canonical:
        return []
    candidates = _find_candidates(canonical)
    if not candidates:
        return []
    body_start_idx = _find_body_start_index(candidates, len(canonical))
    if body_start_idx >= len(candidates):
        return []
    anchor_idx = _find_anchor_index(candidates, body_start_idx)
    accepted = _accept(candidates[anchor_idx:])
    if not accepted:
        return []
    sections: list[Section] = []
    for idx, cand in enumerate(accepted):
        end = accepted[idx + 1].start if idx + 1 < len(accepted) else len(canonical)
        title = _extract_title(canonical, cand.end)
        sections.append(Section(ref=cand.raw_ref, title=title, start=cand.start, end=end))
    return sections


def _true_body_start(canonical: str) -> int | None:
    """The post-TOC heading offset, independent of the accept-chain's anchor.

    This is `candidates[_find_body_start_index(...)].start` -- the first
    heading-like candidate after the TOC-like run(s) -- which is NOT
    necessarily the same heading `_find_anchor_index` later chooses to start
    the numbering chain from. On roughly 58/152 MAUD contracts these differ:
    some other heading-like text (e.g. a stray cross-reference or a
    defined-terms index entry that didn't itself form a dense enough run to
    be filtered as TOC) sits between the true post-TOC start and the
    document's actual "ARTICLE I"/"1.1" anchor. The span between them is
    real, unrecognised body text and must count as *uncovered* by
    `structural_coverage` (spec §A.3), not be excluded from the denominator
    by defining body_start as the first *accepted* section's start (which is
    what an earlier version of this function did, making structural_coverage
    degenerate: covered == body_len identically, since sections tile
    contiguously from their own first start).
    """
    if not canonical:
        return None
    candidates = _find_candidates(canonical)
    if not candidates:
        return None
    body_start_idx = _find_body_start_index(candidates, len(canonical))
    if body_start_idx >= len(candidates):
        return None
    return candidates[body_start_idx].start


def compute_body_bounds(canonical: str, sections: list[Section]) -> tuple[int, int]:
    """Body = the first heading after the TOC, to the first IN WITNESS WHEREOF at/after it.

    The start is deliberately NOT `sections[0].start` (the first *accepted*
    section) -- spec §A.3 defines body as beginning at "the first
    Article/Section heading after the TOC", computed independently of the
    accept-chain via `_true_body_start`. See that function's docstring for
    why the two can differ and why the gap between them must count as
    uncovered.

    Falls back to the end of the document when no signature block is found
    (spec §A.3: "first `IN WITNESS WHEREOF`, else the last heading" -- since
    the final accepted section already runs to the end of the document by
    construction, "else the last heading" is equivalent to not clipping at
    all).
    """
    if not sections:
        return (0, 0)
    body_start = _true_body_start(canonical)
    if body_start is None:
        # `sections` is non-empty, so candidates must have existed when
        # `parse_sections` built it; this should be unreachable, but fall
        # back to the first accepted section rather than raising.
        body_start = sections[0].start
    match = _WITNESS_RE.search(canonical, body_start)
    body_end = match.start() if match else len(canonical)
    return body_start, body_end


@dataclass(frozen=True)
class StructuralMetrics:
    structural_coverage: float
    n_sections: int
    max_section_chars: int
    giant_single_section: bool
    body_start: int
    body_end: int


def compute_structural_metrics(canonical: str, sections: list[Section]) -> StructuralMetrics:
    """Structural coverage/giant-section metrics, defined over `sections` clamped to body.

    `structural_coverage` = fraction of body characters that fall inside a
    recognised, advancing section (spec §A.3). Sections are clamped to the
    body window so that exhibits/post-signature material attached to the
    final section cannot inflate coverage or trigger a false giant.
    """
    if not sections:
        return StructuralMetrics(0.0, 0, 0, False, 0, 0)
    body_start, body_end = compute_body_bounds(canonical, sections)
    body_len = body_end - body_start
    if body_len <= 0:
        return StructuralMetrics(0.0, 0, 0, False, body_start, body_end)
    clamped_lengths: list[int] = []
    covered = 0
    for section in sections:
        clamped_start = max(section.start, body_start)
        clamped_end = min(section.end, body_end)
        if clamped_end <= clamped_start:
            continue
        length = clamped_end - clamped_start
        clamped_lengths.append(length)
        covered += length
    n_sections = len(clamped_lengths)
    max_section_chars = max(clamped_lengths) if clamped_lengths else 0
    giant = max_section_chars > GIANT_SECTION_FRACTION * body_len
    coverage = covered / body_len
    return StructuralMetrics(coverage, n_sections, max_section_chars, giant, body_start, body_end)


def gold_span_section_breakdown(
    sections: list[Section],
    body: tuple[int, int],
    gold_ranges: list[tuple[int, int]],
) -> dict[str, int]:
    """Per-cause counts for where each gold span's midpoint falls, relative to
    the body window and the recognised section map.

    Mutually exclusive and exhaustive over `gold_ranges`:

      - ``hits``: midpoint inside body, inside a recognised NON-giant section
      - ``before_body_start``: midpoint < body_start
      - ``after_signature_block``: midpoint >= body_end
      - ``giant_section``: midpoint inside body, inside a recognised section
        that is itself a giant
      - ``unrecognised_gap``: midpoint inside body, but not inside any
        recognised section
      - ``total``: len(gold_ranges)

    ``hits / total`` reproduces `gold_span_section_rate`'s value -- this
    function is that metric's full causal decomposition, so the corpus-wide
    report can show *why* the rate is what it is (spec Definition of Done,
    amended 2026-09-04: gold_span_section_rate is report-only, and the
    report must carry the miss breakdown by cause).

    Pure, like `gold_span_section_rate`: no import of `select.py` (that would
    be a cycle) -- callers supply already-computed sections and gold ranges.
    """
    counts = {
        "hits": 0,
        "before_body_start": 0,
        "after_signature_block": 0,
        "giant_section": 0,
        "unrecognised_gap": 0,
        "total": len(gold_ranges),
    }
    if not gold_ranges:
        return counts
    body_start, body_end = body
    body_len = body_end - body_start
    if body_len <= 0:
        # Degenerate body: there is no valid body window to be "before" or
        # "inside" relative to, so every span counts as past it.
        counts["after_signature_block"] = len(gold_ranges)
        return counts
    clamped: list[tuple[int, int]] = []
    for section in sections:
        clamped_start = max(section.start, body_start)
        clamped_end = min(section.end, body_end)
        if clamped_end > clamped_start:
            clamped.append((clamped_start, clamped_end))
    giants = {(s, e) for s, e in clamped if (e - s) > GIANT_SECTION_FRACTION * body_len}
    for start, end in gold_ranges:
        mid = (start + end) // 2
        if mid < body_start:
            counts["before_body_start"] += 1
            continue
        if mid >= body_end:
            counts["after_signature_block"] += 1
            continue
        found = False
        for s, e in clamped:
            if s <= mid < e:
                if (s, e) in giants:
                    counts["giant_section"] += 1
                else:
                    counts["hits"] += 1
                found = True
                break
        if not found:
            counts["unrecognised_gap"] += 1
    return counts


def gold_span_section_rate(
    sections: list[Section],
    body: tuple[int, int],
    gold_ranges: list[tuple[int, int]],
) -> float | None:
    """Fraction of `gold_ranges` whose midpoint lies inside a recognised, non-giant section.

    Pure: takes already-computed sections, the (body_start, body_end)
    window, and a flat list of (start, end) gold-span ranges (already
    aligned, e.g. the union of the 12 questions' `AlignResult.ranges` for one
    agreement). This module must not import `select.py`, which is where
    alignment results are computed -- that would be a cycle -- so callers
    supply the ranges directly.

    Returns None when there are no gold ranges to evaluate (undefined rate),
    matching how the caller should treat "no data" differently from "0% in
    scope". Report-only per the amended spec Definition of Done -- this
    value must never be gated by a numeric threshold; see
    `gold_span_section_breakdown` for the causal decomposition that explains
    it.
    """
    if not gold_ranges:
        return None
    breakdown = gold_span_section_breakdown(sections, body, gold_ranges)
    return breakdown["hits"] / breakdown["total"]
