"""Canonical document loader: text + section map + defined-term lookup.

A `Document` wraps one canonical text file under `data/derived/canonical/`
(either a base contract, `contract_0`, or a redacted counterfactual variant,
`contract_103__redacted_q09`) together with its section map under
`data/derived/sections/`. All offsets returned by this module index into
`Document.text`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from dealpoint.config import CANONICAL_DIR, MAX_DEFINITION_CHARS, SECTIONS_DIR
from dealpoint.data.sections import Section, check_parser_version

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(token: str) -> int | None:
    token = token.upper()
    if not token or any(ch not in _ROMAN_VALUES for ch in token):
        return None
    total = 0
    prev = 0
    for ch in reversed(token):
        value = _ROMAN_VALUES[ch]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total


@dataclass(frozen=True)
class Document:
    document_id: str  # "contract_0" or "contract_103__redacted_q09"
    agreement_id: str  # base contract id: document_id.split("__")[0]
    text: str  # canonical text; all offsets index into this string
    sections: list[Section]
    body_start: int | None  # absent for redacted variants
    body_end: int | None


def load_document(document_id: str) -> Document:
    """Load a canonical document (original or redacted variant) with its section map.

    Raises `FileNotFoundError` naming `just data` (or `just index` for a
    redacted variant, which is also produced by `just data`) if the derived
    files are absent, and `StaleDerivedDataError` if the on-disk section map
    was built by a different parser version.
    """
    text_path = CANONICAL_DIR / f"{document_id}.txt"
    sections_path = SECTIONS_DIR / f"{document_id}.json"
    if not text_path.exists() or not sections_path.exists():
        raise FileNotFoundError(
            f"derived data for {document_id!r} not found under {CANONICAL_DIR} / "
            f"{SECTIONS_DIR}. Run `just data` to build it."
        )
    text = text_path.read_text(encoding="utf-8")
    with open(sections_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    check_parser_version(sections_path, payload.get("parser_version"))

    sections = [
        Section(ref=s["ref"], title=s["title"], start=s["start"], end=s["end"])
        for s in payload["sections"]
    ]
    agreement_id = document_id.split("__")[0]
    return Document(
        document_id=document_id,
        agreement_id=agreement_id,
        text=text,
        sections=sections,
        body_start=payload.get("body_start"),
        body_end=payload.get("body_end"),
    )


# --- get_section -------------------------------------------------------

_REF_PREFIX_RE = re.compile(r"^(article|section|sec\.?|§)\s*", re.IGNORECASE)
_TRAILING_PARENS_RE = re.compile(r"(\([^)]*\))+\s*$")
_SECTION_DOTTED_RE = re.compile(r"^\d{1,2}\.\d{1,3}$")
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_BARE_NUM_RE = re.compile(r"^\d{1,3}$")


def _normalise_ref(ref: str) -> tuple[str, object] | None:
    """Normalise a section reference to a comparable (kind, value) key.

    kind is "section" with value (article:int, section:int), "article" with
    value int, or "bare" (an ambiguous bare number with no keyword prefix,
    treated as article-equivalent by `_refs_match`). Returns None for input
    that cannot be parsed as any of these — the caller must treat that as
    "no match", never raise.
    """
    s = ref.strip()
    if not s:
        return None
    kind_hint: str | None = None
    m = _REF_PREFIX_RE.match(s)
    if m:
        kw = m.group(1).lower()
        kind_hint = "article" if kw.startswith("article") else "section"
        s = s[m.end() :]
    s = _TRAILING_PARENS_RE.sub("", s).strip()
    if not s:
        return None
    if _SECTION_DOTTED_RE.match(s):
        n, mnum = s.split(".")
        return ("section", (int(n), int(mnum)))
    if _ROMAN_RE.match(s):
        value = _roman_to_int(s)
        if value is None:
            return None
        return ("article", value)
    if _BARE_NUM_RE.match(s):
        if kind_hint == "article":
            return ("article", int(s))
        if kind_hint == "section":
            return None  # "Section 6" with no minor number is not a valid ref
        return ("bare", int(s))
    return None


def _refs_match(a: tuple[str, object] | None, b: tuple[str, object] | None) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    # A bare number (no keyword prefix) is treated as article-equivalent, so
    # a stored top-level-fallback ref ("1") matches a model query of
    # "Article 1" and vice versa.
    kinds = {a[0], b[0]}
    return kinds == {"bare", "article"} and a[1] == b[1]


def get_section(doc: Document, ref: str) -> Section | None:
    """Look up a section by a model-supplied reference, tolerant of common forms.

    Accepts "6.3", "Section 6.3", "§ 6.3", "Section 6.03", "6.3(b)",
    "Section 6.3(b)(ii)", "Article VI", "ARTICLE 6", mixed case and stray
    whitespace. Returns None on no match — never raises, never guesses a
    neighbour.
    """
    target = _normalise_ref(ref)
    if target is None:
        return None
    for section in doc.sections:
        if _refs_match(_normalise_ref(section.ref), target):
            return section
    return None


# --- defined_term --------------------------------------------------------

_DEFINITION_VERBS = r"(?:means|shall\s+mean|shall\s+have\s+the\s+meaning|has\s+the\s+meaning|have\s+the\s+meaning)"
_PREFIX_WORDS = (
    r"(?:Company|Parent|Target|Purchaser|Buyer|Seller|Acquiror|Acquirer|"
    r"Merger\s+Sub|Surviving\s+Corporation)"
)
_PREFIX_PATTERN = rf"(?:{_PREFIX_WORDS}\s+){{0,2}}"

_GENERIC_DEFINITION_RE = re.compile(
    rf'["\'][^"\']{{1,80}}["\']\s*{_DEFINITION_VERBS}\b', re.IGNORECASE
)


@dataclass(frozen=True)
class DefinedTerm:
    term_as_written: str  # e.g. "Company Material Adverse Effect"
    section_ref: str  # "" if the definition falls outside any recognised section
    start: int
    end: int
    text: str


def _prefix_rank(prefix: str) -> int:
    prefix = prefix.strip().lower()
    if not prefix:
        return 0
    if prefix.startswith("company"):
        return 1
    if prefix.startswith("parent"):
        return 3
    return 2


def _enclosing_section(sections: list[Section], offset: int) -> Section | None:
    best: Section | None = None
    for section in sections:
        if section.start <= offset and (best is None or section.start > best.start):
            best = section
    if best is not None and offset < best.end:
        return best
    return best  # may be None, or a section whose window the offset trails past


def defined_term(doc: Document, term: str) -> DefinedTerm | None:
    """Return the `"Term" means ...` block for `term`, tolerant of quote style,
    case, and a `Company`/`Parent`/... prefix (e.g. `Company Material Adverse
    Effect`). When both a bare/exact and a prefixed form exist, the bare form
    ranks first, then `Company`-style, then any other prefix, `Parent`-style
    last; ties break on lowest offset for determinism.
    """
    term_escaped = re.escape(term.strip())
    if not term_escaped:
        return None
    pattern = re.compile(
        rf'["\'](?P<prefix>{_PREFIX_PATTERN})(?P<term>{term_escaped})["\']\s*{_DEFINITION_VERBS}\b',
        re.IGNORECASE,
    )
    candidates: list[tuple[int, int, re.Match]] = []
    for m in pattern.finditer(doc.text):
        rank = _prefix_rank(m.group("prefix"))
        candidates.append((rank, m.start(), m))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    _, start, match = candidates[0]
    phrase = match.group("prefix") + match.group("term")

    next_start = len(doc.text)
    generic_match = _GENERIC_DEFINITION_RE.search(doc.text, match.end())
    if generic_match is not None:
        next_start = generic_match.start()

    section = _enclosing_section(doc.sections, start)
    section_end = section.end if section is not None else len(doc.text)
    section_ref = section.ref if section is not None else ""

    end = min(next_start, start + MAX_DEFINITION_CHARS, section_end)
    end = max(end, match.end())  # never shorter than the matched phrase itself

    return DefinedTerm(
        term_as_written=phrase,
        section_ref=section_ref,
        start=start,
        end=end,
        text=doc.text[start:end],
    )
