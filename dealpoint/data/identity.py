"""Best-effort agreement party-name / title parsing (spec C2).

Party names are NOT reliably on line 1 of the raw contract text: the large
majority of files start with an exhibit label (``Exhibit 2.1``, ``EX-2.1``,
``Exhibit (d)(10)``, etc.) rather than the agreement title. This module
applies a tolerant regex over the first 20k canonical characters to recover
the agreement title and the "by and among"/"by and between" party list.

This is deliberately best-effort: the build must never fail because a name
could not be parsed. Any of the 20 *selected* agreements the regex misses (or
mis-parses) get a hand-checked entry in the committed override file
``data/agreement_names.json`` (see ``load_agreement_name``).
"""

from __future__ import annotations

import json
import re

from dealpoint.config import AGREEMENT_NAMES_PATH

_TITLE_RE = re.compile(
    r"(AMENDED AND RESTATED AGREEMENT AND PLAN OF MERGER|"
    r"AGREEMENT AND PLAN OF MERGER AND REORGANIZATION|"
    r"AGREEMENT AND PLAN OF REORGANIZATION|"
    r"AGREEMENT AND PLAN OF MERGER|"
    r"PLAN AND AGREEMENT OF MERGER|"
    r"TRANSACTION AGREEMENT|"
    r"AGREEMENT OF MERGER)",
    re.IGNORECASE,
)
_CONNECTOR_RE = re.compile(r"\b(?:by and among|by and between|among|between)\b\s*:?\s*", re.IGNORECASE)
_STOP_RE = re.compile(
    r"\bDated\s+as\s+of\b|\bdated\s+as\s+of\b|\bDATED\s+AS\s+OF\b|WHEREAS|RECITALS|"
    r"NOW,?\s*THEREFORE|\(the\s|\(this\s|This\s+AGREEMENT|This\s+Agreement|"
    r"TABLE OF CONTENTS|_{5,}|ARTICLE\s+[IVX1]|Article\s+[IVX1]",
)

_SEARCH_WINDOW_CHARS = 20_000
_PARTY_WINDOW_CHARS = 3_000
_PARTY_TAIL_CHARS = 500


def parse_agreement_title_and_parties(canonical: str) -> tuple[str, str] | None:
    """Return (title, party_text) parsed from the first 20k canonical chars, or None."""
    window = canonical[:_SEARCH_WINDOW_CHARS]
    title_match = _TITLE_RE.search(window)
    if not title_match:
        return None
    title = title_match.group(0)
    rest = window[title_match.end() : title_match.end() + _PARTY_WINDOW_CHARS]
    connector_match = _CONNECTOR_RE.search(rest)
    if not connector_match:
        return None
    tail = rest[connector_match.end() : connector_match.end() + _PARTY_TAIL_CHARS]
    stop_match = _STOP_RE.search(tail)
    party_text = tail[: stop_match.start()] if stop_match else tail[:300]
    party_text = party_text.strip(" ,.")
    if len(party_text) < 5:
        return None
    return title, party_text


def _load_overrides() -> dict[str, dict[str, str]]:
    if not AGREEMENT_NAMES_PATH.exists():
        return {}
    with open(AGREEMENT_NAMES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_agreement_name(agreement_id: str, canonical: str) -> dict[str, str]:
    """Return the agreement's title/parties dict, preferring the committed override.

    Falls back to the regex parse, and finally to a placeholder that never
    raises -- a missing/unparseable name must not fail the build (spec C2).
    """
    overrides = _load_overrides()
    if agreement_id in overrides:
        return overrides[agreement_id]
    parsed = parse_agreement_title_and_parties(canonical)
    if parsed is not None:
        title, parties = parsed
        return {"title": title, "parties": parties}
    return {"title": "", "parties": ""}
