"""Canonicalisation of raw contract text and model-quote normalisation.

All offsets everywhere in this project index into the string produced by
``canonicalise``. Do not vary the pipeline below: it is what produced the
alignment figures measured during recon (see specs/§0.2 C3-C5, §3.2).
"""

from __future__ import annotations

import re
import unicodedata

# Curly quotes -> ASCII, en/em dash and minus sign -> hyphen, nbsp -> space,
# zero-width space -> removed.
_TRANSLATE_MAP = {
    0x2018: "'",
    0x2019: "'",
    0x201C: '"',
    0x201D: '"',
    0x2013: "-",
    0x2014: "-",
    0x2212: "-",
    0xA0: " ",
    0x200B: None,
}

_WHITESPACE_RE = re.compile(r"\s+")


def canonicalise(raw: str) -> str:
    """Canonicalise raw contract or model-quote text.

    Exact, ordered steps:
      1. strip a leading BOM (``\\ufeff``)
      2. ``unicodedata.normalize("NFKC", s)``
      3. translate curly quotes / dashes / nbsp / zero-width space to ASCII
      4. normalise ``\\r\\n`` and ``\\r`` to ``\\n``
      5. collapse all whitespace runs (including newlines) to a single space
      6. strip leading/trailing whitespace
    """
    s = raw.lstrip("\ufeff")
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_TRANSLATE_MAP)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WHITESPACE_RE.sub(" ", s)
    return s.strip()


def normalise_quote(s: str) -> str:
    """Alias of ``canonicalise``; applied to model output before the verbatim check."""
    return canonicalise(s)
