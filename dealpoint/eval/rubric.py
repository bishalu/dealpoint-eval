"""The M5 judge rubric: `eval/judges/rubrics.md`, hashed and frozen before any judge call.

Mirrors `dealpoint.agent.skill`'s house style: a pure `rubric_text()` /
`rubric_version()` pair plus a merge-write `stamp_rubric_version()`. The
freezing order is non-negotiable (spec §3.3): `stamp_rubric_version()` must
run, and `data/reports/versions.json` must carry the resulting hash, before
the first judge call is made.
"""

from __future__ import annotations

import hashlib
import json

from dealpoint.config import RUBRICS_PATH, VERSIONS_JSON_PATH


def rubric_text() -> str:
    """The frozen rubric file's text."""
    return RUBRICS_PATH.read_text(encoding="utf-8")


def rubric_version() -> str:
    """12-char sha256 of the rubric file's raw bytes."""
    data = RUBRICS_PATH.read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]


def stamp_rubric_version() -> None:
    """Merge-write `rubric_version` into `data/reports/versions.json`.

    Does not touch any other key. Creates the file (with just this key) if
    it does not exist yet -- same convention as
    `dealpoint.agent.skill.stamp_skill_version`.
    """
    existing: dict = {}
    if VERSIONS_JSON_PATH.exists():
        try:
            existing = json.loads(VERSIONS_JSON_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing["rubric_version"] = rubric_version()
    VERSIONS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VERSIONS_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")
