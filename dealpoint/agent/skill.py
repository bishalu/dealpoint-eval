"""The M4 skill: `SKILL.md` + per-question playbooks, injected in arm D only.

Pure, no I/O beyond reading the committed files under `skills/`
(`dealpoint.config.SKILL_DIR`). `skill_block()` is the only function that
composes the skill into a prompt -- and `dealpoint.agent.loop` is the only
caller, so `dealpoint.agent.prompts` stays free of skill content (spec
deliverable 1; see `tests/test_prompts.py`).
"""

from __future__ import annotations

import hashlib
import json

from dealpoint.config import SKILL_DIR, SKILL_PLAYBOOKS_DIR, VERSIONS_JSON_PATH

SKILL_MD_PATH = SKILL_DIR / "SKILL.md"


def _skill_files() -> list:
    """`SKILL.md` + every `playbooks/*.md`, sorted by path relative to `SKILL_DIR`."""
    paths = [SKILL_MD_PATH, *sorted(SKILL_PLAYBOOKS_DIR.glob("*.md"))]
    return [p for p in paths if p.exists()]


def skill_version() -> str:
    """12-char sha256 over `SKILL.md` + every `playbooks/*.md`, sorted by path.

    Hashes `(relative_path, bytes)` pairs, one line per file, so the hash
    changes if any file's content OR set of files changes. Pure -- reads the
    committed files but performs no writes.
    """
    hasher = hashlib.sha256()
    for path in _skill_files():
        rel = path.relative_to(SKILL_DIR).as_posix()
        content = path.read_bytes()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(content)
        hasher.update(b"\x00")
    return hasher.hexdigest()[:12]


def load_skill() -> str:
    """The generic `SKILL.md` procedure text."""
    return SKILL_MD_PATH.read_text(encoding="utf-8")


def load_playbook(question_id: str) -> str | None:
    """The 3-5 line playbook for `question_id`, or `None` for out-of-scope questions."""
    path = SKILL_PLAYBOOKS_DIR / f"{question_id}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def skill_block(question_id: str) -> str:
    """`SKILL.md` + the ONE applicable playbook, as a system-prompt block (arm D only).

    Loads no other playbook. For an out-of-scope question (no playbook file)
    the block is `SKILL.md` alone.
    """
    parts = [load_skill()]
    playbook = load_playbook(question_id)
    if playbook is not None:
        parts.append(playbook)
    return "\n\n".join(parts)


def stamp_skill_version() -> None:
    """Merge-write `skill_version` into `data/reports/versions.json`.

    Does not rebuild the index or touch any other key. Creates the file
    (with just this key) if it does not exist yet.
    """
    existing: dict = {}
    if VERSIONS_JSON_PATH.exists():
        try:
            existing = json.loads(VERSIONS_JSON_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing["skill_version"] = skill_version()
    VERSIONS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VERSIONS_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")
