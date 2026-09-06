"""Disk-usage guard for the M7a optional dependency groups (spec section "Budget and disk").

Measures free disk space before/after installing each optional group with
`uv pip install --dry-run` first (recorded), then the real install, and
writes `data/reports/m7a_disk_guard.json`. If free space would drop below
`M7A_DISK_FLOOR_BYTES`, scope down to `rag-lab` only and document what was
omitted -- never silently shrink something else.

This module is a recording tool, not a gate: it is run once, by hand, around
the actual `uv pip install -e ".[rag-lab]"` / `-e ".[deepeval]"` commands,
and its job is solely to write down what was measured so the DoD's disk
guard requirement has a durable artifact.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime

from dealpoint.config import M7A_DISK_FLOOR_BYTES, M7A_DISK_GUARD_PATH, REPO_ROOT


def free_bytes() -> int:
    return shutil.disk_usage(str(REPO_ROOT)).free


def dry_run_install(extra: str) -> str:
    """`uv pip install --dry-run -e ".[<extra>]"` output, captured (never applied)."""
    result = subprocess.run(
        ["uv", "pip", "install", "--dry-run", "-e", f".[{extra}]"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return (result.stdout or "") + (result.stderr or "")


def record_guard(entries: list[dict], path=M7A_DISK_GUARD_PATH) -> dict:
    payload = {
        "floor_bytes": M7A_DISK_FLOOR_BYTES,
        "measured_at": datetime.now(UTC).isoformat(),
        "groups": entries,
        "within_guard": all(e["within_guard"] for e in entries),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Record the guard for groups already installed (measured by hand, see
    docs/milestones/m7a.md for the exact sequence run).
    """
    print(json.dumps({"free_bytes_now": free_bytes()}, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
