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


def record_guard(entries: list[dict], path=M7A_DISK_GUARD_PATH, install_time_observation: dict | None = None) -> dict:
    payload = {
        "floor_bytes": M7A_DISK_FLOOR_BYTES,
        "groups": entries,
        "install_time_observation": install_time_observation,
        "measured_at": datetime.now(UTC).isoformat(),
        "within_guard": all(e["within_guard"] for e in entries),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Re-measure the disk guard for the already-installed optional groups.

    Both groups are already installed, so `uv pip install --dry-run` reports
    them as already satisfied -- that is recorded honestly under
    `re_measured_at` per group, alongside the dry-run stdout/stderr. The
    original install-time free-space deltas (measured 2026-09-05, before
    either group was installed) are kept separately under
    `install_time_observation` rather than being passed off as this
    measurement.
    """
    entries = []
    for group in ("rag-lab", "deepeval"):
        free_before = free_bytes()
        output = dry_run_install(group)
        free_after = free_bytes()
        entries.append(
            {
                "dry_run_output": output.strip(),
                "free_after_bytes": free_after,
                "free_before_bytes": free_before,
                "group": group,
                "re_measured_at": datetime.now(UTC).isoformat(),
                "within_guard": min(free_before, free_after) >= M7A_DISK_FLOOR_BYTES,
            }
        )
    install_time_observation = {
        "note": (
            "Deltas measured 2026-09-05 during the original live install, before "
            "either optional group was present. `.venv` grew from 340 MB to 451 MB "
            "across both groups combined; neither pulled torch/transformers/nvidia "
            "wheels. Kept for historical record only -- see each group's "
            "`re_measured_at` entry above for the current (already-installed) state."
        ),
        "rag_lab": {"free_after_bytes": 1254531072, "free_before_bytes": 1257095168},
        "deepeval": {"free_after_bytes": 1252827136, "free_before_bytes": 1254473728},
    }
    payload = record_guard(entries, install_time_observation=install_time_observation)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["within_guard"] else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
