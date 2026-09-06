"""`just demo-manifest-record --step N --note ...` -- append one cockpit
session step's result (spec `m7b.md` section 7) into
`data/reports/demo_manifest.json`'s `cockpit_session` list. Never touches
any other manifest key; run AFTER `just braintrust-cockpit` has written the
manifest at least once.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from dealpoint.config import DEMO_MANIFEST_PATH


def record_step(manifest: dict, *, step: int, note: str, url: str | None, tv: str | None) -> dict:
    entries = list(manifest.get("cockpit_session", []))
    entries = [e for e in entries if e.get("step") != step]  # a re-run of a step replaces it, never duplicates
    entries.append(
        {
            "step": step,
            "note": note,
            "url": url,
            "tv": tv,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    )
    entries.sort(key=lambda e: e["step"])
    manifest = dict(manifest)
    manifest["cockpit_session"] = entries
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--note", type=str, required=True)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--tv", type=str, default=None)
    args = parser.parse_args(argv)

    if not DEMO_MANIFEST_PATH.exists():
        print(f"{DEMO_MANIFEST_PATH} does not exist -- run `just braintrust-cockpit` first", file=sys.stderr)
        return 1

    manifest = json.loads(DEMO_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = record_step(manifest, step=args.step, note=args.note, url=args.url, tv=args.tv)
    with open(DEMO_MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        fh.write("\n")
    print(f"recorded step {args.step} in {DEMO_MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
