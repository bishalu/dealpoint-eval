"""Orchestrates the M4 metered sweeps (spec deliverable 5): dev smoke, prompt-
caching probe, the GLM headline (arms A-D, full frozen subset), and the Haiku
replication (arms A/D, tranche_1). Each metered leg appends one entry to
`data/reports/four_arm_manifest.json`, which `dealpoint.eval.report` reads
instead of globbing `data/results/` (spec §6: "prefer an explicit manifest,
it is less fragile").

Run order matters (cheapest/most informative first) -- see `main()`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from dealpoint.config import (
    ARM_ORDER,
    DEFAULT_MODEL,
    FOUR_ARM_JSON_PATH,
    FOUR_ARM_MANIFEST_V1_PATH,
    FOUR_ARM_MD_PATH,
    FOUR_ARM_V1_JSON_PATH,
    FOUR_ARM_V1_MD_PATH,
    M4_1_MILESTONE_TAG,
    REPORTS_DIR,
    WORKHORSE_MODEL,
)
from dealpoint.eval.run import run_eval_set

MANIFEST_PATH = REPORTS_DIR / "four_arm_manifest.json"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_manifest(entry: dict, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


# --- step: dev smoke (spec corrective plan §0.2) ----------------------------


def run_dev_smoke(
    model: str = WORKHORSE_MODEL, arms: tuple[str, ...] = ("C", "D"), limit: int = 3
) -> list[dict]:
    """<= 6 dev cases at the workhorse model, arms C and D -- proves the new

    wiring works end to end before any frozen-subset case runs. NOT a
    reported result (recorded under "dev-loop spend" in the report instead).
    """
    summaries = []
    for arm in arms:
        summaries.append(
            run_eval_set(case_set="dev", arm=arm, model=model, limit=limit, milestone_tag="m4")
        )
    return summaries


# --- step 0: diagnose prompt caching (spec §5 step 0) -----------------------


def run_caching_probe(model: str = DEFAULT_MODEL) -> dict:
    """Run 1 dev case twice at `model`; read `cached_tokens` from the new ledger rows."""
    from dealpoint.eval.spend import read_ledger

    before = len(read_ledger())
    run_eval_set(case_set="dev", arm="B", model=model, limit=1, milestone_tag="m4")
    run_eval_set(case_set="dev", arm="B", model=model, limit=1, milestone_tag="m4")
    rows = read_ledger()[before:]
    cached_tokens = [r.get("cached_tokens", 0) for r in rows]
    input_tokens = [r.get("input_tokens", 0) for r in rows]
    usd = [r.get("usd", 0.0) for r in rows]
    return {
        "model": model,
        "n_calls": len(rows),
        "cached_tokens": cached_tokens,
        "any_cached": any((c or 0) > 0 for c in cached_tokens),
        "measured_static_prefix_input_tokens": input_tokens,
        "cost_per_call_usd": usd,
    }


# --- headline: GLM, arms A-D, full frozen subset ----------------------------


def run_headline_glm(
    model: str = "z-ai/glm-5.3-flash",
    *,
    milestone_tag: str = "m4",
    version: str = "v1",
) -> list[dict]:
    from dealpoint.eval.subset import load_subset_cases

    case_rows = load_subset_cases("test_subset_v1")
    summaries = []
    for arm in ARM_ORDER:
        summary = run_eval_set(
            case_set=f"test_subset_v1_{arm}",
            arm=arm,
            model=model,
            case_rows=case_rows,
            milestone_tag=milestone_tag,
        )
        summaries.append(summary)
        entry = {
            "leg": "headline_glm",
            "arm": arm,
            "model": model,
            "tranche": "full",
            "n_cases": summary["n_cases"],
            "est_usd": summary["est_usd"],
            "realized_usd": summary["realized_usd"],
            "est_basis": summary["est_basis"],
            "results_path": summary["results_path"],
            "summary_path": summary["summary_path"],
            "git_sha7": summary["git_sha7"],
            "index_version": summary["index_version"],
        }
        if version != "v1":
            entry["version"] = version
            entry["milestone"] = milestone_tag
        _append_manifest(entry)
    return summaries


# --- replication: Haiku, arms A/D, tranche_1 --------------------------------


def run_replication_haiku(
    model: str = DEFAULT_MODEL,
    tranche: int = 1,
    *,
    arms: tuple[str, ...] = ("A", "D"),
    milestone_tag: str = "m4",
    version: str = "v1",
) -> list[dict]:
    from dealpoint.eval.subset import load_subset_cases

    case_rows = load_subset_cases("test_subset_v1", tranche=tranche)
    summaries = []
    for arm in arms:
        summary = run_eval_set(
            case_set=f"test_subset_v1_tranche{tranche}_{arm}",
            arm=arm,
            model=model,
            case_rows=case_rows,
            milestone_tag=milestone_tag,
        )
        summaries.append(summary)
        entry = {
            "leg": "replication_haiku",
            "arm": arm,
            "model": model,
            "tranche": f"tranche_{tranche}",
            "n_cases": summary["n_cases"],
            "est_usd": summary["est_usd"],
            "realized_usd": summary["realized_usd"],
            "est_basis": summary["est_basis"],
            "results_path": summary["results_path"],
            "summary_path": summary["summary_path"],
            "git_sha7": summary["git_sha7"],
            "index_version": summary["index_version"],
        }
        if version != "v1":
            entry["version"] = version
            entry["milestone"] = milestone_tag
        _append_manifest(entry)
    return summaries


# --- M4.1: preserve v1 artefacts, then run the v2 sweeps --------------------


def preserve_v1_artifacts() -> dict:
    """Copy the live v1 report/manifest files to their `..._v1_execution_defects`
    / `..._manifest_v1` names, then remove the live manifest so the v2 run
    starts a fresh one (spec deliverable 4). Copies (not moves) the report
    JSON/MD -- `four_arm.json`/`four_arm.md` are regenerated for v2 by
    `dealpoint.eval.report`, so the originals are safe to leave in place
    until that regeneration overwrites them.

    Idempotent: skips a copy whose source is already absent (e.g. re-running
    this after a partial v2 attempt), and never overwrites an already-
    preserved v1 destination file.
    """
    copied: list[str] = []
    skipped: list[str] = []
    for src, dst in (
        (FOUR_ARM_JSON_PATH, FOUR_ARM_V1_JSON_PATH),
        (FOUR_ARM_MD_PATH, FOUR_ARM_V1_MD_PATH),
        (MANIFEST_PATH, FOUR_ARM_MANIFEST_V1_PATH),
    ):
        if dst.exists():
            skipped.append(str(dst))
            continue
        if not src.exists():
            skipped.append(str(src))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst))

    # Fresh manifest for v2 -- if the live manifest still has v1 legs in it,
    # dealpoint.eval.report.load_rows_from_manifest concatenates v1 and v2
    # rows into the same (model, arm) key and every number is wrong.
    manifest_removed = False
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
        manifest_removed = True

    return {"copied": copied, "skipped": skipped, "manifest_removed": manifest_removed}


def run_v2(
    glm_model: str = "z-ai/glm-5.3-flash",
    haiku_model: str = DEFAULT_MODEL,
    *,
    include_haiku_arm_a: bool = True,
) -> dict:
    """v1 preserved, then the v2 headline (GLM A-D, full 32) + replication
    (Haiku D, and Haiku A unless dropped for budget) sweeps, tagged
    `milestone_tag="m4_1"` (spec deliverable 4).

    `include_haiku_arm_a=False` drops the Haiku arm-A re-run first if the
    budget projection would breach the M4.1 absolute -- arm A at Haiku was
    already healthy in v1 (5.6% EXECUTION_FAILED); arm D is the leg this
    milestone exists to fix and is never dropped.
    """
    preserve_info = preserve_v1_artifacts()
    headline = run_headline_glm(glm_model, milestone_tag=M4_1_MILESTONE_TAG, version="v2")
    haiku_arms: tuple[str, ...] = ("A", "D") if include_haiku_arm_a else ("D",)
    replication = run_replication_haiku(
        haiku_model, tranche=1, arms=haiku_arms, milestone_tag=M4_1_MILESTONE_TAG, version="v2"
    )
    return {
        "preserve_v1": preserve_info,
        "headline_glm": headline,
        "replication_haiku": replication,
    }


# --- publish every manifest leg to Braintrust -------------------------------


def publish_to_braintrust(manifest_path: Path = MANIFEST_PATH) -> list[dict]:
    from dealpoint.eval.braintrust_adapter import braintrust_available, run_eval

    if not braintrust_available():
        return []
    results = []
    for entry in load_manifest(manifest_path):
        rows = _read_jsonl(Path(entry["results_path"]))
        if not rows:
            continue
        res = run_eval(
            rows,
            arm=entry["arm"],
            model=entry["model"],
            case_set=f"m4_{entry['leg']}_{entry['tranche']}",
            no_send_logs=False,
        )
        results.append({"leg": entry["leg"], "arm": entry["arm"], **res})
    return results


def main(argv: list[str] | None = None) -> int:
    """Run the full M4 metered phase in spec order and print a summary."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dealpoint.eval.four_arm_sweep")
    parser.add_argument(
        "--v2",
        action="store_true",
        help="M4.1: preserve v1 artefacts, then run the v2 re-sweep (milestone_tag=m4_1) "
        "instead of the original M4 phases",
    )
    parser.add_argument(
        "--skip-dev-smoke", action="store_true", help="skip the <=6-case wiring smoke"
    )
    parser.add_argument("--skip-caching-probe", action="store_true")
    parser.add_argument("--skip-headline", action="store_true")
    parser.add_argument("--skip-replication", action="store_true")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument(
        "--drop-haiku-arm-a",
        action="store_true",
        help="--v2 only: drop the Haiku arm-A re-run first if budget requires it",
    )
    args = parser.parse_args(argv)

    if args.v2:
        out = run_v2(include_haiku_arm_a=not args.drop_haiku_arm_a)
        print(
            json.dumps(
                {
                    "preserve_v1": out["preserve_v1"],
                    "headline_glm": len(out["headline_glm"]),
                    "replication_haiku": len(out["replication_haiku"]),
                }
            )
        )
        return 0

    out: dict = {}
    if not args.skip_dev_smoke:
        out["dev_smoke"] = run_dev_smoke()
    if not args.skip_caching_probe:
        out["caching_probe"] = run_caching_probe()
    if not args.skip_headline:
        out["headline_glm"] = run_headline_glm()
    if not args.skip_replication:
        out["replication_haiku"] = run_replication_haiku()
    if not args.skip_publish:
        out["braintrust"] = publish_to_braintrust()

    print(json.dumps({k: (v if k == "caching_probe" else len(v)) for k, v in out.items()}))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
