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
from pathlib import Path

from dealpoint.config import ARM_ORDER, DEFAULT_MODEL, REPORTS_DIR, WORKHORSE_MODEL
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


def run_headline_glm(model: str = "z-ai/glm-5.3-flash") -> list[dict]:
    from dealpoint.eval.subset import load_subset_cases

    case_rows = load_subset_cases("test_subset_v1")
    summaries = []
    for arm in ARM_ORDER:
        summary = run_eval_set(
            case_set=f"test_subset_v1_{arm}",
            arm=arm,
            model=model,
            case_rows=case_rows,
            milestone_tag="m4",
        )
        summaries.append(summary)
        _append_manifest(
            {
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
        )
    return summaries


# --- replication: Haiku, arms A/D, tranche_1 --------------------------------


def run_replication_haiku(model: str = DEFAULT_MODEL, tranche: int = 1) -> list[dict]:
    from dealpoint.eval.subset import load_subset_cases

    case_rows = load_subset_cases("test_subset_v1", tranche=tranche)
    summaries = []
    for arm in ("A", "D"):
        summary = run_eval_set(
            case_set=f"test_subset_v1_tranche{tranche}_{arm}",
            arm=arm,
            model=model,
            case_rows=case_rows,
            milestone_tag="m4",
        )
        summaries.append(summary)
        _append_manifest(
            {
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
        )
    return summaries


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
        "--skip-dev-smoke", action="store_true", help="skip the <=6-case wiring smoke"
    )
    parser.add_argument("--skip-caching-probe", action="store_true")
    parser.add_argument("--skip-headline", action="store_true")
    parser.add_argument("--skip-replication", action="store_true")
    parser.add_argument("--skip-publish", action="store_true")
    args = parser.parse_args(argv)

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
