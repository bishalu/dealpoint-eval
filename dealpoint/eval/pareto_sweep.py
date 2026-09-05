"""M6 arm-D sweeps, per model, on the frozen 32-case subset (spec deliverable 2).

Writes to `data/reports/pareto_manifest.json` -- a SEPARATE manifest from
`four_arm_manifest.json`, never appended to (M4's report and M5's judged
variants both resolve their result-file paths from the four-arm manifest;
adding M6 rows there would silently change both).

Haiku and GLM are reused from M4/M4.1, never re-run: `reused_legs()` builds
manifest-shaped entries for both, resolved from `four_arm_manifest.json`
(arm D, `version == "v2"`), so the report can tell M6's own new spend apart
from already-paid-for evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from dealpoint.config import (
    M6_STOP_USD,
    PARETO_MANIFEST_PATH,
)
from dealpoint.eval.cases import slugify_model
from dealpoint.eval.spend import PARETO_NEW_MODELS, PARETO_STRETCH_MODELS

FROZEN_INDEX_VERSION = "e2b4a2b97561"

# Rank offset for stretch candidates (they come after the 5 core models in
# dealpoint.eval.pareto_slate.PARETO_CANDIDATES).
_ALL_SWEEP_ORDER: tuple[str, ...] = (*PARETO_NEW_MODELS, *PARETO_STRETCH_MODELS)


def load_pareto_manifest(path: Path = PARETO_MANIFEST_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _append_manifest(entry: dict, path: Path = PARETO_MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_pareto_manifest(path)
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reused_legs() -> list[dict]:
    """Manifest-shaped entries for Haiku (18 cases, tranche_1) and GLM (32
    cases, full), resolved from `four_arm_manifest.json`'s arm-D, v2 rows.
    """
    from dealpoint.eval.four_arm_sweep import MANIFEST_PATH as FOUR_ARM_MANIFEST_PATH
    from dealpoint.eval.four_arm_sweep import load_manifest as load_four_arm_manifest

    four_arm = load_four_arm_manifest(FOUR_ARM_MANIFEST_PATH)
    out: list[dict] = []
    for entry in four_arm:
        if entry.get("arm") != "D" or entry.get("version") != "v2":
            continue
        out.append(
            {
                "leg": "pareto",
                "arm": "D",
                "model": entry["model"],
                "n_cases": entry["n_cases"],
                "est_usd": entry.get("est_usd"),
                "est_basis": entry.get("est_basis"),
                "realized_usd": entry.get("realized_usd"),
                "results_path": entry.get("results_path"),
                "summary_path": entry.get("summary_path"),
                "git_sha7": entry.get("git_sha7"),
                "index_version": entry.get("index_version"),
                "milestone": entry.get("milestone", "m4_1"),
                "rank": 0,
                "reused": True,
            }
        )
    return out


def run_model_sweep(model: str, *, milestone_tag: str = "m6") -> dict:
    """Sweep one new model, arm D, over the full frozen 32-case subset.

    Skips (resume) if a manifest entry for `model` already has a
    `results_path` on disk. Asserts the post-probe per-case basis is
    `ledger:measured(arm,model)` -- if it is not, the probe did not run and
    this model is skipped with a recorded reason rather than swept on a
    guessed price. Reserves the judging cost for this variant
    (`18 * 3 * measured_judge_call_usd`) before starting, on top of the
    sweep's own cap/absolute/stop-floor checks, so a model is never swept
    and then left unjudged.
    """
    from dealpoint.eval.judge_run import measured_judge_call_usd
    from dealpoint.eval.run import _assert_within_milestone_absolute, run_eval_set
    from dealpoint.eval.spend import assert_within_cap, per_case_usd, realized_usd
    from dealpoint.eval.subset import load_subset_cases

    existing = load_pareto_manifest()
    for entry in existing:
        if entry.get("model") == model and entry.get("arm") == "D" and not entry.get("reused"):
            results_path = entry.get("results_path")
            if results_path and Path(results_path).exists():
                return {"model": model, "status": "already_swept", "entry": entry}

    per_case, basis = per_case_usd("D", model)
    if basis != "ledger:measured(arm,model)":
        entry = {
            "leg": "pareto",
            "arm": "D",
            "model": model,
            "status": "skipped (no measured per-case cost -- probe did not run)",
            "est_basis": basis,
            "reused": False,
        }
        _append_manifest(entry)
        return {"model": model, "status": "skipped", "reason": entry["status"], "entry": entry}

    n_cases = 32
    est = per_case * n_cases

    try:
        assert_within_cap(est)
        _assert_within_milestone_absolute(est)
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        entry = {
            "leg": "pareto",
            "arm": "D",
            "model": model,
            "status": f"stopped (spend guard: {exc})",
            "est_usd": round(est, 6),
            "est_basis": basis,
            "reused": False,
        }
        _append_manifest(entry)
        return {"model": model, "status": "stopped", "reason": str(exc), "entry": entry}

    judge_call_usd, _judge_basis = measured_judge_call_usd()
    judge_reserve = 18 * 3 * judge_call_usd
    if realized_usd() + est + judge_reserve > M6_STOP_USD:
        entry = {
            "leg": "pareto",
            "arm": "D",
            "model": model,
            "status": "stopped (stop-floor)",
            "est_usd": round(est, 6),
            "est_basis": basis,
            "judge_reserve_usd": round(judge_reserve, 6),
            "reused": False,
        }
        _append_manifest(entry)
        return {"model": model, "status": "stopped (stop-floor)", "entry": entry}

    case_rows = load_subset_cases("test_subset_v1")
    summary = run_eval_set(
        case_set=f"pareto_D_{slugify_model(model)}",
        arm="D",
        model=model,
        case_rows=case_rows,
        milestone_tag=milestone_tag,
    )

    if summary["index_version"] != FROZEN_INDEX_VERSION:
        raise RuntimeError(
            f"index_version moved: expected {FROZEN_INDEX_VERSION!r}, got "
            f"{summary['index_version']!r} -- arm D was not held fixed; refusing to report this leg"
        )

    rank = (
        _ALL_SWEEP_ORDER.index(model) + 1
        if model in _ALL_SWEEP_ORDER
        else len(_ALL_SWEEP_ORDER) + 1
    )
    entry = {
        "leg": "pareto",
        "arm": "D",
        "model": model,
        "n_cases": summary["n_cases"],
        "est_usd": summary["est_usd"],
        "est_basis": summary["est_basis"],
        "realized_usd": summary["realized_usd"],
        "results_path": summary["results_path"],
        "summary_path": summary["summary_path"],
        "git_sha7": summary["git_sha7"],
        "index_version": summary["index_version"],
        "milestone": "m6",
        "rank": rank,
        "reused": False,
    }
    _append_manifest(entry)
    return {"model": model, "status": "swept", "entry": entry}


def run_sweeps(
    models: tuple[str, ...] | None = None, *, stop_usd: float = M6_STOP_USD
) -> dict:
    """Sweep every model in `models` (default `PARETO_NEW_MODELS`), in order,
    stopping (not failing) the first time a guard fires.
    """
    models = tuple(models) if models is not None else PARETO_NEW_MODELS
    results: list[dict] = []
    for model in models:
        result = run_model_sweep(model)
        results.append(result)
        if result["status"] in ("stopped", "stopped (stop-floor)"):
            break
    return {"stop_usd": stop_usd, "results": results}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from dealpoint.eval.spend import per_case_usd, realized_usd

    parser = argparse.ArgumentParser(prog="python -m dealpoint.eval.pareto_sweep")
    parser.add_argument("--models", default=None, help="comma-separated model ids")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-probe", action="store_true", help="unused; kept for CLI parity")
    args = parser.parse_args(argv)

    models = tuple(args.models.split(",")) if args.models else PARETO_NEW_MODELS

    if args.dry_run:
        table = []
        total = 0.0
        for model in models:
            per_case, basis = per_case_usd("D", model)
            est = per_case * 32
            total += est
            table.append({"model": model, "est_usd": round(est, 6), "basis": basis})
        print(
            json.dumps(
                {
                    "table": table,
                    "projected_total": round(realized_usd() + total, 6),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    out = run_sweeps(models)
    print(json.dumps({"results": [{"model": r["model"], "status": r["status"]} for r in out["results"]]}))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
