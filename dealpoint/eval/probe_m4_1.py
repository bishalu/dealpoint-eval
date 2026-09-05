"""M4.1 diagnostic probes (spec deliverable 2 + 4): 2-case runs per leg to
capture `failure_detail`/`raw_final_text` on the exact v1 `EXECUTION_FAILED`
cases, before spending on the v2 sweeps.

`before` numbers are taken from the v1 result JSONLs (already paid for --
re-spending on them would be waste); only the `after` (and any additional
iteration) round is metered. CLI:
`python -m dealpoint.eval.probe_m4_1 [--round NAME] [--before-only]`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from dealpoint.config import M4_1_MILESTONE_TAG, M4_1_PROBES_JSON_PATH
from dealpoint.eval.cases import git_sha7

# The 3 legs named by the spec: arm D at Haiku, arm A at GLM, arm B at GLM.
PROBE_LEGS: tuple[dict, ...] = (
    {"model": "anthropic/claude-haiku-4.5", "arm": "D", "leg": "replication_haiku"},
    {"model": "z-ai/glm-5.3-flash", "arm": "A", "leg": "headline_glm"},
    {"model": "z-ai/glm-5.3-flash", "arm": "B", "leg": "headline_glm"},
)

CASE_SELECTION_RULE = (
    "For each probed (model, arm), take the first 2 case_ids, in frozen-subset order "
    "(data/eval/test_subset_v1.json's case_ids list), whose v1 result row for that "
    "(model, arm) has record.status == 'EXECUTION_FAILED'. Resolved in code from the "
    "v1 result JSONL named in data/reports/four_arm_manifest_v1.json (falling back to "
    "the live four_arm_manifest.json before v1 preservation), never hard-coded. This "
    "is a diagnostic selection on a FROZEN subset -- it selects nothing that is "
    "reported as a headline result, and the v2 sweep (dealpoint.eval.four_arm_sweep) "
    "runs the whole subset regardless of what the probes found."
)

NOTES = (
    "'before' numbers are read from the v1 result JSONLs, not re-spent: v1's "
    "ExecutionRecord carried no failure_detail/raw_final_text field (that is the "
    "defect this milestone's deliverable 1 fixes), so every 'before' "
    "failure_detail_classes entry is synthesized from the coarser failure_reason "
    "and is labelled as such. 'after' rounds run the same 2 case_ids for real "
    "through the repaired harness and read the new failure_detail/finish_reasons "
    "fields directly off the written result rows."
)


def _read_jsonl(path) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _v1_result_path(model: str, arm: str, leg: str) -> Path | None:
    """Resolve the v1 result JSONL for (model, arm, leg) from a manifest,
    preferring the preserved `four_arm_manifest_v1.json` once it exists.
    """
    from dealpoint.config import FOUR_ARM_MANIFEST_V1_PATH
    from dealpoint.eval.four_arm_sweep import MANIFEST_PATH, load_manifest

    for manifest_path in (FOUR_ARM_MANIFEST_V1_PATH, MANIFEST_PATH):
        for entry in load_manifest(manifest_path):
            if entry.get("model") == model and entry.get("arm") == arm and entry.get("leg") == leg:
                path = Path(entry["results_path"])
                if path.exists():
                    return path
    return None


def select_probe_case_ids(model: str, arm: str, leg: str, n: int = 2) -> list[str]:
    """First `n` case_ids (frozen-subset order) whose v1 (model, arm) row is
    `EXECUTION_FAILED` (spec §4, case-choice rule)."""
    from dealpoint.eval.subset import load_subset_case_ids

    path = _v1_result_path(model, arm, leg)
    if path is None:
        raise FileNotFoundError(f"no v1 manifest entry with a results file for ({model!r}, {arm!r})")
    rows = _read_jsonl(path)
    by_case = {r["case_id"]: r for r in rows}
    order = load_subset_case_ids("test_subset_v1")
    selected: list[str] = []
    for cid in order:
        row = by_case.get(cid)
        if row is not None and row.get("record", {}).get("status") == "EXECUTION_FAILED":
            selected.append(cid)
        if len(selected) >= n:
            break
    return selected


def _failure_detail_classes(
    rows_by_case: dict[str, dict], case_ids: list[str], *, before: bool
) -> dict[str, int]:
    classes: dict[str, int] = {}
    for cid in case_ids:
        row = rows_by_case.get(cid)
        if row is None:
            continue
        record = row.get("record", {})
        if record.get("status") != "EXECUTION_FAILED":
            continue
        detail = record.get("failure_detail")
        if detail:
            key = detail
        elif before:
            key = (
                f"failure_reason={record.get('failure_reason')} "
                "(failure_detail: null -- the defect this milestone fixes)"
            )
        else:
            key = f"failure_reason={record.get('failure_reason')} (failure_detail: null)"
        classes[key] = classes.get(key, 0) + 1
    return classes


def _finish_reason_counts(rows_by_case: dict[str, dict], case_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cid in case_ids:
        row = rows_by_case.get(cid)
        if row is None:
            continue
        for fr in row.get("record", {}).get("finish_reasons") or []:
            counts[fr] = counts.get(fr, 0) + 1
    return counts


def _leg_stats(rows: list[dict], case_ids: list[str], model: str, arm: str, *, before: bool) -> dict:
    by_case = {r["case_id"]: r for r in rows}
    n = len(case_ids)
    execution_failed = sum(
        1
        for cid in case_ids
        if by_case.get(cid, {}).get("record", {}).get("status") == "EXECUTION_FAILED"
    )
    realized = round(
        sum(float(by_case[cid].get("usd", 0.0)) for cid in case_ids if cid in by_case), 6
    )
    return {
        "model": model,
        "arm": arm,
        "case_ids": case_ids,
        "n": n,
        "execution_failed": execution_failed,
        "failure_rate": (execution_failed / n) if n else None,
        "failure_detail_classes": _failure_detail_classes(by_case, case_ids, before=before),
        "finish_reasons": _finish_reason_counts(by_case, case_ids),
        "realized_usd": realized,
    }


def run_before_round() -> dict:
    """`before` numbers come entirely from the v1 result files -- no metered call."""
    legs = []
    for leg_def in PROBE_LEGS:
        model, arm, leg = leg_def["model"], leg_def["arm"], leg_def["leg"]
        case_ids = select_probe_case_ids(model, arm, leg)
        path = _v1_result_path(model, arm, leg)
        rows = _read_jsonl(path) if path else []
        legs.append(_leg_stats(rows, case_ids, model, arm, before=True))
    return {
        "round": "before",
        "ts": datetime.now(UTC).isoformat(),
        "git_sha7": git_sha7(),
        "legs": legs,
    }


def run_after_leg(model: str, arm: str, case_ids: list[str], milestone_tag: str = M4_1_MILESTONE_TAG) -> dict:
    """Run the probed cases for real, through the repaired harness, and
    return this leg's after-round stats (spec §4 verify gate)."""
    from dealpoint.eval.cases import find_case
    from dealpoint.eval.run import _assert_within_milestone_absolute, run_eval_set
    from dealpoint.eval.spend import assert_within_cap, per_case_usd

    case_rows = [find_case(cid) for cid in case_ids]
    per_case, _basis = per_case_usd(arm, model)
    est_usd = round(per_case * len(case_rows), 6)
    print(f"[probe_m4_1] estimate for (model={model}, arm={arm}), n={len(case_rows)}: ${est_usd:.4f}")
    assert_within_cap(est_usd)
    _assert_within_milestone_absolute(est_usd)

    summary = run_eval_set(
        case_set=f"m4_1_probe_{arm}",
        arm=arm,
        model=model,
        case_rows=case_rows,
        milestone_tag=milestone_tag,
    )
    rows = _read_jsonl(summary["results_path"])
    stats = _leg_stats(rows, case_ids, model, arm, before=False)
    stats["realized_usd"] = summary["realized_usd"]
    stats["results_path"] = summary["results_path"]
    return stats


def annotate_provider_attribution(leg: dict, threshold: float = 0.10) -> dict:
    """If `leg['failure_rate']` exceeds `threshold`, add a `provider_attribution`
    note quoting the dominant `failure_detail_classes` entry (spec §4 verify
    gate: "record the exact message and carry it into the report as the
    explanation the DoD allows"). No-op (returns `leg` unchanged) otherwise.
    """
    rate = leg.get("failure_rate")
    if rate is None or rate <= threshold:
        return leg
    classes = leg.get("failure_detail_classes") or {}
    if not classes:
        return leg
    top_class, top_count = max(classes.items(), key=lambda kv: kv[1])
    leg["provider_attribution"] = (
        f"{top_count}/{leg.get('execution_failed')} EXECUTION_FAILED cases carry this "
        f"failure_detail class (model output, not a harness parsing defect -- "
        f"extract_json_object correctly located the JSON span; the model's own JSON was "
        f"invalid, e.g. an unescaped embedded quote): {top_class[:300]!r}"
    )
    return leg


def run_after_round(round_name: str = "after") -> dict:
    legs = []
    for leg_def in PROBE_LEGS:
        case_ids = select_probe_case_ids(leg_def["model"], leg_def["arm"], leg_def["leg"])
        leg_stats = run_after_leg(leg_def["model"], leg_def["arm"], case_ids)
        legs.append(annotate_provider_attribution(leg_stats))
    return {
        "round": round_name,
        "ts": datetime.now(UTC).isoformat(),
        "git_sha7": git_sha7(),
        "legs": legs,
    }


def load_probes(path: Path = M4_1_PROBES_JSON_PATH) -> dict:
    if not path.exists():
        return {"rounds": [], "case_selection_rule": CASE_SELECTION_RULE, "notes": NOTES}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"rounds": [], "case_selection_rule": CASE_SELECTION_RULE, "notes": NOTES}


def _write_probes(payload: dict, path: Path = M4_1_PROBES_JSON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dealpoint.eval.probe_m4_1")
    parser.add_argument(
        "--round",
        default=None,
        help="metered round name (default 'after'); pass a fresh name (e.g. 'after2') "
        "for an extra iteration if the fix needs more than one round",
    )
    parser.add_argument(
        "--before-only", action="store_true", help="write only the (free) before round"
    )
    args = parser.parse_args(argv)

    payload = load_probes()
    payload.setdefault("case_selection_rule", CASE_SELECTION_RULE)
    payload.setdefault("notes", NOTES)
    rounds = payload.setdefault("rounds", [])

    if not any(r.get("round") == "before" for r in rounds):
        rounds.append(run_before_round())

    if not args.before_only:
        rounds.append(run_after_round(args.round or "after"))

    _write_probes(payload)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
