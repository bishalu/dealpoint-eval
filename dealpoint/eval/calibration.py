"""The M5 calibration package: `data/eval/calibration/*` (spec §7, deliverable 5).

Offline, no model calls, idempotent (`just calibration`): rebuilds the
calibration package from the frozen judged subset + result files, reads
`judge_scores.jsonl` and `human_scores.jsonl`, and writes
`data/reports/judges.json` + `judges.md`.
"""

from __future__ import annotations

import json

from dealpoint.config import (
    CALIBRATION_DIR,
    JUDGE_SCORES_PATH,
    JUDGES_JSON_PATH,
    JUDGES_MD_PATH,
)
from dealpoint.eval.rubric import rubric_text, rubric_version

PACKETS_JSONL_PATH = CALIBRATION_DIR / "packets.jsonl"
PACKETS_MD_PATH = CALIBRATION_DIR / "packets.md"
FORM_MD_PATH = CALIBRATION_DIR / "form.md"
HUMAN_SCORES_PATH = CALIBRATION_DIR / "human_scores.jsonl"
HUMAN_SCORES_SCHEMA_PATH = CALIBRATION_DIR / "human_scores.schema.json"
VARIANT_KEY_PATH = CALIBRATION_DIR / "variant_key.json"

HUMAN_SCORES_SCHEMA: dict = {
    "packet_id": "<12-hex>",
    "scorer": "<string>",
    "scored_at": "<ISO-8601>",
    "reasoning": 1,
    "evidence": 1,
    "trajectory": 1,
    "professional": 1,
    "notes": "<string>",
}

BLINDING_LIMITATIONS: tuple[str, ...] = (
    (
        "The packet blinds arm and model identity (the spec's requirement) but cannot blind the "
        "SHAPE of the trajectory: arm A is a single-shot pipeline with no tool calls, arm D is an "
        "agent loop, so a knowledgeable reader can often infer which family of system produced a "
        "trace from trajectory length and tool-call pattern alone."
    ),
    (
        "An empty `options` list (rendered as '(free-form; no fixed option list for this "
        "question)') hints that a case is an out-of-scope probe, even though the exact question_id "
        "and category are never shown."
    ),
)

STATS_NOTES: tuple[str, ...] = (
    (
        "Weighted Cohen's kappa is defined on integer ratings; the mean of three judges is not one. "
        "Spearman rho uses the raw mean-of-judges; kappa uses round() of it (banker's rounding, "
        "Python's default) -- see D5."
    ),
    (
        "A statistic of null means 'not computable' (degenerate input: n<2, zero variance, or a "
        "single distinct rating from both raters) and is always paired with its n; it is never "
        "coerced to 0 or 1."
    ),
)


def _load_jsonl(path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_calibration_package() -> dict:
    """Rebuild every calibration artefact from the frozen judged subset + result files.

    Returns a dict of the in-memory artefacts (packets, variant_key,
    suggested_minimum) for `main()` to write to disk and for tests to
    inspect without touching the filesystem.
    """
    from dealpoint.eval.judge_run import build_all_packets
    from dealpoint.eval.subset import load_subset_payload

    judged_subset = load_subset_payload("judged_subset")
    items = build_all_packets()
    items_sorted = sorted(items, key=lambda it: it["packet"]["packet_id"])

    packets = [it["packet"] for it in items_sorted]
    variant_key = {
        it["packet"]["packet_id"]: {
            "case_id": it["case_id"],
            "variant_id": it["variant_id"],
        }
        for it in items_sorted
    }
    # attach arm/model into variant_key from judged_subset's variants table
    variants_by_id = {v["variant_id"]: v for v in judged_subset["variants"]}
    for entry in variant_key.values():
        v = variants_by_id[entry["variant_id"]]
        entry["arm"] = v["arm"]
        entry["model"] = v["model"]

    rank = judged_subset["rank"]
    # Suggested minimum 12: first 4 (by subset case rank) packet ids per variant.
    by_variant: dict[str, list[str]] = {}
    for it in items_sorted:
        by_variant.setdefault(it["variant_id"], []).append(it["packet"]["packet_id"])
    suggested_minimum: list[str] = []
    for variant_id in sorted(by_variant):
        pids = by_variant[variant_id]
        pids_sorted = sorted(
            pids, key=lambda pid: rank[variant_key[pid]["case_id"]]
        )
        suggested_minimum.extend(pids_sorted[:4])

    return {
        "packets": packets,
        "packet_texts": {it["packet"]["packet_id"]: it["packet_text"] for it in items_sorted},
        "variant_key": variant_key,
        "suggested_minimum": sorted(suggested_minimum),
    }


def render_form_md(package: dict) -> str:
    lines: list[str] = []
    lines.append("# M5 human calibration scoring form")
    lines.append("")
    lines.append(
        "These are **model-judged secondary scores**, budget-scaled, and never objective "
        "truth. Scoring here calibrates the judges against a human reader; it does not replace "
        "the deterministic scores in `four_arm.json`."
    )
    lines.append("")
    lines.append(f"`rubric_version`: `{rubric_version()}`")
    lines.append("")
    lines.append("## Rubric (identical to `eval/judges/rubrics.md`)")
    lines.append("")
    lines.append(rubric_text())
    lines.append("")
    lines.append("## Output contract")
    lines.append("")
    lines.append(
        "For each packet you score, append one line to `human_scores.jsonl` with these keys:"
    )
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(HUMAN_SCORES_SCHEMA, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("Worked example:")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "packet_id": "0123456789ab",
                "scorer": "jane",
                "scored_at": "2026-09-05T12:00:00+00:00",
                "reasoning": 4,
                "evidence": 3,
                "trajectory": 4,
                "professional": 4,
                "notes": "Correctly ties the answer to the cited section; evidence is adjacent "
                "to but not identical to the gold span.",
            },
            indent=2,
        )
    )
    lines.append("```")
    lines.append("")
    lines.append(
        f"## Suggested minimum set ({len(package['suggested_minimum'])} packets, deterministic: "
        "first 4 by subset rank per variant)"
    )
    lines.append("")
    lines.append(
        "Scoring more than these 12 is welcome and there is no upper bound imposed; this is "
        "only a deterministic starting point, not a cap."
    )
    lines.append("")
    for pid in package["suggested_minimum"]:
        lines.append(f"- `{pid}`")
    lines.append("")
    lines.append("## Blinding limitations")
    lines.append("")
    for note in BLINDING_LIMITATIONS:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Statistics notes")
    lines.append("")
    for note in STATS_NOTES:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        "**Do not open `variant_key.json` until you have finished scoring.** It maps each "
        "`packet_id` back to its case, arm and model, and would unblind you mid-scoring."
    )
    lines.append("")
    lines.append("Score packets from `packets.md`, in any order you like.")
    lines.append("")
    return "\n".join(lines)


def render_packets_md(package: dict) -> str:
    lines: list[str] = []
    lines.append("# M5 blinded judge packets")
    lines.append("")
    lines.append(
        "Every packet below is byte-identical in content to what the judge models were shown "
        "(rendered by the same function, `dealpoint.eval.blinding.render_packet_text`)."
    )
    lines.append("")
    lines.append("## Table of contents")
    lines.append("")
    for packet in package["packets"]:
        lines.append(f"- [{packet['packet_id']}](#{packet['packet_id']})")
    lines.append("")
    from dealpoint.eval.blinding import render_packet_text

    for packet in package["packets"]:
        lines.append(f"## {packet['packet_id']}")
        lines.append("")
        lines.append(render_packet_text(packet))
        lines.append("")
        lines.append("**Score:** reasoning=__ evidence=__ trajectory=__ professional=__")
        lines.append("")
    return "\n".join(lines)


def validate_human_scores(rows: list[dict], known_packet_ids: set[str]) -> tuple[list[dict], list[dict]]:
    """Validate raw `human_scores.jsonl` rows against the schema + known packet ids.

    Returns `(valid_rows, errors)` -- errors are collected and surfaced, never
    silently dropped.
    """
    valid: list[dict] = []
    errors: list[dict] = []
    required_keys = ("packet_id", "scorer", "scored_at", "reasoning", "evidence", "trajectory", "professional")
    for i, row in enumerate(rows):
        missing = [k for k in required_keys if k not in row]
        if missing:
            errors.append({"index": i, "row": row, "error": f"missing keys: {missing}"})
            continue
        if row["packet_id"] not in known_packet_ids:
            errors.append({"index": i, "row": row, "error": f"unknown packet_id {row['packet_id']!r}"})
            continue
        bad_dims = []
        for dim in ("reasoning", "evidence", "trajectory", "professional"):
            v = row[dim]
            if isinstance(v, bool) or not isinstance(v, int) or not (1 <= v <= 5):
                bad_dims.append(dim)
        if bad_dims:
            errors.append({"index": i, "row": row, "error": f"invalid dimension(s): {bad_dims}"})
            continue
        valid.append(row)
    return valid, errors


def main(argv: list[str] | None = None) -> int:
    from dealpoint.config import JUDGE_SLATE_PATH
    from dealpoint.eval.cases import git_sha7
    from dealpoint.eval.judges_report import (
        build_judges_report,
        load_judge_slate,
        load_trace_scores,
        render_markdown,
    )
    from dealpoint.eval.rubric import rubric_version
    from dealpoint.eval.spend import realized_by_tag
    from dealpoint.eval.subset import load_subset_payload

    package = build_calibration_package()

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    with open(PACKETS_JSONL_PATH, "w", encoding="utf-8") as fh:
        for packet in package["packets"]:
            fh.write(json.dumps(packet, sort_keys=True, ensure_ascii=False))
            fh.write("\n")

    PACKETS_MD_PATH.write_text(render_packets_md(package), encoding="utf-8")
    FORM_MD_PATH.write_text(render_form_md(package), encoding="utf-8")

    if not HUMAN_SCORES_PATH.exists():
        HUMAN_SCORES_PATH.write_text("", encoding="utf-8")

    with open(HUMAN_SCORES_SCHEMA_PATH, "w", encoding="utf-8") as fh:
        json.dump(HUMAN_SCORES_SCHEMA, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")

    with open(VARIANT_KEY_PATH, "w", encoding="utf-8") as fh:
        json.dump(package["variant_key"], fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")

    judge_rows = _load_jsonl(JUDGE_SCORES_PATH)
    human_rows_raw = _load_jsonl(HUMAN_SCORES_PATH)
    known_packet_ids = set(package["variant_key"])
    human_valid, human_errors = validate_human_scores(human_rows_raw, known_packet_ids)

    judged_subset = load_subset_payload("judged_subset")
    trace_scores = load_trace_scores(judged_subset)
    judge_slate = load_judge_slate(JUDGE_SLATE_PATH)

    from dealpoint.config import M5_MAX_USD, M5_TARGET_USD
    from dealpoint.eval.spend import estimate, realized_usd

    by_tag = realized_by_tag()
    m5_realized = by_tag.get("m5", 0.0)
    m6_realized = by_tag.get("m6", 0.0)
    total_realized = realized_usd()
    judges_est = estimate("judges")
    spend_block = {
        "target_usd": M5_TARGET_USD,
        "absolute_usd": M5_MAX_USD,
        "envelope_usd": 4.00,
        "est_usd": judges_est.get("est_usd"),
        "realized_usd": m5_realized,
        "m6_realized_usd": m6_realized,
        "envelope_realized_usd": total_realized,
        "envelope_headroom_after_m5": round(4.00 - total_realized, 6),
        "envelope_headroom_after_m6": round(4.00 - total_realized, 6),
        "est_basis": judges_est.get("basis"),
        "assumed_input_tokens_per_call": 8000,
        "measured_input_tokens_per_call": (
            round(sum(r.get("input_tokens", 0) for r in judge_rows) / len(judge_rows), 1)
            if judge_rows
            else None
        ),
    }

    report = build_judges_report(
        judge_rows,
        judged_subset=judged_subset,
        trace_scores=trace_scores,
        judge_slate=judge_slate,
        human_rows=human_valid if human_valid else None,
        human_errors=human_errors,
        spend=spend_block,
        rubric_version_value=rubric_version(),
        git_sha7_value=git_sha7(),
    )

    JUDGES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JUDGES_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")

    JUDGES_MD_PATH.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({"n_packets": len(package["packets"]), "n_judge_rows": len(judge_rows)}))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
