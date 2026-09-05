"""M5 judges report: `data/reports/judges.json` + `judges.md` (spec §9).

`build_judges_report` is pure over already-loaded rows (same discipline as
`dealpoint.eval.report.build_report`), so it is unit-testable on fake judge
data with no disk access. Disk-driven loading (reading the judged subset's
result files for `grounded_accuracy`, the judge slate report) lives in
separate loader functions that `dealpoint.eval.calibration.main` calls
before handing everything to `build_judges_report`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from dealpoint.config import (
    M5_MAX_USD,
    M5_TARGET_USD,
)
from dealpoint.eval.agreement import (
    correlate_with_grounded_accuracy,
    pairwise_judge_agreement,
    quadratic_weighted_kappa,
    spearman,
)
from dealpoint.eval.rubric import rubric_version

DIMENSIONS: tuple[str, ...] = ("reasoning", "evidence", "trajectory", "professional")

PENDING_HUMAN_INPUT = "score data/eval/calibration/form.md"

# --- recorded decisions (spec "Factory latitude" clause; §1 of the plan) ---

DECISIONS: tuple[dict, ...] = (
    {
        "id": "D1",
        "decision": "Judged subset is 18 cases (test_subset_v1 tranche_1), not the spec's 12-case default.",
        "reason": (
            "The spec's default is 12 cases (1 per question). The same spec's latitude floor "
            "requires 'every question keeps >= 1 case and counterfactuals stay in'. A 12-case, "
            "one-per-question set drops every counterfactual, so it cannot satisfy the floor. "
            "tranche_1 (12 test cases + 4 redacted + 2 out-of-scope = 18) is the smallest set "
            "that satisfies both; the 6 extra cases cost ~$0.05."
        ),
    },
    {
        "id": "D2",
        "decision": "Three variants judged (A@Haiku, D@Haiku, D@GLM), not the spec's two.",
        "reason": (
            "Spec deliverable 4 names arm A@Haiku and arm D@Haiku. The engineer's 'Judge trio' "
            "instruction separately says 'judge the default and GLM in M5'. Both are satisfied "
            "by judging A@Haiku, D@Haiku, D@GLM: 18 cases x 3 variants = 54 traces, "
            "54 x 3 judges = 162 calls. This costs ~$0.045 more and gives M6 a judged data "
            "point for a model it reuses rather than re-runs."
        ),
    },
    {
        "id": "D3",
        "decision": "Agreement statistics (Spearman, quadratic-weighted kappa, Pearson) are pure Python, not scipy.",
        "reason": (
            "scipy is not installed on this CPU-only VM (~3.5 GB free disk) and must not be "
            "added; the brief's architecture table also keeps scorers framework-free."
        ),
    },
    {
        "id": "D4",
        "decision": "M5 targets a realised spend well under its $0.60 guide.",
        "reason": (
            "Only a small fraction of the whole $4.00 M1-M6 envelope remained by the time M5 "
            "ran; sizing M5's judging (162 calls at the measured per-call cost) leaves headroom "
            "for M6, which is recorded under `spend.envelope_headroom_after_m5` below."
        ),
    },
    {
        "id": "D5",
        "decision": "Mean-of-judges is rounded to the nearest integer for kappa only, not for Spearman.",
        "reason": (
            "Weighted Cohen's kappa is defined on integer ratings; the mean of three judges is "
            "not one. Spearman rho uses the raw mean; kappa uses round() of it (Python's "
            "round-half-to-even)."
        ),
    },
)

BRIEF_DIFFERENCES: tuple[dict, ...] = (
    {
        "id": 1,
        "topic": "scale",
        "difference": (
            "Brief section 2.5 specifies 40 stratified cases, 6 variants, 240 traces, 720 judge "
            "calls, 30 hand-scored traces, est. $15-40. This milestone runs 18 cases, 3 "
            "variants, 54 traces, 162 calls at a small fraction of that cost. Every judged "
            "number here is budget-scaled and is never presented as the brief's benchmark."
        ),
    },
    {
        "id": 2,
        "topic": "judge_families",
        "difference": (
            "Brief section 2.5 gives 'e.g. GPT + Grok + Qwen' as the example judge trio while "
            "also requiring families not in the candidate-agent slate. The M6 candidate slate "
            "now contains OpenAI, xAI and Alibaba/Qwen, so the brief's own illustrative example "
            "would violate the brief's own rule. The engineer's trio (Mistral, NVIDIA, "
            "ByteDance; spare Amazon) satisfies the rule; the brief's rule is honoured, its "
            "illustrative example is not, because the candidate slate changed after the brief "
            "was written."
        ),
    },
    {
        "id": 3,
        "topic": "human_calibration_count",
        "difference": (
            "Brief section 2.5 mandates 30 hand-scored traces (~5 per variant); this milestone's "
            "spec sets 'whatever is judged, target >= 12'. 54 packets are produced, and "
            "form.md names a deterministic suggested minimum of 12. The gap between 12 and the "
            "brief's 30 is a reported difference, carried in this milestone's pending human input."
        ),
    },
    {
        "id": 4,
        "topic": "output_keys",
        "difference": (
            "Brief Appendix C fixes the judge's JSON as {reasoning:int, evidence:int, "
            "trajectory:int, professional:int, notes:str}. Those exact key names are used, with "
            "no renaming and no extra model-supplied keys."
        ),
    },
)

BLINDING_LIMITATIONS: tuple[str, ...] = (
    (
        "The packet blinds arm and model identity (the spec's requirement) but cannot blind the "
        "SHAPE of the trajectory: arm A is a single-shot pipeline with no tool calls, arm D is an "
        "agent loop, so a knowledgeable reader can often infer which family of system produced a "
        "trace from trajectory length and tool-call pattern alone."
    ),
    (
        "An empty `options` list hints that a case is an out-of-scope probe, even though the exact "
        "question_id and category are never shown."
    ),
)

STATS_NOTES: tuple[str, ...] = (
    (
        "Weighted Cohen's kappa is defined on integer ratings; the mean of three judges is not one. "
        "Spearman rho uses the raw mean-of-judges; kappa uses round() of it (D5)."
    ),
    (
        "A statistic of null means 'not computable' (degenerate input) and is always paired with "
        "its n; it is never coerced to 0 or 1."
    ),
)

CAVEATS: tuple[str, ...] = (
    "Judge scores are secondary and model-judged; they are never presented as objective truth.",
    "Budget-scaled: 18 cases x 3 variants x 3 judges, not the brief's 40 x 6 x 3.",
    (
        "Human calibration is pending until data/eval/calibration/human_scores.jsonl carries scores; "
        "no judge-human agreement figure exists yet."
    ),
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_trace_scores(judged_subset: dict) -> dict[tuple[str, str], bool | None]:
    """(case_id, variant_id) -> grounded_accuracy, read from each variant's result file.

    Disk-reading helper, deliberately kept OUT of `build_judges_report` so the
    report builder itself stays pure over already-loaded rows.
    """
    out: dict[tuple[str, str], bool | None] = {}
    case_ids = set(judged_subset.get("case_ids", []))
    for variant in judged_subset.get("variants", []):
        variant_id = variant["variant_id"]
        for row in _load_jsonl(Path(variant["results_path"])):
            case_id = row.get("case_id")
            if not case_id or case_id not in case_ids:
                continue
            scores = row.get("scores") or {}
            out[(case_id, variant_id)] = scores.get("grounded_accuracy")
    return out


def load_judge_slate(path) -> dict | None:
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _trace_key(row: dict) -> tuple[str, str]:
    return (row["case_id"], row["variant_id"])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def build_judges_report(
    judge_rows: list[dict],
    *,
    judged_subset: dict | None = None,
    trace_scores: dict[tuple[str, str], bool | None] | None = None,
    judge_slate: dict | None = None,
    human_rows: list[dict] | None = None,
    human_errors: list[dict] | None = None,
    spend: dict | None = None,
    rubric_version_value: str | None = None,
    git_sha7_value: str | None = None,
) -> dict:
    """Pure assembly of the full `judges.json` payload from already-loaded rows.

    `judge_rows` is `data/eval/judge_scores.jsonl`'s rows (one per
    `(packet_id, judge_model)` call). `trace_scores` maps
    `(case_id, variant_id) -> grounded_accuracy` for the correlation block --
    `None` (or the key absent) means "not computable for this trace", never
    coerced to False.
    """
    trace_scores = trace_scores or {}
    judged_subset = judged_subset or {}

    ok_rows = [r for r in judge_rows if r.get("ok")]
    failed_rows = [r for r in judge_rows if not r.get("ok")]

    judge_models = sorted({r["judge_model"] for r in judge_rows})

    # --- per-dimension aggregates ------------------------------------
    per_dimension: dict[str, dict] = {}
    # trace_mean[dim][(case_id, variant_id)] = mean of judges' ok scores for that trace
    trace_mean_by_dim: dict[str, dict[tuple[str, str], float]] = {}

    for dim in DIMENSIONS:
        by_trace: dict[tuple[str, str], list[float]] = {}
        by_judge_vals: dict[str, list[float]] = {}
        for row in ok_rows:
            value = row.get(dim)
            if value is None:
                continue
            key = _trace_key(row)
            by_trace.setdefault(key, []).append(value)
            by_judge_vals.setdefault(row["judge_model"], []).append(value)

        trace_means = {key: _mean(vals) for key, vals in by_trace.items()}
        trace_mean_by_dim[dim] = {k: v for k, v in trace_means.items() if v is not None}

        overall_values = list(trace_mean_by_dim[dim].values())

        by_variant: dict[str, dict] = {}
        variant_ids = sorted({vid for (_cid, vid) in trace_mean_by_dim[dim]})
        for vid in variant_ids:
            vals = [v for (cid, v_id), v in trace_mean_by_dim[dim].items() if v_id == vid]
            by_variant[vid] = {"mean": _mean(vals), "n": len(vals)}

        by_judge: dict[str, dict] = {}
        for jm in judge_models:
            vals = by_judge_vals.get(jm, [])
            by_judge[jm] = {"mean": _mean(vals), "n": len(vals)}

        per_dimension[dim] = {
            "mean_of_judges": _mean(overall_values),
            "n": len(overall_values),
            "by_variant": by_variant,
            "by_judge": by_judge,
        }

    # --- pairwise judge agreement (per dimension, over packet_id) -----
    pairwise: dict[str, list[dict]] = {}
    for dim in DIMENSIONS:
        scores_by_judge: dict[str, dict[str, int | None]] = {}
        for jm in judge_models:
            scores_by_judge[jm] = {
                r["packet_id"]: r.get(dim) for r in judge_rows if r["judge_model"] == jm
            }
        pairwise[dim] = pairwise_judge_agreement(scores_by_judge)

    # --- correlation with grounded_accuracy ----------------------------
    correlation: dict[str, dict] = {}
    for dim in DIMENSIONS:
        keys = sorted(trace_mean_by_dim[dim])
        quality = [trace_mean_by_dim[dim][k] for k in keys]
        grounded = [trace_scores.get(k) for k in keys]
        correlation[dim] = correlate_with_grounded_accuracy(quality, grounded)

    # --- judges list ----------------------------------------------------
    judges_list: list[dict] = []
    slate_judges = (judge_slate or {}).get("judges", [])
    slate_by_model = {j["model"]: j for j in slate_judges}
    substitutions = (judge_slate or {}).get("substitutions", [])
    substituted_for_by_model: dict[str, str | None] = {}
    for sub in substitutions:
        if "with" in sub:
            substituted_for_by_model[sub["with"]] = sub.get("substituted_for") or sub.get("failed")

    for jm in judge_models:
        rows_for_judge = [r for r in judge_rows if r["judge_model"] == jm]
        slate_entry = slate_by_model.get(jm, {})
        judges_list.append(
            {
                "model": jm,
                "family": slate_entry.get("family")
                or (rows_for_judge[0].get("judge_family") if rows_for_judge else None),
                "prompt_usd_per_token": slate_entry.get("prompt_usd_per_token"),
                "completion_usd_per_token": slate_entry.get("completion_usd_per_token"),
                "price_basis": (judge_slate or {}).get("price_basis"),
                "n_scored": sum(1 for r in rows_for_judge if r.get("ok")),
                "n_failed": sum(1 for r in rows_for_judge if not r.get("ok")),
                "substituted_for": substituted_for_by_model.get(jm),
            }
        )

    # --- judged_subset block ---------------------------------------------
    judged_subset_block = {
        "path": "data/eval/judged_subset.json",
        "subset_hash": judged_subset.get("subset_hash"),
        "n_cases": judged_subset.get("n_cases"),
        "n_traces": judged_subset.get("n_traces"),
        "n_judge_calls": judged_subset.get("n_judge_calls"),
        "seed": judged_subset.get("seed"),
        "rule": judged_subset.get("rule"),
        "variants": judged_subset.get("variants"),
    }

    # --- human calibration -------------------------------------------------
    human_calibration: dict | str
    if human_rows:
        by_packet: dict[str, dict] = {r["packet_id"]: r for r in human_rows}
        human_block: dict[str, dict] = {}
        for dim in DIMENSIONS:
            # mean-of-judges (raw, for rho) and rounded (for kappa) per packet
            by_packet_judge_mean: dict[str, float] = {}
            for row in ok_rows:
                pid = row["packet_id"]
                value = row.get(dim)
                if value is None:
                    continue
                by_packet_judge_mean.setdefault(pid, []).append(value)  # type: ignore[union-attr]
            by_packet_judge_mean = {
                pid: _mean(vals) for pid, vals in by_packet_judge_mean.items()  # type: ignore[arg-type]
            }

            common_pids = sorted(set(by_packet_judge_mean) & set(by_packet))
            judge_means = [by_packet_judge_mean[pid] for pid in common_pids]
            judge_means_rounded = [round(v) for v in judge_means]
            human_vals = [by_packet[pid][dim] for pid in common_pids]

            mean_vs_human = {
                "spearman": spearman(judge_means, human_vals),
                "qwk": quadratic_weighted_kappa(judge_means_rounded, human_vals),
                "n": len(common_pids),
            }

            per_judge_vs_human: dict[str, dict] = {}
            for jm in judge_models:
                jm_scores = {r["packet_id"]: r.get(dim) for r in ok_rows if r["judge_model"] == jm}
                common = sorted(set(jm_scores) & set(by_packet))
                jv = [jm_scores[pid] for pid in common]
                hv = [by_packet[pid][dim] for pid in common]
                per_judge_vs_human[jm] = {
                    "spearman": spearman(jv, hv),
                    "qwk": quadratic_weighted_kappa(jv, hv),
                    "n": len(common),
                }

            human_block[dim] = {
                "mean_of_judges_vs_human": mean_vs_human,
                "per_judge_vs_human": per_judge_vs_human,
            }
        human_calibration = human_block
        pending_human_input: str | None = None
    else:
        human_calibration = "pending"
        pending_human_input = PENDING_HUMAN_INPUT

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha7": git_sha7_value,
        "rubric_version": rubric_version_value or rubric_version(),
        "budget_scaled": True,
        "judges": judges_list,
        "judged_subset": judged_subset_block,
        "per_dimension": per_dimension,
        "pairwise_judge_agreement": pairwise,
        "correlation_with_grounded_accuracy": correlation,
        "human_calibration": human_calibration,
        "pending_human_input": pending_human_input,
        "human_calibration_errors": human_errors or [],
        "spend": spend
        or {
            "target_usd": M5_TARGET_USD,
            "absolute_usd": M5_MAX_USD,
        },
        "decisions": list(DECISIONS),
        "brief_differences": list(BRIEF_DIFFERENCES),
        "blinding_limitations": list(BLINDING_LIMITATIONS),
        "stats_notes": list(STATS_NOTES),
        "caveats": list(CAVEATS),
        "n_judge_rows": len(judge_rows),
        "n_judge_rows_ok": len(ok_rows),
        "n_judge_rows_failed": len(failed_rows),
    }
    return report


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# M5 judges report")
    lines.append("")
    lines.append(
        "These are **model-judged secondary scores, budget-scaled**, and are never presented "
        "as objective truth. Human calibration is **pending** unless stated otherwise below."
    )
    lines.append("")
    lines.append(f"`rubric_version`: `{report.get('rubric_version')}`  ")
    lines.append(f"`git_sha7`: `{report.get('git_sha7')}`")
    lines.append("")

    subset = report.get("judged_subset", {})
    lines.append("## Judged subset")
    lines.append("")
    lines.append(
        f"{subset.get('n_cases')} cases x {len(subset.get('variants') or [])} variants = "
        f"{subset.get('n_traces')} traces; {subset.get('n_judge_calls')} judge calls. "
        f"subset_hash=`{subset.get('subset_hash')}`."
    )
    lines.append("")

    lines.append("## Judges")
    lines.append("")
    lines.append("| model | family | n_scored | n_failed | substituted_for |")
    lines.append("|---|---|---|---|---|")
    for j in report.get("judges", []):
        lines.append(
            f"| {j['model']} | {j.get('family')} | {j.get('n_scored')} | {j.get('n_failed')} | "
            f"{j.get('substituted_for')} |"
        )
    lines.append("")

    lines.append("## Per-dimension aggregates (mean of judges)")
    lines.append("")
    lines.append("| dimension | mean | n |")
    lines.append("|---|---|---|")
    for dim in DIMENSIONS:
        d = report.get("per_dimension", {}).get(dim, {})
        lines.append(f"| {dim} | {d.get('mean_of_judges')} | {d.get('n')} |")
    lines.append("")

    lines.append("## Correlation with grounded_accuracy")
    lines.append("")
    lines.append("| dimension | pearson | spearman | n | caveat |")
    lines.append("|---|---|---|---|---|")
    for dim in DIMENSIONS:
        c = report.get("correlation_with_grounded_accuracy", {}).get(dim, {})
        lines.append(
            f"| {dim} | {c.get('pearson')} | {c.get('spearman')} | {c.get('n')} | "
            f"{c.get('caveat', '')} |"
        )
    lines.append("")

    hc = report.get("human_calibration")
    lines.append("## Human calibration")
    lines.append("")
    if hc == "pending":
        lines.append(f"**pending.** Pending human input: `{report.get('pending_human_input')}`.")
    else:
        lines.append("Computed against the scored human packets:")
        lines.append("")
        lines.append("| dimension | spearman (mean vs human) | qwk (rounded mean vs human) | n |")
        lines.append("|---|---|---|---|")
        for dim in DIMENSIONS:
            block = (hc or {}).get(dim, {}).get("mean_of_judges_vs_human", {})
            lines.append(
                f"| {dim} | {block.get('spearman')} | {block.get('qwk')} | {block.get('n')} |"
            )
    lines.append("")

    lines.append("## Recorded decisions")
    lines.append("")
    for d in report.get("decisions", []):
        lines.append(f"- **{d['id']}**: {d['decision']} -- {d['reason']}")
    lines.append("")

    lines.append("## Brief-vs-spec differences")
    lines.append("")
    for d in report.get("brief_differences", []):
        lines.append(f"- **{d['topic']}**: {d['difference']}")
    lines.append("")

    lines.append("## Blinding limitations")
    lines.append("")
    for note in report.get("blinding_limitations", []):
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Spend")
    lines.append("")
    lines.append(f"```json\n{json.dumps(report.get('spend', {}), indent=2, sort_keys=True)}\n```")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    for c in report.get("caveats", []):
        lines.append(f"- {c}")
    lines.append("")

    return "\n".join(lines)
