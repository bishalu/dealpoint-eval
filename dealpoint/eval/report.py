"""Report generator: `data/reports/four_arm.json` + `four_arm.md`, and the
README Results block (spec deliverable 6, `just report`).

Input is the manifest `dealpoint.eval.four_arm_sweep` writes (one entry per
metered leg: model, arm, tranche, n_cases, est/realized cost, the result
JSONL path) -- preferred over globbing `data/results/` because it names
exactly which files belong to this milestone's headline numbers (spec §6).
`build_report()` itself is pure over already-loaded rows, so it is unit-
testable on fake data without touching disk (gate_m4 DoD: "report generator
tested on fake results").
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

from dealpoint.config import (
    ARM_ORDER,
    ARMS,
    FOUR_ARM_JSON_PATH,
    FOUR_ARM_MD_PATH,
    M4_ENVELOPE_USD,
    M4_MAX_USD,
    M4_TARGET_USD,
    README_PATH,
    VERSIONS_JSON_PATH,
)
from dealpoint.eval.braintrust_adapter import BRAINTRUST_RUNS_PATH
from dealpoint.eval.braintrust_adapter import experiment_name as _bt_experiment_name
from dealpoint.eval.cases import git_sha7
from dealpoint.eval.scorers import SCORE_FIELD_NAMES, majority_baseline, summarise

README_BEGIN = "<!-- BEGIN RESULTS -->"
README_END = "<!-- END RESULTS -->"

ADJACENT_PAIRS: tuple[tuple[str, str], ...] = tuple(itertools.pairwise(ARM_ORDER))

# Below either threshold, a headline pairwise comparison is not supportable
# evidence (corrective task #3: the Haiku A->D 55.6% claim was computed from
# 9 vs 2 scored cases sharing exactly 1 case in common).
MIN_SCORED_N = 5
MIN_COMMON_N = 5

# The per-question columns rendered in four_arm.md's "Per-question metrics"
# section (corrective task #4).
PER_QUESTION_METRICS: tuple[str, ...] = (
    "grounded_accuracy",
    "answer_correct",
    "citation_gold_overlap",
    "gold_seen",
    "required_evidence_met",
    "skill_adherence",
    "execution_failed",
    "cap_hit",
)


# --- reading persisted inputs ------------------------------------------------


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


def _versions() -> dict:
    out: dict = {"git_sha7": git_sha7()}
    if VERSIONS_JSON_PATH.exists():
        try:
            out.update(json.loads(VERSIONS_JSON_PATH.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return out


# --- pure metric computation --------------------------------------------------


def _resolve_experiment_name(entry: dict, braintrust_runs: list[dict]) -> str:
    """experiment_name for a manifest/sweep entry (corrective task #5): prefer
    the name Braintrust actually recorded (by arm/model/git_sha7), falling
    back to recomputing it the same way `braintrust_adapter.run_eval` does.
    """
    arm = entry.get("arm")
    model = entry.get("model")
    sha = entry.get("git_sha7")
    for run in braintrust_runs:
        if run.get("arm") == arm and run.get("model") == model and run.get("git_sha") == sha:
            name = run.get("experiment_name")
            if name:
                return name
    return _bt_experiment_name(arm or "", model or "", entry.get("index_version") or "", sha or "")


def _load_braintrust_runs(path: Path = BRAINTRUST_RUNS_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def config_diff() -> list[dict]:
    diffs = []
    for a_id, b_id in ADJACENT_PAIRS:
        a, b = ARMS[a_id], ARMS[b_id]
        keys = [k for k in a if a[k] != b[k]]
        key = keys[0] if len(keys) == 1 else None
        diffs.append(
            {
                "from": a_id,
                "to": b_id,
                "key": key,
                "from_value": a.get(key) if key else None,
                "to_value": b.get(key) if key else None,
            }
        )
    return diffs


def _score_mean(rows: list[dict], name: str) -> tuple[float | None, int]:
    values = [
        r["scores"].get(name)
        for r in rows
        if r.get("scores") is not None and r["scores"].get(name) is not None
    ]
    numeric = [float(v) for v in values if isinstance(v, bool | int | float)]
    return (sum(numeric) / len(numeric)) if numeric else None, len(numeric)


def arm_metrics(rows: list[dict]) -> dict:
    """Every §2.4 metric, mean + n, over `rows`, plus the brief's aggregate
    abstention names `abstain_recall`/`false_abstain` (corrective task #1;
    reuses `scorers.summarise` rather than duplicating its logic).
    """
    result = {name: dict(zip(("mean", "n"), _score_mean(rows, name))) for name in SCORE_FIELD_NAMES}
    summary = summarise(rows)
    result["abstain_recall"] = summary["abstain_recall"]
    result["false_abstain"] = summary["false_abstain"]
    return result


def per_question_metrics(rows: list[dict]) -> dict:
    by_q: dict[str, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r.get("question_id", ""), []).append(r)
    return {qid: arm_metrics(qrows) for qid, qrows in sorted(by_q.items())}


def skill_adherence_detail_agg(rows: list[dict]) -> dict:
    """Per Appendix-B rule id -> {n_applicable, n_satisfied, rate}, aggregated
    over every row's `skill_rules` detail (spec §3 honesty note)."""
    agg: dict[str, dict] = {}
    for r in rows:
        rules = (r.get("skill_rules") or {}).get("rules") or {}
        for rid, status in rules.items():
            entry = agg.setdefault(rid, {"n_applicable": 0, "n_satisfied": 0})
            if status == "n/a":
                continue
            entry["n_applicable"] += 1
            if status == "satisfied":
                entry["n_satisfied"] += 1
    for entry in agg.values():
        entry["rate"] = (
            entry["n_satisfied"] / entry["n_applicable"] if entry["n_applicable"] else None
        )
    return agg


def paired_flips(rows_a: list[dict], rows_b: list[dict], metric: str = "grounded_accuracy") -> dict:
    """Per-case flips on `metric` between two legs of the SAME model (spec §6:
    \"which cases flipped between arms\")."""
    by_a = {r["case_id"]: r for r in rows_a}
    by_b = {r["case_id"]: r for r in rows_b}
    common = sorted(set(by_a) & set(by_b))
    gained: list[str] = []
    lost: list[str] = []
    unchanged_correct = 0
    unchanged_wrong = 0
    for cid in common:
        va = (by_a[cid].get("scores") or {}).get(metric)
        vb = (by_b[cid].get("scores") or {}).get(metric)
        if va is None or vb is None:
            continue
        if bool(va) == bool(vb):
            if vb:
                unchanged_correct += 1
            else:
                unchanged_wrong += 1
        elif vb and not va:
            gained.append(cid)
        else:
            lost.append(cid)
    return {
        "gained": gained,
        "lost": lost,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _scored_case_ids(rows: list[dict] | None, metric: str) -> set[str]:
    if not rows:
        return set()
    return {
        r["case_id"]
        for r in rows
        if (r.get("scores") or {}).get(metric) is not None and r.get("case_id")
    }


def verdict(
    report: dict,
    model: str,
    from_arm: str,
    to_arm: str,
    metric: str = "grounded_accuracy",
    *,
    rows_a: list[dict] | None = None,
    rows_b: list[dict] | None = None,
) -> dict:
    """The honesty rule, mechanised (spec §7.1): compute the delta and pick one
    of three pre-written sentence templates -- never a hand-written verdict.

    A third template (corrective task #3) fires when either arm's scored `n`
    is below `MIN_SCORED_N`, or the two arms share fewer than `MIN_COMMON_N`
    commonly-scored case ids -- a delta computed from mismatched, mostly-
    disjoint denominators (e.g. n=9 vs n=2 sharing 1 case) is not evidence of
    an improvement or a regression, and must not be reported as either.
    """
    arms_for_model = report.get("arms", {}).get(model, {})
    a = arms_for_model.get(from_arm, {}).get("overall", {}).get(metric, {})
    b = arms_for_model.get(to_arm, {}).get("overall", {}).get(metric, {})
    a_mean, b_mean = a.get("mean"), b.get("mean")
    n_from, n_to = a.get("n") or 0, b.get("n") or 0
    if a_mean is None or b_mean is None:
        return {
            "improves": None,
            "delta": None,
            "metric": metric,
            "sentence": (
                f"Arm {to_arm} vs arm {from_arm} at {model}: insufficient data to compare "
                f"{metric} (missing legs)."
            ),
        }

    n_common = len(_scored_case_ids(rows_a, metric) & _scored_case_ids(rows_b, metric))
    if n_from < MIN_SCORED_N or n_to < MIN_SCORED_N or n_common < MIN_COMMON_N:
        sentence = (
            f"At {model}, arm {to_arm} vs arm {from_arm} on {metric} is not comparable: "
            f"{n_from} vs {n_to} cases scored, {n_common} in common (the rest "
            f"EXECUTION_FAILED/CAP_HIT)."
        )
        return {
            "improves": None,
            "delta": b_mean - a_mean,
            "metric": metric,
            "sentence": sentence,
            "n_from": n_from,
            "n_to": n_to,
            "n_common": n_common,
        }

    delta = b_mean - a_mean
    improves = delta > 0
    if improves:
        sentence = (
            f"At {model}, arm {to_arm} improves {metric} over arm {from_arm} by "
            f"{_pct(delta)} ({_pct(a_mean)} -> {_pct(b_mean)})."
        )
    else:
        sentence = (
            f"At {model}, arm {to_arm} does NOT improve {metric} over arm {from_arm} "
            f"({_pct(a_mean)} -> {_pct(b_mean)}, delta {_pct(delta)}). See adherence, "
            f"fabrication, abstention, trajectory and efficiency deltas below instead."
        )
    return {
        "improves": improves,
        "delta": delta,
        "metric": metric,
        "sentence": sentence,
        "n_from": n_from,
        "n_to": n_to,
        "n_common": n_common,
    }


DELTA_METRICS: tuple[str, ...] = (
    "skill_adherence",
    "fabrication",
    "abstain_correct",
    "tool_calls",
    "wall_ms",
    "usd",
)


def delta_table(report: dict, model: str, from_arm: str, to_arm: str) -> dict:
    arms_for_model = report.get("arms", {}).get(model, {})
    a = arms_for_model.get(from_arm, {}).get("overall", {})
    b = arms_for_model.get(to_arm, {}).get("overall", {})
    out = {}
    for name in DELTA_METRICS:
        am = (a.get(name) or {}).get("mean")
        bm = (b.get(name) or {}).get("mean")
        out[name] = {
            "from": am,
            "to": bm,
            "delta": (bm - am) if (am is not None and bm is not None) else None,
        }
    return out


# --- assembling the full report ---------------------------------------------


def build_report(
    rows_by_arm_model: dict[tuple[str, str], list[dict]],
    *,
    sweeps: list[dict] | None = None,
    caching: dict | None = None,
    dev_loop_spend: dict | None = None,
    m4_ledger_total: float | None = None,
    baseline_cases: list[dict] | None = None,
) -> dict:
    """Pure assembly of the full `four_arm.json` payload from already-scored
    result rows, keyed `(model, arm)`. No I/O beyond what the caller passed in
    -- this is what the offline `gate_m4` report-generator tests exercise.
    """
    sweeps = sweeps or []
    models = sorted({m for (m, _a) in rows_by_arm_model})

    arms: dict[str, dict] = {}
    skill_detail: dict[str, dict] = {}
    for model in models:
        arms[model] = {}
        skill_detail[model] = {}
        for arm in ARM_ORDER:
            rows = rows_by_arm_model.get((model, arm))
            if rows is None:
                continue
            arms[model][arm] = {
                "overall": arm_metrics(rows),
                "per_question": per_question_metrics(rows),
                "n_cases": len(rows),
            }
            skill_detail[model][arm] = skill_adherence_detail_agg(rows)

    report: dict = {
        "sweeps": sweeps,
        "arms": arms,
        "skill_adherence_detail": skill_detail,
        "config_diff": config_diff(),
        "budget_scaled": True,
        "budget_scaled_note": (
            "Every headline number in this report is from a 32-case frozen subset "
            "(18 cases at Haiku), not the brief's full 167-case test set -- budget-scaled "
            "per the engineer's 2026-09-04 instruction, never the full benchmark."
        ),
        "cost": {
            "target_usd": M4_TARGET_USD,
            "absolute_usd": M4_MAX_USD,
            "envelope_usd": M4_ENVELOPE_USD,
            "m4_ledger_total": m4_ledger_total,
            "estimate_vs_realised": [
                {
                    "leg": s.get("leg"),
                    "arm": s.get("arm"),
                    "model": s.get("model"),
                    "tranche": s.get("tranche"),
                    "n_cases": s.get("n_cases"),
                    "est_usd": s.get("est_usd"),
                    "realized_usd": s.get("realized_usd"),
                    "experiment_name": s.get("experiment_name"),
                }
                for s in sweeps
            ],
        },
        "dev_loop_spend": dev_loop_spend or {},
        "caching": caching or {},
        "versions": _versions(),
    }

    if baseline_cases is not None:
        report["majority_baseline"] = majority_baseline(baseline_cases)
    else:
        report["majority_baseline"] = None

    # paired flips + honesty verdicts, per model where both arms of the pair exist
    paired: dict[str, dict] = {}
    verdicts: dict[str, dict] = {}
    deltas: dict[str, dict] = {}
    for model in models:
        paired[model] = {}
        verdicts[model] = {}
        deltas[model] = {}
        for from_arm, to_arm in (*ADJACENT_PAIRS, ("A", "D")):
            rows_a = rows_by_arm_model.get((model, from_arm))
            rows_b = rows_by_arm_model.get((model, to_arm))
            key = f"{from_arm}_to_{to_arm}"
            if rows_a is None or rows_b is None:
                continue
            paired[model][key] = paired_flips(rows_a, rows_b)
            verdicts[model][key] = verdict(
                report, model, from_arm, to_arm, rows_a=rows_a, rows_b=rows_b
            )
            deltas[model][key] = delta_table(report, model, from_arm, to_arm)

    report["paired"] = paired
    report["verdicts"] = verdicts
    report["deltas"] = deltas
    # spec-named top-level convenience field: C -> D specifically (the
    # headline honesty-rule comparison), per model.
    report["arm_d_improves_over_c"] = {
        model: verdicts.get(model, {}).get("C_to_D") for model in models
    }
    return report


# --- markdown rendering -------------------------------------------------


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Four-arm experiment on the frozen test set (M4)")
    lines.append("")
    lines.append(
        "This report covers a budget-scaled, 32-case frozen discriminative subset "
        "(18 cases at Haiku) of MAUD's test set -- not the brief's full 167-case "
        "test set. Task here is document -> (answer, citation); MAUD's published "
        "task is span -> answer, so scores are not comparable to MAUD leaderboard "
        "numbers. Every score in this report is deterministic Python over expert "
        "labels; there is no model judging in M4."
    )
    lines.append("")

    lines.append("## Config diff between adjacent arms")
    lines.append("")
    lines.append("| from | to | key | from value | to value |")
    lines.append("|---|---|---|---|---|")
    for d in report.get("config_diff", []):
        lines.append(
            f"| {d['from']} | {d['to']} | {d['key']} | `{d['from_value']}` | `{d['to_value']}` |"
        )
    lines.append("")

    majority = report.get("majority_baseline") or {}
    lines.append(f"Majority-baseline overall accuracy: {_pct(majority.get('overall'))}")
    lines.append("")

    for model, model_arms in report.get("arms", {}).items():
        lines.append(f"## Model: `{model}`")
        lines.append("")
        lines.append(
            "| arm | n_cases | n_scored | grounded_accuracy | answer_correct | fabrication | "
            "abstain_correct | skill_adherence | execution_failed | cap_hit | tool_calls | usd |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARM_ORDER:
            data = model_arms.get(arm)
            if not data:
                continue
            o = data["overall"]
            n_cases = data["n_cases"]
            n_scored = o["grounded_accuracy"]["n"]
            tool_calls_mean = o["tool_calls"]["mean"]
            tool_calls_str = f"{tool_calls_mean:.2f}" if tool_calls_mean is not None else "n/a"
            usd_mean = o["usd"]["mean"]
            usd_str = f"${usd_mean:.4f}" if usd_mean is not None else "n/a"
            grounded_str = f"{_pct(o['grounded_accuracy']['mean'])} ({n_scored}/{n_cases})"
            lines.append(
                f"| {arm} | {n_cases} | {n_scored} | {grounded_str} | "
                f"{_pct(o['answer_correct']['mean'])} | {_pct(o['fabrication']['mean'])} | "
                f"{_pct(o['abstain_correct']['mean'])} | {_pct(o['skill_adherence']['mean'])} | "
                f"{_pct(o['execution_failed']['mean'])} | {_pct(o['cap_hit']['mean'])} | "
                f"{tool_calls_str} | {usd_str} |"
            )
        lines.append("")

    lines.append("## Per-question metrics")
    lines.append("")
    lines.append(
        "One table per (model, arm), question_id x metric, with the scored n "
        "alongside every mean (corrective task #4). The majority-baseline row "
        "(label-only, never model output) is printed once below, shared by every arm."
    )
    lines.append("")
    majority_by_q = {k: v for k, v in majority.items() if k != "overall"}
    if majority_by_q:
        lines.append(
            "Majority-baseline accuracy per question: "
            + ", ".join(f"{qid}={_pct(acc)}" for qid, acc in sorted(majority_by_q.items()))
        )
        lines.append("")
    for model, model_arms in report.get("arms", {}).items():
        for arm in ARM_ORDER:
            data = model_arms.get(arm)
            if not data:
                continue
            per_q = data.get("per_question") or {}
            if not per_q:
                continue
            lines.append(f"### {model} / arm {arm}")
            lines.append("")
            header = "| question_id | " + " | ".join(PER_QUESTION_METRICS) + " |"
            sep = "|" + "---|" * (len(PER_QUESTION_METRICS) + 1)
            lines.append(header)
            lines.append(sep)
            for qid in sorted(per_q):
                qm = per_q[qid]
                cells = []
                for metric in PER_QUESTION_METRICS:
                    m = qm.get(metric, {})
                    cells.append(f"{_pct(m.get('mean'))} (n={m.get('n', 0)})")
                lines.append(f"| {qid} | " + " | ".join(cells) + " |")
            lines.append("")

    lines.append("## Honesty-rule verdicts")
    lines.append("")
    for model, model_verdicts in report.get("verdicts", {}).items():
        for pair_key, v in model_verdicts.items():
            lines.append(f"- **{model} / {pair_key}**: {v['sentence']}")
    lines.append("")

    lines.append("## Adherence / fabrication / abstention / trajectory / efficiency deltas")
    lines.append("")
    for model, model_deltas in report.get("deltas", {}).items():
        for pair_key, dt in model_deltas.items():
            lines.append(f"### {model} / {pair_key}")
            lines.append("")
            lines.append("| metric | from | to | delta |")
            lines.append("|---|---|---|---|")
            for name, vals in dt.items():
                lines.append(f"| {name} | {vals['from']} | {vals['to']} | {vals['delta']} |")
            lines.append("")

    lines.append("## Paired per-case flips")
    lines.append("")
    for model, model_paired in report.get("paired", {}).items():
        for pair_key, p in model_paired.items():
            lines.append(
                f"- **{model} / {pair_key}**: gained {p['gained']}, lost {p['lost']}, "
                f"unchanged_correct={p['unchanged_correct']}, unchanged_wrong={p['unchanged_wrong']}"
            )
    lines.append("")

    lines.append("## Skill-adherence per-rule detail")
    lines.append("")
    lines.append(
        "Note (honesty, spec §3): arm A lacks several tools rules 2/3/4/6 need, so its "
        "applicable-rule set is structurally smaller than arms B/C/D's. n_applicable and "
        "n_satisfied are printed per rule per arm, so an A-vs-D adherence ratio comparison "
        "is never read as like-for-like."
    )
    lines.append("")
    for model, model_detail in report.get("skill_adherence_detail", {}).items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| arm | rule | n_applicable | n_satisfied | rate |")
        lines.append("|---|---|---|---|---|")
        for arm, rules in model_detail.items():
            for rid in sorted(rules, key=int):
                r = rules[rid]
                lines.append(
                    f"| {arm} | {rid} | {r['n_applicable']} | {r['n_satisfied']} | {_pct(r['rate'])} |"
                )
        lines.append("")

    lines.append("## Cost: estimate vs realised")
    lines.append("")
    cost = report.get("cost", {})
    lines.append(
        f"Target: ${cost.get('target_usd')}, absolute: ${cost.get('absolute_usd')}, "
        f"envelope: ${cost.get('envelope_usd')}, M4 ledger total: ${cost.get('m4_ledger_total')}"
    )
    lines.append("")
    lines.append("| leg | arm | model | tranche | n_cases | est_usd | realized_usd |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in cost.get("estimate_vs_realised", []):
        lines.append(
            f"| {row.get('leg')} | {row.get('arm')} | {row.get('model')} | {row.get('tranche')} | "
            f"{row.get('n_cases')} | {row.get('est_usd')} | {row.get('realized_usd')} |"
        )
    lines.append("")

    caching = report.get("caching") or {}
    if caching:
        lines.append("## Prompt-caching diagnosis")
        lines.append("")
        lines.append(f"```json\n{json.dumps(caching, indent=2, sort_keys=True)}\n```")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Arm C is labelled `agent-hybrid-rrf` here, not the brief's `agent-hybrid-rerank`: "
        "the M3 tournament winner is hybrid RRF *without* rerank (hit@5 0.9138 vs 0.8103 "
        "for dense; also beat hybrid_rrf_rerank on MRR). The measurement governs per the "
        "milestone spec; see dealpoint/config.py for the full rationale."
    )
    lines.append(
        "- Spec-internal inconsistency (not spec-vs-brief): the re-sized deliverable 5 and "
        "the Definition of Done both name the GLM headline as the constant-model comparison "
        "with Haiku as the A/D replication; an older paragraph in the same spec says the "
        "reverse. The later, more specific instruction (deliverable 5 + DoD) is followed."
    )
    lines.append(
        "- Brief §7's relative gate ('no arm regresses vs its previously recorded dev "
        "result') is vacuous this milestone: M2's only recorded run is 3 dev cases at arm B, "
        "not a four-arm baseline to compare against."
    )
    lines.append(
        "- Arms B and C at Haiku: not run (budget). Arithmetic: 2 arms x 18 cases at the "
        "measured Haiku per-case cost (~$0.0227/case) is ~$0.82; adding that to the $0.75 "
        "already realised for M4 would take the milestone to ~$1.57, past the $1.50 target "
        "(though still under the $3.00 absolute) -- so they were deliberately skipped."
    )
    lines.append(
        "- `dealpoint/eval/run.py` gained a per-process retriever cache after the final "
        "sweep finished. This is a runner-only fix (it prevents a second `QdrantClient` "
        "from wedging on the same on-disk index within one process) and does not change "
        "what any model saw or any score computed -- no test case was re-run because of it."
    )
    lines.append(
        f"- {report.get('budget_scaled_note', '')}"
    )
    lines.append("")

    return "\n".join(lines)


def _readme_results_block(report: dict) -> str:
    majority = report.get("majority_baseline") or {}
    first_sentence = (
        "Results below are from a **budget-scaled** 32-case frozen discriminative subset "
        "of MAUD's test set (18 cases at `anthropic/claude-haiku-4.5`), not the brief's "
        "full 167-case design; the task here is document -> (answer, citation) while "
        "MAUD's published task is span -> answer, so scores are **not comparable** to "
        "MAUD leaderboard numbers, and every metric below is **objective** (deterministic "
        "Python over expert labels) -- M4 has no model judging."
    )
    lines = [first_sentence, "", f"Majority-baseline overall accuracy: {_pct(majority.get('overall'))}.", ""]
    for model, model_arms in report.get("arms", {}).items():
        lines.append(f"**Model `{model}`:**")
        lines.append("")
        lines.append("| arm | n_cases | n_scored | grounded_accuracy |")
        lines.append("|---|---|---|---|")
        for arm in ARM_ORDER:
            data = model_arms.get(arm)
            if not data:
                continue
            o = data["overall"]
            n_cases = data["n_cases"]
            n_scored = o["grounded_accuracy"]["n"]
            lines.append(
                f"| {arm} | {n_cases} | {n_scored} | "
                f"{_pct(o['grounded_accuracy']['mean'])} ({n_scored}/{n_cases}) |"
            )
        lines.append("")
        exec_failed_parts = []
        cap_hit_parts = []
        for arm in ARM_ORDER:
            data = model_arms.get(arm)
            if not data:
                continue
            o = data["overall"]
            exec_failed_parts.append(f"{arm}={_pct(o['execution_failed']['mean'])}")
            cap_hit_parts.append(f"{arm}={_pct(o['cap_hit']['mean'])}")
        if exec_failed_parts:
            lines.append(
                f"Overall EXECUTION_FAILED rate by arm: {', '.join(exec_failed_parts)}. "
                f"Overall CAP_HIT rate by arm: {', '.join(cap_hit_parts)}."
            )
            lines.append("")
    for model, model_verdicts in report.get("verdicts", {}).items():
        cd = model_verdicts.get("C_to_D")
        if cd:
            lines.append(f"- {cd['sentence']}")
    return "\n".join(lines)


def regenerate_readme(report: dict, readme_path: Path = README_PATH) -> None:
    block = _readme_results_block(report)
    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
    else:
        text = (
            "# dealpoint-eval\n\n"
            "M&A deal-point review agent evaluation harness (MAUD v1, CC BY 4.0, "
            "Zenodo 7500064).\n\n"
            "## Results\n\n"
            f"{README_BEGIN}\n{README_END}\n"
        )
    pattern = re.compile(re.escape(README_BEGIN) + r".*?" + re.escape(README_END), re.DOTALL)
    replacement = f"{README_BEGIN}\n{block}\n{README_END}"
    if pattern.search(text):
        text = pattern.sub(replacement, text)
    else:
        text = text.rstrip("\n") + f"\n\n## Results\n\n{replacement}\n"
    readme_path.write_text(text, encoding="utf-8")


# --- disk-driven pipeline: manifest -> report -------------------------------


def load_rows_from_manifest(manifest_path: Path) -> dict[tuple[str, str], list[dict]]:
    from dealpoint.eval.four_arm_sweep import load_manifest

    out: dict[tuple[str, str], list[dict]] = {}
    for entry in load_manifest(manifest_path):
        key = (entry["model"], entry["arm"])
        rows = _read_jsonl(Path(entry["results_path"]))
        out.setdefault(key, [])
        out[key].extend(rows)
    return out


def generate_report(
    manifest_path: Path | None = None,
    out_json: Path = FOUR_ARM_JSON_PATH,
    out_md: Path = FOUR_ARM_MD_PATH,
    caching: dict | None = None,
    dev_loop_spend: dict | None = None,
) -> dict:
    from dealpoint.config import REPORTS_DIR
    from dealpoint.eval.cases import find_case
    from dealpoint.eval.four_arm_sweep import MANIFEST_PATH, load_manifest
    from dealpoint.eval.spend import realized_by_tag
    from dealpoint.eval.subset import load_subset_case_ids

    manifest_path = manifest_path or MANIFEST_PATH
    rows_by_arm_model = load_rows_from_manifest(manifest_path)
    sweeps = load_manifest(manifest_path)
    braintrust_runs = _load_braintrust_runs()
    for entry in sweeps:
        entry["experiment_name"] = _resolve_experiment_name(entry, braintrust_runs)

    caching_probe_path = REPORTS_DIR / "caching_probe.json"
    if caching is None and caching_probe_path.exists():
        try:
            caching = json.loads(caching_probe_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            caching = {}

    dev_smoke_path = REPORTS_DIR / "dev_smoke_m4.json"
    if dev_loop_spend is None and dev_smoke_path.exists():
        try:
            dev_loop_spend = json.loads(dev_smoke_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            dev_loop_spend = {}

    baseline_cases = None
    case_ids = load_subset_case_ids("test_subset_v1")
    try:
        baseline_cases = [find_case(cid) for cid in case_ids]
    except KeyError:
        baseline_cases = None

    m4_ledger_total = realized_by_tag().get("m4")

    report = build_report(
        rows_by_arm_model,
        sweeps=sweeps,
        caching=caching or {},
        dev_loop_spend=dev_loop_spend or {},
        m4_ledger_total=m4_ledger_total,
        baseline_cases=baseline_cases,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

    md = render_markdown(report)
    out_md.write_text(md, encoding="utf-8")

    regenerate_readme(report)

    return report


def main(argv: list[str] | None = None) -> int:
    report = generate_report()
    print(
        json.dumps(
            {"n_models": len(report["arms"]), "n_legs": len(report["sweeps"])},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
