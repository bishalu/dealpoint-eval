"""M6 report: `data/reports/pareto.json` + `pareto.md` + `pareto.svg` +
README PARETO block (spec deliverable 4).

Same discipline as `dealpoint.eval.report`: a pure builder over already-
loaded rows (`build_pareto_report`), plus thin disk-driven loaders in
`generate_report`. The Pareto-frontier computation (`pareto_frontier`) is
the spec's named gate_m6 unit-test target: deterministic and total over any
input, including missing axes and exact ties.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from dealpoint.config import (
    M6_ENVELOPE_USD,
    M6_TARGET_USD,
    PARETO_JSON_PATH,
    PARETO_MD_PATH,
    PARETO_SLATE_PATH,
    PARETO_SVG_PATH,
    README_PATH,
)
from dealpoint.eval.report import arm_metrics

DIMENSIONS: tuple[str, ...] = ("reasoning", "evidence", "trajectory", "professional")

README_BEGIN = "<!-- BEGIN PARETO -->"
README_END = "<!-- END PARETO -->"


# --- pure: percentiles -------------------------------------------------


def percentile(values: Sequence[float | None], q: float) -> float | None:
    """Nearest-rank percentile, pure Python (no numpy/scipy). `None` if empty."""
    clean = sorted(v for v in values if isinstance(v, (int, float)) and v is not None)
    n = len(clean)
    if n == 0:
        return None
    rank = max(1, min(n, math.ceil(q / 100 * n)))
    return clean[rank - 1]


# --- pure: the Pareto frontier -------------------------------------------


def pareto_frontier(
    points: list[dict],
    *,
    x_key: str = "usd_per_case",
    y_key: str = "grounded_accuracy",
    id_key: str = "model",
) -> list[dict]:
    """Deterministic, total frontier computation over `points`.

    Minimise `x`, maximise `y`. Returns one dict per input point, in
    ascending-`x` order (points missing either axis sorted last, by id):
    `{id_key: ..., "frontier": bool, "frontier_note": str | None,
      "tied_with": str | None}`.

    Rules (spec DoD -- the named unit-test target):
    - A point with `None` on either axis is excluded, `frontier_note`
      records which axis is missing.
    - Sort comparable points by `(x asc, y desc, id asc)`; sweep keeping the
      running best `y`; a point joins the frontier iff its `y` is strictly
      greater than every `y` seen so far.
    - Exact ties on BOTH axes: the lexicographically-smallest id is kept on
      the frontier; the other(s) are marked `tied_with` that id (never a
      generic "dominated" note, and never a coin flip).
    - Order-independent: shuffling the input list gives the same result.
    """
    missing: list[dict] = []
    comparable: list[dict] = []
    for p in points:
        pid = p[id_key]
        x = p.get(x_key)
        y = p.get(y_key)
        if x is None or y is None:
            axis = x_key if x is None else y_key
            missing.append(
                {
                    id_key: pid,
                    "frontier": False,
                    "frontier_note": f"not comparable (missing {axis})",
                    "tied_with": None,
                }
            )
        else:
            comparable.append({id_key: pid, "_x": x, "_y": y})

    comparable.sort(key=lambda r: (r["_x"], -r["_y"], r[id_key]))

    out: list[dict] = []
    best_y: float | None = None
    winner_at_xy: dict[tuple, str] = {}
    for r in comparable:
        key = (r["_x"], r["_y"])
        if best_y is not None and r["_y"] <= best_y:
            if key in winner_at_xy:
                out.append(
                    {
                        id_key: r[id_key],
                        "frontier": False,
                        "frontier_note": None,
                        "tied_with": winner_at_xy[key],
                    }
                )
            else:
                out.append(
                    {
                        id_key: r[id_key],
                        "frontier": False,
                        "frontier_note": "dominated",
                        "tied_with": None,
                    }
                )
        else:
            out.append(
                {id_key: r[id_key], "frontier": True, "frontier_note": None, "tied_with": None}
            )
            best_y = r["_y"]
            winner_at_xy[key] = r[id_key]

    missing.sort(key=lambda r: r[id_key])
    return out + missing


# --- pure: per-model metrics ---------------------------------------------


def per_model_metrics(rows: list[dict]) -> dict:
    """Every §2.6 objective metric, mean + n (via `report.arm_metrics`), plus
    latency percentiles and realised (never estimated) cost.
    """
    metrics = arm_metrics(rows)
    wall_values: list[float] = []
    usd_values: list[float] = []
    for r in rows:
        scores = r.get("scores") or {}
        wall_ms = scores.get("wall_ms")
        if isinstance(wall_ms, (int, float)):
            wall_values.append(float(wall_ms))
        usd_val = scores.get("usd")
        if isinstance(usd_val, (int, float)):
            usd_values.append(float(usd_val))
    n_cases = len(rows)
    metrics["latency_median_ms"] = percentile(wall_values, 50)
    metrics["latency_p90_ms"] = percentile(wall_values, 90)
    metrics["usd_per_case"] = (sum(usd_values) / n_cases) if n_cases else None
    metrics["total_usd"] = round(sum(usd_values), 6) if usd_values else 0.0
    metrics["n_cases"] = n_cases
    return metrics


def judged_quality(judge_rows: list[dict], variant_id: str) -> dict | None:
    """Mean-of-judges per dimension + overall, for one `variant_id`, over
    `ok: true` rows only. `None` if no judge rows exist for this variant
    (the caller records a reason alongside).
    """
    rows = [r for r in judge_rows if r.get("variant_id") == variant_id and r.get("ok")]
    if not rows:
        return None
    out: dict = {
        "secondary": True,
        "label": "model-judged, secondary, never objective truth",
        "variant_id": variant_id,
    }
    all_vals: list[float] = []
    for dim in DIMENSIONS:
        vals = [r[dim] for r in rows if r.get(dim) is not None]
        out[dim] = {"mean": (sum(vals) / len(vals)) if vals else None, "n": len(vals)}
        all_vals.extend(vals)
    out["overall"] = {
        "mean": (sum(all_vals) / len(all_vals)) if all_vals else None,
        "n": len(all_vals),
    }
    return out


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


# --- pure: partial/abandoned sweep legs (corrective-cycle finding) --------


def _detect_passes(rows: list[dict]) -> list[dict]:
    """Chronological (ts-ordered) per-case-id batches for one model's ledger
    rows: a new pass begins when a `case_id` repeats after already being
    seen in the current batch -- the same wraparound signal
    `dealpoint.eval.spend._batch_by_ts_wraparound` uses to recover
    per-execution totals, applied here to tell a completed run apart from
    an earlier abandoned attempt at the same case_set.

    Returns passes in chronological order: `[{"case_ids": set(...), "usd":
    float}, ...]`.
    """
    ordered = sorted(rows, key=lambda r: r.get("ts", ""))
    passes: list[dict] = []
    current: dict[str, float] = {}
    prev_case_id: str | None = None
    for row in ordered:
        case_id = row.get("case_id")
        if not case_id:
            continue
        usd = float(row.get("usd", 0.0) or 0.0)
        if case_id != prev_case_id and case_id in current:
            passes.append({"case_ids": set(current), "usd": round(sum(current.values()), 6)})
            current = {}
        current[case_id] = current.get(case_id, 0.0) + usd
        prev_case_id = case_id
    if current:
        passes.append({"case_ids": set(current), "usd": round(sum(current.values()), 6)})
    return passes


def compute_partial_runs(
    m6_sweep_rows: list[dict],
    completed_models: dict[str, dict],
    manifest_entries: list[dict],
) -> list[dict]:
    """One entry per model with ledger M6 sweep spend that either (a) has no
    completed 32-case results file, or (b) whose ledger sweep spend exceeds
    the `total_usd` of its completed results file -- i.e. an earlier
    abandoned pass at the same case_set that a later, completed pass
    superseded. Corrective-cycle finding: a model that spent money must
    never be silently absent from every report file.

    `completed_models` maps `model -> {"total_usd": float, "n_cases": int}`
    for every model with a completed (non-reused) results file already
    loaded by the caller. `manifest_entries` is `pareto_manifest.json`'s
    rows, read only for the `status` field (the stop cause) -- every spend
    figure here comes from the ledger, the single source of truth for what
    was actually spent.
    """
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in m6_sweep_rows:
        model = row.get("model")
        if model:
            by_model[model].append(row)

    status_by_model: dict[str, str] = {}
    for entry in manifest_entries:
        status = entry.get("status")
        model = entry.get("model")
        if status and model and model not in status_by_model:
            status_by_model[model] = status

    out: list[dict] = []
    for model in sorted(by_model):
        rows = by_model[model]
        total_ledger_usd = round(sum(float(r.get("usd", 0.0) or 0.0) for r in rows), 6)
        completed = completed_models.get(model)

        if completed is None:
            case_ids = {r.get("case_id") for r in rows if r.get("case_id")}
            reason = status_by_model.get(model) or "started, not completed"
            out.append(
                {
                    "model": model,
                    "n_cases_attempted": len(case_ids),
                    "n_cases_completed": 0,
                    "realized_usd": total_ledger_usd,
                    "reason": reason,
                }
            )
            continue

        excess = round(total_ledger_usd - completed["total_usd"], 6)
        if excess <= 0.001:
            continue

        passes = _detect_passes(rows)
        earlier = passes[:-1] if len(passes) > 1 else []
        earlier_case_ids: set = set()
        for p in earlier:
            earlier_case_ids |= p["case_ids"]
        earlier_usd = round(sum(p["usd"] for p in earlier), 6) if earlier else excess
        reason = status_by_model.get(model) or (
            f"superseded by the completed {completed['n_cases']}-case pass"
        )
        out.append(
            {
                "model": model,
                "n_cases_attempted": len(earlier_case_ids),
                "n_cases_completed": 0,
                "realized_usd": earlier_usd,
                "reason": reason,
            }
        )
    return out


# --- pure: assembling the full report ------------------------------------


def build_pareto_report(
    model_entries: list[dict],
    *,
    judge_rows: list[dict] | None = None,
    not_run: list[dict] | None = None,
    partial_runs: list[dict] | None = None,
    slate: dict | None = None,
    spend: dict | None = None,
    held_fixed: dict | None = None,
    comparability: dict | None = None,
    decisions: list[dict] | None = None,
    brief_differences: list[dict] | None = None,
    caveats: list[str] | None = None,
    generated_at: str | None = None,
    git_sha7: str | None = None,
) -> dict:
    """Pure assembly of `pareto.json` from already-loaded per-model rows.

    `model_entries`: one dict per completed model --
    `{model, family, rank, reused, rows, variant_id, prompt_usd_per_token,
      completion_usd_per_token, price_basis, est_usd, est_basis,
      results_path}`. `rows` is a list of result-row dicts (the runner's
    `data/results/*.jsonl` shape). A model whose rows are ALL
    `EXECUTION_FAILED` still appears (grounded_accuracy.n == 0, mean ==
    None) and is simply excluded from the frontier via the missing-axis
    branch of `pareto_frontier` -- never a crash.
    """
    judge_rows = judge_rows or []

    models_out: dict = {}
    frontier_points: list[dict] = []
    for entry in model_entries:
        model = entry["model"]
        metrics = per_model_metrics(entry.get("rows") or [])
        jq = None
        variant_id = entry.get("variant_id")
        if variant_id:
            jq = judged_quality(judge_rows, variant_id)
        model_block = {
            "family": entry.get("family"),
            "rank": entry.get("rank"),
            "reused": bool(entry.get("reused", False)),
            "n_cases": metrics["n_cases"],
            "prompt_usd_per_token": entry.get("prompt_usd_per_token"),
            "completion_usd_per_token": entry.get("completion_usd_per_token"),
            "price_basis": entry.get("price_basis"),
            "grounded_accuracy": metrics["grounded_accuracy"],
            "abstain_recall": metrics.get("abstain_recall"),
            "execution_failed": metrics["execution_failed"],
            "cap_hit": metrics["cap_hit"],
            "answer_correct": metrics["answer_correct"],
            "citation_gold_overlap": metrics["citation_gold_overlap"],
            "skill_adherence": metrics["skill_adherence"],
            "latency_median_ms": metrics["latency_median_ms"],
            "latency_p90_ms": metrics["latency_p90_ms"],
            "usd_per_case": metrics["usd_per_case"],
            "total_usd": metrics["total_usd"],
            "judged_quality": jq
            if jq is not None
            else {"secondary": True, "reason": "no judge rows for this variant yet"},
            "results_path": entry.get("results_path"),
            "est_usd": entry.get("est_usd"),
            "est_basis": entry.get("est_basis"),
        }
        models_out[model] = model_block
        frontier_points.append(
            {
                "model": model,
                "usd_per_case": metrics["usd_per_case"],
                "grounded_accuracy": (metrics["grounded_accuracy"] or {}).get("mean"),
            }
        )

    frontier_rows = pareto_frontier(frontier_points)
    frontier_by_model = {r["model"]: r for r in frontier_rows}
    for model, block in models_out.items():
        fr = frontier_by_model.get(model, {})
        block["frontier"] = fr.get("frontier", False)
        if fr.get("frontier_note"):
            block["frontier_note"] = fr["frontier_note"]
        if fr.get("tied_with"):
            block["tied_with"] = fr["tied_with"]

    frontier_models = [r["model"] for r in frontier_rows if r["frontier"]]

    spend = dict(spend or {})
    probe_usd = spend.get("m6_probe_usd", 0.0) or 0.0
    sweep_usd = spend.get("m6_sweep_usd", 0.0) or 0.0
    judge_usd = spend.get("m6_judge_usd", 0.0) or 0.0
    spend.setdefault("target_usd", M6_TARGET_USD)
    spend.setdefault("envelope_usd", M6_ENVELOPE_USD)
    spend.setdefault("m6_realized_usd", round(probe_usd + sweep_usd + judge_usd, 6))

    report = {
        "generated_at": generated_at,
        "git_sha7": git_sha7,
        "version": "v1",
        "budget_scaled": True,
        "budget_scaled_note": (
            "32-case frozen subset per model (18 at Haiku); not the brief's full-scale design."
        ),
        "held_fixed": held_fixed
        or {"arm": "D", "sentence": "Only the model varies."},
        "models": models_out,
        "frontier": {
            "rule": (
                "Minimise realised $/case, maximise grounded_accuracy. A point is dominated "
                "iff another has x<=x' and y>=y' with at least one strict inequality. Exact "
                "ties on both axes keep the lexicographically-smallest model id."
            ),
            "x": "usd_per_case",
            "y": "grounded_accuracy",
            "models": frontier_models,
        },
        "not_run": list(not_run or []),
        "partial_runs": list(partial_runs or []),
        "slate": slate or {},
        "spend": spend,
        "comparability": comparability or {},
        "decisions": list(decisions or []),
        "brief_differences": list(brief_differences or []),
        "caveats": list(
            caveats
            or [
                "Judged quality is secondary and model-judged; it is never objective truth.",
                "The reported frontier is a frontier of the cheap tier; no ceiling model was run.",
                (
                    "Scores are not comparable to the MAUD leaderboard (this task is "
                    "document -> (answer, citation); MAUD's published task is span -> answer)."
                ),
            ]
        ),
    }
    return report


# --- markdown rendering ---------------------------------------------------


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Model cost/quality Pareto experiment (M6)")
    lines.append("")
    lines.append(
        "This report is **budget-scaled** (32-case frozen subset per model, 18 at Haiku) -- "
        "grounded_accuracy is **objective** (deterministic Python over expert labels); judged "
        "quality is **secondary and model-judged, never objective truth**; scores are **not "
        "comparable** to the MAUD leaderboard."
    )
    lines.append("")
    hf = report.get("held_fixed") or {}
    if hf.get("sentence"):
        lines.append(hf["sentence"])
        lines.append("")

    lines.append("## Per-model results")
    lines.append("")
    lines.append(
        "| model | n_cases | grounded_accuracy | abstain_recall | execution_failed | cap_hit | "
        "median latency (ms) | p90 latency (ms) | $/case | total $ | frontier |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, m in report.get("models", {}).items():
        ga = m.get("grounded_accuracy") or {}
        n_scored = ga.get("n")
        n_cases = m.get("n_cases")
        ga_str = f"{_pct(ga.get('mean'))} ({n_scored}/{n_cases})"
        usd_per_case = m.get("usd_per_case")
        usd_per_case_str = f"${usd_per_case:.5f}" if usd_per_case is not None else "n/a"
        total_usd = m.get("total_usd")
        total_usd_str = f"${total_usd:.4f}" if total_usd is not None else "n/a"
        lines.append(
            f"| {model} | {n_cases} | {ga_str} | "
            f"{_pct((m.get('abstain_recall') or {}).get('mean'))} | "
            f"{_pct((m.get('execution_failed') or {}).get('mean'))} | "
            f"{_pct((m.get('cap_hit') or {}).get('mean'))} | "
            f"{m.get('latency_median_ms')} | {m.get('latency_p90_ms')} | "
            f"{usd_per_case_str} | {total_usd_str} | "
            f"{'yes' if m.get('frontier') else 'no'} |"
        )
    lines.append("")

    frontier = report.get("frontier") or {}
    lines.append(f"**Frontier** ({frontier.get('rule', '')}): {frontier.get('models')}")
    lines.append("")

    not_run = report.get("not_run") or []
    if not_run:
        lines.append("## Not run")
        lines.append("")
        for nr in not_run:
            lines.append(f"- `{nr.get('model')}`: {nr.get('reason')}")
        lines.append("")

    partial_runs = report.get("partial_runs") or []
    if partial_runs:
        lines.append("## Partially run (metered, not completed)")
        lines.append("")
        lines.append("| model | n_cases_completed/n_cases_attempted | realized_usd | reason |")
        lines.append("|---|---|---|---|")
        for pr in partial_runs:
            n_completed = pr.get("n_cases_completed")
            n_attempted = pr.get("n_cases_attempted")
            realized = pr.get("realized_usd")
            realized_str = f"${realized:.6f}" if realized is not None else "n/a"
            lines.append(
                f"| {pr.get('model')} | {n_completed}/{n_attempted} | {realized_str} | "
                f"{pr.get('reason')} |"
            )
        lines.append("")

    comparability = report.get("comparability") or {}
    if comparability.get("note"):
        lines.append("## Comparability note")
        lines.append("")
        lines.append(comparability["note"])
        lines.append("")

    spend = report.get("spend") or {}
    lines.append("## Spend")
    lines.append("")
    lines.append(f"```json\n{json.dumps(spend, indent=2, sort_keys=True)}\n```")
    lines.append("")

    lines.append("## Recorded decisions")
    lines.append("")
    for d in report.get("decisions", []):
        lines.append(f"- **{d.get('id')}**: {d.get('decision')} -- {d.get('reason')}")
    lines.append("")

    lines.append("## Brief-vs-spec differences")
    lines.append("")
    for d in report.get("brief_differences", []):
        lines.append(f"- **{d.get('topic', d.get('id'))}**: {d.get('difference')}")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    for c in report.get("caveats", []):
        lines.append(f"- {c}")
    lines.append("")

    return "\n".join(lines)


# --- SVG chart (hand-written, no matplotlib; deterministic) --------------


def _short_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def render_svg(report: dict) -> str:
    """A deterministic, hand-written scatter chart: x = realised $/case (log
    scale), y = grounded accuracy, marker radius ~ median latency, frontier
    models joined by a polyline and filled, dominated models hollow.

    No timestamp, no random ids, coordinates rounded to 2 decimals -- two
    calls on the same report produce byte-identical output.
    """
    width, height = 640, 420
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    models = report.get("models", {})
    points = []
    for model, m in models.items():
        x = m.get("usd_per_case")
        y = (m.get("grounded_accuracy") or {}).get("mean")
        if x is None or y is None or x <= 0:
            continue
        points.append(
            {
                "model": model,
                "x": x,
                "y": y,
                "latency": m.get("latency_median_ms") or 0,
                "frontier": bool(m.get("frontier")),
            }
        )

    svg_lines: list[str] = []
    svg_lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    svg_lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    svg_lines.append(
        f'<text x="{width / 2:.2f}" y="20" font-size="14" text-anchor="middle">'
        "Model cost/quality Pareto (M6) -- x: $/case (log10), y: grounded_accuracy</text>"
    )

    if not points:
        svg_lines.append(
            f'<text x="{width / 2:.2f}" y="{height / 2:.2f}" font-size="12" '
            'text-anchor="middle">no comparable points</text>'
        )
        svg_lines.append("</svg>")
        return "\n".join(svg_lines) + "\n"

    log_xs = [math.log10(p["x"]) for p in points]
    x_min, x_max = min(log_xs), max(log_xs)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    y_min, y_max = 0.0, 1.0

    def sx(x: float) -> float:
        lx = math.log10(x)
        return margin + (lx - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    latencies = [p["latency"] for p in points if p["latency"]]
    max_latency = max(latencies) if latencies else 1
    min_radius, max_radius = 4, 18

    def radius(latency: float) -> float:
        if max_latency <= 0:
            return min_radius
        frac = min(1.0, latency / max_latency)
        return round(min_radius + frac * (max_radius - min_radius), 2)

    # axes
    svg_lines.append(
        f'<line x1="{margin}" y1="{margin + plot_h:.2f}" x2="{margin + plot_w:.2f}" '
        f'y2="{margin + plot_h:.2f}" stroke="black"/>'
    )
    svg_lines.append(
        f'<line x1="{margin}" y1="{margin:.2f}" x2="{margin}" y2="{margin + plot_h:.2f}" '
        'stroke="black"/>'
    )

    # frontier polyline (sorted by x), filled markers
    frontier_pts = sorted((p for p in points if p["frontier"]), key=lambda p: p["x"])
    if len(frontier_pts) > 1:
        path = " ".join(f"{sx(p['x']):.2f},{sy(p['y']):.2f}" for p in frontier_pts)
        svg_lines.append(f'<polyline points="{path}" fill="none" stroke="#1a7f37" stroke-width="1.5"/>')

    for p in sorted(points, key=lambda p: p["model"]):
        cx, cy = round(sx(p["x"]), 2), round(sy(p["y"]), 2)
        r = radius(p["latency"])
        fill = "#1a7f37" if p["frontier"] else "white"
        stroke = "#1a7f37" if p["frontier"] else "#555555"
        svg_lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
        svg_lines.append(
            f'<text x="{cx + r + 2:.2f}" y="{cy + 3:.2f}" font-size="9">{_short_name(p["model"])}</text>'
        )

    svg_lines.append(
        f'<text x="{margin}" y="{height - 10}" font-size="9">marker radius ~ median latency '
        f'(min {min_radius}px, max {max_radius}px at {max_latency} ms)</text>'
    )
    svg_lines.append("</svg>")
    return "\n".join(svg_lines) + "\n"


# --- README block ----------------------------------------------------------


def _readme_pareto_block(report: dict) -> str:
    md = render_markdown(report)
    # Strip the top H1 (README already has its own heading structure) but
    # keep everything else -- the README test checks numbers/sentences, not
    # heading levels.
    lines = md.split("\n")
    if lines and lines[0].startswith("# "):
        lines = lines[2:]
    return "\n".join(lines).strip()


def regenerate_readme_pareto(report: dict, readme_path: Path = README_PATH) -> None:
    block = _readme_pareto_block(report)
    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
    else:
        text = "# dealpoint-eval\n\n"
    pattern = re.compile(re.escape(README_BEGIN) + r".*?" + re.escape(README_END), re.DOTALL)
    replacement = f"{README_BEGIN}\n{block}\n{README_END}"
    if pattern.search(text):
        text = pattern.sub(replacement, text)
    else:
        text = (
            text.rstrip("\n")
            + f"\n\n## Model cost/quality Pareto (M6)\n\n{replacement}\n"
        )
    readme_path.write_text(text, encoding="utf-8")


# --- disk-driven pipeline ---------------------------------------------------


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


def generate_report(
    out_json: Path = PARETO_JSON_PATH,
    out_md: Path = PARETO_MD_PATH,
    out_svg: Path = PARETO_SVG_PATH,
) -> dict:
    from datetime import UTC, datetime

    from dealpoint.eval.cases import git_sha7
    from dealpoint.eval.pareto_slate import PARETO_CANDIDATES, PARETO_NOT_RUN
    from dealpoint.eval.pareto_sweep import load_pareto_manifest, reused_legs
    from dealpoint.eval.spend import read_ledger, realized_by_tag, realized_usd
    from dealpoint.eval.subset import load_subset_payload

    slate_payload = None
    if PARETO_SLATE_PATH.exists():
        try:
            slate_payload = json.loads(PARETO_SLATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            slate_payload = None

    price_by_model: dict[str, dict] = {}
    if slate_payload:
        for c in slate_payload.get("candidates", []):
            price_by_model[c["model"]] = c

    judged_subset = None
    try:
        judged_subset = load_subset_payload("judged_subset")
    except (FileNotFoundError, OSError):
        judged_subset = None
    variant_by_model: dict[str, str] = {}
    if judged_subset:
        for v in judged_subset.get("variants", []):
            variant_by_model[v["model"]] = v["variant_id"]

    judge_rows = _read_jsonl(Path("data/eval/judge_scores.jsonl"))

    entries: list[dict] = []
    for leg in reused_legs():
        model = leg["model"]
        price = price_by_model.get(model, {})
        entries.append(
            {
                "model": model,
                "family": price.get("family"),
                "rank": 0,
                "reused": True,
                "rows": _read_jsonl(leg["results_path"]) if leg.get("results_path") else [],
                "variant_id": variant_by_model.get(model),
                "prompt_usd_per_token": price.get("prompt_usd_per_token"),
                "completion_usd_per_token": price.get("completion_usd_per_token"),
                "price_basis": price.get("price_basis") or "reused (M4/M4.1)",
                "est_usd": leg.get("est_usd"),
                "est_basis": leg.get("est_basis"),
                "results_path": leg.get("results_path"),
            }
        )

    for entry in load_pareto_manifest():
        if entry.get("reused") or not entry.get("results_path"):
            continue
        model = entry["model"]
        price = price_by_model.get(model, {})
        entries.append(
            {
                "model": model,
                "family": price.get("family"),
                "rank": entry.get("rank"),
                "reused": False,
                "rows": _read_jsonl(entry["results_path"]),
                "variant_id": variant_by_model.get(model),
                "prompt_usd_per_token": price.get("prompt_usd_per_token"),
                "completion_usd_per_token": price.get("completion_usd_per_token"),
                "price_basis": price.get("price_basis") or entry.get("est_basis"),
                "est_usd": entry.get("est_usd"),
                "est_basis": entry.get("est_basis"),
                "results_path": entry.get("results_path"),
            }
        )

    not_run = list(PARETO_NOT_RUN)
    swept_or_reused = {e["model"] for e in entries}
    excluded_models = set()
    if slate_payload:
        for c in slate_payload.get("candidates", []):
            if c.get("excluded"):
                excluded_models.add(c["model"])
    for cand in PARETO_CANDIDATES:
        model = cand["model"]
        if model in swept_or_reused or model in excluded_models:
            continue
        not_run.append({"model": model, "reason": "not run (budget/stop-floor)"})

    m6_ledger = [r for r in read_ledger() if r.get("milestone_tag") == "m6"]
    probe_usd = sum(float(r.get("usd", 0.0) or 0.0) for r in m6_ledger if r.get("purpose") == "probe")
    judge_usd = sum(float(r.get("usd", 0.0) or 0.0) for r in m6_ledger if r.get("judge"))
    m6_sweep_rows = [
        r for r in m6_ledger if r.get("purpose") != "probe" and not r.get("judge")
    ]
    sweep_usd = sum(float(r.get("usd", 0.0) or 0.0) for r in m6_sweep_rows)

    # Corrective-cycle finding: a model whose ledger sweep spend has no
    # completed 32-case results file (or exceeds it -- an earlier abandoned
    # pass) must never be silently absent from every report file. Compute
    # partial_runs from the ledger, the single source of truth for spend,
    # and re-point any matching not_run entry's reason so no report file
    # ever claims a model that spent money never ran.
    completed_new_models = {
        e["model"]: {
            "total_usd": per_model_metrics(e["rows"])["total_usd"],
            "n_cases": len(e["rows"]),
        }
        for e in entries
        if not e.get("reused")
    }
    partial_runs = compute_partial_runs(m6_sweep_rows, completed_new_models, load_pareto_manifest())
    partial_run_models = {pr["model"] for pr in partial_runs}
    for nr in not_run:
        if nr.get("model") in partial_run_models:
            nr["reason"] = "started, not completed (stop-floor) -- see partial_runs"

    by_tag = realized_by_tag()
    spend = {
        "target_usd": M6_TARGET_USD,
        "envelope_usd": M6_ENVELOPE_USD,
        "m6_probe_usd": round(probe_usd, 6),
        "m6_sweep_usd": round(sweep_usd, 6),
        "m6_judge_usd": round(judge_usd, 6),
        "m6_realized_usd": round(probe_usd + sweep_usd + judge_usd, 6),
        "by_milestone": by_tag,
        "total_realized_usd": realized_usd(),
        "envelope_headroom": round(M6_ENVELOPE_USD - realized_usd(), 6),
    }

    comparability = {
        "haiku_n_cases": 18,
        "others_n_cases": 32,
        "note": (
            "Haiku's arm-D result is 18 cases (tranche_1); every other model's is 32. Any "
            "Haiku-vs-other comparison is over the 18 cases they share and is never compared "
            "to a 32-case mean without saying so."
        ),
    }

    decisions = [
        {
            "id": "PD1",
            "decision": (
                "Haiku and z-ai/glm-5.3-flash are reused from M4/M4.1's frozen arm-D results, "
                "never re-run."
            ),
            "reason": "Spec deliverable 1: reuse, do not re-run.",
        },
        {
            "id": "PD2",
            "decision": (
                "The open-weight slot is deepseek/deepseek-v4-flash, not the spec's named "
                "deepseek/deepseek-v3.2."
            ),
            "reason": (
                "The engineer's 2026-09-04 'Slate and judges' instruction ranks "
                "deepseek/deepseek-v4-flash ahead of deepseek-v3.2 in the candidate slate, and "
                "it is cheaper per case: deepseek-v4-flash is $0.08358/M input, $0.16716/M "
                "output vs deepseek-v3.2's $0.269/M input, $0.400/M output (live prices, "
                "2026-09-05) -- roughly 3x cheaper per token both ways."
            ),
        },
        {
            "id": "PD3",
            "decision": (
                "No ceiling anchor (anthropic/claude-sonnet-5, x-ai/grok-4.3) was run."
            ),
            "reason": (
                "Neither anchor fit the remaining envelope even on the reduced 18-case judged "
                "subset: claude-sonnet-5 is estimated at ~$0.869 for 18 cases and x-ai/grok-4.3 "
                "at ~$0.478 for 18 cases (2026-09-05 live prices), against a total M1-M6 "
                "envelope of $4.00 with a $3.90 stop-floor -- headroom the sweep and judging "
                "legs of the cheap tier had already largely consumed by the time either anchor "
                "would have started."
            ),
        },
    ]
    brief_differences = [
        {
            "id": 1,
            "topic": "scale",
            "difference": (
                "Brief section 2.6 designs the full benchmark over the complete test + "
                "counterfactual sets; this milestone runs the 32-case frozen subset (18 at "
                "Haiku), budget-scaled, per the engineer's 2026-09-04 instruction."
            ),
        },
        {
            "id": 2,
            "topic": "model_count",
            "difference": (
                "Brief section 6 acceptance 6 asks for the model experiment to be complete for "
                ">= 4 of the brief's 5 named models (Sonnet 5 default, Opus 5 ceiling, Haiku, a "
                "Gemini Flash-class model, one open-weight model). This run covers 3 of those 5 "
                "(Haiku, a Gemini Flash-class model, an open-weight model) -- Sonnet 5 and Opus "
                "5 are both absent -- while completing 5 models overall by adding "
                "qwen/qwen3.7-flash and z-ai/glm-5.3-flash to the cheap tier."
            ),
        },
        {
            "id": 3,
            "topic": "ceiling",
            "difference": (
                "The reported Pareto frontier is a frontier of the cheap tier only: no ceiling "
                "model (Sonnet 5, Opus 5, or Grok 4.3) was measured this run, so a stronger, "
                "more expensive model could sit above every point on this frontier. The "
                "frontier and every dominance/tie judgement in this report are scoped to the "
                "models actually completed."
            ),
        },
    ]

    report = build_pareto_report(
        entries,
        judge_rows=judge_rows,
        not_run=not_run,
        partial_runs=partial_runs,
        slate={
            "path": str(PARETO_SLATE_PATH),
            "probe_spend_usd": (slate_payload or {}).get("probe_spend_usd"),
            "excluded": [
                {
                    "model": c["model"],
                    "failure_rate": c.get("failure_rate"),
                    "reason": c.get("failure_details") or "excluded (tool-calling failure)",
                }
                for c in (slate_payload or {}).get("candidates", [])
                if c.get("excluded") and c.get("probed")
            ],
        },
        spend=spend,
        held_fixed={
            "arm": "D",
            "index_version": "e2b4a2b97561",
            "skill_version": "f8d255cc169b",
            "subset": "test_subset_v1",
            "n_cases": 32,
            "sentence": "Arm D is held fixed (frozen index, skill, tools, cases); only the model varies.",
        },
        comparability=comparability,
        decisions=decisions,
        brief_differences=brief_differences,
        generated_at=datetime.now(UTC).isoformat(),
        git_sha7=git_sha7(),
    )

    # Corrective-cycle assertion: every dollar of m6_sweep_usd must be
    # accounted for by either a completed model's total_usd or a
    # partial_runs entry -- no ledger spend may go unattributed in the
    # report.
    accounted = round(
        sum(
            (m.get("total_usd") or 0.0)
            for model, m in report["models"].items()
            if not m.get("reused")
        )
        + sum(pr["realized_usd"] for pr in partial_runs),
        6,
    )
    if abs(accounted - sweep_usd) > 0.001:
        raise ValueError(
            f"m6 sweep spend unaccounted for: completed total_usd + partial_runs "
            f"realized_usd = {accounted} but spend.m6_sweep_usd = {round(sweep_usd, 6)} "
            f"(difference {round(accounted - sweep_usd, 6)})"
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

    out_md.write_text(render_markdown(report), encoding="utf-8")
    out_svg.write_text(render_svg(report), encoding="utf-8")
    regenerate_readme_pareto(report)

    return report


def main(argv: list[str] | None = None) -> int:
    report = generate_report()
    print(json.dumps({"n_models": len(report["models"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
