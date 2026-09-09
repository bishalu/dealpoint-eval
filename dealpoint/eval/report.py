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
    FOUR_ARM_MANIFEST_V1_PATH,
    FOUR_ARM_MD_PATH,
    FOUR_ARM_V1_JSON_PATH,
    FOUR_ARM_V1_MD_PATH,
    M4_1_MAX_USD,
    M4_1_MILESTONE_TAG,
    M4_1_TARGET_USD,
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

# M4.1 (spec deliverable 4): the (model, arm) legs the residual-failure
# honesty check and the failure-rate table apply to.
RESIDUAL_FAILURE_LEGS: tuple[tuple[str, str], ...] = (
    ("z-ai/glm-5.3-flash", "A"),
    ("z-ai/glm-5.3-flash", "B"),
    ("z-ai/glm-5.3-flash", "C"),
    ("z-ai/glm-5.3-flash", "D"),
    ("anthropic/claude-haiku-4.5", "D"),
)
RESIDUAL_FAILURE_THRESHOLD = 0.10

HARNESS_REPAIR_FIXES: tuple[str, ...] = (
    (
        "Capture failure_detail (exception class + first 300 chars, or the validation error) and "
        "raw_final_text (first 1500 chars of the model's final response) on every EXECUTION_FAILED "
        "record, plus finish_reasons per metered call (previously discarded)."
    ),
    (
        "Tolerant JSON extraction (extract_json_object): fenced code blocks, prose-wrapped JSON, a "
        "balanced-brace scan -- truncated/unbalanced JSON still fails, never 'repaired'."
    ),
    (
        "Option normalisation (match_option): canonicalise + casefold + strip a wrapping quote pair "
        "before comparing a model's answer string to the option list; on a match, finding.answer is "
        "rewritten to the exact option string (answer_correct's exact-string comparison unaffected)."
    ),
    (
        "Raised MAX_TOKENS_FINAL 600 -> 1200 uniformly for every arm/model: the v1 ledger showed every "
        "one of the 21 GLM arm-A schema failures dying at output_tokens == 1200 == 2x the old ceiling."
    ),
    (
        "response_format capability table + resolver (response_format_for): json_schema where pinned, "
        "json_object otherwise; a 400 naming response_format/json_schema triggers exactly one "
        "downgrade retry to json_object, recorded in failure_detail."
    ),
    (
        "Assistant tool-call messages send content: '' instead of content: null when the model's "
        "turn had no text -- several providers reject a null content field alongside tool_calls."
    ),
    (
        "Real exponential backoff with jitter (API_RETRY_BASE_DELAY_S, default ~1.0s, honouring "
        "Retry-After) replacing the old 0.05*2^n backoff, which totalled 0.35s across 3 retries."
    ),
)

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


def failure_rate_table(
    v1_rows_by_arm_model: dict[tuple[str, str], list[dict]],
    v2_rows_by_arm_model: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    """One row per (model, arm) present in either v1 or v2: n_cases,
    execution_failed_v1/v2, delta, cap_hit_v1/v2, failure_detail_classes_v2
    (spec deliverable 4/6 -- the v1 -> v2 failure-rate table).
    """
    keys = sorted(set(v1_rows_by_arm_model) | set(v2_rows_by_arm_model))

    def _rate(rows: list[dict] | None, status: str) -> tuple[float | None, int]:
        if not rows:
            return None, 0
        n = len(rows)
        hits = sum(1 for r in rows if (r.get("record") or {}).get("status") == status)
        return hits / n, n

    def _classes(rows: list[dict] | None) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows or []:
            record = r.get("record") or {}
            if record.get("status") != "EXECUTION_FAILED":
                continue
            detail = record.get("failure_detail") or f"failure_reason={record.get('failure_reason')}"
            out[detail] = out.get(detail, 0) + 1
        return out

    rows_out: list[dict] = []
    for model, arm in keys:
        v1_rows = v1_rows_by_arm_model.get((model, arm))
        v2_rows = v2_rows_by_arm_model.get((model, arm))
        ef_v1, n_v1 = _rate(v1_rows, "EXECUTION_FAILED")
        ef_v2, n_v2 = _rate(v2_rows, "EXECUTION_FAILED")
        cap_v1, _ = _rate(v1_rows, "CAP_HIT")
        cap_v2, _ = _rate(v2_rows, "CAP_HIT")
        delta = (ef_v2 - ef_v1) if (ef_v1 is not None and ef_v2 is not None) else None
        rows_out.append(
            {
                "model": model,
                "arm": arm,
                "n_cases_v1": n_v1,
                "n_cases_v2": n_v2,
                "execution_failed_v1": ef_v1,
                "execution_failed_v2": ef_v2,
                "delta": delta,
                "cap_hit_v1": cap_v1,
                "cap_hit_v2": cap_v2,
                "failure_detail_classes_v2": _classes(v2_rows),
            }
        )
    return rows_out


def majority_baseline_note(subset_overall: float | None, full_test_overall: float | None) -> str:
    """Spec deliverable 5: the 0% majority baseline on the frozen subset is
    *by construction* (the subset selection rule prefers cases the majority
    answer gets wrong); give the full-test-set figure for context.
    """
    return (
        f"Majority-baseline accuracy on this subset is {_pct(subset_overall)} "
        "by construction: data/eval/test_subset_v1.json's selection rule prefers cases where "
        "gold_answer != majority_answer (the majority already gets these wrong), so a near-zero "
        "subset baseline is not evidence the majority baseline is generally weak. For context, "
        f"the majority baseline on the full 167-case MAUD test set is {_pct(full_test_overall)}."
    )


def readme_baseline_note(full_test_overall: float | None) -> str:
    """README wording for the majority baseline.

    `majority_baseline_note` (used by data/reports/four_arm.md, the record)
    states the subset figure and explains that it is an artefact of the
    subset selection rule. The README is an introduction, and a bare
    near-zero percentage there reads as a result rather than as the
    definition it actually is, so the README quotes the meaningful number --
    the majority baseline on the FULL test set -- and says in words why the
    subset has no baseline of its own. Same fact, no misleading headline
    figure; the full report keeps both numbers.
    """
    return (
        "For reference, always giving the most common answer scores "
        f"{_pct(full_test_overall)} on the full 167-case test set. The subset selection rule "
        "prefers cases where that common answer is wrong, so the subset has no meaningful "
        "baseline of its own. Compare the arms against each other instead. Both baseline "
        "figures are in the full report."
    )


def _finalized_after_truncated_tool_turn(finish_reasons: list[str]) -> bool:
    """True when the call immediately following this row's LAST 'tool_calls'
    finish_reason itself ended on 'length' -- i.e. MAX_TOKENS_TOOL_TURN (300)
    cut the model off mid tool-turn or mid finalisation, one call before the
    loop stopped issuing tool calls (corrective task, M4.1 review finding).
    """
    last_tool_calls_idx = None
    for i, fr in enumerate(finish_reasons):
        if fr == "tool_calls":
            last_tool_calls_idx = i
    if last_tool_calls_idx is None:
        return False
    next_idx = last_tool_calls_idx + 1
    return next_idx < len(finish_reasons) and finish_reasons[next_idx] == "length"


def tool_turn_truncation_stats(
    rows_by_arm_model: dict[tuple[str, str], list[dict]],
    legs: tuple[tuple[str, str], ...] = RESIDUAL_FAILURE_LEGS,
) -> list[dict]:
    """Per (model, arm) leg, how many of ITS cases (regardless of final
    status) finalised immediately after a tool turn `MAX_TOKENS_TOOL_TURN`
    (300) cut off -- computed from the v2 result rows, never hard-coded
    (corrective task: the M4.1 report previously claimed the probes showed
    no such truncation, which the rows themselves contradict).
    """
    stats: list[dict] = []
    for model, arm in legs:
        rows = rows_by_arm_model.get((model, arm))
        if not rows:
            continue
        n = len(rows)
        count = sum(
            1
            for r in rows
            if _finalized_after_truncated_tool_turn(
                (r.get("record") or {}).get("finish_reasons") or []
            )
        )
        if count:
            stats.append({"model": model, "arm": arm, "n": n, "count": count})
    return stats


def tool_turn_truncation_note(
    stats: list[dict], calls_at_300_output_tokens: int | None
) -> str:
    """Corrective task: replace the prior (false) claim that the M4.1 probes
    showed no tool-turn truncation. `stats` is `tool_turn_truncation_stats`'s
    output (computed from result rows); `calls_at_300_output_tokens` is
    computed from the spend ledger by the caller (generate_report), since
    build_report itself does no I/O.
    """
    per_leg = ", ".join(f"{s['count']}/{s['n']} ({s['model']} {s['arm']})" for s in stats)
    calls_clause = (
        f"and {calls_at_300_output_tokens} v2 sweep calls ended at exactly 300 output tokens "
        "(data/results/spend_ledger.jsonl)"
        if calls_at_300_output_tokens is not None
        else "and a number of v2 sweep calls ended at exactly 300 output tokens "
        "(data/results/spend_ledger.jsonl)"
    )
    return (
        "`MAX_TOKENS_TOOL_TURN` was LEFT at 300 despite evidence that it DOES truncate tool "
        "turns and force premature finalisation. In data/reports/m4_1_probes.json's after2 "
        "round: z-ai/glm-5.3-flash/B (contract_32__q08, contract_39__q01) and "
        "anthropic/claude-haiku-4.5/D (contract_7__q07) each have a finish_reasons entry of "
        "'length' immediately after their last 'tool_calls' turn, with a matching "
        "data/results/spend_ledger.jsonl row at exactly output_tokens == 300, and the loop then "
        "finalised with only 6/8, 4/8 and 3/8 tool calls used respectively. Across the v2 "
        f"sweeps the same pattern recurs at scale (computed from the result rows): {per_leg} "
        f"of this milestone's cases finalised immediately after a tool turn that ended on "
        f"'length', {calls_clause}. The constant was NOT raised because the M4.1 budget "
        "($1.4202 of the $1.50 absolute) leaves no room for another sweep -- this is a "
        "recorded, unfixed limitation of the v2 numbers, not a finding that truncation is "
        "harmless."
    )


def estimate_before_v2_caveat(
    sweeps: list[dict],
    v1_rows_by_arm_model: dict[tuple[str, str], list[dict]] | None,
    haiku_model: str = "anthropic/claude-haiku-4.5",
) -> str | None:
    """Plan §5: the `ledger:measured(arm,model)` estimate basis is biased LOW
    for Haiku arm D because v1's arm-D cases aborted early. Computed from the
    preserved v1 rows (failed count) and this leg's own est/realized sweep
    entry, never hard-coded. Returns `None` if the Haiku-D sweep entry (or
    v1 rows) are not available.
    """
    haiku_d_sweep = next(
        (
            s
            for s in sweeps
            if s.get("leg") == "replication_haiku"
            and s.get("arm") == "D"
            and s.get("model") == haiku_model
        ),
        None,
    )
    if haiku_d_sweep is None:
        return None
    est = haiku_d_sweep.get("est_usd")
    realized = haiku_d_sweep.get("realized_usd")
    v1_rows = (v1_rows_by_arm_model or {}).get((haiku_model, "D"))
    if v1_rows:
        v1_n = len(v1_rows)
        v1_failed = sum(
            1 for r in v1_rows if (r.get("record") or {}).get("status") == "EXECUTION_FAILED"
        )
        aborted_detail = f"{v1_failed}/{v1_n} EXECUTION_FAILED"
    else:
        aborted_detail = "most cases EXECUTION_FAILED"
    return (
        "Caveat: this estimate's `ledger:measured(arm,model)` basis is biased LOW for Haiku "
        f"arm D, because v1's arm-D cases aborted early ({aborted_detail}) -- the mean per-case "
        "cost measured from those short, mostly-failed executions understates what a full "
        f"8-tool-call run costs. This is why the Haiku arm-D leg estimated ${est} and realised "
        f"${realized} -- the only leg in the v2 sweeps to overrun its own estimate."
    )


def residual_failures(
    v2_rows_by_arm_model: dict[tuple[str, str], list[dict]],
    legs: tuple[tuple[str, str], ...] = RESIDUAL_FAILURE_LEGS,
    threshold: float = RESIDUAL_FAILURE_THRESHOLD,
) -> dict:
    """For every (model, arm) leg whose v2 EXECUTION_FAILED mean exceeds
    `threshold`, an explanation sourced from `failure_detail` plus an
    `attribution` ("provider" | "harness") -- spec DoD residual-failure
    honesty rule. `attribution` defaults to "provider" (a residual failure
    surviving the harness repair is presumptively the provider's, unless a
    caller overrides this per-leg after reading the actual failure_detail
    text -- see `dealpoint.eval.report.main`/the report notes for any
    per-leg override).
    """
    out: dict = {}
    for model, arm in legs:
        rows = v2_rows_by_arm_model.get((model, arm))
        if not rows:
            continue
        n = len(rows)
        failed = [r for r in rows if (r.get("record") or {}).get("status") == "EXECUTION_FAILED"]
        rate = len(failed) / n if n else 0.0
        if rate <= threshold:
            continue
        classes: dict[str, int] = {}
        final_verbosity_count = 0
        tool_turn_truncated_count = 0
        for r in failed:
            record = r.get("record") or {}
            detail = record.get("failure_detail") or f"failure_reason={record.get('failure_reason')}"
            classes[detail] = classes.get(detail, 0) + 1
            finish_reasons = record.get("finish_reasons") or []
            if finish_reasons and finish_reasons[-1] == "length":
                final_verbosity_count += 1
            if _finalized_after_truncated_tool_turn(finish_reasons):
                tool_turn_truncated_count += 1
        top_class, top_count = max(classes.items(), key=lambda kv: kv[1]) if classes else ("unknown", 0)
        explanation = (
            f"{len(failed)}/{n} ({_pct(rate)}) EXECUTION_FAILED; most common failure_detail class: "
            f"{top_class!r} ({top_count}/{len(failed)})."
        )
        if final_verbosity_count:
            explanation += (
                f" {final_verbosity_count}/{len(failed)} of these end with finish_reason == 'length' on "
                "both the initial attempt and the schema retry -- the model spends its whole "
                "MAX_TOKENS_FINAL=1200 budget (already doubled from 600 per this milestone's fix) "
                "on prose reasoning before ever emitting the JSON object, so raw_final_text is a "
                "truncated reasoning preamble with no JSON in it at all. This is the model's own "
                "verbosity, not a request-shape defect the harness can fix without an unbounded "
                "token ceiling (out of scope: CAP_HIT-style caps are deliberate, not to be raised "
                "without limit)."
            )
        if tool_turn_truncated_count:
            all_rows_tool_turn_truncated = sum(
                1
                for r in rows
                if _finalized_after_truncated_tool_turn(
                    (r.get("record") or {}).get("finish_reasons") or []
                )
            )
            explanation += (
                f" {tool_turn_truncated_count}/{len(failed)} of these failed cases (and "
                f"{all_rows_tool_turn_truncated}/{n} of this leg's cases overall) finalised "
                "immediately after a tool-calling turn whose finish_reason was 'length' -- "
                "MAX_TOKENS_TOOL_TURN (300) cut that turn off before the model could keep "
                "searching, so the loop moved to finalisation with fewer tool calls used than "
                "it would otherwise have made. This is left-at-300, unfixed for this v2 report "
                "(no budget for another sweep); it is a contributing cause alongside, not "
                "instead of, final-answer verbosity."
            )
        out[f"{model}/{arm}"] = {"explanation": explanation, "attribution": "provider"}
    return out


def build_report(
    rows_by_arm_model: dict[tuple[str, str], list[dict]],
    *,
    sweeps: list[dict] | None = None,
    caching: dict | None = None,
    dev_loop_spend: dict | None = None,
    m4_ledger_total: float | None = None,
    baseline_cases: list[dict] | None = None,
    v1_rows_by_arm_model: dict[tuple[str, str], list[dict]] | None = None,
    full_test_baseline_cases: list[dict] | None = None,
    m4_1_ledger_total: float | None = None,
    total_ledger_usd: float | None = None,
    estimate_before_v2: dict | None = None,
    calls_at_300_output_tokens: int | None = None,
    probes: dict | None = None,
    v1_artifacts: dict | None = None,
    version: str | None = None,
) -> dict:
    """Pure assembly of the full `four_arm.json` payload from already-scored
    result rows, keyed `(model, arm)`. No I/O beyond what the caller passed in
    -- this is what the offline `gate_m4` report-generator tests exercise.

    M4.1 extensions (all optional, default to the pre-M4.1 shape when
    omitted): `v1_rows_by_arm_model` drives `failure_rate_table` and
    `residual_failures` (evaluated over `rows_by_arm_model`, read as v2 once
    `version` is passed); `full_test_baseline_cases` drives
    `majority_baseline_note`'s full-test-set figure; the cost fields extend
    `report["cost"]`; `version`/`probes` populate `report["harness_repair"]`;
    `calls_at_300_output_tokens` (from the spend ledger, computed by the
    caller) feeds `tool_turn_truncation.note`'s ledger-level evidence.
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
            "m4_1_ledger_total": m4_1_ledger_total,
            "m4_1_target_usd": M4_1_TARGET_USD,
            "m4_1_absolute_usd": M4_1_MAX_USD,
            "total_ledger_usd": total_ledger_usd,
            "estimate_before_v2": estimate_before_v2,
            "estimate_before_v2_caveat": estimate_before_v2_caveat(sweeps, v1_rows_by_arm_model)
            if estimate_before_v2 is not None
            else None,
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

    if baseline_cases is not None or full_test_baseline_cases is not None:
        subset_overall = (
            (report["majority_baseline"] or {}).get("overall") if baseline_cases else None
        )
        full_overall = (
            majority_baseline(full_test_baseline_cases).get("overall")
            if full_test_baseline_cases is not None
            else None
        )
        report["majority_baseline_note"] = {
            "subset_overall": subset_overall,
            "full_test_overall": full_overall,
            "sentence": majority_baseline_note(subset_overall, full_overall),
        }

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

    if v1_rows_by_arm_model is not None:
        report["failure_rate_table"] = failure_rate_table(v1_rows_by_arm_model, rows_by_arm_model)
        report["residual_failures"] = residual_failures(rows_by_arm_model)

    if version is not None:
        report["version"] = version
        report["harness_repair"] = {
            "fixes": list(HARNESS_REPAIR_FIXES),
            "probes_path": "data/reports/m4_1_probes.json",
            "probes": probes or {},
        }
        report["v1_artifacts"] = v1_artifacts or {
            "four_arm_json": str(FOUR_ARM_V1_JSON_PATH),
            "four_arm_md": str(FOUR_ARM_V1_MD_PATH),
            "four_arm_manifest": str(FOUR_ARM_MANIFEST_V1_PATH),
        }
        stats = tool_turn_truncation_stats(rows_by_arm_model)
        report["tool_turn_truncation"] = {
            "max_tokens_tool_turn": 300,
            "raised": False,
            "per_leg": stats,
            "calls_at_300_output_tokens": calls_at_300_output_tokens,
            "note": tool_turn_truncation_note(stats, calls_at_300_output_tokens),
        }

    return report


# --- markdown rendering -------------------------------------------------


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Four-arm experiment on the frozen test set (M4)")
    lines.append("")
    version = report.get("version")
    if version:
        lines.append(f"**Version:** {version}.")
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

    harness_repair = report.get("harness_repair")
    if harness_repair:
        lines.append("## Harness repair (M4.1)")
        lines.append("")
        lines.append(f"Probe results: `{harness_repair.get('probes_path')}`.")
        lines.append("")
        for fix in harness_repair.get("fixes", []):
            lines.append(f"- {fix}")
        lines.append("")

    majority = report.get("majority_baseline") or {}
    lines.append(f"Majority-baseline overall accuracy: {_pct(majority.get('overall'))}")
    lines.append("")
    majority_note = report.get("majority_baseline_note")
    if majority_note:
        lines.append(majority_note["sentence"])
        lines.append("")

    failure_table = report.get("failure_rate_table")
    if failure_table:
        lines.append("## Execution-failure rate: v1 -> v2")
        lines.append("")
        lines.append(
            "v1 result files are preserved (see v1_artifacts below); v2 is this report's "
            "live arms/ data."
        )
        lines.append("")
        lines.append(
            "| model | arm | n v1 | n v2 | EXECUTION_FAILED v1 | EXECUTION_FAILED v2 | delta | "
            "CAP_HIT v1 | CAP_HIT v2 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in failure_table:
            lines.append(
                f"| {row['model']} | {row['arm']} | {row.get('n_cases_v1')} | "
                f"{row.get('n_cases_v2')} | {_pct(row.get('execution_failed_v1'))} | "
                f"{_pct(row.get('execution_failed_v2'))} | {_pct(row.get('delta'))} | "
                f"{_pct(row.get('cap_hit_v1'))} | {_pct(row.get('cap_hit_v2'))} |"
            )
        lines.append("")

    residuals = report.get("residual_failures")
    if residuals:
        lines.append("## Residual failures (v2, above the 10% threshold)")
        lines.append("")
        for key, info in residuals.items():
            lines.append(f"- **{key}** ({info['attribution']}): {info['explanation']}")
        lines.append("")

    v1_artifacts = report.get("v1_artifacts")
    if v1_artifacts:
        lines.append("## v1 artefacts (preserved before the v2 re-run)")
        lines.append("")
        for k, v in v1_artifacts.items():
            lines.append(f"- `{k}`: `{v}`")
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
    if cost.get("m4_1_ledger_total") is not None:
        lines.append(
            f"M4.1 ledger total: ${cost.get('m4_1_ledger_total')} "
            f"(target ${cost.get('m4_1_target_usd')}, absolute ${cost.get('m4_1_absolute_usd')}); "
            f"total ledger: ${cost.get('total_ledger_usd')}."
        )
    if cost.get("estimate_before_v2") is not None:
        lines.append(
            f"Estimate before the v2 sweeps: `{json.dumps(cost.get('estimate_before_v2'), sort_keys=True)}`"
        )
        if cost.get("estimate_before_v2_caveat"):
            lines.append("")
            lines.append(cost["estimate_before_v2_caveat"])
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
    if report.get("version"):
        lines.append(
            "- M4.1 request-shape decisions: `MAX_TOKENS_FINAL` raised 600 -> 1200 uniformly "
            "for every arm/model (the v1 evidence for this: every one of the 21 GLM arm-A "
            "schema failures died at output_tokens == 1200, exactly 2x the old ceiling). "
        )
        tool_turn = report.get("tool_turn_truncation")
        if tool_turn and tool_turn.get("note"):
            lines.append(f"- {tool_turn['note']}")
    lines.append("")

    return "\n".join(lines)


def _readme_results_block(report: dict) -> str:
    version = report.get("version")
    first_sentence = (
        "These numbers are budget-scaled. Each model ran a frozen 32-case slice of MAUD's "
        "test set (18 cases for `anthropic/claude-haiku-4.5`), not the full 167-case design "
        "in the brief. Every score is objective. Deterministic Python checks each answer "
        "against the lawyers' own labels, and no model does any judging. The scores are not "
        "comparable to MAUD leaderboard numbers, because MAUD hands the model the passage to "
        "read and this task makes the agent find it."
    )
    if version:
        first_sentence = f"**{version}.** " + first_sentence
    lines = [first_sentence, ""]

    majority_note = report.get("majority_baseline_note")
    if majority_note:
        lines.append(readme_baseline_note(majority_note.get("full_test_overall")))
        lines.append("")

    # README results block (engineer's instruction, 2026-09-05): per model, a
    # table of arms with grounded_accuracy shown as "x% (k of n scored)", plus
    # the one-line baseline note above -- and NOTHING else. No
    # execution-failure/cap-hit line, no v1-vs-v2 table, no residual-failure
    # bullets, no per-arm verdict sentences: those diagnostics stay, complete
    # and honest, in data/reports/four_arm.md and four_arm.json, which the
    # prose immediately below links explicitly. The README is an
    # introduction for readers who do not know the project; the reports are
    # the record.
    for model, model_arms in report.get("arms", {}).items():
        lines.append(f"**Model `{model}`:**")
        lines.append("")
        lines.append("| arm | grounded_accuracy |")
        lines.append("|---|---|")
        for arm in ARM_ORDER:
            data = model_arms.get(arm)
            if not data:
                continue
            o = data["overall"]
            n_cases = data["n_cases"]
            n_scored = o["grounded_accuracy"]["n"]
            lines.append(
                f"| {arm} | {_pct(o['grounded_accuracy']['mean'])} "
                f"({n_scored} of {n_cases} scored) |"
            )
        lines.append("")

    lines.append(
        "Full diagnostics are in [`data/reports/four_arm.md`](data/reports/four_arm.md) and "
        "[`data/reports/four_arm.json`](data/reports/four_arm.json): execution-failure and "
        "cap-hit rates, the v1-vs-v2 comparison, residual-failure attribution, and the "
        "per-arm verdicts."
    )
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
    from dealpoint.config import M4_1_PROBES_JSON_PATH, REPORTS_DIR
    from dealpoint.eval.cases import find_case, load_case_set
    from dealpoint.eval.four_arm_sweep import MANIFEST_PATH, load_manifest
    from dealpoint.eval.spend import read_ledger, realized_by_tag, realized_usd
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

    # M4.1: if the v1 manifest was preserved, this is a v2 report -- add the
    # v1 -> v2 failure-rate table, residual-failure honesty, the full-test-set
    # majority baseline, and the M4.1-specific cost fields (spec deliverable
    # 4/5/6). Absent v1 preservation, every M4.1 field is simply omitted and
    # `build_report`'s pre-M4.1 shape is unchanged.
    v1_rows_by_arm_model = None
    version = None
    probes = None
    v1_artifacts = None
    full_test_baseline_cases = None
    if FOUR_ARM_MANIFEST_V1_PATH.exists():
        v1_rows_by_arm_model = load_rows_from_manifest(FOUR_ARM_MANIFEST_V1_PATH)
        version = "v2 after harness repair"
        v1_artifacts = {
            "four_arm_json": str(FOUR_ARM_V1_JSON_PATH),
            "four_arm_md": str(FOUR_ARM_V1_MD_PATH),
            "four_arm_manifest": str(FOUR_ARM_MANIFEST_V1_PATH),
        }
        try:
            full_test_baseline_cases = load_case_set("test")
        except (FileNotFoundError, KeyError):
            full_test_baseline_cases = None
        if M4_1_PROBES_JSON_PATH.exists():
            try:
                probes = json.loads(M4_1_PROBES_JSON_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                probes = {}

    m4_1_ledger_total = realized_by_tag().get(M4_1_MILESTONE_TAG) if version else None
    total_ledger_usd = realized_usd() if version else None

    estimate_before_v2 = None
    if version:
        estimate_before_v2_path = REPORTS_DIR / "m4_1_estimate_before_v2.json"
        if estimate_before_v2_path.exists():
            try:
                estimate_before_v2 = json.loads(estimate_before_v2_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                estimate_before_v2 = None

    calls_at_300_output_tokens = None
    if version:
        # v2 sweep calls only (milestone_tag m4_1, case_set starting with the
        # frozen subset name, excluding the diagnostic probes) -- corrective
        # task evidence for the tool-turn-truncation note.
        ledger_rows = read_ledger()
        calls_at_300_output_tokens = sum(
            1
            for r in ledger_rows
            if r.get("milestone_tag") == M4_1_MILESTONE_TAG
            and str(r.get("case_set", "")).startswith("test_subset_v1")
            and r.get("output_tokens") == 300
        )

    report = build_report(
        rows_by_arm_model,
        sweeps=sweeps,
        caching=caching or {},
        dev_loop_spend=dev_loop_spend or {},
        m4_ledger_total=m4_ledger_total,
        baseline_cases=baseline_cases,
        v1_rows_by_arm_model=v1_rows_by_arm_model,
        full_test_baseline_cases=full_test_baseline_cases,
        m4_1_ledger_total=m4_1_ledger_total,
        total_ledger_usd=total_ledger_usd,
        estimate_before_v2=estimate_before_v2,
        calls_at_300_output_tokens=calls_at_300_output_tokens,
        probes=probes,
        v1_artifacts=v1_artifacts,
        version=version,
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
