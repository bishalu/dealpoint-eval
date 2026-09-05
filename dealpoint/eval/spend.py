"""Ledger reading, cost estimation, and the spend cap (spec §3).

Two audiences read this module: the eval runner (which must refuse to start
an unattended sweep over budget) and the operator's `just budget <sweep>`
command (which must print a clean JSON estimate for
`adws/adw_modules/spend.py::estimate` to parse). Both share one estimator so
the product's own idea of cost never drifts from the factory's guard.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import dotenv_values

from dealpoint.config import (
    EST_CALLS_PER_CASE,
    PARETO_PROBE_N_CASES,
    RESULTS_DIR,
    SPEND_LEDGER_PATH,
)

CAP_ENV = "MAX_OPENROUTER_SPEND_USD"
LEDGER_PATH = SPEND_LEDGER_PATH

PINNED_PRICES_DATE = "2026-09-04"
# USD per token. Verified 2026-09-04: input_tokens*1e-6 + output_tokens*5e-6
# reproduces `usd` exactly on every M1 ledger row for this model.
PINNED_PRICES: dict[str, dict[str, float]] = {
    "anthropic/claude-haiku-4.5": {"prompt": 1e-6, "completion": 5e-6},
    # z-ai/glm-5.3-flash: the M3+ cheap workhorse for dev-loop/smoke/exploratory
    # metered calls (engineer's instruction, 2026-09-04). Verified live against
    # https://openrouter.ai/api/v1/models on 2026-09-04 (426 models, basis=="live"):
    # prompt $0.000000075/token, completion $0.00000025/token; supports tools,
    # tool_choice, response_format, structured_outputs.
    "z-ai/glm-5.3-flash": {"prompt": 7.5e-08, "completion": 2.5e-07},
}

# M6 (spec deliverable 1, "Cheap workhorse model"/"Slate and judges"
# instructions): the five new Pareto candidates, ranked, verified live on
# OpenRouter 2026-09-05 (tools + response_format/structured_outputs
# supported). Haiku and z-ai/glm-5.3-flash are REUSED from M4/M4.1, never
# re-run, so they are NOT in this list. This is the single source of truth
# for "which new models does the pareto sweep price" -- `dealpoint.eval.
# pareto_slate.PARETO_NEW_MODELS` imports this tuple directly rather than
# keeping its own copy, so the estimate and the actual sweep can never drift
# apart (spec DoD: "one list, never two that can drift").
# Rows 3-7 of specs/milestones/openrouter_sweep_2026-09-04.md's ranked slate
# (rows 1-2, Haiku and GLM, are REUSED from M4/M4.1, never re-run/re-priced
# here). Kept in the sweep file's ranked order -- the cheap tier as a whole
# runs before the two ceiling anchors (rows 11-12); see PARETO_ANCHOR_MODELS.
PARETO_NEW_MODELS: tuple[str, ...] = (
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.7-flash",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-5.6-luna-pro",
    "meta-llama/llama-4-maverick",
)

# Stretch candidates (rows 8-10 of the sweep file): considered in this order
# only while the M6 stop-floor holds after the core five; not priced into
# the pre-build estimate below (their cost is added at sweep time, per
# model, from the measured per-case cost of the models that already ran).
PARETO_STRETCH_MODELS: tuple[str, ...] = (
    "xiaomi/mimo-v2.5",
    "minimax/minimax-m2.5",
    "moonshotai/kimi-k2.5",
)

# Ceiling anchors (rows 11-12): run last, on the 18-case judged subset only,
# and only if the remaining envelope covers them (spec deliverable 1).
PARETO_ANCHOR_MODELS: tuple[str, ...] = (
    "anthropic/claude-sonnet-5",
    "x-ai/grok-4.3",
)

# The three named M5 judge-trio models (engineer's "Judge trio" instruction).
# Shared by the `judges` and `pareto` sweep shapes below so both price the
# same panel from one list.
JUDGE_TRIO_MODELS: tuple[str, ...] = (
    "mistralai/mistral-small-3.2-24b-instruct",
    "nvidia/nemotron-3-super-120b-a12b",
    "bytedance-seed/seed-2.0-mini",
)

# Sweep definitions: shape only, no measured cost. `estimate()` fills the
# cost basis in priority order (spec §3.4). One module-level table so M4-M6
# only need to add/change entries here.
#
# `four_arm` (re-sized 2026-09-04, M4 spec deliverable 5): the old shape here
# was a single Haiku-only 4-arm x 32-case sweep (est $3.09 -- 4x the M4
# target, and the ledger showed 0 cached tokens on 96 calls, so the caching
# saving it implicitly assumed never materialised). The re-sized design is
# two legs: the GLM headline (4 arms x 32 cases, the full frozen subset,
# constant cheap model) plus the Haiku A/D replication, sized by the
# estimator to the largest whole-question tranche (N=18 = tranche_1: 1 case
# per question + 6 redacted + 2 oos) that projects <= $1.00 for that leg.
# The `legs` key is summed by `estimate()`; this makes the estimate the plan
# the runner actually executes, not an upper bound the factory invented.
SWEEP_DEFS: dict[str, dict] = {
    "four_arm": {
        "legs": [
            {"arms": ["A", "B", "C", "D"], "models": ["z-ai/glm-5.3-flash"], "n_cases": 32},
            {"arms": ["A", "D"], "models": ["anthropic/claude-haiku-4.5"], "n_cases": 18},
        ],
    },
    # M5 (spec deliverable 3, engineer's "Judge trio" instruction): the judge
    # panel prices 54 traces (18 judged-subset cases x 3 variants) x 3 named
    # judge models, one call per (trace, judge) pair -- NOT agent-case calls,
    # so pricing directly from token shape x live/pinned price (the
    # `tokens_per_call` leg shape) rather than any ledger branch, which is
    # agent-case shaped and would return nonsense for a judge call. The
    # 8000-in/300-out shape is the sweep file's conservative (upper-bound)
    # assumption; the milestone report records the measured mean alongside it.
    "judges": {
        "arms": [],
        "models": list(JUDGE_TRIO_MODELS),
        "n_cases": 54,  # traces (18 judged-subset cases x 3 variants)
        "tokens_per_call": {"input": 8000, "output": 300},
    },
    # M6 (spec deliverables 1/2/3; corrective-cycle fix): the placeholder
    # 3-models-at-Haiku shape above was priced at the wrong model entirely --
    # Haiku and GLM are REUSED from M4/M4.1 (spec deliverable 1, never
    # re-run), and the actual new candidates are 15-45x cheaper per case.
    # Three legs modelling the real run: (1) a 2-dev-case tool-calling probe
    # per new candidate, (2) the arm-D sweep on the frozen 32-case subset per
    # new candidate, (3) judging -- 18 judged-subset traces per new variant,
    # x3 judges, at the M5-measured judge call shape (NOT an agent-case
    # shape; see dealpoint.eval.pareto_slate.PARETO_NEW_MODELS, the single
    # source of truth this leg's `models` list mirrors so the two can never
    # drift apart).
    "pareto": {
        "legs": [
            {"arms": ["D"], "models": list(PARETO_NEW_MODELS), "n_cases": PARETO_PROBE_N_CASES},
            {"arms": ["D"], "models": list(PARETO_NEW_MODELS), "n_cases": 32},
            {
                "arms": [],
                "models": list(JUDGE_TRIO_MODELS),
                "n_cases": 18 * len(PARETO_NEW_MODELS),
                "tokens_per_call": {"input": 3400, "output": 120},
            },
        ],
    },
}


class SpendCapError(RuntimeError):
    """Raised when a sweep may not proceed: the cap is unset, unparseable, or exceeded."""


def read_ledger(path: Path = LEDGER_PATH) -> list[dict]:
    """Tolerant ledger read: skip blank or corrupt lines rather than raise."""
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def realized_usd(path: Path = LEDGER_PATH) -> float:
    """Sum of `usd` over every ledger row, rounded to 6 places.

    Must agree with `adws/adw_modules/spend.py::realized_usd` to the cent.
    """
    total = 0.0
    for row in read_ledger(path):
        try:
            total += float(row.get("usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return round(total, 6)


def realized_by_tag(path: Path = LEDGER_PATH) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in read_ledger(path):
        tag = row.get("milestone_tag", "unknown")
        try:
            totals[tag] += float(row.get("usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return {tag: round(v, 6) for tag, v in totals.items()}


def fetch_prices(timeout: float = 20) -> tuple[dict, str]:
    """Live OpenRouter prices, falling back to the pinned table on any failure.

    Returns `(prices, basis)` where `basis` is `"live"` or
    `"pinned:<PINNED_PRICES_DATE>"`. Never raises.
    """
    try:
        import requests

        response = requests.get(
            "https://openrouter.ai/api/v1/models", timeout=timeout
        )
        response.raise_for_status()
        payload = response.json()
        prices: dict[str, dict[str, float]] = {}
        for entry in payload.get("data", []):
            model_id = entry.get("id")
            pricing = entry.get("pricing") or {}
            if not model_id or "prompt" not in pricing or "completion" not in pricing:
                continue
            prices[model_id] = {
                "prompt": float(pricing["prompt"]),
                "completion": float(pricing["completion"]),
            }
        if not prices:
            raise ValueError("no usable pricing entries in OpenRouter response")
        return prices, "live"
    except Exception:  # noqa: BLE001 - any failure falls back to the pinned table
        return dict(PINNED_PRICES), f"pinned:{PINNED_PRICES_DATE}"


def _mean_tokens_per_call(rows: list[dict]) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    total_in = sum(float(r.get("input_tokens", 0) or 0) for r in rows)
    total_out = sum(float(r.get("output_tokens", 0) or 0) for r in rows)
    n = len(rows)
    return total_in / n, total_out / n


def _batch_by_ts_wraparound(rows: list[dict]) -> list[float]:
    """ts-ordered fallback grouping for rows with no stable run discriminator.

    Rows are processed oldest-first, accumulating `usd` per `case_id`. Within
    one execution of a case, every one of its (possibly many) LLM calls
    shares the same `case_id` and is contiguous, so consecutive same-`case_id`
    rows are summed. A *new* execution begins when the case_id changes to one
    already accumulated earlier in the current batch (a wrap-around back to a
    case the run already finished) -- at that point the whole in-progress
    batch (one total per case_id seen so far) is closed out as samples, one
    per case, and a fresh batch begins. This recovers the correct per-
    execution totals from a real multi-case sweep's ledger (e.g. ...q01 x5,
    q06 x3, q12 x6, q01 x5, q06 x3, q12 x6... -> 6 samples, one per
    execution), which is the shape every real runner sweep produces.
    """
    ordered = sorted(rows, key=lambda r: r.get("ts", ""))
    samples: list[float] = []
    current_totals: dict[str, float] = {}
    prev_case_id: str | None = None
    for row in ordered:
        case_id = row.get("case_id")
        if not case_id:
            continue
        usd = float(row.get("usd", 0.0) or 0.0)
        if case_id != prev_case_id and case_id in current_totals:
            samples.extend(current_totals.values())
            current_totals = {}
        current_totals[case_id] = current_totals.get(case_id, 0.0) + usd
        prev_case_id = case_id
    samples.extend(current_totals.values())
    return samples


def _per_execution_case_costs(rows: list[dict]) -> list[float]:
    """One total-cost sample per *execution* of a case, not per distinct case_id.

    A case run N times must contribute N independent samples to the mean, not
    one N-times-inflated sample (that was the bug: summing every row for a
    case_id, across every run, then dividing by the count of distinct ids).

    Preferred grouping key is (`case_id`, `git_sha7`) -- `git_sha7` is a
    stable per-run discriminator the eval runner writes into every ledger
    row's `client.context` (dealpoint/eval/run.py). Rows carrying it are
    grouped and summed by that pair directly, no ambiguity. Rows without it
    (older ledger entries, written before this field existed) fall back to
    `_batch_by_ts_wraparound`, which recovers per-execution totals from `ts`
    ordering alone.
    """
    if not rows:
        return []

    with_sha = [r for r in rows if r.get("git_sha7")]
    without_sha = [r for r in rows if not r.get("git_sha7")]

    samples: list[float] = []

    grouped: dict[tuple[str, str], float] = {}
    for row in with_sha:
        case_id = row.get("case_id")
        if not case_id:
            continue
        key = (case_id, row["git_sha7"])
        grouped[key] = grouped.get(key, 0.0) + float(row.get("usd", 0.0) or 0.0)
    samples.extend(grouped.values())

    samples.extend(_batch_by_ts_wraparound(without_sha))

    return samples


def per_case_usd(
    arm: str,
    model: str,
    ledger_path: Path = LEDGER_PATH,
    calls_per_case: int = EST_CALLS_PER_CASE,
) -> tuple[float, str]:
    """Best available per-case cost estimate for (arm, model), and its basis.

    Priority order mirrors `estimate()`'s sweep-level basis resolution
    (spec §3.4), applied to a single (arm, model) pair. `calls_per_case` is
    only used by the branch-4 (tokens x price) fallback -- branches 1-3
    already measure real per-case totals directly.
    """
    ledger = read_ledger(ledger_path)

    # 1. ledger:measured(arm,model) -- rows carrying both arm and case_id
    rows_1 = [
        row
        for row in ledger
        if row.get("arm") == arm and row.get("model") == model and row.get("case_id")
    ]
    samples_1 = _per_execution_case_costs(rows_1)
    if samples_1:
        mean = sum(samples_1) / len(samples_1)
        return mean, "ledger:measured(arm,model)"

    # 2. ledger:measured(model) -- rows with a case_id but no arm
    rows_2 = [
        row
        for row in ledger
        if row.get("model") == model and row.get("case_id") and not row.get("arm")
    ]
    samples_2 = _per_execution_case_costs(rows_2)
    if samples_2:
        mean = sum(samples_2) / len(samples_2)
        return mean, "ledger:measured(model)"

    # 3. results:measured -- mean usd over data/results/*.jsonl rows for (arm, model)
    result_costs: list[float] = []
    if RESULTS_DIR.exists():
        for path in RESULTS_DIR.glob("*.jsonl"):
            if path.name == "spend_ledger.jsonl":
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("arm") == arm and row.get("model") == model and "usd" in row:
                    result_costs.append(float(row["usd"]))
    if result_costs:
        return sum(result_costs) / len(result_costs), "results:measured"

    # 4. ledger:tokens x price -- measured mean tokens/call x price table x calls/case.
    # When this model has NO ledger rows at all (e.g. a model never called
    # before, like z-ai/glm-5.3-flash before its first sweep), fall back to
    # the mean tokens/call across every model's ledger rows rather than
    # silently returning 0 in/0 out -> $0.00 (an estimate of zero passes
    # every cap and measures nothing; the M4 corrective plan requires this
    # fallback so `budget four_arm`'s GLM leg is a real, non-zero estimate).
    model_rows = [r for r in ledger if r.get("model") == model]
    prices, price_basis = fetch_prices()
    if model not in prices:
        raise KeyError(f"no pricing available for model {model!r}")
    if model_rows:
        mean_in, mean_out = _mean_tokens_per_call(model_rows)
        basis_label = f"ledger:tokens x {price_basis}"
    else:
        mean_in, mean_out = _mean_tokens_per_call(ledger)
        basis_label = f"ledger:corpus-tokens x {price_basis}"
    per_call = mean_in * prices[model]["prompt"] + mean_out * prices[model]["completion"]
    return per_call * calls_per_case, basis_label


def _estimate_leg(leg: dict, ledger_path: Path) -> tuple[float, int, int, set[str]]:
    """Sum `per_case_usd` over one leg's arms x models x n_cases. Shared by
    both the flat sweep shape and the multi-`legs` shape.

    A leg carrying `tokens_per_call` (spec deliverable, M5's judge shape) is
    priced directly from `fetch_prices()` for EACH of its `models` -- `n_cases`
    traces x each model x the fixed token shape -- bypassing the ledger
    branches entirely (they are agent-case shaped and would return nonsense
    for a per-trace judge call with no prior ledger history). This is the
    only leg shape where >1 `models` entries are summed independently rather
    than the single-model x n_models_multiplier scaling `pareto` uses.
    """
    arms = leg["arms"] or [""]
    models = leg["models"]
    n_cases_per_combo = leg["n_cases"]

    tokens_per_call = leg.get("tokens_per_call")
    if tokens_per_call is not None:
        prices, price_basis = fetch_prices()
        total_est = 0.0
        total_calls = 0
        total_cases = 0
        bases: set[str] = set()
        for model in models:
            if model not in prices:
                raise KeyError(f"no pricing available for judge model {model!r}")
            per_call = (
                tokens_per_call["input"] * prices[model]["prompt"]
                + tokens_per_call["output"] * prices[model]["completion"]
            )
            total_est += per_call * n_cases_per_combo
            total_calls += n_cases_per_combo
            total_cases += n_cases_per_combo
            bases.add(f"tokens_per_call x {price_basis}")
        return total_est, total_calls, total_cases, bases

    calls_per_case = leg.get("calls_per_case", EST_CALLS_PER_CASE)
    # `pareto` prices 3 models but only one concrete model id is known this
    # early (the other two are M6's slate); scale the known model's per-case
    # cost by this multiplier as a stand-in until the real slate is pinned.
    n_models_multiplier = leg.get("n_models", len(models))

    total_est = 0.0
    total_calls = 0
    total_cases = 0
    bases = set()

    for model in models:
        for arm in arms:
            per_case, basis = per_case_usd(
                arm or "D", model, ledger_path, calls_per_case=calls_per_case
            )
            bases.add(basis)
            n_cases = n_cases_per_combo * (n_models_multiplier if len(models) == 1 else 1)
            total_est += per_case * n_cases
            total_calls += round(calls_per_case * n_cases)
            total_cases += n_cases

    return total_est, total_calls, total_cases, bases


def estimate(sweep_name: str, ledger_path: Path = LEDGER_PATH) -> dict:
    """Cost estimate for a named sweep (spec §3.4).

    Always returns a dict with (at least) `calls` (int) and `est_usd`
    (plain float) -- both mandatory for `adws/adw_modules/spend.py`. A sweep
    may be a single flat shape (`arms`/`models`/`n_cases`, e.g. `judges`,
    `pareto`) or a `legs` list of such shapes summed together (e.g.
    `four_arm`'s GLM-headline + Haiku-replication design, M4 spec
    deliverable 5) -- both produce the same result shape below.
    """
    if sweep_name not in SWEEP_DEFS:
        raise KeyError(f"unknown sweep {sweep_name!r}; expected one of {sorted(SWEEP_DEFS)}")
    sweep = SWEEP_DEFS[sweep_name]
    legs = sweep.get("legs", [sweep])

    total_est = 0.0
    total_calls = 0
    total_cases = 0
    bases: set[str] = set()
    all_arms: list[str] = []
    all_models: list[str] = []

    for leg in legs:
        est, calls_n, cases_n, leg_bases = _estimate_leg(leg, ledger_path)
        total_est += est
        total_calls += calls_n
        total_cases += cases_n
        bases |= leg_bases
        for a in leg["arms"]:
            if a not in all_arms:
                all_arms.append(a)
        for m in leg["models"]:
            if m not in all_models:
                all_models.append(m)

    basis = bases.pop() if len(bases) == 1 else "+".join(sorted(bases))
    calls = total_calls
    arms = all_arms
    models = all_models
    est_usd = round(total_est, 6)
    per_call_usd = round(est_usd / calls, 6) if calls else 0.0
    per_case_usd_value = round(est_usd / total_cases, 6) if total_cases else 0.0

    return {
        "sweep": sweep_name,
        "calls": calls,
        "est_usd": est_usd,
        "per_call_usd": per_call_usd,
        "basis": basis,
        "cases": total_cases,
        "per_case_usd": per_case_usd_value,
        "model": models[0] if len(models) == 1 else models,
        "arms": arms,
        "legs": [
            {"arms": leg["arms"], "models": leg["models"], "n_cases": leg["n_cases"]}
            for leg in legs
        ]
        if "legs" in sweep
        else None,
    }


def cap_usd(dotenv_path: str = ".env") -> float | None:
    """The operator's spend cap: env var first, then `.env` (mirrors adws/spend.py)."""
    raw = os.environ.get(CAP_ENV, "").strip()
    if not raw:
        raw = (dotenv_values(dotenv_path).get(CAP_ENV) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def assert_within_cap(
    est_usd: float, ledger_path: Path = LEDGER_PATH, dotenv_path: str = ".env"
) -> None:
    """Raise `SpendCapError` if the cap is unset/unparseable, or if realized + est > cap."""
    cap = cap_usd(dotenv_path)
    if cap is None:
        raise SpendCapError(
            f"{CAP_ENV} is not set (or is unparseable) -- an unattended sweep cannot "
            f"start without an explicit cap. Set {CAP_ENV} in .env."
        )
    realized = realized_usd(ledger_path)
    projected = realized + est_usd
    if projected > cap:
        raise SpendCapError(
            f"projected spend ${projected:.4f} (realized ${realized:.4f} + estimate "
            f"${est_usd:.4f}) exceeds cap ${cap:.4f} ({CAP_ENV})"
        )
