"""The OpenRouter spend guard: a code phase, never an agent's judgement.

Two numbers decide whether a metered sweep may start unattended: what has
already been spent (the product's ledger, appended by every metered call) and
what the next sweep is projected to cost (the product's own budget command,
which must print JSON). Both are read by code; the cap comes from the
operator's environment. If any of the three is missing the answer is "stop",
because a guard that guesses is not a guard.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

from .data_types import MilestoneSpec, SpendCheck
from .utils import operator_env

CAP_ENV = "MAX_OPENROUTER_SPEND_USD"
LEDGER_PATH = Path("data/results/spend_ledger.jsonl")


def cap_usd() -> Optional[float]:
    """The operator's cap. `.env` is re-read on every call so a cap added while a
    long loop is running takes effect at the next spend gate — `load_dotenv()`
    never overrides a variable the parent process already exported as empty."""
    raw = os.environ.get(CAP_ENV, "").strip()
    if not raw:
        raw = (dotenv_values(".env").get(CAP_ENV) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def realized_usd(ledger: Path = LEDGER_PATH) -> float:
    """Sum of every metered call the product has recorded. Missing ledger = 0."""
    if not ledger.exists():
        return 0.0
    total = 0.0
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        try:
            total += float(json.loads(line).get("usd", 0.0) or 0.0)
        except (ValueError, TypeError):
            continue
    return round(total, 4)


def estimate(run, argv: list[str]) -> dict:
    """Run the product's budget command and parse its JSON. Raises on anything
    but a clean JSON object — an estimate that cannot be read is not an estimate."""
    completed = subprocess.run(argv, cwd=run.repo_root, env=operator_env(),
                               capture_output=True, text=True, timeout=1800)
    if completed.returncode != 0:
        raise RuntimeError(f"budget command exited {completed.returncode}: "
                           f"{(completed.stdout + completed.stderr)[-800:]}")
    text = completed.stdout.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise RuntimeError(f"budget command printed no JSON object: {text[-400:]}")
    data = json.loads(text[start:end + 1])
    if "est_usd" not in data or "calls" not in data:
        raise RuntimeError(f"budget JSON lacks est_usd/calls: {data}")
    return data


def check(run, milestone: MilestoneSpec) -> SpendCheck:
    """The guard. `ok=False` with the reason is a stop condition, not a defect."""
    cap = cap_usd()
    spent = realized_usd()
    if cap is None:
        return SpendCheck(ok=False, cap_usd=None, realized_usd=spent,
                          reason=f"{CAP_ENV} is not set — set it in .env to authorise unattended "
                                 f"OpenRouter spend (realized so far: ${spent:.2f})")
    if not milestone.budget_argv:
        return SpendCheck(ok=False, cap_usd=cap, realized_usd=spent,
                          reason=f"milestone {milestone.id} is spend-gated but declares no budget command")
    data = estimate(run, milestone.budget_argv)
    est = float(data["est_usd"])
    calls = int(data["calls"])
    projected = spent + est
    ok = projected <= cap
    reason = (f"projected ${projected:.2f} (realized ${spent:.2f} + estimate ${est:.2f} for "
              f"{calls} calls) {'<=' if ok else '>'} cap ${cap:.2f}")
    if milestone.absolute_usd is not None and est > milestone.absolute_usd:
        # Advisory here, enforced in the build: the product's budget command predates this
        # milestone's design, so its estimate is an upper bound, not the plan. The per-milestone
        # absolute is handed to the runner through the environment (adw_milestone) and checked
        # there against the real, re-sized sweep — the only place the true estimate exists.
        reason += (f"; NOTE estimate ${est:.2f} exceeds {milestone.id}'s absolute "
                   f"${milestone.absolute_usd:.2f} — the runner must resize before sweeping")
    if "sample_realized_usd" in data and "sample_est_usd" in data:
        reason += (f"; sample check: est ${float(data['sample_est_usd']):.4f} vs "
                   f"realized ${float(data['sample_realized_usd']):.4f}")
    return SpendCheck(ok=ok, reason=reason, cap_usd=cap, realized_usd=spent,
                      estimate_usd=est, calls=calls)
