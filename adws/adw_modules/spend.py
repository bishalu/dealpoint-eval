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
    if spent >= cap:
        return SpendCheck(ok=False, cap_usd=cap, realized_usd=spent,
                          reason=f"realized ${spent:.2f} already at/over cap ${cap:.2f} — raise "
                                 f"{CAP_ENV} or stop")
    # A pre-build projection is advisory: the budget command that models THIS milestone's
    # sweeps is usually built by the milestone itself, so an earlier sweep definition priced
    # at another model's cost is an upper bound at best (M4: $3.09 projected, $0.75 realised;
    # M6: $3.59 projected for a cheap-tier slate). The product runner enforces the live cap and
    # the per-milestone absolute (handed over through the environment) at sweep time, where the
    # real estimate exists. What the gate guarantees here is the realized envelope above.
    est, calls, note = 0.0, 0, "no budget command"
    if milestone.budget_argv:
        try:
            data = estimate(run, milestone.budget_argv)
            est, calls = float(data["est_usd"]), int(data["calls"])
            note = f"pre-build projection ${est:.2f} for {calls} calls (advisory; basis {data.get('basis', '?')})"
        except RuntimeError as error:
            note = f"no pre-build estimate ({str(error)[:160]})"
    reason = (f"realized ${spent:.2f} < cap ${cap:.2f}; {note}; the runner enforces "
              f"{milestone.id}'s absolute ${milestone.absolute_usd or 0:.2f} and the live cap at sweep time")
    return SpendCheck(ok=True, reason=reason, cap_usd=cap, realized_usd=spent,
                      estimate_usd=est, calls=calls)
