"""Metered smoke: 3 dev cases, arm B, anthropic/claude-haiku-4.5 (spec §6.4).

Marked `gate_m2` and `needs_model` ONLY -- never `needs_network`, so the
factory's `-m "gate_m2 and not needs_network"` gate does not silently skip
this test forever. Skips cleanly (not an error) if `OPENROUTER_API_KEY` is
unset, the dataset is absent, or the dense index is missing. Reuses a
persisted result file if it is already current, so re-running the gate does
not re-spend.
"""

from __future__ import annotations

import os

import pytest

from dealpoint.config import (
    INDEX_DIR,
    INDEX_VERSION_TXT_PATH,
    M1_M2_MAX_USD,
    M2_SMOKE_MAX_USD,
)
from dealpoint.eval.spend import realized_usd


def _index_available() -> bool:
    return INDEX_DIR.exists() and INDEX_VERSION_TXT_PATH.exists()


@pytest.mark.gate_m2
@pytest.mark.needs_model
def test_m2_smoke_three_dev_cases_arm_b_haiku(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
    if not _index_available():
        pytest.skip("dense index not built; run `just index`")

    from dealpoint.eval.smoke import run_smoke

    ledger_before = realized_usd()
    summary = run_smoke()
    ledger_after = realized_usd()

    assert summary["status_counts"].get("EXECUTION_FAILED", 0) == 0
    assert summary["n_cases"] == 3

    delta = round(ledger_after - ledger_before, 6)
    # delta may be 0 if the result file was already current (reused, not re-run)
    assert delta >= 0
    assert delta <= M2_SMOKE_MAX_USD
    assert ledger_after <= M1_M2_MAX_USD, (
        f"M1+M2 realised total ${ledger_after:.4f} exceeds the ${M1_M2_MAX_USD} "
        f"absolute per-milestone limit (spec's 'Budget scaling': target $0.50, "
        f"absolute 2x that)"
    )
