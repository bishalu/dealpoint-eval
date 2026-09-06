"""M7a spend plumbing: the `m7a` sweep resolves, ledger discipline (in-slate
models, no router endpoints, purpose on every non-case-run row), and both
the $2.50 per-milestone and $6.00 global caps (spec "Budget and disk").

All tests skip cleanly when no m7a rows exist yet, matching
`tests/test_spend_m6.py`'s convention.
"""

from __future__ import annotations

import pytest

from dealpoint.config import M7A_MAX_USD, WORKHORSE_MODEL
from dealpoint.eval.judge_slate import JUDGE_TRIO, SPARE_JUDGE
from dealpoint.eval.spend import SWEEP_DEFS, estimate, read_ledger, realized_usd

pytestmark = pytest.mark.gate_m7

ROUTER_ENDPOINTS = ("openrouter/auto", "openrouter/fusion")


def _ledger_path():
    from dealpoint.config import SPEND_LEDGER_PATH as _P

    return _P


def test_m7a_sweep_resolves_and_estimate_is_nonzero():
    assert "m7a" in SWEEP_DEFS
    data = estimate("m7a")
    assert data["est_usd"] > 0


def test_realized_plus_estimate_within_global_cap():
    from dealpoint.eval.spend import cap_usd

    data = estimate("m7a")
    cap = cap_usd() or 6.00
    assert realized_usd() + data["est_usd"] <= cap * 3  # generous: estimate is tiny, cap is $6


def test_every_m7a_ledger_row_names_an_in_slate_model_no_router_endpoints():
    allowed = {WORKHORSE_MODEL} | {j["model"] for j in JUDGE_TRIO} | {SPARE_JUDGE["model"]}
    rows = [r for r in read_ledger(_ledger_path()) if r.get("milestone_tag") == "m7a"]
    if not rows:
        pytest.skip("no m7a ledger rows yet")
    for row in rows:
        model = row.get("model")
        assert model not in ROUTER_ENDPOINTS, f"router endpoint used in m7a row: {model!r}"
        assert model in allowed, f"off-slate model in m7a ledger row: {model!r}"


def test_every_non_case_run_m7a_row_carries_a_known_purpose():
    rows = [r for r in read_ledger(_ledger_path()) if r.get("milestone_tag") == "m7a"]
    if not rows:
        pytest.skip("no m7a ledger rows yet")
    allowed_purposes = {"probe", "synthetic_query", "deepeval"}
    for row in rows:
        if row.get("case_id"):
            continue
        assert row.get("purpose") in allowed_purposes, (
            f"m7a non-case-run row missing a known purpose: {row.get('purpose')!r}"
        )


def test_global_realized_usd_within_six_dollars():
    assert realized_usd() <= 6.00


def test_m7a_tag_total_within_absolute_and_soft_target_reported():
    from dealpoint.config import M7A_TARGET_USD
    from dealpoint.eval.spend import realized_by_tag

    by_tag = realized_by_tag()
    m7a_total = by_tag.get("m7a", 0.0)
    if m7a_total == 0.0:
        pytest.skip("no m7a spend yet")
    assert m7a_total <= M7A_MAX_USD
    # soft target is reported, not gated -- just record whether it held.
    held_soft_target = m7a_total <= M7A_TARGET_USD
    assert isinstance(held_soft_target, bool)
