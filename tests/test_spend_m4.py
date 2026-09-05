"""M4 spend extensions: the corpus-tokens fallback, the multi-leg `four_arm`
estimate, and the runner-side milestone spend guard (spec deliverable 5).
"""

from __future__ import annotations

import json

import pytest

from dealpoint.eval.run import MilestoneSpendCapError, _assert_within_milestone_absolute
from dealpoint.eval.spend import SWEEP_DEFS, estimate, per_case_usd

pytestmark = pytest.mark.gate_m4


def _write_ledger(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


def _real_usd(path) -> float:
    """Sum of `usd` over `path`'s ledger rows -- a tiny local reimplementation
    used only to stub `dealpoint.eval.spend.realized_usd` in the milestone-
    guard tests below, isolated from the repo's real spend ledger."""
    total = 0.0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += float(json.loads(line).get("usd", 0.0) or 0.0)
    return round(total, 6)


def test_per_case_usd_falls_back_to_corpus_tokens_when_model_has_no_rows(tmp_path, monkeypatch):
    """A model never seen in the ledger (e.g. a brand-new workhorse model)
    must not silently estimate $0.00 -- it falls back to the corpus-wide
    mean tokens/call across every model's rows (spec corrective plan §9.2).
    """
    # Isolate from this repo's real data/results/*.jsonl rows (branch 3,
    # results:measured), which would otherwise shadow the branch-4 fallback
    # this test targets now that real M4 GLM/Haiku result files exist on disk.
    monkeypatch.setattr("dealpoint.eval.spend.RESULTS_DIR", tmp_path / "empty_results")
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [
            {"model": "some/other-model", "input_tokens": 5000, "output_tokens": 150, "usd": 0.01},
            {"model": "some/other-model", "input_tokens": 5000, "output_tokens": 150, "usd": 0.01},
        ],
    )
    # z-ai/glm-5.3-flash has a pinned price but (in this fixture ledger) no
    # rows of its own -- exactly the case the corpus-tokens fallback exists for.
    per_case, basis = per_case_usd("D", "z-ai/glm-5.3-flash", ledger_path=ledger)
    assert per_case > 0
    assert basis.startswith("ledger:corpus-tokens")


def test_four_arm_sweep_has_two_legs():
    sweep = SWEEP_DEFS["four_arm"]
    assert "legs" in sweep
    assert len(sweep["legs"]) == 2
    glm_leg, haiku_leg = sweep["legs"]
    assert glm_leg["models"] == ["z-ai/glm-5.3-flash"]
    assert set(glm_leg["arms"]) == {"A", "B", "C", "D"}
    assert glm_leg["n_cases"] == 32
    assert haiku_leg["models"] == ["anthropic/claude-haiku-4.5"]
    assert set(haiku_leg["arms"]) == {"A", "D"}
    assert haiku_leg["n_cases"] == 18


def test_four_arm_estimate_sums_both_legs_and_is_nonzero():
    data = estimate("four_arm")
    assert data["est_usd"] > 0
    assert data["cases"] == 32 * 4 + 18 * 2
    assert data["legs"] is not None
    assert len(data["legs"]) == 2


def test_judges_sweep_estimate_unaffected_by_legs_support():
    """The single-leg sweep shape (`judges`, `pareto`) must keep working --
    `legs` is an additive, optional extension (spec §9.1)."""
    data = estimate("judges")
    assert data["est_usd"] > 0
    assert data["legs"] is None


# --- runner-side milestone spend guard --------------------------------------


def test_milestone_guard_noop_when_env_vars_absent(monkeypatch):
    monkeypatch.delenv("DEALPOINT_MILESTONE_ABSOLUTE_USD", raising=False)
    monkeypatch.delenv("DEALPOINT_MILESTONE_SPEND_START_USD", raising=False)
    _assert_within_milestone_absolute(100.0)  # must not raise


def test_milestone_guard_raises_when_projected_exceeds_absolute(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [{"usd": 2.0, "milestone_tag": "m4"}])
    # M4.1: `_assert_within_milestone_absolute` calls `dealpoint.eval.spend.realized_usd()`
    # with no arguments, so its `path` default (bound to `LEDGER_PATH` at import time, not
    # re-read from `dealpoint.config.SPEND_LEDGER_PATH` on each call) is what actually runs --
    # monkeypatching the config attribute alone does not reach it. Patching
    # `dealpoint.eval.spend.realized_usd` itself is what isolates this test from the repo's
    # real spend ledger.
    monkeypatch.setattr("dealpoint.config.SPEND_LEDGER_PATH", ledger)
    monkeypatch.setattr("dealpoint.eval.spend.realized_usd", lambda path=ledger: _real_usd(ledger))
    monkeypatch.setenv("DEALPOINT_MILESTONE_ABSOLUTE_USD", "3.00")
    monkeypatch.setenv("DEALPOINT_MILESTONE_SPEND_START_USD", "0.0000")
    with pytest.raises(MilestoneSpendCapError):
        _assert_within_milestone_absolute(2.0)  # 2.0 (already spent) + 2.0 (est) > 3.00


def test_milestone_guard_passes_when_projected_within_absolute(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [{"usd": 0.5, "milestone_tag": "m4"}])
    monkeypatch.setattr("dealpoint.config.SPEND_LEDGER_PATH", ledger)
    monkeypatch.setattr("dealpoint.eval.spend.realized_usd", lambda path=ledger: _real_usd(ledger))
    monkeypatch.setenv("DEALPOINT_MILESTONE_ABSOLUTE_USD", "3.00")
    monkeypatch.setenv("DEALPOINT_MILESTONE_SPEND_START_USD", "0.0000")
    _assert_within_milestone_absolute(1.0)  # 0.5 + 1.0 <= 3.00, must not raise


def test_milestone_guard_measures_new_spend_since_baseline_not_since_zero(monkeypatch, tmp_path):
    """The guard must subtract the recorded baseline (spend at milestone
    start), not gate on the ledger's raw total -- a milestone started after
    other milestones have already spent money must not be blocked by their
    spend."""
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [{"usd": 2.5, "milestone_tag": "m1"}, {"usd": 0.1, "milestone_tag": "m4"}])
    monkeypatch.setattr("dealpoint.config.SPEND_LEDGER_PATH", ledger)
    monkeypatch.setattr("dealpoint.eval.spend.realized_usd", lambda path=ledger: _real_usd(ledger))
    monkeypatch.setenv("DEALPOINT_MILESTONE_ABSOLUTE_USD", "3.00")
    monkeypatch.setenv("DEALPOINT_MILESTONE_SPEND_START_USD", "2.5000")
    # realized (2.6) - start (2.5) = 0.1 new spend so far; + 1.0 est = 1.1 <= 3.00
    _assert_within_milestone_absolute(1.0)
