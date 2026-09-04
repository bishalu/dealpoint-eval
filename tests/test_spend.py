"""Ledger, estimation and the spend cap (spec §3)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from dealpoint.config import DEFAULT_MODEL
from dealpoint.eval.spend import (
    PINNED_PRICES,
    SpendCapError,
    assert_within_cap,
    cap_usd,
    estimate,
    fetch_prices,
    per_case_usd,
    read_ledger,
    realized_by_tag,
    realized_usd,
)

pytestmark = pytest.mark.gate_m2


def _write_ledger(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


def test_read_ledger_tolerant_of_blank_and_corrupt_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"usd": 0.01}\n\nnot json\n{"usd": 0.02}\n', encoding="utf-8")
    rows = read_ledger(path)
    assert len(rows) == 2
    assert realized_usd(path) == pytest.approx(0.03)


def test_realized_usd_matches_real_ledger():
    from dealpoint.config import SPEND_LEDGER_PATH

    if not SPEND_LEDGER_PATH.exists():
        pytest.skip("no real ledger present")
    total = realized_usd(SPEND_LEDGER_PATH)
    assert total >= 0.0


def test_realized_by_tag(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _write_ledger(
        path,
        [
            {"usd": 0.01, "milestone_tag": "m1"},
            {"usd": 0.02, "milestone_tag": "m1"},
            {"usd": 0.03, "milestone_tag": "m2"},
        ],
    )
    totals = realized_by_tag(path)
    assert totals["m1"] == pytest.approx(0.03)
    assert totals["m2"] == pytest.approx(0.03)


# --- assert_within_cap ---------------------------------------------------


def test_assert_within_cap_raises_on_unset_cap(tmp_path, monkeypatch):
    monkeypatch.delenv("MAX_OPENROUTER_SPEND_USD", raising=False)
    empty_env = tmp_path / ".env.empty"
    empty_env.write_text("", encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(SpendCapError, match="MAX_OPENROUTER_SPEND_USD"):
        assert_within_cap(0.01, ledger_path=ledger, dotenv_path=str(empty_env))


def test_assert_within_cap_raises_on_overrun(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_OPENROUTER_SPEND_USD", "0.10")
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [{"usd": 0.09}])
    with pytest.raises(SpendCapError):
        assert_within_cap(0.05, ledger_path=ledger, dotenv_path=str(tmp_path / "missing.env"))


def test_assert_within_cap_passes_under_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_OPENROUTER_SPEND_USD", "10")
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [{"usd": 0.01}])
    assert_within_cap(0.05, ledger_path=ledger, dotenv_path=str(tmp_path / "missing.env"))


def test_cap_usd_reads_env_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_OPENROUTER_SPEND_USD", "2.5")
    assert cap_usd(dotenv_path=str(tmp_path / "missing.env")) == 2.5


def test_cap_usd_falls_back_to_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("MAX_OPENROUTER_SPEND_USD", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("MAX_OPENROUTER_SPEND_USD=4\n", encoding="utf-8")
    assert cap_usd(dotenv_path=str(env_file)) == 4.0


def test_cap_usd_none_when_unparseable(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_OPENROUTER_SPEND_USD", "not-a-number")
    assert cap_usd(dotenv_path=str(tmp_path / "missing.env")) is None


# --- fetch_prices ----------------------------------------------------------


def test_fetch_prices_falls_back_to_pinned_on_failure(monkeypatch):
    import requests

    def _boom(*args, **kwargs):
        raise requests.ConnectionError("network unreachable")

    monkeypatch.setattr(requests, "get", _boom)
    prices, basis = fetch_prices()
    assert basis.startswith("pinned:")
    assert "anthropic/claude-haiku-4.5" in prices
    assert prices == PINNED_PRICES


def test_pinned_haiku_prices_reproduce_a_real_ledger_row_to_the_cent():
    from dealpoint.config import SPEND_LEDGER_PATH

    if not SPEND_LEDGER_PATH.exists():
        pytest.skip("no real ledger present")
    rows = read_ledger(SPEND_LEDGER_PATH)
    haiku_rows = [r for r in rows if r.get("model") == "anthropic/claude-haiku-4.5"]
    if not haiku_rows:
        pytest.skip("no haiku rows in ledger")
    row = haiku_rows[0]
    prices = PINNED_PRICES["anthropic/claude-haiku-4.5"]
    predicted = row["input_tokens"] * prices["prompt"] + row["output_tokens"] * prices["completion"]
    assert predicted == pytest.approx(row["usd"], abs=0.01)


# --- estimate --------------------------------------------------------------


@pytest.mark.parametrize("sweep_name", ["four_arm", "judges", "pareto"])
def test_estimate_returns_required_keys(sweep_name):
    data = estimate(sweep_name)
    assert isinstance(data["calls"], int)
    assert isinstance(data["est_usd"], float)
    assert "per_call_usd" in data
    assert "basis" in data
    assert data["est_usd"] > 0
    assert data["calls"] > 0


def test_estimate_unknown_sweep_raises():
    with pytest.raises(KeyError):
        estimate("not_a_real_sweep")


def test_per_case_usd_uses_ledger_tokens_fallback_when_no_case_ids():
    per_case, basis = per_case_usd("B", DEFAULT_MODEL)
    assert per_case > 0
    assert basis.startswith(("ledger:", "results:"))


def test_per_case_usd_averages_per_execution_not_per_distinct_case_id(tmp_path):
    """A case run twice (two separate runs, e.g. two smoke invocations at two
    different git_sha7's) must contribute two independent samples to the
    mean, not one execution summed twice under a single case_id key -- a
    case run N times must not read as N times as expensive as running it
    once. `git_sha7` is the per-run discriminator the runner writes into
    every ledger row's context (dealpoint/eval/run.py) precisely so this
    distinction is unambiguous going forward."""
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [
            {
                "arm": "B",
                "model": DEFAULT_MODEL,
                "case_id": "contract_0__q01",
                "git_sha7": "aaaaaaa",
                "usd": 0.02,
                "ts": "2026-09-04T14:00:00+00:00",
            },
            {
                "arm": "B",
                "model": DEFAULT_MODEL,
                "case_id": "contract_0__q01",
                "git_sha7": "bbbbbbb",
                "usd": 0.02,
                "ts": "2026-09-04T15:00:00+00:00",
            },
        ],
    )
    per_case, basis = per_case_usd("B", DEFAULT_MODEL, ledger_path=ledger)
    assert per_case == pytest.approx(0.02)
    assert basis == "ledger:measured(arm,model)"


def test_per_case_usd_ts_fallback_averages_per_execution_no_git_sha7(tmp_path):
    """Same scenario as above, but for legacy rows carrying no `git_sha7` at
    all -- the ts-ordered wraparound fallback must still recover per-
    execution totals correctly from a realistic multi-case sweep shape
    (each case appearing once per run, in the same order, run twice)."""
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [
            {"arm": "B", "model": DEFAULT_MODEL, "case_id": "c__q01", "usd": 0.01, "ts": "t0"},
            {"arm": "B", "model": DEFAULT_MODEL, "case_id": "c__q02", "usd": 0.03, "ts": "t1"},
            {"arm": "B", "model": DEFAULT_MODEL, "case_id": "c__q01", "usd": 0.01, "ts": "t2"},
            {"arm": "B", "model": DEFAULT_MODEL, "case_id": "c__q02", "usd": 0.03, "ts": "t3"},
        ],
    )
    per_case, _basis = per_case_usd("B", DEFAULT_MODEL, ledger_path=ledger)
    # two executions of c__q01 costing 0.01 each, two of c__q02 costing 0.03 each
    # -> mean of 4 per-execution samples [0.01, 0.03, 0.01, 0.03] = 0.02
    assert per_case == pytest.approx(0.02)


def test_per_case_usd_sums_multiple_calls_within_one_execution(tmp_path):
    """Multiple LLM calls for the SAME execution of a case (contiguous, same
    case_id) must be summed into one per-execution total, not averaged
    against each other as if they were separate executions."""
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [
            {"arm": "B", "model": DEFAULT_MODEL, "case_id": "c__q01", "usd": 0.01, "ts": "t0"},
            {"arm": "B", "model": DEFAULT_MODEL, "case_id": "c__q01", "usd": 0.02, "ts": "t1"},
            {"arm": "B", "model": DEFAULT_MODEL, "case_id": "c__q02", "usd": 0.03, "ts": "t2"},
        ],
    )
    per_case, _basis = per_case_usd("B", DEFAULT_MODEL, ledger_path=ledger)
    # one execution of c__q01 costing 0.03 (0.01+0.02), one of c__q02 costing 0.03
    assert per_case == pytest.approx(0.03)


# --- budget CLI --------------------------------------------------------


def test_budget_cli_prints_valid_json_satisfying_adws_contract():
    completed = subprocess.run(
        [sys.executable, "-m", "dealpoint.eval.budget", "four_arm"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    text = completed.stdout.strip()
    start, end = text.find("{"), text.rfind("}")
    assert start != -1 and end > start
    data = json.loads(text[start : end + 1])
    assert "est_usd" in data
    assert "calls" in data
    assert isinstance(data["est_usd"], float)
    assert isinstance(data["calls"], int)


def test_budget_sample_flag_wiring_with_fake_run(monkeypatch, tmp_path):
    """Unit-test `--sample`'s wiring with a stubbed `run_eval_set` (never a real
    metered call) -- the spec forbids invoking `--sample` for real in M2."""
    import dealpoint.eval.budget as budget_mod

    calls = {}

    def fake_run_eval_set(**kwargs):
        calls["kwargs"] = kwargs
        return {"n_cases": 3}

    monkeypatch.setattr("dealpoint.eval.run.run_eval_set", fake_run_eval_set)
    monkeypatch.setattr(budget_mod, "realized_usd", lambda: 0.05)

    data = budget_mod.run_sample("four_arm")
    assert "sample_est_usd" in data
    assert "sample_realized_usd" in data
    assert calls["kwargs"]["fake"] is False  # confirms the code path, not a real spend


def test_budget_cli_unknown_sweep_exits_nonzero_with_names_on_stderr():
    completed = subprocess.run(
        [sys.executable, "-m", "dealpoint.eval.budget", "not_a_sweep"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert "four_arm" in completed.stderr
