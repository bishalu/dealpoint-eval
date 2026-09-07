"""gate_m9: D3/D4, the two spend-gated MLflow-unique steps -- offline guard behaviour only.
Neither step is ever invoked with `--live` here; a real network call would violate the offline suite."""

from __future__ import annotations

import pytest

mlflow = pytest.importorskip("mlflow")

pytestmark = pytest.mark.gate_m9

from dealpoint.eval import mlflow_judges as mj


def test_align_dry_run_prints_estimate_and_does_not_spend(capsys):
    rc = mj.main(["align"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "estimate" in out.lower() or "$" in out
    assert "dry run" in out.lower()


def test_optimize_dry_run_prints_estimate_and_does_not_spend(capsys):
    rc = mj.main(["optimize"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "$" in out
    assert "dry run" in out.lower()


def test_align_live_stops_above_the_cap(monkeypatch, capsys):
    monkeypatch.setenv("MAX_OPENROUTER_SPEND_USD", "0.0")
    rc = mj.main(["align", "--live"])
    assert rc != 0
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "refus" in out.lower() or "cap" in out.lower()


def test_optimize_live_skips_not_fails_above_the_cap(monkeypatch, capsys):
    monkeypatch.setenv("MAX_OPENROUTER_SPEND_USD", "0.0")
    rc = mj.main(["optimize", "--live"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "skip" in out.lower()


def test_score_new_dry_run_does_not_spend(capsys):
    rc = mj.main(["score-new"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry run" in out.lower()
