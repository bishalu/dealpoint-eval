"""The Pareto-frontier computation, on synthetic points only, no disk (M6 spec DoD)."""

from __future__ import annotations

import random

import pytest

from dealpoint.eval.pareto_report import pareto_frontier

pytestmark = pytest.mark.gate_m6


def _by_id(rows):
    return {r["model"]: r for r in rows}


def test_empty_input():
    assert pareto_frontier([]) == []


def test_single_point():
    rows = pareto_frontier([{"model": "a", "usd_per_case": 0.01, "grounded_accuracy": 0.5}])
    assert len(rows) == 1
    assert rows[0]["frontier"] is True


def test_strictly_improving_chain_all_on_frontier():
    points = [
        {"model": "a", "usd_per_case": 0.01, "grounded_accuracy": 0.5},
        {"model": "b", "usd_per_case": 0.02, "grounded_accuracy": 0.6},
        {"model": "c", "usd_per_case": 0.03, "grounded_accuracy": 0.7},
    ]
    out = _by_id(pareto_frontier(points))
    assert all(out[m]["frontier"] for m in ("a", "b", "c"))


def test_strictly_dominated_chain_one_on_frontier():
    points = [
        {"model": "a", "usd_per_case": 0.01, "grounded_accuracy": 0.9},
        {"model": "b", "usd_per_case": 0.02, "grounded_accuracy": 0.5},
        {"model": "c", "usd_per_case": 0.03, "grounded_accuracy": 0.3},
    ]
    out = _by_id(pareto_frontier(points))
    assert out["a"]["frontier"] is True
    assert out["b"]["frontier"] is False
    assert out["c"]["frontier"] is False


def test_tie_on_x_only_higher_y_wins():
    points = [
        {"model": "a", "usd_per_case": 0.01, "grounded_accuracy": 0.5},
        {"model": "b", "usd_per_case": 0.01, "grounded_accuracy": 0.8},
    ]
    out = _by_id(pareto_frontier(points))
    assert out["b"]["frontier"] is True
    assert out["a"]["frontier"] is False


def test_tie_on_y_only_cheaper_wins():
    points = [
        {"model": "a", "usd_per_case": 0.02, "grounded_accuracy": 0.5},
        {"model": "b", "usd_per_case": 0.01, "grounded_accuracy": 0.5},
    ]
    out = _by_id(pareto_frontier(points))
    assert out["b"]["frontier"] is True
    assert out["a"]["frontier"] is False


def test_exact_tie_on_both_axes_lexicographically_smallest_id_wins():
    points = [
        {"model": "zebra", "usd_per_case": 0.01, "grounded_accuracy": 0.5},
        {"model": "alpha", "usd_per_case": 0.01, "grounded_accuracy": 0.5},
    ]
    out = _by_id(pareto_frontier(points))
    assert out["alpha"]["frontier"] is True
    assert out["zebra"]["frontier"] is False
    assert out["zebra"]["tied_with"] == "alpha"
    # never both
    assert not (out["alpha"]["frontier"] and out["zebra"]["frontier"])


def test_none_axis_excluded_with_reason():
    points = [
        {"model": "a", "usd_per_case": None, "grounded_accuracy": 0.5},
        {"model": "b", "usd_per_case": 0.02, "grounded_accuracy": None},
        {"model": "c", "usd_per_case": 0.01, "grounded_accuracy": 0.9},
    ]
    out = _by_id(pareto_frontier(points))
    assert out["a"]["frontier"] is False
    assert "usd_per_case" in out["a"]["frontier_note"]
    assert out["b"]["frontier"] is False
    assert "grounded_accuracy" in out["b"]["frontier_note"]
    assert out["c"]["frontier"] is True


def test_order_independence():
    points = [
        {"model": "a", "usd_per_case": 0.01, "grounded_accuracy": 0.9},
        {"model": "b", "usd_per_case": 0.02, "grounded_accuracy": 0.5},
        {"model": "c", "usd_per_case": 0.005, "grounded_accuracy": 0.95},
        {"model": "d", "usd_per_case": None, "grounded_accuracy": 0.5},
    ]
    baseline = {r["model"]: r["frontier"] for r in pareto_frontier(points)}
    shuffled = list(points)
    random.Random(1).shuffle(shuffled)
    result2 = {r["model"]: r["frontier"] for r in pareto_frontier(shuffled)}
    assert baseline == result2
