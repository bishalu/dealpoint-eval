"""M6 Pareto slate verification, driven fully offline with a FakeClient (spec DoD)."""

from __future__ import annotations

import json

import pytest

from dealpoint.eval.pareto_slate import (
    PARETO_CANDIDATES,
    PROBE_CASE_IDS,
    probe_candidate,
    verify_slate,
)
from dealpoint.llm.client import FakeClient, ScriptedToolCall, ScriptedTurn

pytestmark = pytest.mark.gate_m6


@pytest.fixture(autouse=True)
def _results_in_tmp(tmp_path, monkeypatch):
    """`verify_slate(..., fake=True)` drives `run_eval_set` with its default
    `out_dir`, which is the real `data/results/`. Without this the suite left
    `pareto_probe_<model>_..._offline_<sha>.jsonl` files in the repo on every
    run, which the factory's write gate then attributed to whichever agent
    ran the tests. Point the writer at tmp_path instead."""
    import dealpoint.eval.run as run_mod

    monkeypatch.setattr(run_mod, "RESULTS_DIR", tmp_path / "results")

_MODELS = [c["model"] for c in PARETO_CANDIDATES]


def _valid_answer_json() -> str:
    # ABSTAIN is valid regardless of a question's option list (empty
    # evidence is all the schema requires alongside it), so one script
    # works for both probe case ids without a per-question option mismatch.
    return json.dumps({"answer": "ABSTAIN", "evidence": [], "rationale": "Not addressed."})


def _tool_call_then_answer_script(n_cases: int = 2) -> list:
    """One tool-call turn, one stop turn, one finalization turn, per case --
    the shape `dealpoint.agent.loop.run_agent` actually consumes (see
    tests/test_agent_loop.py's own 4-turn script for the same shape)."""
    script = []
    for _ in range(n_cases):
        script.append(
            ScriptedTurn(
                tool_calls=[
                    ScriptedToolCall(name="search_agreement", arguments={"query": "consideration"})
                ]
            )
        )
        script.append(ScriptedTurn(content=None))
        script.append(ScriptedTurn(content=_valid_answer_json()))
    return script


def _no_tool_call_script(n_cases: int = 2) -> list:
    """Model answers directly with no tool call at all."""
    return [ScriptedTurn(content=_valid_answer_json()) for _ in range(n_cases)]


_PRICES = {m: {"prompt": 1e-7, "completion": 4e-7} for m in _MODELS}


def test_every_candidate_available_and_tool_calling_all_ok(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    model = _MODELS[0]
    fake = FakeClient(script=_tool_call_then_answer_script(len(PROBE_CASE_IDS)))
    payload = verify_slate(fake, models=(model,), prices=_PRICES, fake=True)
    cand = next(c for c in payload["candidates"] if c["model"] == model)
    assert cand["ok"] is True
    assert cand["excluded"] is False


def test_candidate_absent_from_price_table_excluded_no_call_made(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    model = _MODELS[0]
    prices_without = {m: p for m, p in _PRICES.items() if m != model}
    fake = FakeClient(script=[])
    payload = verify_slate(fake, models=(model,), prices=prices_without, fake=True)
    cand = next(c for c in payload["candidates"] if c["model"] == model)
    assert cand["available"] is False
    assert cand["excluded"] is True
    assert fake.calls == []


def test_candidate_with_no_tool_call_excluded_not_swapped(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    model = _MODELS[0]
    fake = FakeClient(script=_no_tool_call_script(len(PROBE_CASE_IDS)))
    payload = verify_slate(fake, models=(model,), prices=_PRICES, fake=True)
    cand = next(c for c in payload["candidates"] if c["model"] == model)
    assert cand["excluded"] is True
    assert cand["failure_rate"] is not None
    # surviving list is one shorter, never back-filled with a substitute
    assert payload["survivors"] == []


def test_probe_candidate_raises_for_model_outside_slate():
    fake = FakeClient(script=[])
    with pytest.raises(ValueError):
        probe_candidate(fake, "openrouter/auto")
    with pytest.raises(ValueError):
        probe_candidate(fake, "openrouter/fusion")


def test_prices_and_basis_recorded_on_every_candidate(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    model = _MODELS[0]
    fake = FakeClient(script=_tool_call_then_answer_script(len(PROBE_CASE_IDS)))
    payload = verify_slate(fake, models=(model,), prices=_PRICES, fake=True)
    assert payload["price_basis"] == "provided"
    for cand in payload["candidates"]:
        if cand.get("available"):
            assert cand["prompt_usd_per_token"] > 0
            assert cand["completion_usd_per_token"] > 0
