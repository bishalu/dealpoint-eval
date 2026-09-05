"""Judge slate run-time verification: family constraint + fallback ladder (spec §5.2)."""

from __future__ import annotations

import pytest

from dealpoint.eval.judge_slate import (
    CANDIDATE_FAMILIES,
    JUDGE_FAMILIES,
    JUDGE_TRIO,
    SPARE_JUDGE,
    verify_slate,
)
from dealpoint.llm.client import FakeClient, ScriptedTurn

pytestmark = pytest.mark.gate_m5

_JSON_OK = '{"reasoning": 3, "evidence": 3, "trajectory": 3, "professional": 3, "notes": "ok"}'

_PRICES = {
    "mistralai/mistral-small-3.2-24b-instruct": {"prompt": 7.5e-8, "completion": 2.0e-7},
    "nvidia/nemotron-3-super-120b-a12b": {"prompt": 8.5e-8, "completion": 4.0e-7},
    "bytedance-seed/seed-2.0-mini": {"prompt": 1.0e-7, "completion": 4.0e-7},
    "amazon/nova-lite-v1": {"prompt": 6.0e-8, "completion": 2.4e-7},
    "some-other-family/cheap-model": {"prompt": 1.0e-8, "completion": 1.0e-8},
}


def test_judge_families_disjoint_from_candidate_families():
    assert set(JUDGE_FAMILIES).isdisjoint(set(CANDIDATE_FAMILIES))


def test_verify_slate_selects_all_three_named_judges_when_all_succeed():
    fake = FakeClient(script=[ScriptedTurn(content=_JSON_OK) for _ in range(len(JUDGE_TRIO))])
    result = verify_slate(fake, prices=_PRICES)
    models = {j["model"] for j in result["judges"]}
    assert models == {j["model"] for j in JUDGE_TRIO}
    assert result["price_basis"] == "provided"


def test_verify_slate_falls_back_to_spare_when_one_named_judge_fails():
    # First named judge fails (bad JSON), spare succeeds, then the remaining two succeed.
    script = [
        ScriptedTurn(content="not json"),  # judge 1 fails
        ScriptedTurn(content=_JSON_OK),  # spare succeeds
        ScriptedTurn(content=_JSON_OK),  # judge 2 succeeds
        ScriptedTurn(content=_JSON_OK),  # judge 3 succeeds
    ]
    fake = FakeClient(script=script)
    result = verify_slate(fake, prices=_PRICES)
    models = {j["model"] for j in result["judges"]}
    assert SPARE_JUDGE["model"] in models
    assert any(sub.get("with") == SPARE_JUDGE["model"] for sub in result["substitutions"])


def test_verify_slate_raises_when_fewer_than_two_families_survive():
    # Every named judge AND the spare fail every time -> only the dynamic
    # fallback (`some-other-family/cheap-model`) could succeed, but scripting
    # every attempt to fail forces < 2 families to survive.
    n_attempts = len(JUDGE_TRIO) * 3  # named + spare + one dynamic fallback attempt each
    script = [ScriptedTurn(content="not json") for _ in range(n_attempts + 5)]
    fake = FakeClient(script=script)
    with pytest.raises(RuntimeError):
        verify_slate(fake, prices=_PRICES)


def test_verify_slate_records_prices_and_basis():
    fake = FakeClient(script=[ScriptedTurn(content=_JSON_OK) for _ in range(len(JUDGE_TRIO))])
    result = verify_slate(fake, prices=_PRICES)
    for judge in result["judges"]:
        assert judge["prompt_usd_per_token"] > 0
        assert judge["completion_usd_per_token"] > 0
    assert "verified_at" in result
