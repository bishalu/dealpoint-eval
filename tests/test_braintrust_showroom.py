"""Showroom + org scoping: the score ledger and showroom manifest derive their paths
from the org the active API key belongs to, so a fresh org never inherits another
org's ledger state or overwrites its manifest. Offline: no test touches the network."""

from __future__ import annotations

from itertools import pairwise

import pytest

from dealpoint.config import ARM_ORDER, ARMS


def test_org_slug_from_name_is_deterministic_and_filesystem_safe():
    from dealpoint.eval.braintrust_adapter import org_slug_from_name

    assert org_slug_from_name("Vibeset Technologies") == "vibeset-technologies"
    assert org_slug_from_name("bishal.ai") == "bishal-ai"
    assert org_slug_from_name("  Weird//Name  ") == "weird-name"


def test_org_slug_without_a_key_is_local_and_never_calls_the_network(monkeypatch):
    import dealpoint.eval.braintrust_adapter as adapter

    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)

    def _boom(*a, **k):
        raise AssertionError("no key -> no network")

    monkeypatch.setattr("requests.get", _boom)
    assert adapter.org_slug() == "local"


def test_org_slug_resolves_the_single_org_bound_to_the_key(monkeypatch):
    import dealpoint.eval.braintrust_adapter as adapter

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"objects": [{"name": "bishal.ai", "id": "x"}]}

    calls: list[str] = []

    def _get(url, headers=None, timeout=None):
        calls.append(headers["Authorization"])
        return _Resp()

    monkeypatch.setattr("requests.get", _get)
    adapter._ORG_SLUG_CACHE.clear()
    assert adapter.org_slug("sk-demo") == "bishal-ai"
    assert adapter.org_slug("sk-demo") == "bishal-ai"
    assert calls == ["Bearer sk-demo"], "one lookup per key; the slug is cached"
    adapter._ORG_SLUG_CACHE.clear()


def test_ledger_and_manifest_paths_scope_to_the_active_org(monkeypatch):
    import dealpoint.eval.braintrust_adapter as adapter
    import dealpoint.eval.braintrust_cockpit as cockpit
    import dealpoint.eval.braintrust_showroom as showroom

    monkeypatch.delenv("BRAINTRUST_LEDGER_FILE", raising=False)
    monkeypatch.setattr(adapter, "org_slug", lambda key=None: "bishal-ai")
    monkeypatch.setattr(showroom, "LEDGER_PATH", None)
    monkeypatch.setattr(showroom, "MANIFEST_PATH", None)
    monkeypatch.setattr(cockpit, "SCORE_LEDGER_PATH", None)

    ledger = showroom._ledger_path()
    assert ledger.parts[-3:] == ("orgs", "bishal-ai", "braintrust_score_ledger.jsonl")
    assert showroom._manifest_path().parts[-3:] == ("orgs", "bishal-ai", "showroom_manifest.json")
    assert cockpit._score_ledger_path() == ledger, "cockpit and showroom share one ledger per org"

    monkeypatch.setattr(adapter, "org_slug", lambda key=None: "vibeset-technologies")
    assert showroom._ledger_path() != ledger


def test_ledger_file_override_still_wins(monkeypatch, tmp_path):
    import dealpoint.eval.braintrust_adapter as adapter
    import dealpoint.eval.braintrust_cockpit as cockpit
    import dealpoint.eval.braintrust_showroom as showroom

    monkeypatch.setattr(adapter, "org_slug", lambda key=None: "bishal-ai")
    monkeypatch.setattr(showroom, "LEDGER_PATH", None)
    monkeypatch.setattr(cockpit, "SCORE_LEDGER_PATH", None)
    override = tmp_path / "custom_ledger.jsonl"
    monkeypatch.setenv("BRAINTRUST_LEDGER_FILE", str(override))
    assert showroom._ledger_path() == override
    assert cockpit._score_ledger_path() == override


def test_arm_parameter_sets_match_the_arms_in_config():
    from dealpoint.eval.braintrust_showroom import arm_parameter_sets

    sets = {s["arm"]: s for s in arm_parameter_sets()}
    assert tuple(sets) == ARM_ORDER
    for arm in ARM_ORDER:
        truth, s = ARMS[arm], sets[arm]
        assert s["agent_loop"] == (truth["loop"] == "agent"), arm
        assert s["retriever"] == truth["retriever"]["name"], arm
        assert s["skill"] == truth["skill"], arm
        assert (s["max_tool_calls"] > 0) == s["agent_loop"], arm
    for prev, cur in pairwise(ARM_ORDER):
        changed = [k for k in ("agent_loop", "retriever", "skill") if sets[prev][k] != sets[cur][k]]
        assert len(changed) == 1, f"{prev}->{cur} must change exactly one key, changed {changed}"


@pytest.mark.parametrize("arm", ARM_ORDER)
def test_arm_parameter_descriptions_do_not_contradict_the_loop(arm):
    from dealpoint.eval.braintrust_showroom import arm_parameter_sets

    s = {x["arm"]: x for x in arm_parameter_sets()}[arm]
    desc = s["description"].lower()
    if s["agent_loop"]:
        assert "single shot" not in desc and "agent" in desc
    else:
        assert "agent loop" not in desc
