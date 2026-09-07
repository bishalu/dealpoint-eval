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


def test_is_root_uses_is_root_then_span_parents_never_root_span_id():
    from dealpoint.eval.braintrust_showroom import _is_root

    # what the fetch API actually returns for a root: is_root True, span_id != root_span_id
    assert _is_root({"is_root": True, "span_id": "a", "root_span_id": "b", "span_parents": None})
    assert not _is_root({"is_root": False, "span_id": "a", "root_span_id": "a", "span_parents": ["x"]})
    assert _is_root({"span_id": "a", "root_span_id": "b", "span_parents": []})
    assert not _is_root({"span_id": "a", "root_span_id": "a", "span_parents": ["p"]})


def test_agent_run_log_plan_covers_every_stored_run_once():
    """Offline over the committed result files: the extra log plan is exactly the stored agent
    rows minus the judged pairs, and never duplicates a (case, variant) under either id form."""
    from dealpoint.eval.braintrust_cockpit import _judged_subset
    from dealpoint.eval.braintrust_showroom import agent_run_log_plan
    from dealpoint.eval.braintrust_sync import _agent_experiment_plans, _judged_variant_id_for

    subset = _judged_subset()
    judged = {(c, v["variant_id"]) for v in subset["variants"] for c in subset["case_ids"]}
    extra = agent_run_log_plan(set(judged))
    stored = {(r["case_id"], f"{p['metadata']['arm']}@{p['metadata']['model']}") for p in _agent_experiment_plans() for r in p["rows"]}
    planned = {(p["case_id"], p["variant_id"]) for p in extra}
    assert len(planned) == len(extra), "no duplicates"
    assert planned <= stored
    for case_id, variant_id in planned:
        arm, _, model = variant_id.partition("@")
        short = _judged_variant_id_for(arm, model)
        assert (case_id, short) not in judged, f"{case_id} {variant_id} is already a judged trace"
    assert len(stored) - len(planned) == len(judged), "judged pairs are the only ones removed"
    assert all(p["category"] == "agent" and p["row"] for p in extra)


def test_replay_plan_entry_resolves_short_and_full_variant_ids_and_never_collides_with_the_ledger():
    from dealpoint.eval.braintrust_showroom import replay_plan_entry

    a = replay_plan_entry("contract_144__q05", "D@glm")
    b = replay_plan_entry("contract_144__q05", "D@z-ai/glm-5.3-flash")
    assert a["row"]["case_id"] == b["row"]["case_id"] == "contract_144__q05"
    assert a["category"].startswith("live-replay-") and b["category"].startswith("live-replay-")
    with pytest.raises(SystemExit):
        replay_plan_entry("contract_144__q05", "Z@nope")


@pytest.mark.parametrize("name,axis,arm,model,cases", [
    ("A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc", "system", "A", "glm", "test-32"),
    ("D-anthropic_claude-haiku-4.5-e2b4a2b97561-e3ee9cc", "system", "D", "haiku", "test-18"),
    ("D-qwen_qwen3.7-flash-e2b4a2b97561-3fcdae7", "model", "D", "qwen3.7-flash", "test-32"),
    ("pareto-deepseek_deepseek-v4-flash", "model", "D", "deepseek-v4-flash", "test-32"),
    ("judge-D@glm", "judge", "D", "glm", "judged-18"),
    ("rag-m3-hybrid", "retrieval", None, None, "dev-58"),
    ("rag-m7-synthetic", "retrieval", None, None, "synthetic-106"),
    ("deepeval-crosscheck", "crosscheck", None, None, "judged-18"),
    ("playground-arm-A-terse", "prompt", "A", "glm", "judged-18"),
])
def test_classify_experiment_gives_every_experiment_one_factorial_schema(name, axis, arm, model, cases):
    from dealpoint.eval.braintrust_showroom import classify_experiment

    c = classify_experiment(name)
    assert c["axis"] == axis and c.get("arm") == arm and c.get("model") == model and c["cases"] == cases
    for key in ("varies", "holds", "scorers", "description"):
        assert c[key]
    assert f"axis:{axis}" in c["tags"] and f"cases:{cases}" in c["tags"]
    if arm:
        assert c["loop"] == ARMS[arm]["loop"] and c["retriever"] == ARMS[arm]["retriever"]["name"]
        assert c["skill"] == ("on" if ARMS[arm]["skill"] else "off")


def test_system_axis_descriptions_name_the_one_key_that_changes():
    from dealpoint.eval.braintrust_showroom import classify_experiment

    d = {arm: classify_experiment(f"{arm}-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc")["description"] for arm in ARM_ORDER}
    assert "control" in d["A"] and "loop" in d["B"] and "retriever" in d["C"] and "skill" in d["D"]


def test_playground_rows_are_the_exact_arm_a_packets_with_gold_expected():
    from dealpoint.eval.braintrust_showroom import playground_prompts, playground_rows

    rows = playground_rows()
    assert len(rows) == 18
    hero = next(r for r in rows if r["id"] == "contract_144__q05")
    assert "Knowledge" in hero["input"] and "Retrieved passages from this agreement:" in hero["input"]
    assert hero["metadata"]["n_passages"] == 5 and hero["input"].count("[section") == 5
    assert hero["expected"].startswith("Actual knowledge") and "Gold span:" in hero["expected"]
    redacted = next(r for r in rows if r["id"] == "contract_39__redacted_q05")
    assert redacted["expected"] == "ABSTAIN"
    prompts = playground_prompts()
    assert [p["slug"] for p in prompts] == ["arm-a-prompt-base", "arm-a-prompt-terse", "arm-a-prompt-cite-first", "arm-a-prompt-abstain-first"]
    assert all(p["messages"][1]["content"].startswith("{{input}}") and p["messages"][0]["role"] == "system" for p in prompts)
