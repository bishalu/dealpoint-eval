"""gate_m9c: every MLflow tab filled from the stored rows (specs/milestones/m9c.md, D11-D15).

Offline, no network. Live steps (`--live`) run against a temporary SQLite tracking URI with a real
client, matching gate_m9/gate_m9b's own convention; Databricks-only failures (D12's `.start()`, D13's
label schemas/labeling sessions) are caught by the steps themselves and asserted-present-but-recorded,
never a test failure.
"""

from __future__ import annotations

import pytest

mlflow = pytest.importorskip("mlflow")

pytestmark = pytest.mark.gate_m9c

from dealpoint.eval import mlflow_tabs as t

SPINE_CASE, TWIN_CASE = "contract_144__q05", "contract_39__redacted_q05"


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory):
    d = tmp_path_factory.mktemp("mlflow_tabs")
    return f"sqlite:///{d}/m.db"


# --- D11: sessions ---------------------------------------------------------------------------------


def test_session_plan_covers_every_trace():
    plan = t.session_plan()
    assert len(plan) == 738
    assert all(e["session"] == e["case_id"] for e in plan)
    assert all(e["user"] for e in plan)


def test_spine_and_twin_sessions_hold_the_same_comparable_variants():
    by_case = t.sessions_by_case()
    spine = by_case[SPINE_CASE]
    twin = by_case[TWIN_CASE]
    spine_comparable = sorted(e["variant_id"] for e in spine if e["comparable"] == 1)
    twin_comparable = sorted(e["variant_id"] for e in twin if e["comparable"] == 1)
    assert spine_comparable == twin_comparable
    assert len(spine_comparable) == 9  # the 6 judged-18 + 3 GLM test-32 system@model combos
    judged18 = {"A@haiku", "D@haiku", "D@glm", "D@deepseek-v4-flash", "D@qwen3.7-flash", "D@gemini-3.1-flash-lite"}
    assert judged18 <= set(spine_comparable)


def test_retrieval_and_prompt_variant_traces_group_under_their_case():
    plan = t.session_plan()
    retrieval = [e for e in plan if e["category"] == "retrieval"]
    prompt_variant = [e for e in plan if e["category"] == "prompt-variant"]
    assert len(retrieval) == 406
    assert len(prompt_variant) == 67
    assert all(e["session"] == e["case_id"] for e in retrieval + prompt_variant)


# --- D12: judges -------------------------------------------------------------------------------


def test_judge_registration_plan_has_ten_scorers_with_descriptions():
    plan = t.judge_registration_plan()
    assert len(plan) == 10
    assert {p["name"] for p in plan if p["kind"] == "deterministic"} == {
        "grounded_accuracy", "answer_correct", "citation_gold_overlap", "citation_verbatim", "abstain_correct", "skill_adherence",
    }
    assert {p["name"] for p in plan if p["kind"] == "judge"} == {f"judge-{d}" for d in t.JUDGE_DIMS}
    assert all(p["description"] for p in plan)


def test_online_scoring_rule_targets_live_replay_category():
    assert t.ONLINE_SCORING_RULE["name"] == "judge-professional"
    assert "live-replay" in t.ONLINE_SCORING_RULE["sampling_filter"]


def test_replay_dry_run_tags_category_live_replay():
    plan = t.replay_trace(SPINE_CASE, "D@glm", live=False)
    assert plan["category"].startswith("live-replay-")


# --- D13: review ---------------------------------------------------------------------------------


def test_label_schemas_are_the_five_m5_dimensions_plus_gold_answer():
    names = {s["name"] for s in t.LABEL_SCHEMAS}
    assert names == {"reasoning", "evidence", "trajectory", "professional", "gold_answer"}
    assert sum(1 for s in t.LABEL_SCHEMAS if s["type"] == "expectation") == 1


def test_review_queues_hold_12_open_and_24_done():
    queues = t.review_queue_plan()
    by_name = {q["name"]: q for q in queues}
    assert len(by_name["Lawyer calibration review"]["items"]) == 12
    assert all(it["status"] == "PENDING" for it in by_name["Lawyer calibration review"]["items"])
    assert len(by_name["Lawyer-scored packets (24)"]["items"]) == 24
    assert all(it["status"] == "DONE" for it in by_name["Lawyer-scored packets (24)"]["items"])


# --- D14: agent versions -----------------------------------------------------------------------


def test_logged_model_plan_has_fourteen_entries():
    plan = t.logged_model_plan()
    assert len(plan) == 14
    agent_configs = [p for p in plan if p["kind"] == "agent-config"]
    prompt_variants = [p for p in plan if p["kind"] == "prompt-variant"]
    assert len(agent_configs) == 10
    assert len(prompt_variants) == 4
    for p in agent_configs:
        assert set(p["metrics"]) >= {"safe_accuracy", "usd_per_case", "net_accuracy"}


def test_registry_aliases_point_at_judged18_configs():
    for pool, config in t.REGISTRY_ALIASES_D14.values():
        assert pool == "judged-18"
        assert config


# --- D15: gateway and playground -----------------------------------------------------------------


def test_gateway_plan_has_five_named_endpoints():
    plan = t.gateway_plan()
    assert plan["connection"] == "openrouter"
    assert len(plan["endpoints"]) == 5
    assert plan["endpoints"]["dealpoint-glm"] == "z-ai/glm-5.3-flash"
    assert plan["endpoints"]["judge-bytedance"] == "bytedance-seed/seed-2.0-mini"


def test_every_prompt_gets_an_endpoint():
    plan = t.prompt_endpoint_plan()
    assert len(plan) == 8
    assert all(p["endpoint"] in t.GATEWAY_ENDPOINTS for p in plan)
    by_name = {p["name"]: p["endpoint"] for p in plan}
    assert by_name["arm-a-prompt-cite-first"] == "dealpoint-glm"
    assert by_name["judge-panel-rubric"] == "judge-mistral"


def test_gateway_smoke_dry_run_never_calls_openrouter(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("OpenRouterClient must not be constructed on a dry run")

    monkeypatch.setattr("dealpoint.llm.client.OpenRouterClient", _boom)
    result = t.gateway_smoke_test(live=False)
    assert result["planned_calls"] == 5


# --- no model calls except --gateway-smoke ------------------------------------------------------


def test_dry_run_main_makes_no_model_calls(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("no step should touch OpenRouterClient on a dry run")

    monkeypatch.setattr("dealpoint.llm.client.OpenRouterClient", _boom)
    assert t.main([]) == 0


def test_live_sessions_step_is_idempotent(tracking_uri, tmp_path, monkeypatch):
    """`--live` twice creates no new session tags the second time (spec gate: `--live` twice creates
    nothing new). Never touches the real `data/reports/*.json` manifests: both mm's and t's manifest
    paths are monkeypatched to a temp file for the duration of this test."""
    from dealpoint.eval import mlflow_mirror as mm

    monkeypatch.setattr(mm, "MANIFEST_PATH", tmp_path / "mlflow_manifest.json")
    monkeypatch.setattr(t, "MANIFEST_PATH", tmp_path / "mlflow_tabs_manifest.json")

    client = mm._client(tracking_uri)
    manifest0: dict = {}
    mm.step_traces(client, True, manifest0, limit=5)
    manifest: dict = {}
    t.step_sessions(client, True, manifest)
    first = manifest["sessions"]["tagged_this_run"]
    manifest2: dict = {}
    t.step_sessions(client, True, manifest2)
    assert manifest2["sessions"]["tagged_this_run"] == 0
    assert first >= 0
