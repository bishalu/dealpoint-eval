"""gate_m9: the MLflow mirror of the Braintrust showroom (specs/milestones/m9.md).

Offline, no network: MLflow runs against a temporary SQLite tracking URI with a real client (no
fake), per the build plan section 0.1 (proven to work with no server running).
"""

from __future__ import annotations

import json

import pytest

mlflow = pytest.importorskip("mlflow")

pytestmark = pytest.mark.gate_m9

from dealpoint.eval import mlflow_mirror as m

SPINE_CASE, SPINE_VARIANT = "contract_144__q05", "D@glm"
TWIN_CASE, TWIN_VARIANT = "contract_39__redacted_q05", "D@glm"


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory):
    d = tmp_path_factory.mktemp("mlflow")
    return f"sqlite:///{d}/m.db"


@pytest.fixture(scope="module")
def client(tracking_uri):
    return m._client(tracking_uri)


# --- pure planning: exact counts ---------------------------------------------------------------


def test_run_plan_count_and_classification():
    plans = m.mirror_run_plan()
    assert len(plans) == 33
    from dealpoint.eval.braintrust_showroom import classify_experiment

    for p in plans:
        axis = classify_experiment(p["name"])["axis"]
        assert axis != "other", f"{p['name']} did not classify into a known axis"


def test_dataset_plan_count():
    plan = m.dataset_plan()
    assert set(plan) == set(m.DATASET_NAMES)
    assert len(plan) == 9


def test_prompt_plan_count():
    assert len(m.prompt_plan_m9()) == 8


def test_agent_trace_plan_count():
    plan = m.agent_trace_plan()
    assert len(plan) == 265
    assert sum(1 for p in plan if p["category"] == "judged") == 108


def test_retrieval_trace_plan_count():
    assert len(m.retrieval_trace_plan()) == 406


def test_prompt_variant_trace_plan_count():
    plan = m.prompt_variant_trace_plan()
    assert len(plan) == 67
    assert all(p["run_name"].startswith("playground-arm-A-") for p in plan)


def test_root_trace_total_is_738():
    total = len(m.agent_trace_plan()) + len(m.retrieval_trace_plan()) + len(m.prompt_variant_trace_plan())
    assert total == 738


def test_assessment_counts_derived_not_hardcoded():
    counts = m.assessment_counts()
    assert counts["llm_judge"] == 1296
    assert counts["human"] == 96
    assert counts["deepeval"] == 432
    assert counts["retrieval_code"] == 1044
    assert counts["li_code"] == 812


# --- dry run: writes nothing -------------------------------------------------------------------


def test_main_dry_run_by_default_writes_no_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "MANIFEST_PATH", tmp_path / "manifest.json")
    rc = m.main(["--only", "datasets,prompts,runs"])
    assert rc == 0
    assert not (tmp_path / "manifest.json").exists()


def test_dry_run_always_wins_over_live(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "MANIFEST_PATH", tmp_path / "manifest.json")
    rc = m.main(["--live", "--dry-run", "--only", "datasets"])
    assert rc == 0
    assert not (tmp_path / "manifest.json").exists()


# --- live + idempotency -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_manifest(tracking_uri, client, monkeypatch_module, tmp_path_factory):
    import os

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    monkeypatch_module.setattr(m, "VIEWS_PATH", tmp_path_factory.mktemp("views") / "mlflow_views.json")
    manifest: dict = {}
    m.step_datasets(client, True, manifest)
    m.step_prompts(client, True, manifest)
    m.step_runs(client, True, manifest)
    m.step_traces(client, True, manifest, limit=None)
    m.step_assessments(client, True, manifest, limit=None)
    m.step_scorers(client, True, manifest)
    m.step_registry(client, True, manifest)
    m.step_views(client, True, manifest)
    return manifest


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _counts(client, exp_id):
    n_runs = len(client.search_runs([exp_id], max_results=5000))
    n_traces = len(mlflow.search_traces(locations=[exp_id], max_results=5000, return_type="list"))
    return n_runs, n_traces


def test_live_creates_planned_objects(client, live_manifest):
    exp_id = m._ensure_experiment(client)
    n_runs, n_traces = _counts(client, exp_id)
    assert n_runs == 33
    assert n_traces == 738


def test_spine_and_twin_traces_are_real_span_trees(client, live_manifest):
    exp_id = m._ensure_experiment(client)
    for case_id, variant_id in ((SPINE_CASE, SPINE_VARIANT), (TWIN_CASE, TWIN_VARIANT)):
        key = f"{case_id}:{variant_id}:judged"
        trace = m._find_trace(client, exp_id, key)
        assert trace is not None
        assert len(trace.data.spans) > 1, f"{key} is a flat single-span trace"
        assert any(s.parent_id is not None for s in trace.data.spans), f"{key} has no nested span"


def test_live_second_run_is_idempotent(client, tracking_uri):
    import os

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    exp_id = m._ensure_experiment(client)
    before_runs, before_traces = _counts(client, exp_id)
    before_versions = {name: len(client.search_prompt_versions(name)) for name in
                       ("agent-base-system-prompt", "arm-a-prompt-base", "judge-panel-rubric")}

    manifest: dict = {}
    m.step_datasets(client, True, manifest)
    m.step_prompts(client, True, manifest)
    m.step_runs(client, True, manifest)
    m.step_traces(client, True, manifest, limit=None)

    after_runs, after_traces = _counts(client, exp_id)
    after_versions = {name: len(client.search_prompt_versions(name)) for name in before_versions}

    assert after_runs == before_runs
    assert after_traces == before_traces
    assert after_versions == before_versions
    assert manifest["runs"]["created_this_run"] == 0
    assert manifest["traces"]["written_this_run"] == 0


def test_spine_and_twin_assessment_equality(client, live_manifest):
    exp_id = m._ensure_experiment(client)
    ctx = m.mirror_context()
    agent = m.agent_trace_plan()

    for case_id, variant_id in ((SPINE_CASE, SPINE_VARIANT), (TWIN_CASE, TWIN_VARIANT)):
        entry = next((e for e in agent if e["case_id"] == case_id and e["variant_id"] == variant_id and e["category"] == "judged"), None)
        if entry is None:
            pytest.skip(f"{case_id}:{variant_id} is not in the judged plan")
        key = f"{case_id}:{variant_id}:judged"
        trace = m._find_trace(client, exp_id, key)
        assert trace is not None
        assess = {a.name: a.value for a in trace.info.assessments}
        mirror = m.mirror_for(case_id, variant_id, entry["row"], ctx, category="judged")

        if mirror.get("ga_scored") is not None:
            assert assess.get("obj/grounded_accuracy") == pytest.approx(float(bool(mirror["ga_scored"])))
        if "judge/mistral/evidence" in assess and mirror.get("judge_mistral_evidence") is not None:
            assert assess["judge/mistral/evidence"] == pytest.approx(mirror["judge_mistral_evidence"])
        if "human/professional" in assess and mirror.get("human_professional") is not None:
            assert assess["human/professional"] == pytest.approx(mirror["human_professional"])
        if "deepeval/task_completion" in assess:
            from dealpoint.eval.braintrust_sync import _deepeval_scores_by_trace

            dp = _deepeval_scores_by_trace().get((case_id, variant_id), {})
            tc = (dp.get("task_completion") or {}).get("score") if isinstance(dp.get("task_completion"), dict) else dp.get("task_completion")
            if tc is not None:
                assert assess["deepeval/task_completion"] == pytest.approx(tc)


def test_registry_has_champion_alias(client, live_manifest):
    mv = client.get_model_version_by_alias("dealpoint-agent", "champion")
    assert mv.tags.get("variant_id") == "D@gemini-3.1-flash-lite"


def test_views_step_wrote_run_comparisons(tmp_path, client, monkeypatch):
    out = tmp_path / "views.json"
    monkeypatch.setattr(m, "VIEWS_PATH", out)
    manifest: dict = {}
    m.step_views(client, True, manifest)
    assert out.exists()
    entries = json.loads(out.read_text())
    assert len(entries) > 0
    assert {"question", "tag_filter", "metric", "title"} <= set(entries[0])


# --- scorers and judges -------------------------------------------------------------------------


def test_six_deterministic_scorers_reproduce_stored_values():
    scorers = m.build_deterministic_scorers()
    assert set(scorers) == set(m.DETERMINISTIC_SCORER_NAMES)

    from dealpoint.eval.braintrust_cockpit import _judged_subset, _variant_results

    subset = _judged_subset()
    row = _variant_results(subset, SPINE_VARIANT).get(SPINE_CASE)
    assert row is not None
    outputs = {"finding": row.get("finding"), "record": (row.get("record") or {})}
    for name, scorer in scorers.items():
        expected = (row.get("scores") or {}).get(name)
        if expected is None:
            continue
        result = scorer(outputs=outputs, metadata={"case_id": SPINE_CASE})
        assert result == expected, name


def test_mlflow_tour_names_every_unique_feature_and_gap():
    from pathlib import Path

    text = Path('docs/mlflow-tour.md').read_text(encoding='utf-8')
    for stop in ('Stop 1', 'Stop 2', 'Stop 3', 'Stop 4', 'Stop 5', 'Stop 6', 'Stop 7', 'Stop 8', 'Stop 9'):
        assert stop in text
    assert 'What MLflow adds' in text
    assert 'What only Braintrust has here' in text
    for feature in ('in-process', 'judge alignment', 'prompt optimization', 'model registry', 'timestamps'):
        assert feature.lower() in text.lower()
    for gap in ('Topics', 'Patterns', 'Loop', 'labeling', 'chart', 'online scoring'):
        assert gap.lower() in text.lower()
    demo_text = Path('docs/demo-tour.md').read_text(encoding='utf-8')
    assert 'mlflow-tour.md' in demo_text


def test_judges_constructed_with_openrouter_base_url_and_never_called(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("a judge must never be invoked in the offline suite")

    judges = m.build_judges()
    assert set(judges) == set(m.JUDGE_DIMS)
    for dim, judge in judges.items():
        assert judge.model.startswith("openai:/")
        assert m.JUDGE_MODEL_FOR_DIM[dim] in judge.model
        monkeypatch.setattr(judge, "__call__", _boom, raising=False)
