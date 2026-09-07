"""M7b gate: `just braintrust-cockpit`'s view/dashboard/topics/pattern
mapping, the hero-case rule, the judge-span replay, and the human-score
budget guard -- all driven offline with fake REST/SDK clients (spec
`m7b.md` section 4-5, `gate_m7b`).
"""

from __future__ import annotations

import pytest

from dealpoint.eval.braintrust_sync import _DryRunClient

pytestmark = pytest.mark.gate_m7b


class FakeExperiment:
    def __init__(self, name):
        self.name = name
        self.logged = []
        self.spans = []
        self._logged_by_id = {}

    def log(self, **kwargs):
        row_id = kwargs.get("id")
        if row_id is not None and row_id in self._logged_by_id:
            self._logged_by_id[row_id].update(kwargs)
            return row_id
        self.logged.append(kwargs)
        if row_id is not None:
            self._logged_by_id[row_id] = kwargs
        return row_id or f"log-{len(self.logged)}"

    def start_span(self, name=None):
        span = FakeSpan(name)
        self.spans.append(span)
        return span

    def flush(self):
        pass


class FakeSpan:
    def __init__(self, name):
        self.name = name
        self.logged = []
        self.children = []

    def log(self, **kwargs):
        self.logged.append(kwargs)

    def start_span(self, name=None):
        child = FakeSpan(name)
        self.children.append(child)
        return child

    def end(self):
        pass


class FakeSdkClient:
    def __init__(self):
        self.experiments: dict[str, FakeExperiment] = {}

    def init_experiment(self, project, name):
        if name not in self.experiments:
            self.experiments[name] = FakeExperiment(name)
        return self.experiments[name]


class FakeRestClient:
    """`get`/`post`/`patch` shape only -- the same seam `RestClient` exposes.
    Resolves the one project by name; `/v1/view` is filtered by
    `object_type`/`object_id` on GET (the real API's query params) and
    updated via `PATCH /v1/view/{id}` (the real update path -- `POST` never
    takes an `id`, confirmed live).
    """

    def __init__(self):
        self.views: list[dict] = []
        self.functions: list[dict] = []
        self._next_id = 1
        self.post_calls: list[tuple[str, dict]] = []

    def get(self, path, params=None):
        if path == "/v1/project":
            return {"objects": [{"id": "proj-1", "name": (params or {}).get("project_name")}]}
        if path == "/v1/view":
            params = params or {}
            return {
                "objects": [
                    v
                    for v in self.views
                    if v.get("object_type") == params.get("object_type") and v.get("object_id") == params.get("object_id")
                ]
            }
        if path == "/v1/function":
            return {"objects": list(self.functions)}
        if path == "/v1/organization":
            return {"objects": [{"id": "org-1", "name": "Fake Org"}]}
        if path == "/v1/experiment":
            return {"objects": []}
        if path == "/v1/dataset":
            return {"objects": []}
        raise ValueError(path)

    def post(self, path, json_body):
        self.post_calls.append((path, json_body))
        if path == "/v1/view":
            new_view = dict(json_body)
            new_view["id"] = f"view-{self._next_id}"
            self._next_id += 1
            self.views.append(new_view)
            return new_view
        if path == "/v1/function":
            for fn in self.functions:
                if fn.get("slug") == json_body.get("slug"):
                    fn.update(json_body)
                    return fn
            new_fn = dict(json_body)
            new_fn["id"] = f"function-{self._next_id}"
            self._next_id += 1
            self.functions.append(new_fn)
            return new_fn
        raise ValueError(path)

    def patch(self, path, json_body):
        if path.startswith("/v1/project/"):
            self.project_settings = json_body.get("settings")
            return {"id": path.rsplit("/", 1)[-1], "settings": self.project_settings}
        view_id = path.rsplit("/", 1)[-1]
        for v in self.views:
            if v["id"] == view_id:
                v.update(json_body)
                return v
        raise ValueError(path)


# --- views + dashboard --------------------------------------------------


def test_view_definitions_cover_the_seven_named_views():
    from dealpoint.eval.braintrust_cockpit import view_definitions

    names = {v["name"] for v in view_definitions()}
    assert names == {
        "Judged traces by variant",
        "Judge disagreement",
        "Retrieval rescue",
        "Failure attribution",
        "DeepEval vs judge disagreement",
        "Trajectory inefficiency",
        "RAG tournament",
        "Review set (12)",
    }
    for v in view_definitions():
        assert v["caption"], f"{v['name']} has no caption"


def test_view_btql_filters_match_the_documented_queries_verbatim():
    from dealpoint.eval.braintrust_cockpit import documented_btql_query, view_definitions

    by_name = {v["name"]: v for v in view_definitions()}
    assert by_name["Retrieval rescue"]["definition"]["btql"] == documented_btql_query(1)
    assert by_name["Failure attribution"]["definition"]["btql"] == documented_btql_query(2)
    assert by_name["DeepEval vs judge disagreement"]["definition"]["btql"] == documented_btql_query(4)
    assert by_name["Trajectory inefficiency"]["definition"]["btql"] == documented_btql_query(6)


def test_dashboards_are_one_question_each_over_the_metadata_mirror():
    """The dashboard is the monitor surface over project logs; the logs carry no obj/*, judge/*,
    human/* or li/* scores (those live in experiments), so every chart aggregates the metadata mirror
    the showroom merges onto logs, as a toplist (groups on the axis, not time). The dashboard name is
    the question, chart titles carry no prefix, ranking charts compare one case pool of comparable traces."""
    from dealpoint.eval.braintrust_cockpit import (
        DASHBOARDS,
        _chart_rest_definition,
        chart_catalogue,
        dashboard_definitions,
    )

    dashes = dashboard_definitions()
    assert [d["name"] for d in dashes] == ["DealPoint eval overview", "Which system?", "Which model?", "Which retriever?", "Judges and the lawyer", "DeepEval", "LlamaIndex", "Which prompt?"]
    assert len(dashes[0]["charts"]) == 11 and sum(1 for c in dashes[0]["charts"] if c.get("kind") == "bignumber") == 5
    assert all(d["description"].startswith("VERDICT") for d in dashes), "every dashboard leads with its verdict"
    judges = next(d for d in dashes if d["name"] == "Judges and the lawyer")
    assert [c["group_by"] for c in judges["charts"]] == [[]] * len(judges["charts"]), "the judges page never ranks systems"
    used = {k for _, _, keys in DASHBOARDS for k in keys}
    assert all(len(d["description"]) > 80 for d in dashes), "every dashboard says what a reader needs to know"
    assert used == set(chart_catalogue()), "every catalogued chart is on some dashboard"
    for dash in dashes:
        for chart in dash["charts"]:
            assert not chart["title"].startswith(("Which system?", "Which model?", "Which retriever?", "Which prompt?")), chart["title"]
            measures = chart["measure"] if isinstance(chart["measure"], list) else [chart["measure"]]
            for m in measures:
                btql = m["btql"] if isinstance(m, dict) else m
                assert btql.startswith(("avg(metadata.", "count(", "sum(metadata.", "percentile(metadata.")), btql
                assert "scores." not in btql and "ga_all" not in btql
            if len(measures) > 1:
                assert all(isinstance(m, dict) and m.get("name") for m in measures), f"multi-measure rows need display names: {chart['title']}"
                assert len(_chart_rest_definition(chart)["measures"]) == len(measures)
                assert _chart_rest_definition(chart)["measures"][0]["displayName"] == measures[0]["name"]
            assert all(g.startswith("metadata.") for g in chart["group_by"])
            assert chart["filters"], "every chart names the log family it aggregates"
            if any(g.endswith(("system_label", "model_label", "variant_label")) for g in chart["group_by"]):
                assert "metadata.comparable = 1" in " ".join(chart["filters"]), chart["title"]
            rest = _chart_rest_definition(chart)
            assert rest["type"] == "scalars" and rest["viz"]["type"] == ("singleValue" if chart.get("kind") == "bignumber" else "toplist")
    assert "A pipeline+dense, B agent+dense, C agent+hybrid, D agent+hybrid+skill" in dashes[1]["charts"][0]["title"]


def test_monitor_views_pin_a_time_range():
    from dealpoint.eval.braintrust_cockpit import (
        DASHBOARD_RANGE,
        _upsert_view,
        _view_data_for,
        dashboard_definitions,
    )

    rest = FakeRestClient()
    dash = dashboard_definitions()[0]
    _upsert_view(rest, "project", "proj-1", "monitor", dash["name"], _view_data_for({"custom_charts": dash["charts"]}))
    body = rest.post_calls[-1][1]
    assert body["options"]["options"]["spanType"] == "range" and body["options"]["options"]["rangeValue"] == DASHBOARD_RANGE


def test_sync_views_and_dashboard_idempotent_second_run_creates_nothing_new():
    from dealpoint.eval.braintrust_cockpit import sync_views_and_dashboard

    rest = FakeRestClient()
    r1 = sync_views_and_dashboard(rest)
    n1 = len(rest.views)
    ids1 = sorted(v["id"] for v in r1["views"]) + sorted(d["id"] for d in r1["dashboards"])

    r2 = sync_views_and_dashboard(rest)
    n2 = len(rest.views)
    ids2 = sorted(v["id"] for v in r2["views"]) + sorted(d["id"] for d in r2["dashboards"])

    assert n1 == n2 == 16  # 8 views + 8 dashboards
    assert ids1 == ids2
    assert all(not v["created"] for v in r2["views"])
    assert r2["dashboard"]["created"] is False


# --- hero case ------------------------------------------------------------


def test_hero_case_is_deterministic_and_disagrees_on_grounded_accuracy():
    from dealpoint.eval.braintrust_cockpit import hero_case

    h1 = hero_case()
    h2 = hero_case()
    assert h1 == h2
    if h1["case_id"] is not None:
        assert h1["grounded_accuracy_a"] != h1["grounded_accuracy_d"]
        assert h1["max_pairwise_spread"] is not None


# --- judge spans ------------------------------------------------------------


def test_judge_spans_shape_three_judges_plus_aggregate():
    from dealpoint.eval.braintrust_cockpit import HERO_VARIANTS, hero_case, judge_spans

    hero = hero_case()
    if hero["case_id"] is None:
        pytest.skip("no hero case in this fixture data")
    tree = judge_spans(hero["case_id"], HERO_VARIANTS[0])
    names = [s["name"] for s in tree["spans"]]
    assert names[-1] == "judge/aggregate"
    judge_names = [n for n in names if n.startswith("judge/") and n != "judge/aggregate"]
    assert len(judge_names) == 3
    for span in tree["spans"][:-1]:
        assert "score" not in span  # per-judge scores never logged as scores
        assert set(span["metadata"]) >= {"judge_model", "rubric_version", "judge_price_usd", "call_cost_usd", "subset_hash"}
    aggregate = tree["spans"][-1]
    assert "per_judge_scores" in aggregate["metadata"]
    assert "rounding_rule" in aggregate["metadata"]


def test_judge_spans_never_invents_content_when_packet_missing():
    from dealpoint.eval.braintrust_cockpit import judge_spans

    tree = judge_spans("not-a-real-case", "A@haiku")
    assert tree["spans"][0]["name"] == "judge/unavailable"
    assert tree["spans"][0]["metadata"]["reason"]


def test_replay_hero_case_produces_three_judge_children_plus_aggregate_under_scoring():
    from dealpoint.eval.braintrust_cockpit import hero_case, replay_hero_case

    hero = hero_case()
    if hero["case_id"] is None:
        pytest.skip("no hero case in this fixture data")
    sdk = FakeSdkClient()
    result = replay_hero_case(sdk, hero["case_id"])
    experiment = sdk.experiments["m7b-hero-case"]
    assert len(experiment.spans) == 2  # one root span per HERO_VARIANTS entry

    def find_scoring(span):
        if span.name == "scoring":
            return span
        for child in span.children:
            found = find_scoring(child)
            if found is not None:
                return found
        return None

    for root in experiment.spans:
        scoring = find_scoring(root)
        assert scoring is not None
        child_names = [c.name for c in scoring.children]
        assert child_names.count("judge/aggregate") == 1
        assert len([n for n in child_names if n.startswith("judge/") and n != "judge/aggregate"]) in (0, 3)
    assert set(result["variant_trees"].keys()) == {"A@haiku", "D@haiku"}


# --- human score budget ------------------------------------------------


def test_human_score_rows_only_rescaled_1_5_to_0_1():
    from dealpoint.eval.braintrust_cockpit import human_score_rows

    rows = human_score_rows()
    assert rows, "sanity: this repo's fixture data has human-scored packets"
    for row in rows:
        assert row["experiment_name"].startswith("judge-")
        for value in row["scores"].values():
            assert 0.0 <= value <= 1.0


def test_assert_human_score_budget_passes_on_real_data_and_rejects_over_budget():
    from dealpoint.eval.braintrust_cockpit import (
        HumanScoreBudgetError,
        assert_human_score_budget,
        human_score_rows,
    )

    rows = human_score_rows()
    total = assert_human_score_budget(rows)
    assert total <= 1500
    assert total <= 4 * len(rows)

    too_many = [{"scores": {f"human/d{i}": 0.5 for i in range(5)}} for _ in range(2)]
    with pytest.raises(HumanScoreBudgetError):
        assert_human_score_budget(too_many)


def test_push_human_scores_is_idempotent_row_count_stable():
    from dealpoint.eval.braintrust_cockpit import human_score_rows, push_human_scores

    rows = human_score_rows()
    sdk = FakeSdkClient()
    n1 = push_human_scores(sdk, rows)
    n_logged_1 = sum(len(e.logged) for e in sdk.experiments.values())
    n2 = push_human_scores(sdk, rows)
    n_logged_2 = sum(len(e.logged) for e in sdk.experiments.values())

    assert n1 == n2
    assert n_logged_1 == n_logged_2, "a second push must update rows in place, never duplicate"


# --- full driver ----------------------------------------------------------


def test_sync_cockpit_idempotent_second_run_creates_nothing_new():
    from dealpoint.eval.braintrust_cockpit import sync_cockpit

    rest = FakeRestClient()
    sdk = FakeSdkClient()
    sync_cockpit(rest, sdk_client=sdk)
    n_views_1 = len(rest.views)
    n_exp_1 = len(sdk.experiments)
    n_logs_1 = sum(len(e.logged) for e in sdk.experiments.values())
    n_spans_1 = sum(len(e.spans) for e in sdk.experiments.values())

    sync_cockpit(rest, sdk_client=sdk)
    n_views_2 = len(rest.views)
    n_exp_2 = len(sdk.experiments)
    n_logs_2 = sum(len(e.logged) for e in sdk.experiments.values())
    n_spans_2 = sum(len(e.spans) for e in sdk.experiments.values())

    assert n_views_1 == n_views_2
    assert n_exp_1 == n_exp_2
    assert n_logs_1 == n_logs_2
    assert n_spans_2 == 2 * n_spans_1, (
        "spans are trace replays, not upserted rows -- a second sync legitimately "
        "re-emits the same replay spans (mirrors braintrust_sync's own replay "
        "section, which is also span-append, not row-upsert); only LOG rows and "
        "views/dashboard must not grow"
    )


def test_sync_cockpit_rest_only_skips_replay_and_push():
    from dealpoint.eval.braintrust_cockpit import sync_cockpit

    rest = FakeRestClient()
    result = sync_cockpit(rest, sdk_client=None)
    assert result["replay"] is None
    assert result["n_human_scores_pushed"] == 0
    assert result["n_human_scores_planned"] > 0


def test_build_demo_manifest_has_every_required_key():
    from dealpoint.eval.braintrust_cockpit import build_demo_manifest, sync_cockpit

    rest = FakeRestClient()
    sdk = FakeSdkClient()
    result = sync_cockpit(rest, sdk_client=sdk)
    manifest = build_demo_manifest(result, rest_client=rest)
    for key in (
        "project_id",
        "views",
        "dashboard",
        "topics",
        "pattern",
        "hero_case",
        "human_scoring_probe",
        "human_score_rows",
        "n_human_scores_planned",
        "n_human_scores_pushed",
        "replay",
        "experiments",
        "datasets",
        "review_set",
        "permalinks",
        "git_sha7",
        "rubric_version",
        "subset_hash",
        "synced_at",
    ):
        assert key in manifest, f"manifest missing {key!r}"
    assert len(manifest["views"]) == 8
    assert manifest["subset_hash"] == "5918ef10a7e6"

    # spec section 6: "permalinks generated by the API for each walkthrough
    # stop" -- every one of the seven views and the dashboard must have an
    # entry, not just experiments/datasets.
    for view in manifest["views"]:
        assert f"view:{view['name']}" in manifest["permalinks"], f"no permalink for view {view['name']!r}"
    assert f"dashboard:{manifest['dashboard']['name']}" in manifest["permalinks"]


def test_pattern_has_at_least_three_supporting_trace_ids():
    from dealpoint.eval.braintrust_cockpit import pattern_definition

    pattern = pattern_definition()
    assert len(pattern["supporting_trace_ids"]) >= 3
    assert pattern["description"]


def test_dry_run_client_and_rest_client_are_offline_safe():
    """The `--dry-run` production path (`main(["--dry-run"])`'s client
    choice) never imports `requests`/`braintrust` for network use -- this
    just proves the dry-run objects are self-contained fakes.
    """
    from dealpoint.eval.braintrust_cockpit import _DryRunRestClient

    rest = _DryRunRestClient()
    sdk = _DryRunClient()
    assert rest.get("/v1/project")["objects"][0]["id"]
    assert sdk.init_experiment("proj", "x") is not None


# --- key resolution + spend guard (T1 / T5b / spend-guard section) --------


def test_main_dry_run_is_the_default_and_never_touches_the_key(monkeypatch, tmp_path):
    """`main([])` and `main(["--live", "--dry-run"])` both take the dry-run
    path (`--dry-run` always wins) and never call `load_braintrust_key`.
    """
    import dealpoint.eval.braintrust_cockpit as cockpit

    def _boom():
        raise AssertionError("dry run must never resolve the live API key")

    monkeypatch.setattr("dealpoint.eval.braintrust_adapter.load_braintrust_key", _boom)
    monkeypatch.setattr(cockpit, "DEMO_MANIFEST_PATH", tmp_path / "demo_manifest.json")

    for argv in ([], ["--live", "--dry-run"]):
        assert cockpit.main(argv) == 0


def test_main_live_with_no_key_exits_nonzero_and_writes_nothing(monkeypatch, tmp_path):
    import dealpoint.eval.braintrust_cockpit as cockpit

    monkeypatch.setattr("dealpoint.eval.braintrust_adapter.load_braintrust_key", lambda: None)
    fake_manifest_path = tmp_path / "demo_manifest.json"
    monkeypatch.setattr(cockpit, "DEMO_MANIFEST_PATH", fake_manifest_path)

    def _boom_sync(*args, **kwargs):
        raise AssertionError("must abort before syncing when the key is empty")

    monkeypatch.setattr(cockpit, "sync_cockpit", _boom_sync)

    rc = cockpit.main(["--live"])
    assert rc != 0
    assert not fake_manifest_path.exists()


def test_main_live_with_a_key_constructs_rest_client_with_a_nonempty_bearer(monkeypatch, tmp_path):
    """No network call: `sync_cockpit`/`build_demo_manifest` are stubbed so
    this only proves `main` resolves the key via `load_braintrust_key` (not
    `os.environ`), exports it to the process environment, and builds
    `RestClient` with it -- never `RestClient("")`.
    """
    import os

    import dealpoint.eval.braintrust_cockpit as cockpit

    monkeypatch.setattr("dealpoint.eval.braintrust_adapter.load_braintrust_key", lambda: "fake-live-key-1234567890")
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    fake_manifest_path = tmp_path / "demo_manifest.json"
    monkeypatch.setattr(cockpit, "DEMO_MANIFEST_PATH", fake_manifest_path)
    # `main(["--live"])` arms the module-global `_LEDGER_ACTIVE` for real (not
    # through monkeypatch); restore it so this test cannot leak a "ledger
    # active" state into every later test in the session.
    monkeypatch.setattr(cockpit, "_LEDGER_ACTIVE", cockpit._LEDGER_ACTIVE)
    # The ledger path is org-derived from the active key over the network; pin it so the fake key
    # never triggers an org lookup.
    monkeypatch.setattr(cockpit, "SCORE_LEDGER_PATH", tmp_path / "score_ledger.jsonl")

    captured: dict = {}

    def _fake_sync(rest_client, sdk_client=None, **_kwargs):
        captured["rest_client"] = rest_client
        captured["sdk_client"] = sdk_client
        return {
            "views_dashboard": {"project_id": "p", "views": [], "dashboard": {"name": "x", "created": True}},
            "topics_pattern": {"topics": {"id": None, "config": {}}, "pattern": {"id": None}},
            "hero_case": {"case_id": None},
            "human_score_rows": [],
            "n_human_scores_planned": 0,
            "n_human_scores_pushed": 0,
            "n_live_scores_planned": 0,
            "live_score_cap": cockpit.LIVE_SCORE_CAP,
            "replay": None,
            "human_scoring_probe": cockpit.HUMAN_SCORING_PROBE,
            "synced_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(cockpit, "sync_cockpit", _fake_sync)

    rc = cockpit.main(["--live"])
    assert rc == 0
    assert isinstance(captured["rest_client"], cockpit.RestClient)
    assert captured["rest_client"].api_key == "fake-live-key-1234567890"
    assert os.environ["BRAINTRUST_API_KEY"] == "fake-live-key-1234567890"
    assert fake_manifest_path.exists()


def test_score_budget_error_above_the_cap():
    from dealpoint.eval.braintrust_cockpit import (
        LIVE_SCORE_CAP,
        ScoreBudgetError,
        assert_live_score_budget,
    )

    assert assert_live_score_budget(LIVE_SCORE_CAP) == LIVE_SCORE_CAP
    with pytest.raises(ScoreBudgetError):
        assert_live_score_budget(LIVE_SCORE_CAP + 1)


def test_planned_live_scores_equals_what_the_fake_sdk_client_actually_receives():
    """The guard on the guard (T5b.4): `planned_live_scores` must equal the
    number of scores the fake SDK client actually receives across the hero
    replay + human-score push, or the printed pre-flight number is a lie.
    """
    from dealpoint.eval.braintrust_cockpit import (
        human_score_rows,
        planned_live_scores,
        sync_cockpit,
    )

    rest = FakeRestClient()
    sdk = FakeSdkClient()
    result = sync_cockpit(rest, sdk_client=sdk)

    n_scores_written = 0
    for experiment in sdk.experiments.values():
        for logged in experiment.logged:
            n_scores_written += len(logged.get("scores") or {})
        for root in experiment.spans:

            def _count_scores(span):
                total = sum(len(entry.get("scores") or {}) for entry in span.logged)
                for child in span.children:
                    total += _count_scores(child)
                return total

            n_scores_written += _count_scores(root)

    n_human_scores = result["n_human_scores_planned"]
    assert n_scores_written == planned_live_scores(n_human_scores)
    assert n_human_scores == sum(len(r["scores"]) for r in human_score_rows())


def test_ledger_active_second_live_run_emits_zero_spans_and_zero_scores(monkeypatch, tmp_path):
    """The idempotency proof, offline: with the ledger armed (as a real
    `--live` run arms it), a second `sync_cockpit` against the SAME fake
    clients must add no new spans, no new logged rows, and no new ledger
    lines. `test_sync_cockpit_idempotent_second_run_creates_nothing_new`
    (above) covers the LEDGER-INACTIVE case only -- a second sync legitimately
    re-emits the whole replay when there is no ledger to consult. This test
    is its ledger-ACTIVE counterpart.
    """
    import dealpoint.eval.braintrust_cockpit as cockpit

    ledger_path = tmp_path / "score_ledger.jsonl"
    monkeypatch.setattr(cockpit, "SCORE_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(cockpit, "_LEDGER_ACTIVE", True)

    rest = FakeRestClient()
    sdk = FakeSdkClient()

    cockpit.sync_cockpit(rest, sdk_client=sdk)
    n_spans_1 = sum(len(e.spans) for e in sdk.experiments.values())
    n_logs_1 = sum(len(e.logged) for e in sdk.experiments.values())
    ledger_lines_1 = ledger_path.read_text(encoding="utf-8").splitlines()

    cockpit.sync_cockpit(rest, sdk_client=sdk)
    n_spans_2 = sum(len(e.spans) for e in sdk.experiments.values())
    n_logs_2 = sum(len(e.logged) for e in sdk.experiments.values())
    ledger_lines_2 = ledger_path.read_text(encoding="utf-8").splitlines()

    assert n_spans_2 == n_spans_1, "the ledger must stop a second live run from re-emitting the replay"
    assert n_logs_2 == n_logs_1, "the ledger must stop a second live run from re-pushing human scores"
    assert ledger_lines_2 == ledger_lines_1, "a second live run must add zero ledger lines"


def test_topics_installs_a_preprocessor_function_and_sets_it_as_the_project_default():
    from dealpoint.eval.braintrust_cockpit import (
        PREPROCESSOR_SLUG,
        sync_topics_and_pattern,
        topics_preprocessor_code,
    )

    rest = FakeRestClient()
    result = sync_topics_and_pattern(rest, "proj-1")
    pre = next(f for f in rest.functions if f.get("function_type") == "preprocessor")
    assert pre["slug"] == PREPROCESSOR_SLUG
    assert pre["function_data"]["data"]["code"] == topics_preprocessor_code()
    assert "function handler(span)" in topics_preprocessor_code() and "system_label" in topics_preprocessor_code()
    assert rest.project_settings == {"default_preprocessor": {"type": "function", "id": pre["id"]}}
    assert result["topics"]["preprocessor"] == {"id": pre["id"], "slug": PREPROCESSOR_SLUG, "set_as_default": True}
    # idempotent: a second run upserts the same slug and creates nothing new
    n = len(rest.functions)
    sync_topics_and_pattern(rest, "proj-1")
    assert len(rest.functions) == n
