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


def test_dashboard_has_five_charts_in_spec_order():
    from dealpoint.eval.braintrust_cockpit import dashboard_definition

    dash = dashboard_definition()
    assert dash["name"] == "DealPoint eval overview"
    titles = [c["title"] for c in dash["charts"]]
    assert len(titles) == 5
    assert "grounded_accuracy" in titles[0]
    assert "judge/" in titles[1]
    assert "human" in titles[2] or "judge" in titles[2]
    assert "$/case" in titles[3]
    assert "DeepEval" in titles[4]
    assert dash["charts"][2].get("caption_if_empty") == "pending human calibration"


def test_sync_views_and_dashboard_idempotent_second_run_creates_nothing_new():
    from dealpoint.eval.braintrust_cockpit import sync_views_and_dashboard

    rest = FakeRestClient()
    r1 = sync_views_and_dashboard(rest)
    n1 = len(rest.views)
    ids1 = sorted(v["id"] for v in r1["views"]) + [r1["dashboard"]["id"]]

    r2 = sync_views_and_dashboard(rest)
    n2 = len(rest.views)
    ids2 = sorted(v["id"] for v in r2["views"]) + [r2["dashboard"]["id"]]

    assert n1 == n2 == 8  # 7 views + 1 dashboard
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
    manifest = build_demo_manifest(result)
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
        "git_sha7",
        "rubric_version",
        "subset_hash",
        "synced_at",
    ):
        assert key in manifest, f"manifest missing {key!r}"
    assert len(manifest["views"]) == 7
    assert manifest["subset_hash"] == "5918ef10a7e6"


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
