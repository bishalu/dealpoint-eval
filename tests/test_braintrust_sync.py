"""M7a gate: `just braintrust-sync`'s mapping, driven fully offline with a
fake client (spec section 3, "offline tests with a fake client cover the
mapping").
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.gate_m7


class FakeDataset:
    def __init__(self, name):
        self.name = name
        self.rows = []

    def insert(self, input, expected=None, metadata=None):
        self.rows.append({"input": input, "expected": expected, "metadata": metadata})
        return f"row-{len(self.rows)}"

    def flush(self):
        pass


class FakeExperiment:
    def __init__(self, name):
        self.name = name
        self.logged = []

    def log(self, **kwargs):
        self.logged.append(kwargs)
        return f"log-{len(self.logged)}"

    def flush(self):
        pass


class FakeSpan:
    def __init__(self, name):
        self.name = name
        self.logged = []

    def log(self, **kwargs):
        self.logged.append(kwargs)

    def end(self):
        pass


class FakeBraintrustClient:
    """Records init_dataset / init_experiment / start_span / log calls."""

    def __init__(self):
        self.datasets: dict[str, FakeDataset] = {}
        self.experiments: dict[str, FakeExperiment] = {}
        self.spans: list[FakeSpan] = []
        self.calls: list[tuple] = []
        self.model_client_touched = False

    def init_dataset(self, project, name):
        self.calls.append(("init_dataset", name))
        if name not in self.datasets:
            self.datasets[name] = FakeDataset(name)
        return self.datasets[name]

    def init_experiment(self, project, name):
        self.calls.append(("init_experiment", name))
        if name not in self.experiments:
            self.experiments[name] = FakeExperiment(name)
        return self.experiments[name]

    def start_span(self, name=None):
        self.calls.append(("start_span", name))
        span = FakeSpan(name)
        self.spans.append(span)
        return span


def test_sync_creates_every_dataset_and_experiment_exactly_once():
    from dealpoint.eval.braintrust_sync import DATASET_PLAN_NAMES, experiment_plan, sync

    client = FakeBraintrustClient()
    result = sync(client, dry_run=True)

    assert set(client.datasets.keys()) == {f"maud-dealpoint-{n}" for n in DATASET_PLAN_NAMES}
    expected_experiment_names = {p["name"] for p in experiment_plan()}
    assert set(client.experiments.keys()) == expected_experiment_names
    assert result["n_experiments"] == len(expected_experiment_names)


def test_sync_is_idempotent_second_call_creates_nothing_new():
    from dealpoint.eval.braintrust_sync import sync

    client = FakeBraintrustClient()
    sync(client, dry_run=True)
    n_datasets_1 = len(client.datasets)
    n_experiments_1 = len(client.experiments)

    sync(client, dry_run=True)
    n_datasets_2 = len(client.datasets)
    n_experiments_2 = len(client.experiments)

    assert n_datasets_1 == n_datasets_2
    assert n_experiments_1 == n_experiments_2


def test_score_budget_passes_at_12_on_60_cases_raises_at_13():
    from dealpoint.eval.braintrust_sync import ScoreBudgetError, assert_score_budget

    ok_plan = {"name": "x", "stage": "eval", "n_cases": 60, "score_names": [f"s{i}" for i in range(12)]}
    assert_score_budget(ok_plan)  # must not raise

    bad_plan = {"name": "x", "stage": "eval", "n_cases": 60, "score_names": [f"s{i}" for i in range(13)]}
    with pytest.raises(ScoreBudgetError):
        assert_score_budget(bad_plan)


def test_score_budget_m4_m6_sweep_keeps_six():
    from dealpoint.eval.braintrust_sync import ScoreBudgetError, assert_score_budget

    plan_seven = {"name": "x", "stage": "m4_sweep", "n_cases": 32, "score_names": [f"s{i}" for i in range(7)]}
    with pytest.raises(ScoreBudgetError):
        assert_score_budget(plan_seven)

    plan_six = {"name": "x", "stage": "m6_sweep", "n_cases": 32, "score_names": [f"s{i}" for i in range(6)]}
    assert_score_budget(plan_six)  # must not raise


def test_every_logged_row_metadata_carries_all_ten_common_keys():
    from dealpoint.eval.braintrust_sync import COMMON_METADATA_KEYS, common_metadata

    row = {"case_id": "contract_0__q01", "arm": "D", "model": "z-ai/glm-5.3-flash", "index_version": "e2b4a2b97561"}
    metadata = common_metadata(row, stage="agent")
    assert set(COMMON_METADATA_KEYS) <= set(metadata.keys())
    for key in COMMON_METADATA_KEYS:
        assert key in metadata


def test_score_names_are_namespaced():
    from dealpoint.eval.braintrust_sync import SCORE_NAMESPACES, score_namespace

    assert score_namespace("grounded_accuracy") == "obj/grounded_accuracy"
    assert score_namespace("hit_rate") == "li/hit_rate"
    assert score_namespace("reasoning") == "judge/reasoning"
    assert score_namespace("task_completion") == "deepeval/task_completion"
    for name in ("grounded_accuracy", "hit_rate", "reasoning", "task_completion"):
        ns = score_namespace(name)
        assert re.match(r"^(" + "|".join(SCORE_NAMESPACES) + r")/", ns)


def test_log_hierarchy_produces_expected_nesting_with_no_model_call():
    from dealpoint.eval.braintrust_sync import log_hierarchy

    row = {
        "case_id": "doc_x__q01",
        "scores": {"grounded_accuracy": True},
        "finding": {"answer": "All Cash", "evidence": [], "rationale": "ok"},
        "record": {
            "trajectory": [
                {"tool": "search_agreement", "args": {"query": "x"}, "char_ranges": []},
                {"tool": "lookup_defined_term", "args": {"term": "Knowledge"}, "char_ranges": []},
            ]
        },
    }
    hierarchy = log_hierarchy(row, {}, None)
    assert hierarchy["name"] == "case"
    agent = hierarchy["children"][0]
    assert agent["name"] == "agent"
    names = [c["name"] for c in agent["children"]]
    assert "search_agreement" in names
    assert "lookup_defined_term" in names
    assert "final_answer" in names
    assert "scoring" in names
    scoring = next(c for c in agent["children"] if c["name"] == "scoring")
    assert scoring["provenance"]["obj"]["grounded_accuracy"] is True


def test_representative_cases_selects_one_per_category_deterministically():
    from dealpoint.eval.braintrust_sync import representative_cases

    first = representative_cases()
    second = representative_cases()
    assert first == second
    categories = [s["category"] for s in first["selections"]]
    assert categories == [
        "successful_direct",
        "retrieval_rescue",
        "defined_term_cross_ref",
        "inefficient_trajectory",
        "wrong_answer",
        "abstention_counterfactual",
    ]


def test_review_set_satisfies_rule_and_extends_unchanged_to_30():
    from dealpoint.eval.braintrust_sync import review_set

    rs12 = review_set(12)
    if not rs12:
        pytest.skip("no judged-subset packets available in this environment")
    assert len(rs12) == 12
    reasoning_types = {r["reasoning_type"] for r in rs12}
    assert len(reasoning_types) >= 2
    variants = {r["variant_id"] for r in rs12}
    assert len(variants) >= 2
    n_abstained = sum(1 for r in rs12 if r["status"] == "ABSTAINED")
    assert n_abstained >= 1

    rs30 = review_set(30)
    # same ordering rule: rs12 must be a prefix of rs30
    assert rs12 == rs30[:12]
