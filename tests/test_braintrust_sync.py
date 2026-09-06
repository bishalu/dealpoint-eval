"""M7a gate: `just braintrust-sync`'s mapping, driven fully offline with a
fake client (spec section 3, "offline tests with a fake client cover the
mapping").
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.gate_m7


class FakeDataset:
    """Mirrors the real SDK's `id=` + implicit-update semantics: inserting
    with an `id` that was already seen UPDATES that row in place rather
    than appending a new one -- exactly what `_stable_dataset_row_id`
    (`braintrust_sync.py`) relies on for a second sync to be idempotent.
    Rows without an `id` always append (no identity to dedupe on), matching
    the real SDK's behaviour of minting a fresh row id when none is given.
    """

    def __init__(self, name):
        self.name = name
        self.rows = []
        self._rows_by_id: dict[str, dict] = {}

    def insert(self, input, expected=None, metadata=None, id=None):
        row = {"input": input, "expected": expected, "metadata": metadata, "id": id}
        if id is not None and id in self._rows_by_id:
            existing = self._rows_by_id[id]
            existing.update(row)
            return id
        self.rows.append(row)
        if id is not None:
            self._rows_by_id[id] = row
        return id or f"row-{len(self.rows)}"

    def flush(self):
        pass


class FakeExperiment:
    """Mirrors the real SDK's `id=` + `update=True` upsert semantics for
    `.log(...)`: a second call with the SAME `id` updates that logged row
    in place instead of appending a duplicate -- this is what makes the
    idempotency test below able to actually fail if `sync()` regresses.
    """

    def __init__(self, name):
        self.name = name
        self.logged = []
        self.spans: list[FakeSpanWithChildren] = []
        self._logged_by_id: dict[str, dict] = {}

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
        span = FakeSpanWithChildren(name)
        self.spans.append(span)
        return span

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


class FakeSpanWithChildren(FakeSpan):
    def __init__(self, name):
        super().__init__(name)
        self.children: list[FakeSpanWithChildren] = []

    def start_span(self, name=None):
        child = FakeSpanWithChildren(name)
        self.children.append(child)
        return child


class FakeBraintrustClient:
    """Records init_dataset / init_experiment / start_span / log calls, plus
    the simpler fake-client shape for scorers/prompts/parameters (a real
    `braintrust` module instead exposes `.projects.create(...).scorers...`).
    """

    def __init__(self):
        self.datasets: dict[str, FakeDataset] = {}
        self.experiments: dict[str, FakeExperiment] = {}
        self.spans: list[FakeSpanWithChildren] = []
        self.calls: list[tuple] = []
        self.model_client_touched = False
        self.scorers_registered: list[str] = []
        self.prompts_registered: list[str] = []
        self.parameters_registered: list[str] = []

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
        span = FakeSpanWithChildren(name)
        self.spans.append(span)
        return span

    def register_scorer(self, name, import_path, threshold):
        self.calls.append(("register_scorer", name))
        if name not in self.scorers_registered:
            self.scorers_registered.append(name)

    def register_prompt(self, name, source_hash):
        self.calls.append(("register_prompt", name))
        if name not in self.prompts_registered:
            self.prompts_registered.append(name)

    def register_parameters(self, name, schema):
        self.calls.append(("register_parameters", name))
        if name not in self.parameters_registered:
            self.parameters_registered.append(name)


def test_sync_creates_every_dataset_and_experiment_exactly_once():
    from dealpoint.eval.braintrust_sync import DATASET_PLAN_NAMES, experiment_plan, sync

    client = FakeBraintrustClient()
    result = sync(client, dry_run=True)

    expected_dataset_names = {f"maud-dealpoint-{n}" for n in DATASET_PLAN_NAMES} | {
        "maud-dealpoint-review-set"
    }
    assert set(client.datasets.keys()) == expected_dataset_names
    expected_experiment_names = {p["name"] for p in experiment_plan()} | {"m7-representative-traces"}
    assert set(client.experiments.keys()) == expected_experiment_names
    assert result["n_experiments"] == len(expected_experiment_names) - 1


def test_sync_is_idempotent_second_call_creates_nothing_new():
    """A second `sync()` must not grow TOTAL row/log counts anywhere.

    Container counts alone (`len(client.datasets)`/`len(client.experiments)`)
    are insufficient: the fake client already dedupes CONTAINERS by name, so
    that assertion passes even if every container's ROWS silently double on
    a second sync. `FakeDataset.insert`/`FakeExperiment.log` now upsert on
    `id=` (mirroring the real SDK's `id=` + `update=True` semantics
    `braintrust_sync.py` passes), so this test actually exercises whether
    `sync()`'s stable-id repair works -- it must fail if row/log counts grow.
    """
    from dealpoint.eval.braintrust_sync import sync

    client = FakeBraintrustClient()
    sync(client, dry_run=True)
    n_datasets_1 = len(client.datasets)
    n_experiments_1 = len(client.experiments)
    n_dataset_rows_1 = sum(len(d.rows) for d in client.datasets.values())
    n_experiment_logs_1 = sum(len(e.logged) for e in client.experiments.values())
    assert n_dataset_rows_1 > 0, "sanity: the first sync must actually insert rows"
    assert n_experiment_logs_1 > 0, "sanity: the first sync must actually log rows"

    sync(client, dry_run=True)
    n_datasets_2 = len(client.datasets)
    n_experiments_2 = len(client.experiments)
    n_dataset_rows_2 = sum(len(d.rows) for d in client.datasets.values())
    n_experiment_logs_2 = sum(len(e.logged) for e in client.experiments.values())

    assert n_datasets_1 == n_datasets_2
    assert n_experiments_1 == n_experiments_2
    assert n_dataset_rows_2 == n_dataset_rows_1, (
        f"dataset rows grew on a second sync: {n_dataset_rows_1} -> {n_dataset_rows_2}"
    )
    assert n_experiment_logs_2 == n_experiment_logs_1, (
        f"experiment logs grew on a second sync: {n_experiment_logs_1} -> {n_experiment_logs_2}"
    )


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


def test_sync_registers_scorers_prompts_parameters_review_set_exactly_once():
    from dealpoint.eval.braintrust_sync import prompt_plan, scorer_plan, sync

    client = FakeBraintrustClient()
    result = sync(client, dry_run=True)

    expected_scorer_names = {s["slug"] for s in scorer_plan()}
    assert set(client.scorers_registered) == expected_scorer_names
    assert result["scorers"]["n"] == len(expected_scorer_names)

    expected_prompt_names = {p["slug"] for p in prompt_plan()}
    assert set(client.prompts_registered) == expected_prompt_names
    assert result["prompts"]["n"] == len(expected_prompt_names)

    assert client.parameters_registered == ["dealpoint-runtime-parameters"]
    assert result["parameters"]["registered"] == ["dealpoint-runtime-parameters"]

    assert "maud-dealpoint-review-set" in client.datasets
    assert result["review_set"]["dataset"] == "maud-dealpoint-review-set"

    n_scorers_1 = len(client.scorers_registered)
    n_prompts_1 = len(client.prompts_registered)
    n_params_1 = len(client.parameters_registered)
    sync(client, dry_run=True)
    assert len(client.scorers_registered) == n_scorers_1
    assert len(client.prompts_registered) == n_prompts_1
    assert len(client.parameters_registered) == n_params_1


def test_sync_replays_full_span_tree_from_a_fixture_row(monkeypatch, tmp_path):
    """The replayed representative-case span tree must reflect the ACTUAL
    stored result row for that selection (not an empty-trajectory stub).
    """
    import json

    import dealpoint.eval.braintrust_sync as bs

    fixture_row = {
        "case_id": "contract_0__q01",
        "arm": "A",
        "model": "z-ai/glm-5.3-flash",
        "scores": {"grounded_accuracy": True},
        "finding": {"answer": "All Cash", "evidence": [], "rationale": "ok"},
        "record": {
            "trajectory": [
                {"tool": "search_agreement", "args": {"query": "x"}, "char_ranges": []},
            ]
        },
    }
    fixture_path = tmp_path / "fixture.jsonl"
    fixture_path.write_text(json.dumps(fixture_row) + "\n", encoding="utf-8")

    fake_selections = {
        "rule": "fixture rule",
        "selections": [
            {
                "category": "successful_direct",
                "case_id": "contract_0__q01",
                "variant_id": "A@z-ai/glm-5.3-flash",
                "experiment_name": None,
                "results_path": str(fixture_path),
            }
        ],
    }
    monkeypatch.setattr(bs, "representative_cases", lambda: fake_selections)
    monkeypatch.setattr(bs, "experiment_plan", list)
    monkeypatch.setattr(bs, "dataset_rows", lambda name: [])
    monkeypatch.setattr(bs, "review_set", lambda n=12: [])

    client = FakeBraintrustClient()
    result = bs.sync(client, dry_run=True)

    assert result["replayed_traces"] == 1
    replay_experiment = client.experiments["m7-representative-traces"]
    assert len(replay_experiment.spans) == 1
    case_span = replay_experiment.spans[0]
    assert case_span.name == "case"
    agent_span = case_span.children[0]
    assert agent_span.name == "agent"
    names = [c.name for c in agent_span.children]
    assert "search_agreement" in names
    assert "final_answer" in names
    assert "scoring" in names
    search_span = next(c for c in agent_span.children if c.name == "search_agreement")
    assert len(search_span.children) == 1


def test_actual_logged_scores_never_exceed_the_declared_plan():
    """For every experiment `sync()` logs, the union of score names actually
    logged across its rows must be a subset of `plan['score_names']` --
    closes the gap where `_log_scores_for_row` could log more names than
    the plan declared (D7a).
    """
    from dealpoint.eval.braintrust_sync import experiment_plan, sync

    client = FakeBraintrustClient()
    sync(client, dry_run=True)

    for exp_plan in experiment_plan():
        experiment = client.experiments[exp_plan["name"]]
        declared = set(exp_plan["score_names"])
        actual: set[str] = set()
        for logged in experiment.logged:
            actual.update((logged.get("scores") or {}).keys())
        assert actual <= declared, f"{exp_plan['name']} logged undeclared scores: {actual - declared}"


def test_rag_experiments_log_real_per_case_scores_not_placeholder_rows():
    from dealpoint.eval.braintrust_sync import experiment_plan, sync

    client = FakeBraintrustClient()
    sync(client, dry_run=True)

    rag_plans = [p for p in experiment_plan() if p["stage"] == "rag"]
    if not rag_plans:
        pytest.skip("no li_rag_eval.json on disk in this environment")
    for exp_plan in rag_plans:
        experiment = client.experiments[exp_plan["name"]]
        n_with_scores = sum(1 for logged in experiment.logged if logged.get("scores"))
        assert n_with_scores > 0, f"{exp_plan['name']} logged zero rows with any score"


def test_review_set_never_leaks_variant_id_to_the_pushed_dataset():
    from dealpoint.eval.braintrust_sync import sync

    client = FakeBraintrustClient()
    result = sync(client, dry_run=True)
    if result["review_set"]["n"] == 0:
        pytest.skip("no judged-subset packets available in this environment")

    review_dataset = client.datasets["maud-dealpoint-review-set"]
    for row in review_dataset.rows:
        assert "variant_id" not in (row["metadata"] or {})
        assert row["expected"], "expected blinded packet_text to be pushed as the row body"


def test_parameters_schema_includes_skill_version_and_model_alias():
    from dealpoint.eval.braintrust_sync import parameters_schema

    schema = parameters_schema()
    assert "skill_version" in schema
    assert "model_alias" in schema
    assert "arms" not in schema, "schema should expose compact arm names/retrievers, not the full nested ARMS config"


def test_dataset_rows_carry_source_hashes_for_every_set():
    from dealpoint.eval.braintrust_sync import DATASET_PLAN_NAMES, dataset_rows

    for name in DATASET_PLAN_NAMES:
        rows = dataset_rows(name)
        if not rows:
            continue
        for row in rows[:3]:
            assert "source_hashes" in row["metadata"]
            assert row["metadata"]["source_hashes"]


def test_make_scorer_handler_end_to_end_bridges_output_to_canonical_scorer():
    """`_make_scorer_handler` must actually import and call the named
    canonical `dealpoint.eval.scorers` function, reconstructing
    `(case, finding, record, canonical_text)` from `(output, metadata)` --
    previously this bridge had zero test coverage (D7h).
    """
    from dealpoint.eval.braintrust_sync import _make_scorer_handler

    handler = _make_scorer_handler("grounded_accuracy")
    output = {
        "case_id": "contract_0__q01",
        "finding": {
            "answer": "All Cash",
            "evidence": [
                {
                    "section_ref": "1.1",
                    "quote": '"Per Share Cash Amount" means $115.00 in cash per share of Company Common Stock.',
                }
            ],
            "rationale": "ok",
        },
        "record": {"status": "ANSWERED", "trajectory": []},
    }
    result = handler(input=None, output=output, expected=None, metadata={"case_id": "contract_0__q01"})
    assert isinstance(result, bool)


def test_make_scorer_handler_returns_none_without_case_id():
    from dealpoint.eval.braintrust_sync import _make_scorer_handler

    handler = _make_scorer_handler("grounded_accuracy")
    assert handler(input=None, output={}, expected=None, metadata={}) is None


def test_sync_scorers_documents_publish_limitation_not_hidden():
    from dealpoint.eval.braintrust_sync import sync

    client = FakeBraintrustClient()
    result = sync(client, dry_run=True)
    assert result["scorers"]["published"] is False
    assert "braintrust push" in result["scorers"]["limitation"]


def test_sync_prompts_and_parameters_are_published():
    from dealpoint.eval.braintrust_sync import sync

    client = FakeBraintrustClient()
    result = sync(client, dry_run=True)
    assert result["prompts"]["published"] is True
    assert result["parameters"]["published"] is True


def test_judge_dimension_scores_are_rescaled_into_0_1_before_logging():
    """Braintrust rejects scores outside [0, 1]; judge dimensions are a 1-5
    Likert scale and must be rescaled, not logged raw (hit live during
    this repair with 'score values must be between 0 and 1').
    """
    from dealpoint.eval.braintrust_sync import _log_scores_for_row

    row = {"scores": {"reasoning": 5, "evidence": 1, "grounded_accuracy": True}}
    scores_to_log, _secondary = _log_scores_for_row(
        row, allowed_score_names={"judge/reasoning", "judge/evidence", "obj/grounded_accuracy"}
    )
    assert scores_to_log["judge/reasoning"] == pytest.approx(1.0)
    assert scores_to_log["judge/evidence"] == pytest.approx(0.0)
    assert scores_to_log["obj/grounded_accuracy"] == pytest.approx(1.0)


def test_assert_actual_score_budget_catches_overlogging():
    from dealpoint.eval.braintrust_sync import ScoreBudgetError, assert_actual_score_budget

    plan = {"name": "x", "stage": "agent", "n_cases": 32, "score_names": ["obj/a"]}
    assert_actual_score_budget(plan, {"obj/a"})  # within declared -- ok
    with pytest.raises(ScoreBudgetError):
        assert_actual_score_budget(plan, {"obj/a", "obj/b", "obj/c", "obj/d", "obj/e", "obj/f", "obj/g", "obj/h", "obj/i", "obj/j", "obj/k", "obj/l", "obj/m"})


def test_representative_cases_json_is_written_by_main(tmp_path, monkeypatch):
    import dealpoint.config as config_mod
    import dealpoint.eval.braintrust_sync as bs

    fake_path = tmp_path / "representative_cases.json"
    monkeypatch.setattr(config_mod, "REPRESENTATIVE_CASES_PATH", fake_path)

    rep_cases = {"rule": "x", "selections": []}
    bs._record_representative_cases(rep_cases)
    assert fake_path.exists()
    import json as json_mod

    assert json_mod.loads(fake_path.read_text(encoding="utf-8")) == rep_cases


def test_m7b_human_score_budget_extends_m7a_budget_without_touching_it():
    """M7b's only score addition is `human/<dimension>` (spec "Budget and
    disk": "M7b adds at most human/<dimension> (4) per human-scored trace on
    the review set (12 to 54 traces) and nothing else"). Asserted here, next
    to M7a's own budget tests, without modifying `braintrust_sync.py` itself
    -- M7b is additive, M7a's checkpoint is untouched.
    """
    from dealpoint.eval.braintrust_cockpit import (
        JUDGE_DIMENSIONS,
        assert_human_score_budget,
        human_score_rows,
    )

    rows = human_score_rows()
    total = assert_human_score_budget(rows)
    assert 0 < len(rows) <= 54
    assert total <= 4 * len(rows)
    for row in rows:
        assert len(row["scores"]) <= len(JUDGE_DIMENSIONS)
        assert all(name.startswith("human/") for name in row["scores"])


def test_m7b_openrouter_ledger_unchanged():
    """M7b makes no model calls: no spend-ledger row carries `milestone_tag:
    m7b` (spec "Budget and disk": "OpenRouter ledger unchanged by this
    milestone").
    """
    import json as json_mod
    from pathlib import Path

    from dealpoint.config import RESULTS_DIR

    ledger_path = Path(RESULTS_DIR) / "spend_ledger.jsonl"
    if not ledger_path.exists():
        return
    with open(ledger_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json_mod.loads(line)
            assert row.get("milestone_tag") != "m7b"
