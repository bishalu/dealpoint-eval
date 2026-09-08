"""M9d gate: Braintrust baseline, aggregate scores, the regressions dataset, first-class log tags
and the live-app design note -- all driven offline with a fake REST client (spec `m9d.md` section 3,
`gate_m9d`). No test constructs a real `Api`/`RestClient`, and none touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import dealpoint.eval.braintrust_showroom as showroom

pytestmark = pytest.mark.gate_m9d


# --- fake REST client (Api's relative-path convention) ----------------------------


class FakeApi(showroom.Api):
    """`get`/`post`/`patch`/`experiments`/`fetch_rows` over `Api`'s relative paths
    (no `/v1` prefix), plus enough state to drive the four new steps end to end. Subclasses the
    real `Api` (never calling `requests`) so it type-checks wherever a step expects one."""

    def __init__(self):
        super().__init__("fake-key")
        self.project_settings: dict = {"default_preprocessor": {"type": "function", "id": "pp-1"}}
        self.project_scores: dict[str, dict] = {}
        self._score_seq = 0
        self.experiments_by_name: dict[str, dict] = {}
        self._exp_seq = 0
        self.log_events: dict[str, dict] = {}
        self.posts: list[tuple[str, dict]] = []
        self.patches: list[tuple[str, dict]] = []

    # -- seeding helpers, not part of Api's real shape --
    def add_experiment(self, name: str, rows: list[dict]) -> str:
        self._exp_seq += 1
        eid = f"exp-{self._exp_seq}"
        self.experiments_by_name[name] = {"id": eid, "rows": rows}
        return eid

    def add_score(self, name: str, score_type: str, **rest) -> str:
        self._score_seq += 1
        sid = f"score-{self._score_seq}"
        self.project_scores[sid] = {"id": sid, "name": name, "score_type": score_type, **rest}
        return sid

    def add_log(self, log_id: str, metadata: dict, tags: list[str] | None = None) -> None:
        self.log_events[log_id] = {"id": log_id, "is_root": True, "metadata": metadata, "tags": tags or []}

    # -- Api's real shape --
    def get(self, path: str, params: dict | None = None) -> dict:
        params = params or {}
        if path == "project":
            return {"objects": [{"id": "proj-1", "name": params.get("project_name"), "settings": dict(self.project_settings)}]}
        if path == "project_score":
            return {"objects": list(self.project_scores.values())}
        if path == "experiment":
            name = params.get("experiment_name")
            if name:
                e = self.experiments_by_name.get(name)
                return {"objects": [{"id": e["id"], "name": name}] if e else []}
            return {"objects": [{"id": e["id"], "name": n} for n, e in self.experiments_by_name.items()]}
        raise ValueError(path)

    def post(self, path: str, body: dict) -> dict:
        self.posts.append((path, body))
        if path == "project_score":
            sid = self.add_score(body["name"], body["score_type"])
            self.project_scores[sid] = {**body, "id": sid}
            return self.project_scores[sid]
        if path.startswith("project_logs/") and path.endswith("/fetch"):
            return {"events": list(self.log_events.values()) if not body.get("cursor") else [], "cursor": None}
        if path.startswith("project_logs/") and path.endswith("/insert"):
            for ev in body.get("events", []):
                cur = self.log_events.setdefault(ev["id"], {"id": ev["id"], "is_root": True, "metadata": {}, "tags": []})
                if "tags" in ev:
                    cur["tags"] = ev["tags"]
                if "metadata" in ev:
                    cur["metadata"] = {**cur.get("metadata", {}), **ev["metadata"]}
            return {"row_ids": [ev["id"] for ev in body.get("events", [])]}
        if path.startswith("experiment/") and path.endswith("/fetch"):
            eid = path.split("/")[1]
            exp = next((e for e in self.experiments_by_name.values() if e["id"] == eid), None)
            rows = exp["rows"] if exp else []
            return {"events": rows if not body.get("cursor") else [], "cursor": None}
        if path.startswith("experiment/") and path.endswith("/insert"):
            eid = path.split("/")[1]
            exp = next((e for e in self.experiments_by_name.values() if e["id"] == eid), None)
            if exp:
                by_id = {r["id"]: r for r in exp["rows"]}
                for ev in body.get("events", []):
                    row = by_id.get(ev["id"])
                    if row is not None and "scores" in ev:
                        row.setdefault("scores", {}).update(ev["scores"])
            return {"row_ids": [ev["id"] for ev in body.get("events", [])]}
        raise ValueError(path)

    def patch(self, path: str, body: dict) -> dict:
        self.patches.append((path, body))
        if path.startswith("project/"):
            self.project_settings.update(body.get("settings", {}))
            return {"id": path.split("/")[1], "settings": dict(self.project_settings)}
        if path.startswith("project_score/"):
            # Confirmed live 2026-09-08: PATCH /v1/project_score/{id} rejects `project_id`
            # ("Extraneous key") and validates `categories` by a discriminated union keyed on
            # `score_type` -- minimum/maximum want a plain array of names, weighted wants
            # {name: weight}. Enforced here so an offline test would catch the real bug.
            assert "project_id" not in body, "PATCH /v1/project_score rejects project_id"
            categories = body.get("categories")
            if body.get("score_type") in ("minimum", "maximum"):
                assert isinstance(categories, list), "minimum/maximum categories must be an array of names"
            elif body.get("score_type") == "weighted":
                assert isinstance(categories, dict), "weighted categories must be a {name: weight} object"
            sid = path.split("/")[1]
            self.project_scores[sid].update(body)
            return self.project_scores[sid]
        raise ValueError(path)

    def experiments(self) -> list[dict]:
        return [{"id": e["id"], "name": n} for n, e in self.experiments_by_name.items()]

    def fetch_rows(self, experiment_id: str) -> list[dict]:
        exp = next((e for e in self.experiments_by_name.values() if e["id"] == experiment_id), None)
        return list(exp["rows"]) if exp else []


# --- fake cockpit RestClient (absolute /v1/... paths), for the two saved views ----


class FakeRestClient:
    def __init__(self):
        self.views: list[dict] = []
        self._next_id = 1

    def get(self, path, params=None):
        if path == "/v1/view":
            params = params or {}
            return {"objects": [v for v in self.views
                                if v.get("object_type") == params.get("object_type") and v.get("object_id") == params.get("object_id")]}
        raise ValueError(path)

    def post(self, path, json_body):
        if path == "/v1/view":
            new_view = dict(json_body)
            new_view["id"] = f"view-{self._next_id}"
            self._next_id += 1
            self.views.append(new_view)
            return new_view
        raise ValueError(path)

    def patch(self, path, json_body):
        view_id = path.rsplit("/", 1)[-1]
        for v in self.views:
            if v["id"] == view_id:
                v.update(json_body)
                return v
        raise ValueError(path)


class FakeDataset:
    def __init__(self, name: str):
        self.name = name
        self.rows: dict[str, dict] = {}

    def insert(self, *, id, input, expected, metadata):
        self.rows[id] = {"input": input, "expected": expected, "metadata": metadata}

    def flush(self) -> None:
        pass


class FakeBraintrustModule:
    def __init__(self):
        self.datasets: dict[str, FakeDataset] = {}

    def init_dataset(self, project, name, description=None):
        if name not in self.datasets:
            self.datasets[name] = FakeDataset(name)
        return self.datasets[name]


@pytest.fixture(autouse=True)
def _ledger_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(showroom, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(showroom, "PROJECT_ID", "proj-1")
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)


# --- D18: baseline + comparison key -----------------------------------------------


def test_dry_run_prints_the_baseline_name_and_the_comparison_key(capsys):
    manifest: dict = {}
    showroom.step_baseline(FakeApi(), False, manifest)
    out = capsys.readouterr().out
    assert showroom.BASELINE_EXPERIMENT in out
    assert "comparison_key" in out and "input" in out

    fake = FakeApi()
    fake.add_experiment(showroom.BASELINE_EXPERIMENT, [{"id": "row-1", "is_root": True, "input": "contract_1__q01", "metadata": {"case_id": "contract_1__q01"}}])
    manifest2: dict = {}
    showroom.step_baseline(fake, True, manifest2)
    out2 = capsys.readouterr().out
    exp_id = fake.experiments_by_name[showroom.BASELINE_EXPERIMENT]["id"]
    assert exp_id in out2
    assert fake.project_settings["baseline_experiment_id"] == exp_id
    assert manifest2["baseline"]["experiment_id"] == exp_id


def test_comparison_key_stays_input_because_every_non_retrieval_experiment_keys_on_the_case_id():
    good = {
        "A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc": [
            {"input": "contract_1__q01", "metadata": {"case_id": "contract_1__q01"}},
            {"input": "contract_2__q02", "metadata": {"case_id": "contract_2__q02"}},
        ],
        "rag-m3-hybrid": [{"input": "some retrieval query", "metadata": {}}],
    }
    ok, offenders = showroom.comparison_key_is_safe(good)
    assert ok is True and offenders == []

    bad = dict(good)
    bad["judge-D@glm"] = [{"input": "not-the-case-id", "metadata": {"case_id": "contract_3__q03"}}]
    ok2, offenders2 = showroom.comparison_key_is_safe(bad)
    assert ok2 is False and offenders2 == ["judge-D@glm"]


# --- D19: aggregate scores ---------------------------------------------------------


def test_aggregate_score_defs_are_exactly_derivable_from_the_six_obj_scores():
    six = {"obj/grounded_accuracy", "obj/answer_correct", "obj/citation_gold_overlap",
           "obj/citation_verbatim", "obj/abstain_correct", "obj/skill_adherence"}
    defs = showroom.aggregate_score_defs()
    by_name = {d["name"]: d for d in defs}
    for d in defs:
        assert set(d["inputs"]) <= six
    assert by_name["grounded and verbatim"]["weights"] is None
    for name in ("citation quality", "headline composite"):
        weights = by_name[name]["weights"]
        assert abs(sum(weights) - 1.0) < 1e-9
        assert "0.5" in by_name[name]["description"]
    headline = by_name["headline composite"]
    assert headline["weights"] == [0.5, 0.2, 0.2, 0.1]
    assert "composite" in headline["name"]
    assert "accuracy" not in headline["name"]


def test_dry_run_says_which_trust_metrics_are_not_expressible(capsys):
    showroom.step_aggscores(FakeApi(), False, {})
    out = capsys.readouterr().out
    assert "safe accuracy" in out and "precision when answering" in out and "net accuracy" in out
    assert "abstain_correct" in out and "answered" in out


def test_project_scores_are_created_once_and_never_touch_the_existing_four():
    fake = FakeApi()
    for name in ("reasoning", "evidence", "trajectory", "professional"):
        fake.add_score(name, "slider", config={"multi_select": False})
    fake.add_score("online: judge-professional on new logs", "online", config={"online": {"sampling_rate": 1}})
    before = {sid: dict(rec) for sid, rec in fake.project_scores.items()}

    showroom.step_aggscores(fake, True, {})
    assert len(fake.project_scores) == 8
    ids_after_first = {name: sid for sid, rec in fake.project_scores.items() for name in [rec["name"]] if name in
                       ("grounded and verbatim", "citation quality", "headline composite")}
    for sid, rec in before.items():
        assert fake.project_scores[sid] == rec, "pre-existing scores must be byte-identical"

    showroom.step_aggscores(fake, True, {})
    assert len(fake.project_scores) == 8
    ids_after_second = {name: sid for sid, rec in fake.project_scores.items() for name in [rec["name"]] if name in
                        ("grounded and verbatim", "citation quality", "headline composite")}
    assert ids_after_first == ids_after_second
    for sid, rec in before.items():
        assert fake.project_scores[sid] == rec


def test_score_trust_triple_without_live_prints_the_count_and_writes_nothing(capsys):
    fake = FakeApi()
    showroom.step_score_trust_triple(fake, False, {})
    out = capsys.readouterr().out
    n = len(showroom.trust_triple_plan())
    assert f"3 x {n} = {3 * n}" in out
    assert fake.posts == []
    assert showroom._ledger_keys() == set()


def _seed_trust_triple_experiments(fake: FakeApi) -> None:
    plan = showroom.trust_triple_plan()
    by_exp: dict[str, set[str]] = {}
    for p in plan:
        by_exp.setdefault(p["experiment"], set()).add(p["case_id"])
    for name, case_ids in by_exp.items():
        rows = [{"id": f"{name}::{cid}", "is_root": True, "input": cid, "metadata": {"case_id": cid}} for cid in case_ids]
        fake.add_experiment(name, rows)


def test_score_trust_triple_with_live_writes_exactly_that_many_and_ledgers_it():
    fake = FakeApi()
    _seed_trust_triple_experiments(fake)
    plan = showroom.trust_triple_plan()
    manifest: dict = {}
    showroom.step_score_trust_triple(fake, True, manifest)
    assert manifest["trust_triple"]["scores"] == 3 * len(plan)
    assert manifest["trust_triple"]["written"] == 3 * len(plan)
    ledger_rows = [json.loads(line) for line in showroom._ledger_path().read_text(encoding="utf-8").splitlines()]
    total_ledgered = sum(r["n_scores"] for r in ledger_rows if r["key"] == "trust-triple")
    assert total_ledgered == 3 * len(plan)
    # every row actually carries the three scores now
    for exp in fake.experiments_by_name.values():
        for row in exp["rows"]:
            assert set(row.get("scores", {})) == {"trust/safe", "trust/net", "trust/correct_outcome"}


# --- D20: the regressions dataset ---------------------------------------------------


def test_regression_rows_are_deterministic_and_33_with_the_rule_counts():
    from dealpoint.eval.braintrust_cockpit import _judged_subset, _variant_results

    rows1 = showroom.regression_rows()
    rows2 = showroom.regression_rows()
    assert rows1 == rows2
    assert len(rows1) == 33

    # recompute each rule's raw hit count independently of the merge, to check 25/2/6/2
    subset = _judged_subset()
    cap_hits = 0
    for v in subset["variants"]:
        rows = _variant_results(subset, v["variant_id"])
        for case_id in subset["case_ids"]:
            row = rows.get(case_id)
            if row and row.get("case_set") == "counterfactual" and (row.get("record") or {}).get("status") == "CAP_HIT":
                cap_hits += 1
    assert cap_hits == 25

    variant_key = json.loads(Path("data/eval/calibration/variant_key.json").read_text(encoding="utf-8"))
    assert all(pid in variant_key for pid in showroom.LAWYER_BEATS_JUDGES_PACKETS)
    assert len(showroom.LAWYER_BEATS_JUDGES_PACKETS) == 2

    rows_a = _variant_results(subset, "A@haiku")
    misleading = 0
    for case_id in subset["case_ids"]:
        row = rows_a.get(case_id)
        if not row:
            continue
        status = (row.get("record") or {}).get("status")
        scores = row.get("scores") or {}
        correct_all = bool(scores.get("grounded_accuracy")) or (row.get("case_set") == "counterfactual" and bool(scores.get("abstain_correct")))
        if status == "ANSWERED" and not correct_all:
            misleading += 1
    assert misleading == 6
    assert len(showroom.HERO_PAIR) == 2

    merged = [r for r in rows1 if "," in r["metadata"]["reason"]]
    assert len(merged) == 2
    ids = [r["id"] for r in rows1]
    assert len(ids) == len(set(ids))


def test_every_regression_row_has_a_gold_expectation():
    for r in showroom.regression_rows():
        expected = r["expected"]
        assert isinstance(expected["answer"], str) and expected["answer"]
        if r["metadata"]["case_set"] == "counterfactual":
            assert expected["answer"] == "ABSTAIN"
            assert expected["gold_spans"] == []
        elif r["metadata"]["case_set"] == "test":
            assert len(expected["gold_spans"]) >= 1


def test_the_hero_pair_is_present():
    rows = showroom.regression_rows()
    by_key = {(r["input"], r["metadata"]["source_variant"]): r for r in rows}
    for case_id, variant_id in showroom.HERO_PAIR:
        row = by_key[(case_id, variant_id)]
        assert "hero_pair" in row["metadata"]["rule"] or "hero case" in row["metadata"]["reason"] or "redacted twin" in row["metadata"]["reason"]


def test_every_regression_row_resolves_to_a_log_id():
    rows = showroom.regression_rows()
    fake = FakeApi()
    for i, r in enumerate(rows):
        fake.add_log(f"log-{i}", {"case_id": r["input"], "variant_id": r["metadata"]["source_variant"], "category": "judged"})
    n = showroom.resolve_source_log_ids(fake, rows)
    assert n == len(rows)
    assert all(r["metadata"]["source_log_id"] is not None for r in rows)


def test_regressions_dataset_is_idempotent(monkeypatch):
    import sys

    fake_bt = FakeBraintrustModule()
    monkeypatch.setitem(sys.modules, "braintrust", fake_bt)
    rows = showroom.regression_rows()
    fake_api = FakeApi()
    for i, r in enumerate(rows):
        fake_api.add_log(f"log-{i}", {"case_id": r["input"], "variant_id": r["metadata"]["source_variant"], "category": "judged"})

    manifest: dict = {}
    showroom.step_regressions(fake_api, True, manifest)
    ds = fake_bt.datasets[showroom.REGRESSIONS_DATASET]
    assert set(ds.rows) == {r["id"] for r in rows}
    n_rows_first = len(ds.rows)

    manifest2: dict = {}
    showroom.step_regressions(fake_api, True, manifest2)
    assert len(ds.rows) == n_rows_first
    assert set(ds.rows) == {r["id"] for r in rows}


# --- D21: first-class tags on logs --------------------------------------------------


def test_tags_for_log_covers_every_family():
    judged_cap_hit = {"status": "CAP_HIT", "arm": "D", "model_label": "glm", "category": "judged",
                      "case_id": "contract_39__redacted_q05", "variant_id": "D@glm"}
    tags = showroom.tags_for_log(judged_cap_hit)
    for t in ("status:CAP_HIT", "system:D", "model:glm", "pool:judged-18", "category:judged"):
        assert t in tags

    retrieval = {"category": "retrieval"}
    assert set(showroom.tags_for_log(retrieval)) == {"pool:retrieval", "category:retrieval"}

    representative = {"category": "successful_direct", "status": "ANSWERED", "arm": "D", "model": "openai/gpt-5.6-luna-pro"}
    rep_tags = showroom.tags_for_log(representative)
    assert not any(t.startswith("pool:") for t in rep_tags)

    # a real D20 (case, variant) pair -- the hero case
    hero = {"case_id": "contract_144__q05", "variant_id": "A@haiku", "status": "ABSTAINED", "arm": "A",
            "model_label": "haiku", "category": "judged"}
    assert "regression" in showroom.tags_for_log(hero)


def test_tag_plan_counts_by_key_and_is_printed(capsys):
    plan = showroom.tag_plan()
    total_from_by_key = sum(plan["by_key"]["category"].values())
    assert total_from_by_key == plan["planned"]
    assert plan["by_key"]["regression"] == 33

    showroom.step_logtags(FakeApi(), False, {})
    out = capsys.readouterr().out
    assert str(plan["planned"]) in out
    assert "status=" in out and "system=" in out and "pool=" in out and "category=" in out and "regression=" in out


def test_log_tags_are_merged_in_one_pass_and_are_idempotent(monkeypatch):
    fake = FakeApi()
    fake.add_log("log-1", {"status": "CAP_HIT", "arm": "D", "model_label": "glm", "category": "judged",
                           "case_id": "contract_39__redacted_q05", "variant_id": "D@glm"})
    fake.add_log("log-2", {"category": "retrieval"})
    monkeypatch.setattr("dealpoint.eval.braintrust_cockpit.RestClient", lambda key: FakeRestClient())
    monkeypatch.setattr(showroom, "load_braintrust_key", lambda: "fake-key")

    manifest: dict = {}
    showroom.step_logtags(fake, True, manifest)
    assert all("tags" in e for e in fake.log_events.values())
    for path, body in fake.posts:
        if path.startswith("project_logs/") and path.endswith("/insert"):
            for e in body["events"]:
                assert "scores" not in e
                assert e.get("_is_merge") is True
    first_tags = {lid: list(e["tags"]) for lid, e in fake.log_events.items()}

    manifest2: dict = {}
    showroom.step_logtags(fake, True, manifest2)
    second_tags = {lid: list(e["tags"]) for lid, e in fake.log_events.items()}
    assert first_tags == second_tags


def test_two_saved_logs_views_exist_and_are_idempotent(monkeypatch):
    fake = FakeApi()
    rc = FakeRestClient()
    monkeypatch.setattr("dealpoint.eval.braintrust_cockpit.RestClient", lambda key: rc)
    monkeypatch.setattr(showroom, "load_braintrust_key", lambda: "fake-key")

    showroom.step_logtags(fake, True, {})
    names = {v["name"] for v in rc.views}
    assert names == {"Cap-hits", "Regressions"}
    for v in rc.views:
        assert v["view_type"] == "logs"
    by_name = {v["name"]: v for v in rc.views}
    assert "status:CAP_HIT" in by_name["Cap-hits"]["view_data"]["search"]["filter"][0]["btql"]
    assert "regression" in by_name["Regressions"]["view_data"]["search"]["filter"][0]["btql"]
    n_views_first = len(rc.views)

    showroom.step_logtags(fake, True, {})
    assert len(rc.views) == n_views_first


# --- D19/D20/D21 write no scores ----------------------------------------------------


def test_every_other_live_write_carries_no_scores(monkeypatch):
    import sys

    fake = FakeApi()
    fake.add_experiment(showroom.BASELINE_EXPERIMENT, [{"id": "row-1", "is_root": True, "input": "contract_1__q01", "metadata": {"case_id": "contract_1__q01"}}])
    rows = showroom.regression_rows()
    for i, r in enumerate(rows):
        fake.add_log(f"log-{i}", {"case_id": r["input"], "variant_id": r["metadata"]["source_variant"], "category": "judged"})
    monkeypatch.setattr("dealpoint.eval.braintrust_cockpit.RestClient", lambda key: FakeRestClient())
    monkeypatch.setattr(showroom, "load_braintrust_key", lambda: "fake-key")
    monkeypatch.setitem(sys.modules, "braintrust", FakeBraintrustModule())

    manifest: dict = {}
    for step_name in ("baseline", "logtags", "aggscores", "regressions"):
        getattr(showroom, f"step_{step_name}")(fake, True, manifest)

    for _path, body in fake.posts:
        for e in body.get("events", []) if isinstance(body, dict) else []:
            assert "scores" not in e
    for line in showroom._ledger_path().read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["key"] != "trust-triple":
            assert row["n_scores"] == 0


# --- D22: the design note + tour edits ----------------------------------------------


def test_design_note_names_the_three_seams_and_every_braintrust_touching_route():
    text = Path("docs/braintrust-live-app.md").read_text(encoding="utf-8")
    for seam in ("init_logger", "_traced", "_maybe_wrap_openai"):
        assert seam in text
    for route in ("/api/agreements", "/api/agreements/{id}/text", "/api/questions", "/api/cases/",
                  "/api/run", "/api/market/", "/api/reports"):
        assert route in text
    assert "no FastAPI app" in text or "no `web/`" in text or "does not exist" in text


def test_tour_stops_3_and_4_carry_the_new_actions():
    text = Path("docs/demo-tour.md").read_text(encoding="utf-8")
    stop3 = text.split("### Stop 3.", 1)[1].split("### Stop 4.", 1)[0]
    stop4 = text.split("### Stop 4.", 1)[1].split("### Stop 5.", 1)[0]
    assert showroom.BASELINE_EXPERIMENT in stop3
    assert showroom.REGRESSIONS_DATASET in stop4
