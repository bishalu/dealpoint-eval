"""M7b gate: `just demo-walkthrough` generates `docs/demo-walkthrough.md`
from the manifest with no `[M7b]` markers left, only `[cockpit session]`
where spec section 7 applies, under the 1,400-word cap, and every named
object resolves against the manifest (spec `m7b.md` section 6, `gate_m7b`).
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.gate_m7b


def _fake_manifest() -> dict:
    return {
        "project_id": "proj-1",
        "views": [
            {"name": "Judged traces by variant", "view_type": "experiments", "id": "view-1", "caption": "x"},
            {"name": "Judge disagreement", "view_type": "experiment", "id": "view-2", "caption": "x"},
            {"name": "Retrieval rescue", "view_type": "logs", "id": "view-3", "caption": "x"},
            {"name": "Failure attribution", "view_type": "logs", "id": "view-4", "caption": "x"},
            {"name": "DeepEval vs judge disagreement", "view_type": "logs", "id": "view-5", "caption": "x"},
            {"name": "Trajectory inefficiency", "view_type": "logs", "id": "view-6", "caption": "x"},
            {"name": "Review set (12)", "view_type": "for_review_experiments", "id": "view-7", "caption": "x"},
        ],
        "dashboard": {"name": "DealPoint eval overview", "id": "view-8"},
        "topics": {"config": {"name": "x", "facet_name": "trace-outcome-summary"}},
        "pattern": {"definition": {"name": "Trajectory inefficiency pattern", "supporting_trace_ids": ["a", "b", "c"]}},
        "hero_case": {
            "case_id": "contract_32__q04",
            "grounded_accuracy_a": False,
            "grounded_accuracy_d": True,
            "max_pairwise_spread": 4,
            "n_candidates": 2,
            "rule": "x",
        },
        "human_scoring_probe": {
            "decision": "local_form",
            "finding": "x",
            "decision_reason": "x",
        },
        "human_score_rows": [{"case_id": "c1"}],
        "n_human_scores_planned": 96,
        "n_human_scores_pushed": 0,
        "replay": {"experiment_name": "m7b-hero-case", "variants": ["A@haiku", "D@haiku"]},
        "git_sha7": "abc1234",
        "rubric_version": "cfda9f8cc401",
        "subset_hash": "5918ef10a7e6",
        "synced_at": "2026-09-06T00:00:00+00:00",
    }


def test_generate_walkthrough_leaves_no_m7b_markers():
    from dealpoint.config import DEMO_WALKTHROUGH_DRAFT_PATH
    from dealpoint.eval.demo_walkthrough import generate_walkthrough

    draft = DEMO_WALKTHROUGH_DRAFT_PATH.read_text(encoding="utf-8")
    generated = generate_walkthrough(_fake_manifest(), draft)
    assert "[M7b]" not in generated
    assert "[cockpit session]" in generated


def test_generate_walkthrough_under_word_cap():
    from dealpoint.config import DEMO_WALKTHROUGH_DRAFT_PATH, M7B_MAX_WORDS
    from dealpoint.eval.demo_walkthrough import _word_count, generate_walkthrough

    draft = DEMO_WALKTHROUGH_DRAFT_PATH.read_text(encoding="utf-8")
    generated = generate_walkthrough(_fake_manifest(), draft)
    assert _word_count(generated) <= M7B_MAX_WORDS


def test_generate_walkthrough_is_idempotent_from_the_frozen_draft():
    """Regenerating from the SAME frozen draft twice must produce the exact
    same document -- the draft is never mutated in place.
    """
    from dealpoint.config import DEMO_WALKTHROUGH_DRAFT_PATH
    from dealpoint.eval.demo_walkthrough import generate_walkthrough

    draft = DEMO_WALKTHROUGH_DRAFT_PATH.read_text(encoding="utf-8")
    manifest = _fake_manifest()
    first = generate_walkthrough(manifest, draft)
    second = generate_walkthrough(manifest, draft)
    assert first == second


def test_generate_walkthrough_cites_the_manifests_hero_case_and_views():
    from dealpoint.config import DEMO_WALKTHROUGH_DRAFT_PATH
    from dealpoint.eval.demo_walkthrough import generate_walkthrough

    draft = DEMO_WALKTHROUGH_DRAFT_PATH.read_text(encoding="utf-8")
    manifest = _fake_manifest()
    generated = generate_walkthrough(manifest, draft)
    assert manifest["hero_case"]["case_id"] in generated
    for view in manifest["views"]:
        assert view["name"] in generated
    assert manifest["dashboard"]["name"] in generated


def test_real_manifest_generates_a_valid_walkthrough_when_present():
    """If `data/reports/demo_manifest.json` already exists (a prior
    `--dry-run`), the real doc it produces must also satisfy the gate --
    this is the actual reproducibility check, not just the fake fixture.
    """
    from dealpoint.config import DEMO_MANIFEST_PATH, DEMO_WALKTHROUGH_DRAFT_PATH, M7B_MAX_WORDS
    from dealpoint.eval.demo_walkthrough import _word_count, generate_walkthrough

    if not DEMO_MANIFEST_PATH.exists():
        pytest.skip("data/reports/demo_manifest.json not present -- run `just braintrust-cockpit-dry-run` first")
    manifest = json.loads(DEMO_MANIFEST_PATH.read_text(encoding="utf-8"))
    draft = DEMO_WALKTHROUGH_DRAFT_PATH.read_text(encoding="utf-8")
    generated = generate_walkthrough(manifest, draft)
    assert "[M7b]" not in generated
    assert _word_count(generated) <= M7B_MAX_WORDS
    assert manifest["hero_case"].get("case_id") is None or manifest["hero_case"]["case_id"] in generated


def test_disk_manifest_is_a_live_sync_not_a_dry_run_simulation():
    """The milestone's deliverable is that every persistent Braintrust
    object stops being a dry-run simulation (M7b attempt 3 corrective
    task). This is the test that would have caught attempt 2 shipping
    `project_id: "dry-run-project-id"` and `view-1..view-8` placeholder
    ids as if they were real Braintrust objects.
    """
    from dealpoint.config import DEMO_MANIFEST_PATH

    if not DEMO_MANIFEST_PATH.exists():
        pytest.skip("data/reports/demo_manifest.json not present")
    manifest = json.loads(DEMO_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert not manifest["project_id"].startswith("dry-run")
    for view in manifest["views"]:
        assert not view["id"].startswith("view-"), f"{view['name']} still has a dry-run placeholder id"
    assert not manifest["dashboard"]["id"].startswith("view-")
    assert manifest["topics"]["id"], "Topics facet must be created live (POST /v1/function)"
    assert manifest["experiments"], "experiments must resolve from the live project"
    assert manifest["datasets"], "datasets must resolve from the live project"
    assert manifest["permalinks"], "permalinks must be built from a resolved org/project"
    assert manifest["replay"]["experiment_name"] == "m7b-hero-case"
    assert set(manifest["replay"]["variants"]) == {"A@haiku", "D@haiku"}


# Prefixes a backtick-quoted doc token must start with to be treated as a
# candidate experiment name for the reverse-resolution check below.
_DOC_EXPERIMENT_NAME_PREFIXES = ("A-", "B-", "C-", "D-", "judge-", "judged-", "m7-", "m7b-", "rag-", "pareto-")


def test_every_named_object_in_the_walkthrough_resolves_against_the_manifest():
    """Every experiment/dataset name the doc cites is one the manifest
    actually resolved (spec section 6: "every link in the generated doc is
    one the manifest can regenerate"). This also fails, going the OTHER
    direction, when the doc names an experiment or dataset the manifest did
    NOT resolve -- the exact gap that let 14 rag-/pareto-/A- experiment
    names and a placeholder dataset glob survive M7b attempt 3.
    """
    import re

    from dealpoint.config import DEMO_MANIFEST_PATH, DEMO_WALKTHROUGH_PATH

    if not DEMO_MANIFEST_PATH.exists() or not DEMO_WALKTHROUGH_PATH.exists():
        pytest.skip("live manifest/doc not present")
    manifest = json.loads(DEMO_MANIFEST_PATH.read_text(encoding="utf-8"))
    doc = DEMO_WALKTHROUGH_PATH.read_text(encoding="utf-8")

    for view in manifest["views"]:
        assert view["name"] in doc
    assert manifest["dashboard"]["name"] in doc
    if manifest["hero_case"].get("case_id"):
        assert manifest["hero_case"]["case_id"] in doc

    experiment_names = {e["name"] for e in manifest["experiments"]}
    dataset_names = {d["name"] for d in manifest["datasets"]}
    # Strip fenced ```...``` blocks first -- their triple-backtick delimiters
    # otherwise pair up with unrelated single backticks elsewhere in the doc
    # and the inline-code regex below swallows huge, wrong "tokens".
    doc_prose = re.sub(r"```.*?```", "", doc, flags=re.DOTALL)
    for token in re.findall(r"`([^`]+)`", doc_prose):
        # Skip placeholders/globs ("judge-<variant_id>", "maud-dealpoint-*")
        # -- these are patterns the doc explains, not literal object names.
        if "<" in token or "*" in token:
            continue
        if token.startswith(_DOC_EXPERIMENT_NAME_PREFIXES):
            assert token in experiment_names, f"doc cites experiment {token!r} absent from manifest['experiments']"
        elif token.startswith("maud-dealpoint-"):
            assert token in dataset_names, f"doc cites dataset {token!r} absent from manifest['datasets']"
