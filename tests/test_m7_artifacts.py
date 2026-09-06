"""M7a gate: every DoD artifact exists with a `.gitignore` allowlist entry,
frozen-integrity assertions hold, and `docs/demo-walkthrough.md` names only
paths that actually resolve on disk (spec DoD, "walkthrough reproducibility").

Every test skips cleanly rather than fails when a metered artifact has not
been produced yet, matching `test_readme_results.py`/`test_spend_m6.py`.
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from dealpoint.config import (
    ARM_C_RETRIEVER,
    BTQL_INVESTIGATIONS_PATH,
    DEEPEVAL_CROSSCHECK_JSON_PATH,
    DEEPEVAL_CROSSCHECK_MD_PATH,
    FRAMEWORK_VERSIONS_PATH,
    LI_RAG_EVAL_JSON_PATH,
    LI_RAG_EVAL_MD_PATH,
    M7A_DISK_FLOOR_BYTES,
    M7A_DISK_GUARD_PATH,
    REPO_ROOT,
    REPRESENTATIVE_CASES_PATH,
    TEST_SUBSET_V1_PATH,
    TOURNAMENT_JSON_PATH,
    VERSIONS_JSON_PATH,
)

pytestmark = pytest.mark.gate_m7

M7A_REPORT_PATHS = (
    LI_RAG_EVAL_JSON_PATH,
    LI_RAG_EVAL_MD_PATH,
    DEEPEVAL_CROSSCHECK_JSON_PATH,
    DEEPEVAL_CROSSCHECK_MD_PATH,
    BTQL_INVESTIGATIONS_PATH,
    REPRESENTATIVE_CASES_PATH,
    FRAMEWORK_VERSIONS_PATH,
    M7A_DISK_GUARD_PATH,
)


def _gitignore_text() -> str:
    return (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")


def test_every_m7a_report_path_has_a_gitignore_allowlist_line():
    gitignore = _gitignore_text()
    for path in M7A_REPORT_PATHS:
        rel = path.relative_to(REPO_ROOT).as_posix()
        assert f"!{rel}" in gitignore, f"{rel} missing its !allowlist line in .gitignore"


def test_framework_versions_json_exists_and_has_expected_keys():
    if not FRAMEWORK_VERSIONS_PATH.exists():
        pytest.skip("framework_versions.json not generated yet")
    payload = json.loads(FRAMEWORK_VERSIONS_PATH.read_text(encoding="utf-8"))
    assert "python" in payload
    assert "git_sha7" in payload


def test_disk_guard_was_written_by_record_guard():
    if not M7A_DISK_GUARD_PATH.exists():
        pytest.skip("m7a_disk_guard.json not generated yet; run `just disk-guard`")
    raw = M7A_DISK_GUARD_PATH.read_text(encoding="utf-8")
    payload = json.loads(raw)

    # record_guard() writes with sort_keys=True and a trailing newline.
    assert raw == json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert list(json.loads(raw).keys()) == sorted(json.loads(raw).keys())

    assert payload["floor_bytes"] == M7A_DISK_FLOOR_BYTES
    assert payload["measured_at"] != "2026-09-05T00:00:00+00:00"
    assert payload["within_guard"] is True
    for group in payload["groups"]:
        assert "re_measured_at" in group
        assert min(group["free_before_bytes"], group["free_after_bytes"]) >= M7A_DISK_FLOOR_BYTES


def test_li_rag_eval_frozen_assertions_match_disk():
    if not LI_RAG_EVAL_JSON_PATH.exists():
        pytest.skip("li_rag_eval.json not generated yet; run `just li-rag-eval`")
    report = json.loads(LI_RAG_EVAL_JSON_PATH.read_text(encoding="utf-8"))
    fa = report["frozen_assertions"]

    versions = json.loads(VERSIONS_JSON_PATH.read_text(encoding="utf-8"))
    assert fa["chunk_version"] == versions["chunk_version"] == "8e5e8ba56765"
    assert fa["index_version"] == versions["index_version"] == "e2b4a2b97561"

    test_subset_hash = hashlib.sha256(TEST_SUBSET_V1_PATH.read_bytes()).hexdigest()
    assert fa["test_subset_v1_sha256"] == test_subset_hash

    tournament_hash = hashlib.sha256(TOURNAMENT_JSON_PATH.read_bytes()).hexdigest()
    assert fa["tournament_json_sha256"] == tournament_hash

    assert ARM_C_RETRIEVER["name"] == "hybrid_rrf"


def test_deepeval_crosscheck_has_one_of_three_classifications():
    if not DEEPEVAL_CROSSCHECK_JSON_PATH.exists():
        pytest.skip("deepeval_crosscheck.json not generated yet")
    report = json.loads(DEEPEVAL_CROSSCHECK_JSON_PATH.read_text(encoding="utf-8"))
    assert report["classification"] in (
        "KEEP_CORE_DIAGNOSTIC",
        "KEEP_OPTIONAL_ANALYSIS",
        "REMOVE_NO_ADDED_SIGNAL",
    )
    assert report["resolved_evaluator"]["model"]


def test_deepeval_crosscheck_run_quality_not_silently_degraded():
    if not DEEPEVAL_CROSSCHECK_JSON_PATH.exists():
        pytest.skip("deepeval_crosscheck.json not generated yet")
    report = json.loads(DEEPEVAL_CROSSCHECK_JSON_PATH.read_text(encoding="utf-8"))

    n_traces = report["subset"]["n_traces"]
    assert len(report["per_trace_scores"]) == n_traces

    coverage = report["coverage"]
    for name in ("task_completion", "tool_correctness", "argument_correctness", "step_efficiency"):
        assert name in coverage
        c = coverage[name]
        assert c["n_scored"] + c["n_null"] == n_traces
        # a coverage floor: every metric must have a non-null score on a stated
        # majority of traces, so a silently-degraded run (e.g. step_efficiency's
        # pre-repair 93/108 nulls) fails the gate instead of passing it.
        assert c["n_scored"] > n_traces / 2, f"{name} scored on only {c['n_scored']}/{n_traces} traces"

    step_eff_scores = {
        row["step_efficiency"]["score"] for row in report["per_trace_scores"] if row["step_efficiency"]["score"] is not None
    }
    assert len(step_eff_scores) > 1, "step_efficiency shows no variation -- looks degraded"

    assert report["resolved_evaluator"].get("provider")
    assert report["resolved_evaluator"].get("deepeval_version")

    vs_judge = report["comparisons"]["vs_judge"]
    for val in vs_judge.values():
        assert val["n"] <= n_traces

    assert report["decisions"], "decisions must not be empty"
    assert any("judge-trio" in d["decision"] or "independence" in d["topic"] for d in report["decisions"])

    spend = report["spend"]
    assert "m7a_realized_usd" in spend
    assert "deepeval_realized_usd" in spend
    assert "global_realized_usd" in spend


def test_btql_investigations_has_six_entries():
    if not BTQL_INVESTIGATIONS_PATH.exists():
        pytest.skip("btql_investigations.json not generated yet")
    entries = json.loads(BTQL_INVESTIGATIONS_PATH.read_text(encoding="utf-8"))
    assert len(entries) == 6
    for e in entries:
        assert "row_count" in e
        assert "btql" in e


def test_demo_walkthrough_exists_with_role_diagram():
    path = REPO_ROOT / "docs" / "demo-walkthrough.md"
    if not path.exists():
        pytest.skip("docs/demo-walkthrough.md not written yet")
    text = path.read_text(encoding="utf-8")
    assert "custom Python" in text
    assert "LlamaIndex" in text
    assert "DeepEval" in text
    assert "Braintrust" in text
    assert "benchmark truth" in text


def test_demo_walkthrough_named_artifacts_resolve_on_disk():
    """Mechanised version of the reviewer's \"walkthrough reproducibility\"
    check: every backtick-quoted path under data/, docs/ or specs/ that the
    walkthrough names must exist on disk.
    """
    path = REPO_ROOT / "docs" / "demo-walkthrough.md"
    if not path.exists():
        pytest.skip("docs/demo-walkthrough.md not written yet")
    text = path.read_text(encoding="utf-8")

    candidates = set(re.findall(r"`((?:data|docs|specs)/[^`]+)`", text))
    missing = []
    for rel in candidates:
        # strip any trailing punctuation a sentence might have glued on
        clean_rel = rel.rstrip(".,;:")
        if not (REPO_ROOT / clean_rel).exists():
            missing.append(clean_rel)
    assert not missing, f"demo-walkthrough.md names paths that do not exist: {missing}"
