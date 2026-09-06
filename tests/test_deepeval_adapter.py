"""M7a gate: DeepEval adapter -- evaluator resolution, the pure row_to_test_case
mapping, expected_tools derivation, comparisons/classification. All of this
must work with `deepeval` absent (spec section 2/4: "the mapping is
framework-free by design").
"""

from __future__ import annotations

import json

import pytest

from dealpoint.corpus.document import Document
from dealpoint.data.sections import Section

pytestmark = pytest.mark.gate_m7


def _fake_doc() -> Document:
    text = "Article I THE MERGER Section 1.1 Some text about Knowledge and cash consideration follows here." * 3
    return Document(
        document_id="doc_x",
        agreement_id="doc_x",
        text=text,
        sections=[Section(ref="1.1", title="the merger", start=0, end=len(text))],
        body_start=0,
        body_end=len(text),
    )


def test_resolve_evaluator_model_picks_non_candidate_family(tmp_path, monkeypatch):
    from dealpoint.eval import deepeval_adapter as da

    slate_path = tmp_path / "judge_slate.json"
    slate_path.write_text(
        json.dumps(
            {
                "price_basis": "live",
                "judges": [
                    {"model": "mistralai/mistral-small-3.2-24b-instruct", "family": "Mistral", "ok": True},
                    {"model": "nvidia/nemotron-3-super-120b-a12b", "family": "NVIDIA", "ok": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    import dealpoint.config as config_mod

    monkeypatch.setattr(config_mod, "JUDGE_SLATE_PATH", slate_path)

    resolved = da.resolve_evaluator_model(env_value=None)
    assert resolved["model"] == "mistralai/mistral-small-3.2-24b-instruct"
    assert resolved["family"] == "Mistral"
    assert resolved["policy_checked"] is True


def test_resolve_evaluator_model_raises_when_all_candidates_are_excluded(tmp_path, monkeypatch):
    from dealpoint.eval import deepeval_adapter as da
    from dealpoint.eval.judge_slate import CANDIDATE_FAMILIES

    slate_path = tmp_path / "judge_slate.json"
    slate_path.write_text(
        json.dumps(
            {
                "price_basis": "live",
                "judges": [
                    {"model": "anthropic/claude-haiku-4.5", "family": "Anthropic", "ok": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    import dealpoint.config as config_mod

    monkeypatch.setattr(config_mod, "JUDGE_SLATE_PATH", slate_path)
    assert "Anthropic" in CANDIDATE_FAMILIES

    with pytest.raises(RuntimeError):
        da.resolve_evaluator_model(env_value=None)


def test_resolve_evaluator_model_honours_env_but_still_checks_policy():
    from dealpoint.eval import deepeval_adapter as da

    resolved = da.resolve_evaluator_model(env_value="mistralai/mistral-small-3.2-24b-instruct")
    assert resolved["model"] == "mistralai/mistral-small-3.2-24b-instruct"
    assert resolved["policy_checked"] is True
    assert resolved["source"].startswith("env:")

    with pytest.raises(RuntimeError):
        da.resolve_evaluator_model(env_value="anthropic/claude-haiku-4.5")


def test_row_to_test_case_mapping_answered_case():
    from dealpoint.eval import deepeval_adapter as da

    case = {
        "case_id": "doc_x__q01",
        "question_id": "q01",
        "required_evidence": None,
    }
    row = {
        "finding": {"answer": "All Cash", "evidence": [], "rationale": "clear"},
        "record": {
            "trajectory": [
                {"tool": "search_agreement", "args": {"query": "consideration"}, "char_ranges": [[0, 50]]},
            ]
        },
    }
    doc = _fake_doc()
    payload = da.row_to_test_case(row, case, doc)

    assert payload["actual_output"] == "answer: All Cash\nrationale: clear"
    assert payload["tools_called"] == [{"name": "search_agreement", "input_parameters": {"query": "consideration"}}]
    assert payload["expected_tools"] == [{"name": "search_agreement"}]
    assert len(payload["retrieval_context"]) == 1
    assert payload["case_id"] == "doc_x__q01"


def test_row_to_test_case_mapping_finding_is_none():
    from dealpoint.eval import deepeval_adapter as da

    case = {"case_id": "doc_x__q05", "question_id": "q05", "required_evidence": 'defined term "Knowledge"'}
    row = {"finding": None, "record": {"trajectory": []}}
    doc = _fake_doc()
    payload = da.row_to_test_case(row, case, doc)

    assert payload["actual_output"] == "The system produced no finding for this case."
    assert payload["retrieval_context"] == []
    assert payload["expected_tools"] == [{"name": "search_agreement"}, {"name": "lookup_defined_term"}]


def test_derive_expected_tools_defined_term_case():
    from dealpoint.eval.deepeval_adapter import derive_expected_tools

    case = {"required_evidence": 'defined term "Material Adverse Effect" (or "Company MAE")'}
    tools = derive_expected_tools(case)
    assert {t["name"] for t in tools} == {"search_agreement", "lookup_defined_term"}


def test_derive_expected_tools_none_case():
    from dealpoint.eval.deepeval_adapter import derive_expected_tools

    case = {"required_evidence": None}
    tools = derive_expected_tools(case)
    assert tools == [{"name": "search_agreement"}]


def test_compare_to_deterministic_handles_none_without_coercion():
    from dealpoint.eval.deepeval_adapter import compare_to_deterministic

    deepeval_scores = [
        {"task_completion": {"score": 0.8}},
        {"task_completion": {"score": None}},
    ]
    det_rows = [
        {"grounded_accuracy": True, "required_evidence_met": None, "tool_calls": 3},
        {"grounded_accuracy": None, "required_evidence_met": None, "tool_calls": 8},
    ]
    result = compare_to_deterministic(deepeval_scores, det_rows)
    assert result["task_completion_vs_grounded_accuracy"]["n"] == 1
    assert result["n_at_or_over_tool_cap"] == 1


def test_compare_to_human_reports_pending_n_zero(tmp_path, monkeypatch):
    import dealpoint.config as config_mod
    from dealpoint.eval import deepeval_adapter as da

    monkeypatch.setattr(config_mod, "CALIBRATION_DIR", tmp_path)
    result = da.compare_to_human()
    assert result["n"] == 0
    assert result["status"] == "pending"
    assert result["task_completion_vs_human"] == "pending"
    assert result["step_efficiency_vs_human_trajectory_quality"] == "pending"


def test_compare_to_human_computes_real_statistics_when_scores_exist():
    """With real human rows on disk, both cells must carry a numeric n and
    a numeric spearman -- never the placeholder string 'available'."""
    from dealpoint.eval.deepeval_adapter import compare_to_human

    deepeval_scores = [
        {"case_id": "c1", "variant_id": "v1", "task_completion": {"score": 0.8}, "step_efficiency": {"score": 0.6}},
        {"case_id": "c2", "variant_id": "v1", "task_completion": {"score": 0.4}, "step_efficiency": {"score": 0.9}},
        {"case_id": "c3", "variant_id": "v1", "task_completion": {"score": 1.0}, "step_efficiency": {"score": 0.2}},
    ]
    human_means = {
        ("c1", "v1"): {"reasoning": 4.0, "trajectory": 3.0},
        ("c2", "v1"): {"reasoning": 2.0, "trajectory": 5.0},
        ("c3", "v1"): {"reasoning": 5.0, "trajectory": 1.0},
    }
    result = compare_to_human(deepeval_scores, human_means)
    assert result["status"] == "available"
    assert result["n"] == 3
    for key in ("task_completion_vs_human", "step_efficiency_vs_human_trajectory_quality"):
        cell = result[key]
        assert isinstance(cell, dict), f"{key} must be a dict, not a placeholder string"
        assert isinstance(cell["n"], int)
        assert isinstance(cell["spearman"], float)


def test_load_human_dimension_means_joins_through_variant_key(tmp_path, monkeypatch):
    import json as json_mod

    import dealpoint.config as config_mod
    from dealpoint.eval.deepeval_adapter import load_human_dimension_means

    monkeypatch.setattr(config_mod, "CALIBRATION_DIR", tmp_path)
    (tmp_path / "variant_key.json").write_text(
        json_mod.dumps({"abc123": {"case_id": "contract_0__q01", "variant_id": "D@haiku"}}),
        encoding="utf-8",
    )
    (tmp_path / "human_scores.jsonl").write_text(
        json_mod.dumps(
            {
                "packet_id": "abc123",
                "scorer": "lawyer_1",
                "reasoning": 4,
                "evidence": 3,
                "trajectory": 5,
                "professional": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    means = load_human_dimension_means(tmp_path / "human_scores.jsonl")
    assert means[("contract_0__q01", "D@haiku")]["reasoning"] == 4
    assert means[("contract_0__q01", "D@haiku")]["trajectory"] == 5


@pytest.mark.gate_m7
def test_deepeval_crosscheck_vs_human_is_numeric_when_human_scores_exist():
    """Whenever `human_scores.jsonl` is non-empty, `vs_human` on the actual
    persisted report must carry a numeric n and spearman for both
    comparisons -- guards against the placeholder-string regression."""
    import json as json_mod

    from dealpoint.config import CALIBRATION_DIR, DEEPEVAL_CROSSCHECK_JSON_PATH

    human_path = CALIBRATION_DIR / "human_scores.jsonl"
    if not human_path.exists() or human_path.stat().st_size == 0:
        pytest.skip("human_scores.jsonl is empty in this environment")
    if not DEEPEVAL_CROSSCHECK_JSON_PATH.exists():
        pytest.skip("deepeval_crosscheck.json not generated yet")

    report = json_mod.loads(DEEPEVAL_CROSSCHECK_JSON_PATH.read_text(encoding="utf-8"))
    vs_human = report["comparisons"]["vs_human"]
    assert vs_human["status"] == "available"
    for key in ("task_completion_vs_human", "step_efficiency_vs_human_trajectory_quality"):
        cell = vs_human[key]
        assert isinstance(cell, dict), f"{key} is {cell!r}, not a numeric statistic"
        assert isinstance(cell["n"], int)
        assert isinstance(cell["spearman"], float)

    for d in report["brief_differences"]:
        if d["topic"] == "scale_inherited":
            assert "'pending'" not in d["difference"] or "n=0" not in d["difference"]
            assert str(vs_human["n"]) in d["difference"]


def test_classify_returns_one_of_three_allowed_strings():
    from dealpoint.eval.deepeval_adapter import classify

    strong = {"vs_deterministic": {"task_completion_vs_grounded_accuracy": {"spearman": 0.9, "n": 50}}}
    weak = {"vs_deterministic": {"task_completion_vs_grounded_accuracy": {"spearman": 0.1, "n": 50}}}
    small_n = {"vs_deterministic": {"task_completion_vs_grounded_accuracy": {"spearman": 0.9, "n": 3}}}

    allowed = {"KEEP_CORE_DIAGNOSTIC", "KEEP_OPTIONAL_ANALYSIS", "REMOVE_NO_ADDED_SIGNAL"}
    label_strong, _ = classify(strong)
    label_weak, _ = classify(weak)
    label_small, _ = classify(small_n)
    assert label_strong == "REMOVE_NO_ADDED_SIGNAL"
    assert label_weak == "KEEP_CORE_DIAGNOSTIC"
    assert label_small == "KEEP_OPTIONAL_ANALYSIS"
    assert {label_strong, label_weak, label_small} <= allowed


def test_classify_downgrades_when_evaluator_shares_model_with_judge_trio():
    from dealpoint.eval.deepeval_adapter import classify

    weak = {"vs_deterministic": {"task_completion_vs_grounded_accuracy": {"spearman": 0.1, "n": 50}}}
    label, msg = classify(weak, resolved_evaluator={"shares_model_with_judge_trio": True})
    assert label == "KEEP_OPTIONAL_ANALYSIS"
    assert "judge trio" in msg

    label2, _ = classify(weak, resolved_evaluator={"shares_model_with_judge_trio": False})
    assert label2 == "KEEP_CORE_DIAGNOSTIC"


def test_compute_coverage_counts_null_reasons_by_error_type():
    from dealpoint.eval.deepeval_adapter import compute_coverage

    rows = [
        {
            "task_completion": {"score": 0.8},
            "tool_correctness": {"score": None, "error_type": "RateLimitError"},
            "argument_correctness": {"score": 1.0},
            "step_efficiency": {"score": None, "error_type": "RateLimitError"},
        },
        {
            "task_completion": {"score": None, "error_type": "ValueError"},
            "tool_correctness": {"score": 1.0},
            "argument_correctness": {"score": None, "error_type": "unknown"},
            "step_efficiency": {"score": 0.5},
        },
    ]
    coverage = compute_coverage(rows)
    assert coverage["task_completion"] == {"n_scored": 1, "n_null": 1, "null_reasons": {"ValueError": 1}}
    assert coverage["tool_correctness"]["n_scored"] == 1
    assert coverage["step_efficiency"]["null_reasons"] == {"RateLimitError": 1}


def test_compare_to_judge_n_is_paired_non_none_count():
    from dealpoint.eval.deepeval_adapter import compare_to_judge

    deepeval_scores = [
        {"case_id": "c1", "variant_id": "v1", "step_efficiency": {"score": 0.7}, "task_completion": {"score": None}},
        {"case_id": "c2", "variant_id": "v1", "step_efficiency": {"score": None}, "task_completion": {"score": 0.9}},
    ]
    judge_means = {
        ("c1", "v1"): {"trajectory": 4.0, "reasoning": 3.0, "evidence": None, "professional": None},
        ("c2", "v1"): {"trajectory": 2.0, "reasoning": 4.0, "evidence": None, "professional": None},
    }
    result = compare_to_judge(deepeval_scores, judge_means)
    # step_efficiency is None for c2 -> only 1 paired value
    assert result["step_efficiency_vs_judge_trajectory"]["n"] == 1
    # task_completion is None for c1 -> only 1 paired value
    assert result["task_completion_vs_judge_reasoning"]["n"] == 1


def test_resolve_evaluator_model_records_provider_and_shares_model_flag(tmp_path, monkeypatch):
    from dealpoint.eval import deepeval_adapter as da

    slate_path = tmp_path / "judge_slate.json"
    slate_path.write_text(
        json.dumps(
            {
                "price_basis": "live",
                "judges": [
                    {"model": "mistralai/mistral-small-3.2-24b-instruct", "family": "Mistral", "ok": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    import dealpoint.config as config_mod

    monkeypatch.setattr(config_mod, "JUDGE_SLATE_PATH", slate_path)

    resolved = da.resolve_evaluator_model(env_value=None)
    assert resolved["provider"] == "mistralai"
    assert resolved["shares_model_with_judge_trio"] is True
    assert "deepeval_version" in resolved


def test_main_from_cache_regenerates_without_spending(tmp_path, monkeypatch):
    """`main(["--from-cache"])` must reproduce the cached report's spend,
    coverage and decisions unchanged, recomputing only comparisons (which
    are pure functions of already-stored data) -- and must never construct
    an OpenRouterClient or call any metric's `.measure(...)`.
    """
    import json as json_mod

    import dealpoint.config as config_mod
    from dealpoint.config import DEEPEVAL_CROSSCHECK_JSON_PATH
    from dealpoint.eval import deepeval_adapter as da

    if not DEEPEVAL_CROSSCHECK_JSON_PATH.exists():
        pytest.skip("deepeval_crosscheck.json not generated yet")
    original = json_mod.loads(DEEPEVAL_CROSSCHECK_JSON_PATH.read_text(encoding="utf-8"))

    tmp_json = tmp_path / "deepeval_crosscheck.json"
    tmp_md = tmp_path / "deepeval_crosscheck.md"
    tmp_json.write_text(json_mod.dumps(original), encoding="utf-8")
    monkeypatch.setattr(config_mod, "DEEPEVAL_CROSSCHECK_JSON_PATH", tmp_json)
    monkeypatch.setattr(da, "DEEPEVAL_CROSSCHECK_JSON_PATH", tmp_json)
    monkeypatch.setattr(config_mod, "DEEPEVAL_CROSSCHECK_MD_PATH", tmp_md)
    monkeypatch.setattr(da, "DEEPEVAL_CROSSCHECK_MD_PATH", tmp_md)

    def _boom(*args, **kwargs):
        raise AssertionError("main(--from-cache) must never touch OpenRouterClient")

    monkeypatch.setattr("dealpoint.llm.client.OpenRouterClient", _boom, raising=False)

    exit_code = da.main(["--from-cache"])
    assert exit_code == 0

    regenerated = json_mod.loads(tmp_json.read_text(encoding="utf-8"))
    assert regenerated["spend"] == original["spend"]
    assert regenerated["coverage"] == original["coverage"]
    assert regenerated["decisions"] == original["decisions"]
    assert regenerated["classification"] == original["classification"]
    assert regenerated["per_trace_scores"] == original["per_trace_scores"]
    assert tmp_md.exists()


def test_module_importable_without_deepeval(monkeypatch):
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "deepeval" or mod_name.startswith("deepeval."):
            monkeypatch.setitem(sys.modules, mod_name, None)
    import importlib

    import dealpoint.eval.deepeval_adapter as da

    importlib.reload(da)
    assert callable(da.row_to_test_case)
