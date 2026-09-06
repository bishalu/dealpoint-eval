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


def test_module_importable_without_deepeval(monkeypatch):
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "deepeval" or mod_name.startswith("deepeval."):
            monkeypatch.setitem(sys.modules, mod_name, None)
    import importlib

    import dealpoint.eval.deepeval_adapter as da

    importlib.reload(da)
    assert callable(da.row_to_test_case)
