"""M5 Braintrust judged-subset path: offline, no_send_logs=True (spec deliverable 6).

Mirrors `tests/test_braintrust_adapter.py::test_run_eval_replays_rows_without_reinvoking_the_agent`
but drives `run_judge_eval` over fake judge-aggregate rows instead of `run_eval` over fake
agent-result rows. Skips cleanly when `braintrust` (or a key) is unavailable so this stays
green on a machine without the package.
"""

from __future__ import annotations

import json

import pytest

from dealpoint.eval.braintrust_adapter import (
    BRAINTRUST_RUNS_PATH,
    SCORE_NAMES,
    braintrust_available,
    experiment_name,
)

pytestmark = pytest.mark.gate_m5


def _row(
    packet_id: str,
    case_id: str,
    *,
    reasoning: float | None = 4.0,
    evidence: float | None = 3.0,
    trajectory: float | None = 4.0,
    professional: float | None = 4.0,
    status: str = "ANSWERED",
    grounded_accuracy: bool | None = True,
) -> dict:
    return {
        "packet_id": packet_id,
        "case_id": case_id,
        "question_id": "q01",
        "rubric_version": "abc123def456",
        "status": status,
        "grounded_accuracy": grounded_accuracy,
        "index_version": "e2b4a2b97561",
        "git_sha7": "e3ee9cc",
        "dimension_means": {
            "reasoning": reasoning,
            "evidence": evidence,
            "trajectory": trajectory,
            "professional": professional,
        },
        "per_judge_scores": {
            "mistralai/mistral-small-3.2-24b-instruct": {
                "reasoning": 4,
                "evidence": 3,
                "trajectory": 4,
                "professional": 4,
            },
            "nvidia/nemotron-3-super-120b-a12b": {
                "reasoning": 4,
                "evidence": 3,
                "trajectory": 4,
                "professional": 4,
            },
        },
    }


def _fake_rows() -> list[dict]:
    return [
        _row("packet_aaa", "contract_1__q01"),
        _row("packet_bbb", "contract_2__q01", reasoning=3.0, evidence=None, status="CAP_HIT", grounded_accuracy=None),
        _row("packet_ccc", "contract_3__q01", reasoning=2.0, evidence=2.0, trajectory=3.0, professional=3.0),
    ]


def test_run_judge_eval_offline_no_send():
    if not braintrust_available():
        pytest.skip("braintrust package/key not available in this environment")

    from dealpoint.eval.braintrust_adapter import JUDGE_SCORE_NAMES, run_judge_eval

    # (1) JUDGE_SCORE_NAMES is the four-dimension judge budget; SCORE_NAMES
    # (the M2 six-score budget) is untouched by this M5 extension.
    assert JUDGE_SCORE_NAMES == (
        "judge_reasoning",
        "judge_evidence",
        "judge_trajectory",
        "judge_professional",
    )
    assert len(SCORE_NAMES) == 6
    assert SCORE_NAMES == (
        "grounded_accuracy",
        "answer_correct",
        "citation_gold_overlap",
        "citation_verbatim",
        "abstain_correct",
        "skill_adherence",
    )

    runs_path_existed = BRAINTRUST_RUNS_PATH.exists()
    before_bytes = BRAINTRUST_RUNS_PATH.read_bytes() if runs_path_existed else None

    rows = _fake_rows()
    result = run_judge_eval(
        rows,
        arm="D",
        model="anthropic/claude-haiku-4.5",
        case_set="judged_subset_m5",
        no_send_logs=True,
    )

    # (2) experiment_name shape: judged-{experiment_name(...)}.
    expected_name = f"judged-{experiment_name('D', 'anthropic/claude-haiku-4.5', 'e2b4a2b97561', 'e3ee9cc')}"
    assert result["experiment_name"].startswith("judged-")
    assert result["experiment_name"] == expected_name

    # (5) no_send_logs=True must never append to braintrust_runs.json.
    after_bytes = BRAINTRUST_RUNS_PATH.read_bytes() if BRAINTRUST_RUNS_PATH.exists() else None
    assert after_bytes == before_bytes


def test_row_to_judge_eval_case_normalises_1_5_mean_to_0_1_and_none_stays_none():
    from dealpoint.eval.braintrust_adapter import _row_to_judge_eval_case

    row = _row("packet_ddd", "contract_4__q01", reasoning=3.0, evidence=None)
    eval_case = _row_to_judge_eval_case(row)

    # (3) (mean - 1) / 4: 3.0 -> 0.5; a None mean stays None, never 0.
    assert eval_case["input"]["judge_scores_0_1"]["reasoning"] == pytest.approx(0.5)
    assert eval_case["input"]["judge_scores_0_1"]["evidence"] is None


def test_row_to_judge_eval_case_metadata_carries_per_judge_and_raw_means_only():
    from dealpoint.eval.braintrust_adapter import _row_to_judge_eval_case

    row = _row("packet_eee", "contract_5__q01")
    eval_case = _row_to_judge_eval_case(row)

    # (4) per-case metadata carries the per-judge breakdown and the raw 1-5
    # means; the four LOGGED scores (input.judge_scores_0_1) carry only the
    # aggregate -- no individual judge model id appears there.
    metadata = eval_case["metadata"]
    assert metadata["per_judge_scores"] == row["per_judge_scores"]
    assert metadata["dimension_means_1_5"] == row["dimension_means"]

    input_payload_text = json.dumps(eval_case["input"])
    for judge_model in row["per_judge_scores"]:
        assert judge_model not in input_payload_text


def test_run_judge_eval_never_calls_record_braintrust_run_when_no_send_logs(monkeypatch):
    if not braintrust_available():
        pytest.skip("braintrust package/key not available in this environment")

    import dealpoint.eval.braintrust_adapter as adapter

    called = {"n": 0}

    def _fail_if_called(*args, **kwargs):
        called["n"] += 1

    monkeypatch.setattr(adapter, "_record_braintrust_run", _fail_if_called)

    adapter.run_judge_eval(
        _fake_rows(),
        arm="D",
        model="anthropic/claude-haiku-4.5",
        case_set="judged_subset_m5",
        no_send_logs=True,
    )

    assert called["n"] == 0
