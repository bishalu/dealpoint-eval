"""The eval runner: `--fake` end-to-end for both arms (spec §4)."""

from __future__ import annotations

import json

import pytest

from dealpoint.eval.run import result_stem, run_eval_set
from dealpoint.eval.scorers import SCORE_FIELD_NAMES

pytestmark = pytest.mark.gate_m2


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@pytest.mark.parametrize("arm", ["A", "B", "C", "D"])
def test_fake_runner_produces_result_rows_with_every_score(tmp_path, arm, dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    summary = run_eval_set(
        case_set="dev",
        arm=arm,
        model="anthropic/claude-haiku-4.5",
        limit=3,
        fake=True,
        out_dir=tmp_path,
    )
    results_path = tmp_path / f"{result_stem('dev', arm, 'anthropic/claude-haiku-4.5', 'offline', summary['git_sha7'])}.jsonl"
    assert results_path.exists()
    rows = _read_jsonl(results_path)
    assert len(rows) == 3

    for row in rows:
        assert set(row["scores"].keys()) == set(SCORE_FIELD_NAMES)
        assert "usd" in row
        assert row["case_set"] == "dev"
        assert row["arm"] == arm

    assert summary["est_usd"] is None  # --fake never estimates a real cost
    assert "realized_usd" in summary
    assert "majority_baseline" in summary
    assert summary["n_cases"] == 3


def test_fake_runner_writes_no_ledger_rows(tmp_path, dataset_available, monkeypatch):
    if not dataset_available:
        pytest.skip("dataset not present")
    ledger_path = tmp_path / "spend_ledger.jsonl"
    monkeypatch.setattr("dealpoint.config.SPEND_LEDGER_PATH", ledger_path)
    run_eval_set(
        case_set="dev",
        arm="B",
        model="anthropic/claude-haiku-4.5",
        limit=2,
        fake=True,
        out_dir=tmp_path / "results",
    )
    assert not ledger_path.exists()


def test_fake_runner_cases_arg_selects_specific_ids(tmp_path, dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    summary = run_eval_set(
        case_set="dev",
        arm="B",
        model="anthropic/claude-haiku-4.5",
        cases="contract_0__q01,contract_0__q06",
        fake=True,
        out_dir=tmp_path,
    )
    assert summary["n_cases"] == 2
    stem = result_stem("dev", "B", "anthropic/claude-haiku-4.5", "offline", summary["git_sha7"])
    rows = _read_jsonl(tmp_path / f"{stem}.jsonl")
    assert {r["case_id"] for r in rows} == {"contract_0__q01", "contract_0__q06"}


def test_offline_chunk_retriever_returns_first_k_chunks(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    from dealpoint.eval.run import OfflineChunkRetriever

    retriever = OfflineChunkRetriever()
    chunks = retriever.search("contract_0", "irrelevant query", k=3)
    assert len(chunks) <= 3
    assert all(c.agreement_id == "contract_0" for c in chunks)
