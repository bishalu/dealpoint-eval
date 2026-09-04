"""Metered smoke: 2 dev cases, arm B, anthropic/claude-haiku-4.5 (spec §6.4).

Marked `gate_m1` and `needs_model` ONLY -- never `needs_network`, so the
factory's `-m "gate_m1 and not needs_network"` gate does not silently skip
this test forever. Skips cleanly (not an error) if `OPENROUTER_API_KEY` is
unset or the dense index has not been built.
"""

from __future__ import annotations

import os

import pytest

from dealpoint.config import (
    DEFAULT_MODEL,
    INDEX_DIR,
    INDEX_VERSION_TXT_PATH,
    M1_SMOKE_CASE_IDS,
    M1_SMOKE_MAX_USD,
    SPEND_LEDGER_PATH,
)
from dealpoint.data.canonical import normalise_quote


def _ledger_total() -> float:
    if not SPEND_LEDGER_PATH.exists():
        return 0.0
    import json

    total = 0.0
    for line in SPEND_LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += float(json.loads(line).get("usd", 0.0) or 0.0)
    return total


def _index_available() -> bool:
    return INDEX_DIR.exists() and INDEX_VERSION_TXT_PATH.exists()


@pytest.mark.gate_m1
@pytest.mark.needs_model
def test_m1_smoke_two_dev_cases_arm_b_haiku(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
    if not _index_available():
        pytest.skip("dense index not built; run `just index`")

    from dealpoint.agent.loop import run_agent
    from dealpoint.corpus.document import load_document
    from dealpoint.corpus.retrievers import DenseRetriever
    from dealpoint.data.questions import QUESTION_BY_ID
    from dealpoint.llm.client import OpenRouterClient

    case_lookup: dict[str, dict] = {}
    import json

    from dealpoint.config import DEV_JSONL_PATH

    with open(DEV_JSONL_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_lookup[row["case_id"]] = row

    cases = [case_lookup[cid] for cid in M1_SMOKE_CASE_IDS if cid in case_lookup]
    assert len(cases) == len(M1_SMOKE_CASE_IDS), "smoke case ids missing from dev.jsonl"

    client = OpenRouterClient()
    retriever = DenseRetriever()

    ledger_before = _ledger_total()

    n_execution_failed = 0
    for case in cases:
        doc = load_document(case["agreement_id"])
        question = QUESTION_BY_ID[case["question_id"]]
        finding, record = run_agent(case, doc, retriever, client, question, DEFAULT_MODEL)

        if record.status == "EXECUTION_FAILED":
            n_execution_failed += 1
            continue

        if record.status == "ANSWERED":
            assert finding is not None
            assert len(finding.evidence) >= 1
            has_verbatim = any(
                normalise_quote(ev.quote) in doc.text for ev in finding.evidence
            )
            assert has_verbatim, (
                f"case {case['case_id']}: no evidence quote is a verbatim substring "
                f"of the canonical text"
            )

    assert n_execution_failed == 0, "smoke expects 0 EXECUTION_FAILED"

    ledger_after = _ledger_total()
    delta = ledger_after - ledger_before
    assert delta > 0, "ledger did not gain rows with usd > 0 during the smoke"
    assert delta <= M1_SMOKE_MAX_USD, (
        f"smoke cost ${delta:.4f} exceeds the ${M1_SMOKE_MAX_USD} ceiling -- an "
        f"expensive smoke is a defect, not a threshold to raise"
    )
