"""Retrieval sanity check (spec Definition of Done): NOT a quality gate.

Skips cleanly if the dense index or index_version.txt has not been built
yet. Samples 5 dev cases deterministically and asserts that at least one of
them gets a `search_agreement` hit overlapping its gold span.
"""

from __future__ import annotations

import json

import pytest

from dealpoint.config import DEV_JSONL_PATH, INDEX_DIR, INDEX_VERSION_TXT_PATH
from dealpoint.corpus.document import load_document


def _index_available() -> bool:
    return INDEX_DIR.exists() and INDEX_VERSION_TXT_PATH.exists()


def _sample_dev_cases(n: int = 5) -> list[dict]:
    rows = []
    with open(DEV_JSONL_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("gold_spans"):
                rows.append(row)
    rows.sort(key=lambda r: r["case_id"])
    # Deterministic sample: every Nth case rather than random, seed=42 is
    # already baked into how dev.jsonl itself was built.
    step = max(1, len(rows) // n)
    return rows[::step][:n]


@pytest.mark.gate_m1
def test_search_agreement_hits_gold_span_for_at_least_one_sampled_case(dataset_available):
    if not dataset_available:
        pytest.skip("dataset not present")
    if not _index_available():
        pytest.skip("dense index not built; run `just index`")

    from dealpoint.data.questions import QUESTION_BY_ID

    try:
        from dealpoint.corpus.retrievers import DenseRetriever
    except ImportError:
        pytest.skip("retrieval extra not installed")

    retriever = DenseRetriever()

    sampled = _sample_dev_cases(5)
    assert len(sampled) > 0, "no dev cases with gold spans found"

    any_hit = False
    for case in sampled:
        question = QUESTION_BY_ID[case["question_id"]]
        doc = load_document(case["agreement_id"])
        chunks = retriever.search(doc.document_id, question.canonical_query, k=5)
        gold_ranges = [(s["start"], s["end"]) for s in case["gold_spans"]]
        for chunk in chunks:
            for g_start, g_end in gold_ranges:
                overlap = min(chunk.end, g_end) - max(chunk.start, g_start)
                if overlap >= 50:
                    any_hit = True
                    break
            if any_hit:
                break
        if any_hit:
            break

    assert any_hit, "no sampled dev case's search_agreement hit its gold span (sanity check)"
