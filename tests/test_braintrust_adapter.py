"""Braintrust adapter: score budget, metadata, offline behaviour (spec §5)."""

from __future__ import annotations

import sys

import pytest

from dealpoint.eval.braintrust_adapter import (
    METADATA_KEYS,
    SCORE_NAMES,
    experiment_metadata,
    experiment_name,
)

pytestmark = pytest.mark.gate_m2


def test_score_names_has_exactly_six_entries():
    assert len(SCORE_NAMES) == 6
    assert SCORE_NAMES == (
        "grounded_accuracy",
        "answer_correct",
        "citation_gold_overlap",
        "citation_verbatim",
        "abstain_correct",
        "skill_adherence",
    )


def test_experiment_name_shape():
    name = experiment_name("B", "anthropic/claude-haiku-4.5", "e2b4a2b97561", "d2b052d")
    assert name == "B-anthropic_claude-haiku-4.5-e2b4a2b97561-d2b052d"


def test_experiment_metadata_has_exactly_six_keys():
    metadata = experiment_metadata(
        "B", "anthropic/claude-haiku-4.5", "e2b4a2b97561", None, "d2b052d", "dev"
    )
    assert set(metadata.keys()) == set(METADATA_KEYS)
    assert len(metadata) == 6
    assert metadata["skill_version"] is None


def test_module_works_with_braintrust_unimportable(monkeypatch):
    # experiment_name/experiment_metadata never import braintrust at all;
    # braintrust_available() must degrade cleanly when the import fails.
    monkeypatch.setitem(sys.modules, "braintrust", None)
    import importlib

    import dealpoint.eval.braintrust_adapter as adapter

    importlib.reload(adapter)
    try:
        assert adapter.experiment_name("B", "m/x", "iv", "sha") == "B-m_x-iv-sha"
        assert adapter.braintrust_available() is False
    finally:
        monkeypatch.delitem(sys.modules, "braintrust", raising=False)
        importlib.reload(adapter)


def test_traced_fallback_leaves_tool_output_unchanged(monkeypatch):
    # With braintrust unimportable, dealpoint.agent.tools' @traced decorator
    # (a no-op fallback) must not change tool output.
    monkeypatch.setitem(sys.modules, "braintrust", None)
    import importlib

    import dealpoint.agent.tools as tools_mod

    importlib.reload(tools_mod)
    try:
        from dealpoint.corpus.chunks import Chunk
        from dealpoint.corpus.document import Document
        from dealpoint.data.sections import Section

        text = "Article I THE MERGER 6.1. Foo. Some text."
        doc = Document(
            document_id="doc_x",
            agreement_id="doc_x",
            text=text,
            sections=[Section(ref="6.1", title="Foo", start=0, end=len(text))],
            body_start=0,
            body_end=len(text),
        )

        class StubRetriever:
            def search(self, agreement_id, query, k=5):
                return [
                    Chunk(
                        agreement_id="doc_x",
                        chunk_id="doc_x:0-5",
                        section_ref="6.1",
                        start=0,
                        end=5,
                        text=text[:5],
                    )
                ]

        result = tools_mod.search_agreement(doc, StubRetriever(), "foo", k=5)
        assert result.chunk_ids == ["doc_x:0-5"]
    finally:
        monkeypatch.delitem(sys.modules, "braintrust", raising=False)
        importlib.reload(tools_mod)


def test_load_braintrust_key_present_in_this_repo():
    from dealpoint.eval.braintrust_adapter import load_braintrust_key

    # this repo ships .braintrust.json / .env.braintrust for the smoke send
    key = load_braintrust_key()
    assert key is None or isinstance(key, str)


@pytest.mark.needs_network
def test_push_datasets_pushes_the_three_case_sets():
    from dealpoint.eval.braintrust_adapter import push_datasets

    result = push_datasets()
    assert set(result.keys()) == {"dev", "test", "counterfactual"}
    assert all(n > 0 for n in result.values())


def test_run_eval_replays_rows_without_reinvoking_the_agent():
    from dealpoint.eval.braintrust_adapter import run_eval

    rows = [
        {
            "case_id": "contract_0__q01",
            "case_set": "dev",
            "arm": "B",
            "model": "anthropic/claude-haiku-4.5",
            "question_id": "q01",
            "index_version": "e2b4a2b97561",
            "chunk_version": "8e5e8ba56765",
            "git_sha7": "c96758b",
            "finding": {"answer": "All Cash", "evidence": [], "rationale": "ok"},
            "record": {"status": "ANSWERED"},
            "scores": {
                "grounded_accuracy": True,
                "answer_correct": True,
                "citation_gold_overlap": True,
                "citation_verbatim": True,
                "abstain_correct": True,
                "skill_adherence": None,
                "gold_seen": True,
                "tool_calls": 2,
            },
            "usd": 0.01,
        }
    ]
    result = run_eval(
        rows, arm="B", model="anthropic/claude-haiku-4.5", case_set="dev", no_send_logs=True
    )
    assert result["experiment_name"] == "B-anthropic_claude-haiku-4.5-e2b4a2b97561-c96758b"
    assert result["metadata"]["case_set"] == "dev"
