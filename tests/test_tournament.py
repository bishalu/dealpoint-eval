"""M3 gate: `dealpoint.eval.tournament` — synthetic tiny run, determinism,
the test-set prohibition, and the committed full-dev-set report + freeze
(spec §5.2).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dealpoint.config import (
    ARM_C_RETRIEVER,
    INDEX_VERSION_TXT_PATH,
    VERSIONS_JSON_PATH,
)
from dealpoint.corpus.chunks import Chunk, chunk_version
from dealpoint.corpus.retrievers import (
    DenseRetriever,
    arm_c_index_version,
    index_version,
)
from dealpoint.data.questions import QUESTION_BY_ID
from dealpoint.eval import tournament as tournament_module
from dealpoint.eval.tournament import DEFAULT_CONFIGS, main, render_markdown, run_tournament

pytestmark = pytest.mark.gate_m3


def _fake_embed(text: str, dims: int = 8) -> list[float]:
    vec = [0.0] * dims
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dims] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


class FakeEmbedder:
    def query_embed(self, texts):
        return [_fake_embed(t) for t in texts]

    def passage_embed(self, texts, batch_size=64):
        return [_fake_embed(t) for t in texts]


def _make_corpus() -> list[Chunk]:
    doc_a_texts = [
        "apple banana orchard fruit harvest season begins early autumn",
        "cherry date fig grove seasonal picking crew arrives each year",
        "grape honeydew melon vineyard summer irrigation schedule set",
        "kiwi lemon lime citrus grove watered daily during dry months",
        "mango nectarine papaya tropical greenhouse climate controlled",
        "orange peach pear autumn harvest festival held every October",
    ]
    doc_b_texts = [
        "rocket satellite orbit launch pad crew final countdown begins",
        "telescope nebula galaxy observatory night sky survey program",
        "asteroid comet meteor shower visible from northern hemisphere",
        "spaceship engine thruster fuel tank pressure nominal readings",
        "astronaut spacesuit oxygen tank pressure checked before walk",
        "planet moon eclipse solar system alignment rare event tonight",
    ]
    chunks: list[Chunk] = []
    for doc_id, texts in (("doc_a", doc_a_texts), ("doc_b", doc_b_texts)):
        offset = 0
        for text in texts:
            start = offset
            end = start + len(text)
            chunks.append(
                Chunk(
                    agreement_id=doc_id,
                    chunk_id=f"{doc_id}:{start}-{end}",
                    section_ref="",
                    start=start,
                    end=end,
                    text=text,
                )
            )
            offset = end + 1
    return chunks


def _build_dense_retriever(chunks: list[Chunk]) -> DenseRetriever:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="test_m3_tournament",
        vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(
            id=i,
            vector=_fake_embed(c.text),
            payload={
                "agreement_id": c.agreement_id,
                "chunk_id": c.chunk_id,
                "section_ref": c.section_ref,
                "start": c.start,
                "end": c.end,
                "text": c.text,
            },
        )
        for i, c in enumerate(chunks)
    ]
    client.upsert(collection_name="test_m3_tournament", points=points)
    return DenseRetriever(client=client, collection="test_m3_tournament", embedder=FakeEmbedder())


class BM25LikeRetriever:
    """Deterministic offline stand-in for BM25: term-overlap ranking, no bm25s import."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]:
        q_tokens = set(query.lower().split())
        candidates = [c for c in self._chunks if c.agreement_id == agreement_id]

        def score(c: Chunk) -> int:
            return len(q_tokens & set(c.text.lower().split()))

        ranked = sorted(candidates, key=lambda c: (-score(c), c.chunk_id))
        return ranked[:k]


def _fake_scorer(query: str, texts: list[str]) -> list[float]:
    # Deterministic, cheap: longer text scores lower so ordering is stable
    # and non-trivial (never a real cross-encoder).
    return [float(-len(t)) for t in texts]


def _make_synthetic_cases() -> tuple[list[dict], list[Chunk]]:
    chunks = _make_corpus()
    question_ids = ["q01", "q02", "q03", "q04", "q05"]
    cases: list[dict] = []
    for i in range(10):
        doc_id = "doc_a" if i % 2 == 0 else "doc_b"
        doc_chunks = [c for c in chunks if c.agreement_id == doc_id]
        target = doc_chunks[i % len(doc_chunks)]
        qid = question_ids[i % len(question_ids)]
        cases.append(
            {
                "case_id": f"synthetic_{i}",
                "agreement_id": doc_id,
                "question_id": qid,
                "gold_spans": [{"start": target.start, "end": target.end}],
            }
        )
    return cases, chunks


def _build_configs_and_retrievers():
    cases, chunks = _make_synthetic_cases()
    dense = _build_dense_retriever(chunks)
    sparse = BM25LikeRetriever(chunks)
    variants_by_query: dict[str, list[str]] = {}
    for qid in {c["question_id"] for c in cases}:
        q = QUESTION_BY_ID[qid]
        variants_by_query[q.canonical_query] = [q.canonical_query + " variant one"]
        variants_by_query[q.maud_question] = [q.maud_question + " variant one"]
    return cases, dense, sparse, variants_by_query


def test_run_tournament_synthetic_end_to_end():
    cases, dense, sparse, variants_by_query = _build_configs_and_retrievers()
    configs = list(DEFAULT_CONFIGS)

    report = run_tournament(
        cases,
        configs,
        dense=dense,
        sparse=sparse,
        variants_by_query=variants_by_query,
        scorer=_fake_scorer,
        k=10,
    )

    results = report["results"]
    assert set(results.keys()) == {c.name for c in configs}
    for cfg in configs:
        for query_type in ("canonical", "maud"):
            row = results[cfg.name][query_type]
            for key in ("hit_at_5", "hit_at_10", "mrr", "n_cases", "wall_seconds"):
                assert key in row
            assert 0.0 <= row["mrr"] <= 1.0
            assert row["hit_at_5"] <= row["hit_at_10"]
            assert row["n_cases"] == len(cases)

    per_case = report["per_case"]
    assert len(per_case) == len(cases) * len(configs) * 2
    stage_keys = {"dense_rank", "bm25_rank", "fused_rank", "reranked_rank", "surfaced_by", "rerank_effect"}
    for row in per_case:
        assert stage_keys == set(row["stages"].keys())

    winner = report["winner"]
    assert winner["config"] in {c.name for c in configs}


def test_wall_seconds_charges_rerank_configs_not_the_stage_pass():
    """Regression: a config's `wall_seconds` must measure its own
    `retriever.search` cost, not the config-independent stage-instrumentation
    pass (`_compute_stages`), which used to be timed inside the first
    config's loop and then served free to every later config via a shared
    memoized scorer.
    """
    import time

    cases, dense, sparse, variants_by_query = _build_configs_and_retrievers()
    configs = list(DEFAULT_CONFIGS)

    def slow_scorer(query: str, texts: list[str]) -> list[float]:
        time.sleep(0.05)
        return [float(-len(t)) for t in texts]

    report = run_tournament(
        cases,
        configs,
        dense=dense,
        sparse=sparse,
        variants_by_query=variants_by_query,
        scorer=slow_scorer,
        k=10,
    )
    results = report["results"]

    assert (
        results["hybrid_rrf_rerank"]["canonical"]["wall_seconds"]
        > results["hybrid_rrf"]["canonical"]["wall_seconds"]
    )
    assert (
        results["dense"]["canonical"]["wall_seconds"]
        < results["multi_query_fusion_rerank"]["canonical"]["wall_seconds"]
    )


def test_run_tournament_is_deterministic():
    configs = list(DEFAULT_CONFIGS)

    def run_once():
        # Fresh dense/sparse instances each run: a single Qdrant :memory:
        # client can be reused, but rebuilding keeps this test independent
        # of any shared-state assumption.
        cases2, dense2, sparse2, variants2 = _build_configs_and_retrievers()
        report = run_tournament(
            cases2,
            configs,
            dense=dense2,
            sparse=sparse2,
            variants_by_query=variants2,
            scorer=_fake_scorer,
            k=10,
        )
        for cfg_results in report["results"].values():
            for row in cfg_results.values():
                row.pop("wall_seconds", None)
        return report["results"], report["per_case"]

    results_a, per_case_a = run_once()
    results_b, per_case_b = run_once()
    assert results_a == results_b
    assert per_case_a == per_case_b


def test_render_markdown_from_report():
    cases, dense, sparse, variants_by_query = _build_configs_and_retrievers()
    configs = list(DEFAULT_CONFIGS)
    report = run_tournament(
        cases,
        configs,
        dense=dense,
        sparse=sparse,
        variants_by_query=variants_by_query,
        scorer=_fake_scorer,
        k=10,
    )
    report["generated_from"] = {
        "chunk_version": "x",
        "index_version": "y",
        "dataset_version": "z",
        "git_sha7": "abc1234",
        "n_cases": len(cases),
        "case_set": "dev",
    }
    report["llamaindex"] = {"adopted": False, "measured_install_mb": 178, "marginal_mb": 100, "reason": "r"}
    md = render_markdown(report)
    assert isinstance(md, str) and len(md) > 0
    assert "canonical queries" in md
    assert "maud queries" in md


# --- The test.jsonl prohibition, two independent enforcements --------------


def test_source_never_names_the_frozen_holdout_set():
    src = Path(tournament_module.__file__).read_text(encoding="utf-8")
    assert "test.jsonl" not in src
    assert "TEST_JSONL_PATH" not in src


def test_main_refuses_set_test():
    with pytest.raises(SystemExit) as exc_info:
        main(["--set", "test"])
    assert exc_info.value.code != 0


def test_main_refuses_set_counterfactual():
    with pytest.raises(SystemExit) as exc_info:
        main(["--set", "counterfactual"])
    assert exc_info.value.code != 0


# --- The committed report ----------------------------------------------------


def test_committed_tournament_json_exists_and_covers_full_dev_set():
    from dealpoint.config import TOURNAMENT_JSON_PATH

    assert TOURNAMENT_JSON_PATH.exists(), (
        "data/reports/tournament.json must be committed (DoD requires the artifact)"
    )
    with open(TOURNAMENT_JSON_PATH, encoding="utf-8") as fh:
        report = json.load(fh)

    generated_from = report["generated_from"]
    assert generated_from["case_set"] == "dev"
    assert generated_from["n_cases"] == 58
    assert generated_from["chunk_version"] == chunk_version()

    results = report["results"]
    assert len(results) == 6
    for config_results in results.values():
        assert set(config_results.keys()) == {"canonical", "maud"}

    winner = report["winner"]
    assert winner["hit_at_5"] >= winner["dense_hit_at_5"]


def test_committed_tournament_md_exists_and_nonempty():
    from dealpoint.config import TOURNAMENT_MD_PATH

    assert TOURNAMENT_MD_PATH.exists()
    content = TOURNAMENT_MD_PATH.read_text(encoding="utf-8")
    assert len(content.strip()) > 0


# --- The freeze --------------------------------------------------------------


def test_arm_c_freeze_consistent_with_tournament_winner():
    from dealpoint.config import TOURNAMENT_JSON_PATH

    assert INDEX_VERSION_TXT_PATH.exists()
    on_disk_index_version = INDEX_VERSION_TXT_PATH.read_text(encoding="utf-8").strip()
    assert on_disk_index_version == arm_c_index_version()

    assert VERSIONS_JSON_PATH.exists()
    with open(VERSIONS_JSON_PATH, encoding="utf-8") as fh:
        versions = json.load(fh)
    assert versions["arm_c_index_version"] == arm_c_index_version()
    assert versions["arm_c_retriever"] == ARM_C_RETRIEVER
    assert versions["index_version"] == "e2b4a2b97561"
    assert index_version() == "e2b4a2b97561"

    with open(TOURNAMENT_JSON_PATH, encoding="utf-8") as fh:
        report = json.load(fh)
    assert ARM_C_RETRIEVER["name"] == report["winner"]["config"]
