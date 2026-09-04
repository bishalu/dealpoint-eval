"""One-off dense-index builder: `python -m dealpoint.corpus.build_index [--force]`.

Embeds and indexes the 20 selected agreements + the 30 redacted counterfactual
variants into Qdrant local (on-disk) mode under `data/index/`, then writes
`data/reports/index_version.txt`, `data/reports/versions.json`, and
`data/reports/chunk_report.json`.

This is deliberately a CLI, never something a test imports and runs: building
the index downloads ONNX embedding weights and takes several minutes over
~12k chunks on a CPU-only VM.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
import sys
import time
from collections.abc import Sequence

from dealpoint.config import (
    CHUNK_REPORT_PATH,
    COUNTERFACTUAL_JSONL_PATH,
    DATASET_VERSION_TXT_PATH,
    EMBEDDING_MODEL,
    INDEX_DIR,
    INDEX_VERSION_TXT_PATH,
    PARSER_VERSION_TXT_PATH,
    QDRANT_COLLECTION,
    SELECTION_PATH,
    VERSIONS_JSON_PATH,
)
from dealpoint.corpus.chunks import Chunk, chunk_document, chunk_version
from dealpoint.corpus.document import load_document
from dealpoint.corpus.retrievers import index_version
from dealpoint.data.sections import PARSER_VERSION


def _resolve_document_ids() -> list[str]:
    with open(SELECTION_PATH, encoding="utf-8") as fh:
        selection = json.load(fh)
    document_ids = list(selection["selected"])

    with open(COUNTERFACTUAL_JSONL_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") == "redacted":
                document_ids.append(row["case_id"])

    return sorted(set(document_ids))


def _chunk_report(all_chunks: list[Chunk]) -> dict:
    from fastembed import TextEmbedding

    embedder = TextEmbedding(EMBEDDING_MODEL)
    sample = all_chunks[:: max(1, len(all_chunks) // 500)][:500] or all_chunks
    char_counts = [len(c.text) for c in sample]
    token_counts = [embedder.token_count(c.text) for c in sample]
    ratios = [c / t for c, t in zip(char_counts, token_counts) if t > 0]

    def dist(values: Sequence[float]) -> dict:
        if not values:
            return {"mean": None, "median": None, "p90": None}
        ordered = sorted(values)
        p90_idx = min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p90": ordered[p90_idx],
        }

    return {
        "n_sampled_chunks": len(sample),
        "n_total_chunks": len(all_chunks),
        "chars_per_token_ratio_mean": statistics.mean(ratios) if ratios else None,
        "chunk_chars": dist(char_counts),
        "chunk_tokens": dist(token_counts),
    }


def build_index(force: bool = False) -> None:
    from qdrant_client import QdrantClient, models

    if force and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    document_ids = _resolve_document_ids()
    print(f"Resolved {len(document_ids)} documents to index")

    all_chunks: list[Chunk] = []
    for doc_id in document_ids:
        doc = load_document(doc_id)
        all_chunks.extend(chunk_document(doc))
    print(f"Chunked into {len(all_chunks)} chunks")

    from fastembed import TextEmbedding

    vector_size = TextEmbedding.get_embedding_size(EMBEDDING_MODEL)

    client = QdrantClient(path=str(INDEX_DIR))
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )

    # fastembed's ONNX session grows resident memory over a long cumulative
    # run (measured during build: the process was OOM-killed partway
    # through this ~14k-chunk corpus with a single long-lived embedder).
    # Recreating the embedder every GROUP_SIZE texts keeps resident memory
    # from climbing unbounded (measured: plateaus instead of growing once
    # past the first couple of groups).
    GROUP_SIZE = 300
    UPSERT_BATCH = 64
    point_id = 0
    n_embedded = 0
    embedder = None
    for group_start in range(0, len(all_chunks), GROUP_SIZE):
        group_chunks = all_chunks[group_start : group_start + GROUP_SIZE]
        if embedder is None:
            embedder = TextEmbedding(EMBEDDING_MODEL)
        group_texts = [c.text for c in group_chunks]
        group_vectors = list(embedder.passage_embed(group_texts, batch_size=UPSERT_BATCH))

        batch: list[models.PointStruct] = []
        for chunk, vector in zip(group_chunks, group_vectors):
            batch.append(
                models.PointStruct(
                    id=point_id,
                    vector=list(vector),
                    payload={
                        "agreement_id": chunk.agreement_id,
                        "chunk_id": chunk.chunk_id,
                        "section_ref": chunk.section_ref,
                        "start": chunk.start,
                        "end": chunk.end,
                        "text": chunk.text,
                    },
                )
            )
            point_id += 1
            if len(batch) >= UPSERT_BATCH:
                client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
                batch = []
        if batch:
            client.upsert(collection_name=QDRANT_COLLECTION, points=batch)

        n_embedded += len(group_chunks)
        # Recreate the embedder every group (see comment above) and force a
        # GC pass so the freed ONNX session's memory is actually reclaimed.
        del embedder, group_vectors, group_texts
        embedder = None  # noqa: F841 - reassigned so the next loop iteration recreates it
        gc.collect()
        print(
            f"  embedded {n_embedded}/{len(all_chunks)} chunks "
            f"({time.time() - t0:.1f}s elapsed)",
            flush=True,
        )

    elapsed = time.time() - t0

    idx_version = index_version(collection=QDRANT_COLLECTION)
    INDEX_VERSION_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_VERSION_TXT_PATH, "w", encoding="utf-8") as fh:
        fh.write(idx_version + "\n")

    dataset_version = ""
    if DATASET_VERSION_TXT_PATH.exists():
        dataset_version = DATASET_VERSION_TXT_PATH.read_text(encoding="utf-8").strip()
    parser_version_on_disk = PARSER_VERSION
    if PARSER_VERSION_TXT_PATH.exists():
        parser_version_on_disk = PARSER_VERSION_TXT_PATH.read_text(encoding="utf-8").strip()

    versions_payload = {
        "parser_version": parser_version_on_disk,
        "dataset_version": dataset_version,
        "chunk_version": chunk_version(),
        "index_version": idx_version,
        "embedding_model": EMBEDDING_MODEL,
        "n_documents": len(document_ids),
        "n_chunks": len(all_chunks),
    }
    with open(VERSIONS_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(versions_payload, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")

    report = _chunk_report(all_chunks)
    CHUNK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNK_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"documents={len(document_ids)} chunks={len(all_chunks)} index_version={idx_version}")
    print(f"elapsed={elapsed:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dealpoint.corpus.build_index")
    parser.add_argument("--force", action="store_true", help="rebuild the index from scratch")
    args = parser.parse_args(argv)
    build_index(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
