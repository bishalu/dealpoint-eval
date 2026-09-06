"""Evaluate every frozen M3 candidate on the frozen synthetic-query set.

Secondary and never benchmark truth (spec section 1.B): this module only
reads the already-frozen `data/eval/synthetic_dev_queries.jsonl`; it never
generates queries itself (that is `dealpoint.rag_lab.synthetic`'s job) and
never writes to it.
"""

from __future__ import annotations

import json

from dealpoint.config import SYNTHETIC_DEV_QUERIES_PATH
from dealpoint.corpus.chunks import chunk_document
from dealpoint.corpus.document import load_document
from dealpoint.corpus.retrievers import Retriever, Scorer, build_retriever


def load_synthetic_queries(path=SYNTHETIC_DEV_QUERIES_PATH) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_synthetic_set(
    *,
    dense: Retriever,
    sparse: Retriever,
    variants_by_query: dict[str, list[str]] | None = None,
    scorer: Scorer | None = None,
    k: int = 5,
) -> dict:
    """Every frozen M3 candidate, scored against the frozen synthetic queries.

    `expected_id` for a synthetic query is the SINGLE generating chunk id
    (the chunk the question was generated from) -- never a set, per the
    spec's "using the generating chunk id as the single expected_id".
    """
    from dealpoint.eval.tournament import DEFAULT_CONFIGS, _first_hit_rank

    rows = load_synthetic_queries()
    if not rows:
        return {
            "n_queries": 0,
            "results": {},
            "verdict": "no synthetic queries frozen yet",
        }

    results: dict[str, dict] = {}
    for config in DEFAULT_CONFIGS:
        retriever = build_retriever(
            config, dense=dense, sparse=sparse, variants_by_query=variants_by_query, scorer=scorer
        )
        li_hits: list[float] = []
        obj_hits: list[float] = []
        for row in rows:
            agreement_id = row["agreement_id"]
            expected_id = row["chunk_id"]
            chunks = retriever.search(agreement_id, row["question"], k=k)
            retrieved_ids = [c.chunk_id for c in chunks]
            li_hits.append(1.0 if expected_id in retrieved_ids else 0.0)

            doc_chunks = chunk_document(load_document(agreement_id))
            target_chunk = next((c for c in doc_chunks if c.chunk_id == expected_id), None)
            golds = [(target_chunk.start, target_chunk.end)] if target_chunk else []
            rank = _first_hit_rank(chunks, golds) if golds else None
            obj_hits.append(1.0 if rank is not None and rank <= 5 else 0.0)

        n = len(rows)
        results[config.name] = {
            "li_hit_rate": sum(li_hits) / n if n else 0.0,
            "obj_hit_at_5": sum(obj_hits) / n if n else 0.0,
            "n": n,
        }

    winner_name = "hybrid_rrf"
    winner_result = results.get(winner_name, {})
    winner_hit = winner_result.get("li_hit_rate")
    dense_hit = results.get("dense", {}).get("li_hit_rate")
    stays_strong = (
        winner_hit is not None
        and dense_hit is not None
        and winner_hit >= dense_hit
    )
    verdict = (
        f"Frozen winner `{winner_name}` "
        f"{'stays strong' if stays_strong else 'does NOT clearly stay strong'} under the "
        f"synthetic query distribution: li_hit_rate={winner_hit} vs dense={dense_hit}."
    )

    return {
        "n_queries": len(rows),
        "results": results,
        "verdict": verdict,
    }
