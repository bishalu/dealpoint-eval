"""Native LlamaIndex evaluation on canonical dev retrieval, beside project metrics.

MAUD gold-span overlap remains benchmark truth. `gold_bearing_chunk_ids` uses
`dealpoint.eval.scorers.overlap_chars`/`gold_ranges`/`MIN_GOLD_OVERLAP_CHARS`
verbatim -- the exact rule `dealpoint.eval.tournament._first_hit_rank` already
uses to decide whether a chunk is a "hit". These ARE `expected_ids`: nothing
here reimplements the geometry.
"""

from __future__ import annotations

import hashlib
import json

from dealpoint.config import VERSIONS_JSON_PATH
from dealpoint.corpus.chunks import Chunk, chunk_document, chunk_version
from dealpoint.corpus.document import load_document
from dealpoint.corpus.retrievers import Retriever, RetrieverConfig, Scorer, index_version
from dealpoint.eval.scorers import MIN_GOLD_OVERLAP_CHARS, gold_ranges, overlap_chars

Range = tuple[int, int]

# Fixed disagreement-cause vocabulary (spec section 3.2).
#
# "reranking" is deliberately absent. The spec lists it as a candidate cause
# label, but `find_disagreements` never computes a pre-rerank ranking to
# compare against (that would require threading the fused-but-unreranked
# chunk list through `evaluate_retriever_obj`/`evaluate_retriever_li`, which
# neither function currently exposes), so `classify_disagreement`'s
# `rerank_would_hit`/`current_hit` parameters are never supplied by any
# caller and the branch that would emit "reranking" is unreachable outside a
# unit test. Describing a rule the pipeline cannot apply would misrepresent
# the evidence, so the label is dropped rather than left as dead code path.
CAUSE_VOCAB: tuple[str, ...] = (
    "chunk_identity",
    "partial_overlap",
    "duplicate_relevant_chunks",
    "section_boundary",
)

CAUSE_RULE_TEXT = (
    "Deterministic disagreement-cause rule, applied in this priority order to a "
    "disagreeing (case, config, direction): "
    "'partial_overlap' when the top-ranked retrieved chunk overlaps a gold span by "
    f"0 < overlap < MIN_GOLD_OVERLAP_CHARS ({MIN_GOLD_OVERLAP_CHARS}) chars; "
    "'duplicate_relevant_chunks' when >= 2 chunks in the case's full chunk list each "
    f"overlap a gold span by >= {MIN_GOLD_OVERLAP_CHARS} chars (expected_ids has >= 2 "
    "entries), so which one a retriever happens to surface first is underdetermined; "
    "'section_boundary' when a single gold span is covered (any overlap > 0) by chunks "
    "carrying more than one distinct section_ref; "
    "'chunk_identity' otherwise (the default: the two metrics disagree for a reason not "
    "captured by the other three labels -- typically the retrieved list and expected_ids "
    "simply differ). A 'reranking' label was considered (spec section 3.2) but is not "
    "emitted: the pipeline never computes the pre-rerank ranking needed to detect it, so "
    "including it would describe a rule this code cannot actually apply."
)


def assert_versions_match(versions_path=VERSIONS_JSON_PATH) -> dict:
    """Raise if the live chunk/index identity has drifted from the frozen M3 stamp.

    This is the spec's "same chunks and indexes (chunk_version/index_version
    asserted)" requirement -- LlamaIndex must evaluate the identical frozen
    artifacts the M3 tournament and M4-M6 sweeps used, never a re-derived one.
    """
    payload = json.loads(versions_path.read_text(encoding="utf-8"))
    live_chunk = chunk_version()
    live_index = index_version()
    if payload.get("chunk_version") != live_chunk:
        raise RuntimeError(
            f"chunk_version mismatch: {versions_path} has {payload.get('chunk_version')!r}, "
            f"live is {live_chunk!r} -- the M3 frozen chunks must not move under M7a"
        )
    if payload.get("index_version") != live_index:
        raise RuntimeError(
            f"index_version mismatch: {versions_path} has {payload.get('index_version')!r}, "
            f"live is {live_index!r} -- the M3 frozen index must not move under M7a"
        )
    return {"chunk_version": live_chunk, "index_version": live_index}


def gold_bearing_chunk_ids(case: dict, chunks: list[Chunk]) -> list[str]:
    """Chunk ids overlapping >= MIN_GOLD_OVERLAP_CHARS of any of `case`'s gold spans.

    The canonical rule, verbatim from `dealpoint.eval.tournament._first_hit_rank`
    (which itself calls `dealpoint.eval.scorers.overlap_chars`) -- this IS
    `expected_ids` for the LlamaIndex `RetrieverEvaluator`. A case with no gold
    spans yields `[]`.
    """
    golds = gold_ranges(case)
    if not golds:
        return []
    ids: list[str] = []
    for chunk in chunks:
        span = (chunk.start, chunk.end)
        if any(overlap_chars(span, g) >= MIN_GOLD_OVERLAP_CHARS for g in golds):
            ids.append(chunk.chunk_id)
    return ids


def _canonical_query_for(case: dict) -> str:
    from dealpoint.data.questions import QUESTION_BY_ID

    return QUESTION_BY_ID[case["question_id"]].canonical_query


def evaluate_retriever_li(
    cases: list[dict],
    config: RetrieverConfig,
    *,
    dense: Retriever,
    sparse: Retriever,
    variants_by_query: dict[str, list[str]] | None = None,
    scorer: Scorer | None = None,
    k: int = 5,
) -> dict:
    """LlamaIndex `RetrieverEvaluator` hit_rate/mrr for one frozen config, over `cases`.

    Cases with no gold spans (and therefore no `expected_ids`) are skipped, same
    convention as `dealpoint.eval.tournament.evaluate_config`. One
    `ProjectRetrieverAdapter` per case (agreement_id varies per case; the
    underlying `Retriever` -- and any Qdrant client it holds -- is shared and
    built exactly once by the caller).
    """
    from llama_index.core.evaluation import RetrieverEvaluator

    from dealpoint.eval.cases import resolve_document_id
    from dealpoint.rag_lab.adapters import build_li_retriever

    hits: list[float] = []
    mrrs: list[float] = []
    per_case: list[dict] = []
    for case in cases:
        golds = gold_ranges(case)
        if not golds:
            continue
        doc_id = resolve_document_id(case)
        chunks = chunk_document(load_document(doc_id))
        expected_ids = gold_bearing_chunk_ids(case, chunks)
        if not expected_ids:
            continue
        query = _canonical_query_for(case)
        adapter = build_li_retriever(
            config,
            dense=dense,
            sparse=sparse,
            variants_by_query=variants_by_query,
            scorer=scorer,
            agreement_id=doc_id,
            k=k,
        )
        evaluator = RetrieverEvaluator.from_metric_names(["hit_rate", "mrr"], retriever=adapter)
        result = evaluator.evaluate(query=query, expected_ids=expected_ids)
        vals = result.metric_vals_dict
        hits.append(vals["hit_rate"])
        mrrs.append(vals["mrr"])
        per_case.append(
            {
                "case_id": case["case_id"],
                "expected_ids": expected_ids,
                "retrieved_ids": list(result.retrieved_ids),
                "hit_rate": vals["hit_rate"],
                "mrr": vals["mrr"],
            }
        )
    n = len(hits)
    return {
        "hit_rate": (sum(hits) / n) if n else 0.0,
        "mrr": (sum(mrrs) / n) if n else 0.0,
        "n": n,
        "per_case": per_case,
    }


def evaluate_native_bm25_li(
    cases: list[dict],
    *,
    k: int = 5,
) -> dict:
    """Evaluate a genuinely native `llama_index.retrievers.bm25.BM25Retriever`
    (see `dealpoint.rag_lab.adapters.build_native_bm25_retriever`) with
    LlamaIndex's own `RetrieverEvaluator`, over the same canonical chunks and
    the same `expected_ids` rule as `evaluate_retriever_li`.

    Unlike `evaluate_retriever_li` (which wraps the project's own bm25s
    ranking), this retriever's tokenizer and scoring are entirely
    LlamaIndex's -- so its disagreements against `obj/` are real, not
    guaranteed-zero by construction.
    """
    from llama_index.core.evaluation import RetrieverEvaluator

    from dealpoint.eval.cases import resolve_document_id
    from dealpoint.rag_lab.adapters import build_native_bm25_retriever

    hits: list[float] = []
    mrrs: list[float] = []
    per_case: list[dict] = []
    for case in cases:
        golds = gold_ranges(case)
        if not golds:
            continue
        doc_id = resolve_document_id(case)
        chunks = chunk_document(load_document(doc_id))
        expected_ids = gold_bearing_chunk_ids(case, chunks)
        if not expected_ids:
            continue
        query = _canonical_query_for(case)
        retriever = build_native_bm25_retriever(chunks, k=k)
        evaluator = RetrieverEvaluator.from_metric_names(["hit_rate", "mrr"], retriever=retriever)
        result = evaluator.evaluate(query=query, expected_ids=expected_ids)
        vals = result.metric_vals_dict
        hits.append(vals["hit_rate"])
        mrrs.append(vals["mrr"])
        per_case.append(
            {
                "case_id": case["case_id"],
                "expected_ids": expected_ids,
                "retrieved_ids": list(result.retrieved_ids),
                "hit_rate": vals["hit_rate"],
                "mrr": vals["mrr"],
            }
        )
    n = len(hits)
    return {
        "hit_rate": (sum(hits) / n) if n else 0.0,
        "mrr": (sum(mrrs) / n) if n else 0.0,
        "n": n,
        "per_case": per_case,
    }


def evaluate_retriever_obj(
    cases: list[dict],
    config: RetrieverConfig,
    *,
    dense: Retriever,
    sparse: Retriever,
    variants_by_query: dict[str, list[str]] | None = None,
    scorer: Scorer | None = None,
    k: int = 5,
) -> dict:
    """`obj/gold_span_hit@k` / `obj/gold_span_mrr` for one config, via the
    EXISTING `dealpoint.eval.tournament.evaluate_config` -- never a fresh
    reimplementation of the metric.

    Always retrieves at `TOURNAMENT_K` regardless of the `k` argument (which
    controls the LlamaIndex side's `top_k` in the caller), so hit@5, hit@10
    and MRR reproduce `data/reports/tournament.json` exactly -- passing a
    shallower `k` here would truncate hit@10 and MRR to match hit@5.
    """
    from dealpoint.config import TOURNAMENT_K
    from dealpoint.corpus.retrievers import build_retriever
    from dealpoint.eval.tournament import evaluate_config

    retriever = build_retriever(
        config, dense=dense, sparse=sparse, variants_by_query=variants_by_query, scorer=scorer
    )

    per_case: list[dict] = []

    def _on_case(case: dict, query: str, rank: int | None) -> None:
        per_case.append({"case_id": case["case_id"], "first_hit_rank": rank})

    metrics = evaluate_config(cases, retriever, _canonical_query_for, k=TOURNAMENT_K, _on_case=_on_case)
    return {
        "hit_at_5": metrics["hit_at_5"],
        "hit_at_10": metrics["hit_at_10"],
        "mrr": metrics["mrr"],
        "n": metrics["n_cases"],
        "per_case": per_case,
    }


def classify_disagreement(
    *,
    config: RetrieverConfig,
    expected_ids: list[str],
    retrieved_ids: list[str],
    chunks: list[Chunk],
    golds: list[Range],
) -> str:
    """Deterministic cause label for one disagreeing (case, config, direction).

    See `CAUSE_RULE_TEXT` for the full documented rule; this function is its
    single implementation. `config` is accepted for a consistent call
    signature across configs (including rerank configs) even though no
    branch currently inspects it -- see `CAUSE_RULE_TEXT`'s note on why
    "reranking" is not an emitted label.
    """
    del config
    chunk_by_id = {c.chunk_id: c for c in chunks}
    if retrieved_ids:
        top_chunk = chunk_by_id.get(retrieved_ids[0])
        if top_chunk is not None:
            span = (top_chunk.start, top_chunk.end)
            for g in golds:
                ov = overlap_chars(span, g)
                if 0 < ov < MIN_GOLD_OVERLAP_CHARS:
                    return "partial_overlap"

    if len(expected_ids) >= 2:
        return "duplicate_relevant_chunks"

    section_refs_by_span: dict[Range, set[str]] = {}
    for chunk in chunks:
        span = (chunk.start, chunk.end)
        for g in golds:
            if overlap_chars(span, g) > 0:
                section_refs_by_span.setdefault(g, set()).add(chunk.section_ref)
    if any(len(refs) >= 2 for refs in section_refs_by_span.values()):
        return "section_boundary"

    return "chunk_identity"


def find_disagreements(
    cases: list[dict],
    config: RetrieverConfig,
    li_result: dict,
    obj_result: dict,
    *,
    k: int = 5,
) -> list[dict]:
    """LI-hit/MAUD-miss and MAUD-hit/LI-miss disagreements for one config.

    By construction (same underlying retriever, same `expected_ids` rule),
    these two metrics should agree almost everywhere -- see
    `dealpoint.rag_lab.evaluate`'s module docstring. This function is what
    would surface the rare cases where they do not.
    """
    from dealpoint.eval.cases import find_case, resolve_document_id

    li_by_case = {row["case_id"]: row for row in li_result.get("per_case", [])}
    obj_by_case = {row["case_id"]: row for row in obj_result.get("per_case", [])}

    disagreements: list[dict] = []
    for case_id, li_row in li_by_case.items():
        obj_row = obj_by_case.get(case_id)
        if obj_row is None:
            continue
        li_hit = bool(li_row["hit_rate"])
        obj_hit = obj_row["first_hit_rank"] is not None and obj_row["first_hit_rank"] <= k
        if li_hit == obj_hit:
            continue

        case = find_case(case_id)
        golds = gold_ranges(case)
        doc_id = resolve_document_id(case)
        chunks = chunk_document(load_document(doc_id))
        direction = "li_hit_maud_miss" if li_hit and not obj_hit else "maud_hit_li_miss"

        cause = classify_disagreement(
            config=config,
            expected_ids=li_row["expected_ids"],
            retrieved_ids=li_row["retrieved_ids"],
            chunks=chunks,
            golds=golds,
        )
        disagreements.append(
            {
                "case_id": case_id,
                "config": config.name,
                "direction": direction,
                "cause": cause,
                "detail": {
                    "expected_ids": li_row["expected_ids"],
                    "retrieved_ids": li_row["retrieved_ids"],
                    "li_hit_rate": li_row["hit_rate"],
                    "obj_first_hit_rank": obj_row["first_hit_rank"],
                },
            }
        )
    return disagreements


def worked_examples(disagreements: list[dict], cases_by_id: dict[str, dict], n: int = 2) -> list[dict]:
    """1-2 worked examples with real case ids (spec section 3.2)."""
    examples: list[dict] = []
    for d in disagreements[:n]:
        case = cases_by_id.get(d["case_id"])
        query = _canonical_query_for(case) if case else None
        examples.append(
            {
                "case_id": d["case_id"],
                "config": d["config"],
                "direction": d["direction"],
                "cause": d["cause"],
                "query": query,
                "top_retrieved_chunk_ids": d["detail"]["retrieved_ids"][:3],
                "gold_ranges": gold_ranges(case) if case else [],
            }
        )
    return examples


def frozen_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
