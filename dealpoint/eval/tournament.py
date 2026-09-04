"""M3 — the LLM-free retrieval tournament (spec deliverable 2).

Every dev case is scored against every retriever candidate under two query
types: the question's canonical query (Appendix A) and the raw MAUD
question text. Metrics are hit\u00405, hit\u004010 and MRR against the case's
aligned gold spans; per-stage instrumentation records which component
(dense, bm25, the fused list, the reranked list) first surfaced the gold
chunk.

This module never reads the frozen holdout set: it only calls
`dealpoint.eval.cases.load_case_set("dev")`, and `main` refuses any
`--set` value other than `"dev"` with a non-zero exit. Do not import a path
constant for the frozen holdout set here -- `dealpoint.eval.cases` already
names it, this module must not.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from dealpoint.config import (
    HYBRID_FETCH_K,
    REPORTS_DIR,
    RERANK_MODEL,
    RRF_K,
    TOURNAMENT_K,
    TOURNAMENT_QUERIES_PATH,
)
from dealpoint.corpus.retrievers import (
    Retriever,
    RetrieverConfig,
    Scorer,
    build_retriever,
    reciprocal_rank_fusion,
)
from dealpoint.data.questions import QUESTION_BY_ID
from dealpoint.eval.cases import git_sha7, resolve_document_id
from dealpoint.eval.scorers import gold_ranges, overlap_chars

Range = tuple[int, int]

# The six tournament arms (brief §2.3): dense, bm25, hybrid-rrf,
# hybrid-rrf+rerank, hybrid+multiquery-fusion, hybrid+multiquery-fusion+rerank.
DEFAULT_CONFIGS: tuple[RetrieverConfig, ...] = (
    RetrieverConfig(name="dense", kind="dense"),
    RetrieverConfig(name="bm25", kind="bm25"),
    RetrieverConfig(name="hybrid_rrf", kind="hybrid_rrf"),
    RetrieverConfig(name="hybrid_rrf_rerank", kind="hybrid_rrf", rerank_model=RERANK_MODEL),
    RetrieverConfig(name="multi_query_fusion", kind="hybrid_rrf", multi_query=True),
    RetrieverConfig(
        name="multi_query_fusion_rerank",
        kind="hybrid_rrf",
        multi_query=True,
        rerank_model=RERANK_MODEL,
    ),
)


def _first_hit_rank(chunks: list, golds: list[Range]) -> int | None:
    """1-based rank of the first chunk overlapping a gold span, else `None`."""
    from dealpoint.config import MIN_GOLD_OVERLAP_CHARS

    for i, chunk in enumerate(chunks, start=1):
        span = (chunk.start, chunk.end)
        if any(overlap_chars(span, g) >= MIN_GOLD_OVERLAP_CHARS for g in golds):
            return i
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_config(
    cases: list[dict],
    retriever: Retriever,
    query_for: Callable[[dict], str],
    k: int = TOURNAMENT_K,
    _on_case: Callable[[dict, str, int | None], None] | None = None,
) -> dict:
    """hit\u00405 / hit\u004010 / MRR / n_cases / wall_seconds for one (config, query_type).

    Cases with no gold spans are skipped (dev has none, but be safe).
    `wall_seconds` accumulates only the cost of `retriever.search(...)` plus
    the cheap rank/metric bookkeeping for each case -- `_on_case` (the
    stage-instrumentation hook `run_tournament` supplies) is deliberately
    called *after* every case has already been timed and is never inside
    the timed region, so a config's reported wall-clock never includes the
    config-independent dense+bm25+RRF+rerank stage pass. `_on_case` is an
    internal hook `run_tournament` uses to record per-case detail without a
    second pass of `retriever.search` calls -- it is not part of the public
    two-value contract other callers rely on.
    """
    hits5: list[bool] = []
    hits10: list[bool] = []
    mrrs: list[float] = []
    n = 0
    wall = 0.0
    deferred: list[tuple[dict, str, int | None]] = []
    for case in cases:
        golds = gold_ranges(case)
        if not golds:
            continue
        n += 1
        query = query_for(case)
        agreement_id = resolve_document_id(case)

        t0 = time.time()
        chunks = retriever.search(agreement_id, query, k=k)
        rank = _first_hit_rank(chunks, golds)
        wall += time.time() - t0

        hits5.append(rank is not None and rank <= 5)
        hits10.append(rank is not None and rank <= 10)
        mrrs.append(1.0 / rank if rank else 0.0)
        if _on_case is not None:
            deferred.append((case, query, rank))

    if _on_case is not None:
        for case, query, rank in deferred:
            _on_case(case, query, rank)

    return {
        "hit_at_5": _mean([1.0 if h else 0.0 for h in hits5]),
        "hit_at_10": _mean([1.0 if h else 0.0 for h in hits10]),
        "mrr": _mean(mrrs),
        "n_cases": n,
        "wall_seconds": round(wall, 3),
    }


def _compute_stages(
    case: dict,
    query: str,
    dense: Retriever,
    sparse: Retriever,
    scorer: Scorer | None,
    fetch_k: int,
    rrf_k: int,
) -> dict:
    """Which component surfaced / promoted the gold chunk for one (case, query).

    Independent of any tournament config: always runs the canonical
    dense -> bm25 -> RRF-fuse -> (optionally) rerank pipeline, so every
    config's per-case row can carry the same diagnostic regardless of which
    candidate actually produced its own ranked list.
    """
    golds = gold_ranges(case)
    agreement_id = resolve_document_id(case)
    dense_hits = dense.search(agreement_id, query, k=fetch_k)
    sparse_hits = sparse.search(agreement_id, query, k=fetch_k)
    dense_rank = _first_hit_rank(dense_hits, golds)
    bm25_rank = _first_hit_rank(sparse_hits, golds)

    fused = reciprocal_rank_fusion([dense_hits, sparse_hits], rrf_k=rrf_k)
    fused_top = fused[:fetch_k]
    fused_rank = _first_hit_rank(fused_top, golds)

    if scorer is not None and fused_top:
        texts = [c.text for c in fused_top]
        scores = scorer(query, texts)
        ordered = sorted(
            zip(fused_top, scores, strict=True), key=lambda pair: (-pair[1], pair[0].chunk_id)
        )
        reranked = [chunk for chunk, _score in ordered]
        reranked_rank = _first_hit_rank(reranked, golds)
    else:
        reranked_rank = None

    if dense_rank is not None and bm25_rank is not None:
        surfaced_by = "both"
    elif dense_rank is not None:
        surfaced_by = "dense"
    elif bm25_rank is not None:
        surfaced_by = "bm25"
    else:
        surfaced_by = "none"

    if scorer is None or (fused_rank is None and reranked_rank is None):
        rerank_effect = "n/a"
    elif fused_rank is None and reranked_rank is not None:
        rerank_effect = "promoted"
    elif fused_rank is not None and reranked_rank is None:
        rerank_effect = "demoted"
    elif fused_rank is not None and reranked_rank is not None and reranked_rank < fused_rank:
        rerank_effect = "promoted"
    elif fused_rank is not None and reranked_rank is not None and reranked_rank > fused_rank:
        rerank_effect = "demoted"
    else:
        rerank_effect = "unchanged"

    return {
        "dense_rank": dense_rank,
        "bm25_rank": bm25_rank,
        "fused_rank": fused_rank,
        "reranked_rank": reranked_rank,
        "surfaced_by": surfaced_by,
        "rerank_effect": rerank_effect,
    }


def _memoized_scorer(scorer: Scorer) -> Scorer:
    """Dedupe scorer calls across identical (query, doc-text) inputs.

    The stage baseline (`_compute_stages`) and a `*_rerank` config's own
    `RerankRetriever` frequently rescore the exact same hybrid top-`fetch_k`
    candidate set for the same case -- memoizing avoids paying the
    cross-encoder's ~1.9s/call cost twice for that pair.
    """
    cache: dict[tuple[str, tuple[str, ...]], list[float]] = {}

    def wrapped(query: str, texts: list[str]) -> list[float]:
        key = (query, tuple(texts))
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = scorer(query, texts)
        cache[key] = result
        return result

    return wrapped


def _make_lazy_scorer(model: str) -> Scorer:
    """A scorer that constructs one real `TextCrossEncoder` on first use."""
    state: dict[str, object] = {}

    def scorer(query: str, texts: list[str]) -> list[float]:
        ce = state.get("ce")
        if ce is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            ce = TextCrossEncoder(model)
            state["ce"] = ce
        return list(ce.rerank(query, texts))  # type: ignore[union-attr]

    return scorer


def _select_winner(configs: list[RetrieverConfig], results: dict[str, dict[str, dict]]) -> str:
    """Highest hit\u00405 on canonical queries; ties -> MRR canonical, hit\u00405 maud,
    lower wall-clock, then config name.
    """

    def sort_key(cfg: RetrieverConfig):
        r = results[cfg.name]
        canon = r["canonical"]
        maud = r["maud"]
        wall_total = canon["wall_seconds"] + maud["wall_seconds"]
        return (-canon["hit_at_5"], -canon["mrr"], -maud["hit_at_5"], wall_total, cfg.name)

    ordered = sorted(configs, key=sort_key)
    return ordered[0].name


def run_tournament(
    cases: list[dict],
    configs: list[RetrieverConfig],
    *,
    dense: Retriever,
    sparse: Retriever,
    variants_by_query: dict[str, list[str]] | None = None,
    scorer: Scorer | None = None,
    k: int = TOURNAMENT_K,
    fetch_k: int = HYBRID_FETCH_K,
    rrf_k: int = RRF_K,
    compute_rerank_stage: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Score every config x query-type over `cases`. Pure w.r.t. its inputs.

    `dense`/`sparse` are already-constructed, shared retriever instances
    (Qdrant local mode allows exactly one client per path -- never build a
    fresh `DenseRetriever` per config here).
    """
    variants_by_query = variants_by_query or {}
    # Two independently-memoized wrappers around the same raw `scorer`: one
    # feeds the configs' own retrievers (via `build_retriever`), the other
    # feeds the config-independent stage baseline (`_compute_stages`). They
    # must not share a cache -- if the stage pass's cache entries were also
    # visible to a `*_rerank` config's own `RerankRetriever`, that config's
    # timed `retriever.search()` calls would silently hit a cache the stage
    # pass warmed for it and look artificially fast.
    config_scorer = _memoized_scorer(scorer) if scorer is not None else None
    stage_scorer = (
        _memoized_scorer(scorer) if scorer is not None and compute_rerank_stage else None
    )

    stage_cache: dict[tuple[str, str], dict] = {}
    per_case: list[dict] = []
    results: dict[str, dict[str, dict]] = {}

    query_fns: dict[str, Callable[[dict], str]] = {
        "canonical": lambda case: QUESTION_BY_ID[case["question_id"]].canonical_query,
        "maud": lambda case: QUESTION_BY_ID[case["question_id"]].maud_question,
    }

    for config in configs:
        retriever = build_retriever(
            config,
            dense=dense,
            sparse=sparse,
            variants_by_query=variants_by_query,
            scorer=config_scorer,
        )
        results[config.name] = {}
        for query_type, query_fn in query_fns.items():
            t0 = time.time()

            def _on_case(
                case: dict,
                query: str,
                rank: int | None,
                _qt: str = query_type,
                _config_name: str = config.name,
            ) -> None:
                cache_key = (case["case_id"], _qt)
                stages = stage_cache.get(cache_key)
                if stages is None:
                    stages = _compute_stages(
                        case, query, dense, sparse, stage_scorer, fetch_k, rrf_k
                    )
                    stage_cache[cache_key] = stages
                per_case.append(
                    {
                        "case_id": case["case_id"],
                        "question_id": case["question_id"],
                        "agreement_id": resolve_document_id(case),
                        "config": _config_name,
                        "query_type": _qt,
                        "first_hit_rank": rank,
                        "stages": stages,
                    }
                )

            metrics = evaluate_config(cases, retriever, query_fn, k=k, _on_case=_on_case)
            results[config.name][query_type] = metrics
            if progress is not None:
                progress(
                    f"  {config.name:>28s} / {query_type:<10s} "
                    f"hit@5={metrics['hit_at_5']:.3f} hit@10={metrics['hit_at_10']:.3f} "
                    f"mrr={metrics['mrr']:.3f} ({time.time() - t0:.1f}s)"
                )

    winner_name = _select_winner(list(configs), results)
    dense_results = results.get("dense", {}).get("canonical", {"hit_at_5": 0.0})
    winner_hit5 = results[winner_name]["canonical"]["hit_at_5"]
    dense_hit5 = dense_results["hit_at_5"]
    winner = {
        "config": winner_name,
        "query_type_basis": "canonical",
        "hit_at_5": winner_hit5,
        "dense_hit_at_5": dense_hit5,
        "margin": winner_hit5 - dense_hit5,
    }

    return {"results": results, "per_case": per_case, "winner": winner}


def render_markdown(report: dict) -> str:
    """Human-readable tables, regenerable from `report` (the JSON payload) alone."""
    lines: list[str] = []
    generated_from = report.get("generated_from", {})
    lines.append("# M3 retrieval tournament\n")
    total_wall_seconds = report.get("total_wall_seconds")
    lines.append(
        f"Case set: `{generated_from.get('case_set')}` "
        f"({generated_from.get('n_cases')} cases), "
        f"chunk_version=`{generated_from.get('chunk_version')}`, "
        f"index_version=`{generated_from.get('index_version')}`, "
        f"git_sha7=`{generated_from.get('git_sha7')}`, "
        f"total_wall_seconds=`{total_wall_seconds}`\n"
    )

    results = report.get("results", {})
    config_names = list(results.keys())
    for query_type in ("canonical", "maud"):
        lines.append(f"## {query_type} queries\n")
        lines.append("| config | hit@5 | hit@10 | MRR | wall (s) |")
        lines.append("|---|---|---|---|---|")
        for name in config_names:
            row = results[name].get(query_type)
            if row is None:
                continue
            lines.append(
                f"| {name} | {row['hit_at_5']:.3f} | {row['hit_at_10']:.3f} | "
                f"{row['mrr']:.3f} | {row['wall_seconds']:.1f} |"
            )
        lines.append("")

    per_case = report.get("per_case", [])
    surfaced_counts: dict[str, int] = {}
    effect_counts: dict[str, int] = {}
    seen_stage_keys: set[tuple] = set()
    for row in per_case:
        stage_key = (row["case_id"], row["query_type"])
        if stage_key in seen_stage_keys:
            continue
        seen_stage_keys.add(stage_key)
        stages = row["stages"]
        surfaced_counts[stages["surfaced_by"]] = surfaced_counts.get(stages["surfaced_by"], 0) + 1
        effect_counts[stages["rerank_effect"]] = effect_counts.get(stages["rerank_effect"], 0) + 1

    lines.append("## which component found it\n")
    lines.append("| surfaced_by | count |")
    lines.append("|---|---|")
    for key in sorted(surfaced_counts):
        lines.append(f"| {key} | {surfaced_counts[key]} |")
    lines.append("")
    lines.append("| rerank_effect | count |")
    lines.append("|---|---|")
    for key in sorted(effect_counts):
        lines.append(f"| {key} | {effect_counts[key]} |")
    lines.append("")

    winner = report.get("winner", {})
    lines.append(
        f"## Winner: `{winner.get('config')}` "
        f"(hit@5={winner.get('hit_at_5'):.3f} vs dense {winner.get('dense_hit_at_5'):.3f}, "
        f"margin={winner.get('margin'):+.3f}, basis={winner.get('query_type_basis')})\n"
    )

    llamaindex = report.get("llamaindex", {})
    lines.append("## LlamaIndex\n")
    lines.append(
        f"adopted={llamaindex.get('adopted')}; measured standalone install "
        f"{llamaindex.get('measured_install_mb')} MB, marginal into this repo's venv "
        f"~{llamaindex.get('marginal_mb')} MB. {llamaindex.get('reason', '')}\n"
    )

    return "\n".join(lines) + "\n"


def _load_variants_by_query() -> dict[str, list[str]]:
    with open(TOURNAMENT_QUERIES_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    variants_by_query: dict[str, list[str]] = {}
    for question_id, entry in payload["queries"].items():
        question = QUESTION_BY_ID[question_id]
        variants_by_query[question.canonical_query] = list(entry["variants"])
        variants_by_query[question.maud_question] = list(entry["variants"])
    return variants_by_query


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dealpoint.eval.tournament")
    parser.add_argument("--set", default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--configs", default=None, help="comma-separated config names")
    parser.add_argument("--k", type=int, default=TOURNAMENT_K)
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args(argv)

    if args.set != "dev":
        print(
            f"error: --set {args.set!r} refused. M3's tournament is LLM-free and dev-only "
            "(specs/milestones/m3.md: nothing here may read the frozen holdout set); "
            "only --set dev is permitted.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    from dealpoint.eval.cases import load_case_set

    all_cases = sorted(load_case_set(args.set), key=lambda c: c["case_id"])
    if args.limit is not None:
        all_cases = all_cases[: args.limit]

    configs = list(DEFAULT_CONFIGS)
    if args.configs:
        wanted = {c.strip() for c in args.configs.split(",") if c.strip()}
        configs = [c for c in configs if c.name in wanted]
    if args.no_rerank:
        configs = [c for c in configs if c.rerank_model is None]

    variants_by_query = _load_variants_by_query()

    from dealpoint.corpus.retrievers import BM25Retriever, DenseRetriever, index_version

    print(f"Building shared dense + bm25 retrievers over {len(all_cases)} dev cases ...")
    dense = DenseRetriever()
    sparse = BM25Retriever()

    scorer = None
    if not args.no_rerank:
        scorer = _make_lazy_scorer(RERANK_MODEL)

    t_total0 = time.time()
    core = run_tournament(
        all_cases,
        configs,
        dense=dense,
        sparse=sparse,
        variants_by_query=variants_by_query,
        scorer=scorer,
        k=args.k,
        compute_rerank_stage=not args.no_rerank,
        progress=lambda msg: print(msg, flush=True),
    )

    from dealpoint.config import DATASET_VERSION_TXT_PATH
    from dealpoint.corpus.chunks import chunk_version

    dataset_version = ""
    if DATASET_VERSION_TXT_PATH.exists():
        dataset_version = DATASET_VERSION_TXT_PATH.read_text(encoding="utf-8").strip()

    elapsed_total = time.time() - t_total0

    report = {
        "schema_version": 1,
        "generated_from": {
            "chunk_version": chunk_version(),
            "index_version": index_version(),
            "dataset_version": dataset_version,
            "git_sha7": git_sha7(),
            "n_cases": len(all_cases),
            "case_set": args.set,
        },
        "k": args.k,
        "rrf_k": RRF_K,
        "fetch_k": HYBRID_FETCH_K,
        "rerank_model": RERANK_MODEL,
        "total_wall_seconds": round(elapsed_total, 3),
        "configs": [dataclasses.asdict(c) for c in configs],
        "results": core["results"],
        "per_case": core["per_case"],
        "winner": core["winner"],
        "llamaindex": {
            "adopted": False,
            "measured_install_mb": 178,
            "marginal_mb": 100,
            "reason": (
                "llama-index-core + llama-index-retrievers-bm25 measured (uv venv probe): "
                "178 MB standalone, no torch/transformers/nvidia wheels, ~100 MB marginal into "
                "this repo's venv (numpy/pillow/pydantic already present). It fits the ~2 GB "
                "disk budget but was declined: bm25s gives BM25 in three calls, RRF is eight "
                "lines of pure Python, and fastembed already ships the named cross-encoder, so "
                "LlamaIndex would add install weight for no material capability this milestone "
                "needs."
            ),
        },
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tournament.json"
    md_path = out_dir / "tournament.md"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))

    print(f"Wrote {json_path} and {md_path} in {elapsed_total:.1f}s total")
    winner = report["winner"]
    print(
        f"Winner: {winner['config']} "
        f"(hit@5={winner['hit_at_5']:.3f} vs dense {winner['dense_hit_at_5']:.3f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
