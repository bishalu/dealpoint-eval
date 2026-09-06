"""`data/reports/li_rag_eval.json` + `.md` (spec section 1, output).

Builds a pure-ish report from already-computed LI/obj evaluation results;
the disk-driven `main()` builds the shared dense/bm25 retrievers exactly
once (Qdrant local mode allows one client per path) and evaluates every
frozen M3 candidate on the full 58-case dev set, plus (when the synthetic
set has already been frozen by `dealpoint.rag_lab.synthetic`) the same
comparison over the synthetic query distribution.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from dealpoint.config import (
    LI_RAG_EVAL_JSON_PATH,
    LI_RAG_EVAL_MD_PATH,
    SYNTHETIC_DEV_QUERIES_PATH,
    SYNTHETIC_GENERATION_RUN_PATH,
    TEST_SUBSET_V1_PATH,
    TOURNAMENT_JSON_PATH,
)
from dealpoint.eval.agreement import spearman
from dealpoint.eval.cases import git_sha7
from dealpoint.rag_lab.evaluate import (
    CAUSE_RULE_TEXT,
    assert_versions_match,
    evaluate_retriever_li,
    evaluate_retriever_obj,
    find_disagreements,
    worked_examples,
)

JUDGED_SUBSET_PATH_NAME = "data/eval/judged_subset.json"

BRIEF_DIFFERENCES = [
    {
        "id": 1,
        "topic": "llamaindex_scope",
        "difference": (
            "Brief section 3 allows LlamaIndex 'only behind the Retriever interface'; "
            "section 4 excludes LlamaIndex agents/workflows. M7a additionally uses it for "
            "framework-native evaluation (RetrieverEvaluator) and synthetic-query generation "
            "(DatasetGenerator). Agents/workflows remain excluded -- neither is used anywhere "
            "in this module. M3 measured LlamaIndex and declined it for retrieval composition "
            "(tournament.json's llamaindex.adopted: false, 178 MB reason); M7a adopts it for a "
            "different purpose (evaluation/synthesis, not composition) -- the earlier decision "
            "is superseded for that purpose only, not reversed for retrieval itself."
        ),
    },
    {
        "id": 2,
        "topic": "m6_envelope_test_scoping",
        "difference": (
            "tests/test_spend_m6.py::test_realized_usd_within_envelope originally asserted "
            "realized_usd() (the WHOLE ledger) <= 4.00. Once M7a's own ledger rows exist that "
            "assertion would go false for a reason having nothing to do with M6. Repaired to "
            "sum only the M1-M6 milestone tags via the existing realized_by_tag() helper, "
            "which keeps M6's own claim exactly as strong as it was measured, rather than "
            "either weakening the envelope (raising 4.00, which the spec explicitly forbids) "
            "or leaving a spurious cross-milestone failure. M7a's own ledger discipline is "
            "asserted separately in tests/test_spend_m7.py against the $6.00 global cap."
        ),
    },
    {
        "id": 3,
        "topic": "score_budget",
        "difference": (
            "Brief section 2.7 caps Braintrust logging at <= 6 scores/case. M7a permits "
            "<= 12 scores/case on subsets <= 60 cases as a ceiling, not a target, with M4/M6 "
            "sweeps still logging exactly six -- enforced at sync time by "
            "dealpoint.eval.braintrust_sync.assert_score_budget."
        ),
    },
]


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(
    per_config_results: dict[str, dict],
    *,
    versions: dict,
    n_cases: int,
    dataset_version: str,
    framework_versions: dict,
    synthetic: dict | None = None,
) -> dict:
    """Pure builder over already-computed per-config LI/obj results.

    `per_config_results[config_name]` = {"li": {...}, "obj": {...},
    "disagreements": [...], "config": RetrieverConfig-as-dict}.
    """
    config_names = list(per_config_results.keys())

    li_hit_ranks_input = [per_config_results[c]["li"]["hit_rate"] for c in config_names]
    obj_hit_ranks_input = [per_config_results[c]["obj"]["hit_at_5"] for c in config_names]
    li_mrr_input = [per_config_results[c]["li"]["mrr"] for c in config_names]
    obj_mrr_input = [per_config_results[c]["obj"]["mrr"] for c in config_names]

    ranking_agreement = {
        "hit_rate_vs_hit_at_5": {
            "spearman": spearman(li_hit_ranks_input, obj_hit_ranks_input),
            "li_hit_rate_by_config": dict(zip(config_names, li_hit_ranks_input, strict=True)),
            "obj_hit_at_5_by_config": dict(zip(config_names, obj_hit_ranks_input, strict=True)),
        },
        "mrr_vs_mrr": {
            "spearman": spearman(li_mrr_input, obj_mrr_input),
            "li_mrr_by_config": dict(zip(config_names, li_mrr_input, strict=True)),
            "obj_mrr_by_config": dict(zip(config_names, obj_mrr_input, strict=True)),
        },
    }

    all_disagreements: list[dict] = []
    for c in config_names:
        all_disagreements.extend(per_config_results[c]["disagreements"])

    n_li_hit_maud_miss = sum(1 for d in all_disagreements if d["direction"] == "li_hit_maud_miss")
    n_maud_hit_li_miss = sum(1 for d in all_disagreements if d["direction"] == "maud_hit_li_miss")

    retrievers = {
        c: {
            "li": {
                "hit_rate": per_config_results[c]["li"]["hit_rate"],
                "mrr": per_config_results[c]["li"]["mrr"],
                "n": per_config_results[c]["li"]["n"],
                "per_case": per_config_results[c]["li"].get("per_case", []),
            },
            "obj": {
                "gold_span_hit_at_5": per_config_results[c]["obj"]["hit_at_5"],
                "gold_span_hit_at_10": per_config_results[c]["obj"]["hit_at_10"],
                "gold_span_mrr": per_config_results[c]["obj"]["mrr"],
                "n": per_config_results[c]["obj"]["n"],
                "per_case": per_config_results[c]["obj"].get("per_case", []),
            },
        }
        for c in config_names
    }

    report = {
        "schema_version": 1,
        "generated_from": {
            "chunk_version": versions["chunk_version"],
            "index_version": versions["index_version"],
            "dataset_version": dataset_version,
            "git_sha7": git_sha7(),
            "n_cases": n_cases,
            "case_set": "dev",
        },
        "framework_versions": framework_versions,
        "retrievers": retrievers,
        "ranking_agreement": ranking_agreement,
        "disagreements": all_disagreements,
        "disagreement_summary": {
            "li_hit_maud_miss": n_li_hit_maud_miss,
            "maud_hit_li_miss": n_maud_hit_li_miss,
            "note": (
                "Near-zero disagreement is expected and reported plainly, not manufactured: "
                "expected_ids for the LlamaIndex RetrieverEvaluator ARE the project's "
                "gold-bearing chunk ids (dealpoint.rag_lab.evaluate.gold_bearing_chunk_ids uses "
                "the exact same overlap_chars/MIN_GOLD_OVERLAP_CHARS rule as "
                "dealpoint.eval.tournament._first_hit_rank), so the two metrics differ only "
                "where k-truncation or multi-hit averaging bites."
            )
            if (n_li_hit_maud_miss + n_maud_hit_li_miss) == 0
            else None,
        },
        "cause_rule": CAUSE_RULE_TEXT,
        "worked_examples": [],
        "synthetic": synthetic,
        "frozen_assertions": {
            "chunk_version": versions["chunk_version"],
            "index_version": versions["index_version"],
            "test_subset_v1_sha256": _sha256_file(TEST_SUBSET_V1_PATH),
            "tournament_json_sha256": _sha256_file(TOURNAMENT_JSON_PATH),
        },
        "decisions": [
            {
                "id": 1,
                "topic": "m6_envelope_test_scoping",
                "decision": (
                    "tests/test_spend_m6.py's envelope assertion scoped to M1-M6 tags via "
                    "realized_by_tag() rather than the whole ledger; see brief_differences[1]."
                ),
            },
        ],
        "brief_differences": BRIEF_DIFFERENCES,
    }
    return report


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# M7a -- LlamaIndex RAG lab: native evaluation on canonical dev retrieval\n")
    gf = report.get("generated_from", {})
    lines.append(
        f"Case set: `{gf.get('case_set')}` ({gf.get('n_cases')} cases), "
        f"chunk_version=`{gf.get('chunk_version')}`, index_version=`{gf.get('index_version')}`, "
        f"git_sha7=`{gf.get('git_sha7')}`\n"
    )
    lines.append(
        "LlamaIndex owns retriever composition/evaluation and the synthetic-query study; "
        "MAUD gold-span overlap remains benchmark truth. LlamaIndex node ids never replace it.\n"
    )

    lines.append("## Per-retriever: LI vs obj/\n")
    lines.append("| config | li/hit_rate | li/mrr | obj/gold_span_hit\u00405 | obj/gold_span_mrr | n |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in report.get("retrievers", {}).items():
        li = r["li"]
        obj = r["obj"]
        lines.append(
            f"| {name} | {_pct(li['hit_rate'])} | {li['mrr']:.3f} | "
            f"{_pct(obj['gold_span_hit_at_5'])} | {obj['gold_span_mrr']:.3f} | {li['n']} |"
        )
    lines.append("")

    ra = report.get("ranking_agreement", {})
    hr = ra.get("hit_rate_vs_hit_at_5", {})
    mr = ra.get("mrr_vs_mrr", {})
    lines.append("## Ranking-order agreement (Spearman over retriever ranks)\n")
    lines.append(f"hit_rate vs hit\u00405: rho = {hr.get('spearman')}")
    lines.append("")
    lines.append(f"mrr vs mrr: rho = {mr.get('spearman')}")
    lines.append("")

    ds = report.get("disagreement_summary", {})
    lines.append("## Disagreements\n")
    lines.append(f"LI-hit/MAUD-miss: {ds.get('li_hit_maud_miss')}; MAUD-hit/LI-miss: {ds.get('maud_hit_li_miss')}")
    if ds.get("note"):
        lines.append("")
        lines.append(ds["note"])
    lines.append("")
    lines.append("Cause rule:\n")
    lines.append(f"> {report.get('cause_rule')}")
    lines.append("")

    synth = report.get("synthetic")
    lines.append("## Synthetic-query robustness (secondary)\n")
    if synth is None:
        lines.append("Not generated yet -- run `just synth-queries` then regenerate this report.")
    else:
        lines.append(
            f"Generator: `{synth.get('generator')}` ({synth.get('generator_version')}), "
            f"model=`{synth.get('model')}`, n_chunks_attempted={synth.get('n_chunks_attempted')}, "
            f"n_chunks_kept={synth.get('n_chunks_kept')} (filter: `{synth.get('kept_filter')}`), "
            f"n_queries={synth.get('n_queries')}, prompt_hash=`{synth.get('prompt_hash')}`"
        )
        lines.append("")

        def _usd(x):
            return f"${x:.6f}" if x is not None else "n/a"

        lines.append(
            f"kept_questions_usd: {_usd(synth.get('kept_questions_usd'))} | "
            f"frozen_run_usd: {_usd(synth.get('frozen_run_usd'))} | "
            f"ledger_purpose_total_usd: {_usd(synth.get('ledger_purpose_total_usd'))}"
        )
        if synth.get("cost_note"):
            lines.append("")
            lines.append(synth["cost_note"])
        calibration = synth.get("calibration")
        estimate = synth.get("estimate")
        if calibration or estimate:
            lines.append("")
            lines.append("Calibration/estimate provenance:")
            lines.append(
                f"```json\n{json.dumps({'calibration': calibration, 'estimate': estimate}, indent=2, sort_keys=True)}\n```"
            )
        lines.append("")
        lines.append(f"Verdict: {synth.get('verdict', 'n/a')}")
        results = synth.get("results")
        if results:
            lines.append("")
            lines.append("| config | li/hit_rate (synthetic) | obj/hit\u00405 (synthetic) |")
            lines.append("|---|---|---|")
            for name, r in results.items():
                lines.append(f"| {name} | {_pct(r['li_hit_rate'])} | {_pct(r['obj_hit_at_5'])} |")
    lines.append("")

    lines.append("## Frozen-integrity assertions\n")
    fa = report.get("frozen_assertions", {})
    lines.append(f"```json\n{json.dumps(fa, indent=2, sort_keys=True)}\n```\n")

    lines.append("## Decisions\n")
    for d in report.get("decisions", []):
        lines.append(f"- **{d['topic']}**: {d['decision']}")
    lines.append("")

    lines.append("## Brief-vs-spec differences\n")
    for d in report.get("brief_differences", []):
        lines.append(f"- **{d['topic']}**: {d['difference']}")
    lines.append("")

    return "\n".join(lines) + "\n"


def _framework_versions() -> dict:
    from dealpoint.eval.framework_versions import framework_versions

    return framework_versions()


def main(argv: list[str] | None = None) -> int:
    from dealpoint.config import DATASET_VERSION_TXT_PATH, RERANK_MODEL
    from dealpoint.corpus.retrievers import BM25Retriever, DenseRetriever
    from dealpoint.eval.cases import load_case_set
    from dealpoint.eval.tournament import (
        DEFAULT_CONFIGS,
        _load_variants_by_query,
        _make_lazy_scorer,
    )

    versions = assert_versions_match()

    all_cases = sorted(load_case_set("dev"), key=lambda c: c["case_id"])
    variants_by_query = _load_variants_by_query()

    print(f"Building shared dense + bm25 retrievers over {len(all_cases)} dev cases ...")
    dense = DenseRetriever()
    sparse = BM25Retriever()
    scorer = _make_lazy_scorer(RERANK_MODEL)

    t0 = time.time()
    per_config: dict[str, dict] = {}
    for config in DEFAULT_CONFIGS:
        li = evaluate_retriever_li(
            all_cases, config, dense=dense, sparse=sparse, variants_by_query=variants_by_query,
            scorer=scorer, k=5,
        )
        obj = evaluate_retriever_obj(
            all_cases, config, dense=dense, sparse=sparse, variants_by_query=variants_by_query,
            scorer=scorer, k=5,
        )
        disagreements = find_disagreements(all_cases, config, li, obj, k=5)
        per_config[config.name] = {"li": li, "obj": obj, "disagreements": disagreements}
        print(
            f"  {config.name:>28s} li/hit_rate={li['hit_rate']:.3f} obj/hit\u00405={obj['hit_at_5']:.3f} "
            f"disagreements={len(disagreements)} ({time.time() - t0:.1f}s)"
        )

    print("Evaluating native llama_index.retrievers.bm25.BM25Retriever (li_native_bm25) ...")
    import dataclasses

    from dealpoint.rag_lab.evaluate import evaluate_native_bm25_li

    bm25_config = next(c for c in DEFAULT_CONFIGS if c.name == "bm25")
    native_bm25_config = dataclasses.replace(bm25_config, name="li_native_bm25")
    li_native = evaluate_native_bm25_li(all_cases, k=5)
    obj_for_native = per_config["bm25"]["obj"]
    native_disagreements = find_disagreements(
        all_cases, native_bm25_config, li_native, obj_for_native, k=5
    )
    per_config["li_native_bm25"] = {
        "li": li_native,
        "obj": obj_for_native,
        "disagreements": native_disagreements,
    }
    print(
        f"  li_native_bm25 native li/hit_rate={li_native['hit_rate']:.3f} "
        f"disagreements={len(native_disagreements)}"
    )

    dataset_version = ""
    if DATASET_VERSION_TXT_PATH.exists():
        dataset_version = DATASET_VERSION_TXT_PATH.read_text(encoding="utf-8").strip()

    framework_versions = _framework_versions()

    synthetic = None
    if SYNTHETIC_DEV_QUERIES_PATH.exists():
        from dealpoint.rag_lab.synthetic_eval import evaluate_synthetic_set, load_synthetic_queries

        synth_rows = load_synthetic_queries()
        synth_eval = evaluate_synthetic_set(
            dense=dense, sparse=sparse, variants_by_query=variants_by_query, scorer=scorer
        )
        first_row = synth_rows[0] if synth_rows else {}

        generation_run = {}
        if SYNTHETIC_GENERATION_RUN_PATH.exists():
            try:
                generation_run = json.loads(
                    SYNTHETIC_GENERATION_RUN_PATH.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                generation_run = {}

        # total_usd: prefer the persisted generation-run record (which the
        # metered run / cost backfill wrote); fall back to summing the rows
        # on disk so a manually-backfilled file (no run record yet) still
        # reports a real total rather than silently omitting it.
        row_usd_values: list[float] = [
            float(r["usd"]) for r in synth_rows if r.get("usd") is not None
        ]
        kept_questions_usd = generation_run.get("total_usd")
        if kept_questions_usd is None and row_usd_values:
            kept_questions_usd = round(sum(row_usd_values), 6)
        n_rows_missing_usd = sum(1 for r in synth_rows if r.get("usd") is None)

        n_chunks_kept = len({r["chunk_id"] for r in synth_rows})
        n_chunks_attempted = generation_run.get("n_chunks", n_chunks_kept)

        from dealpoint.eval.spend import read_ledger

        ledger_rows = read_ledger()
        ledger_purpose_rows = [r for r in ledger_rows if r.get("purpose") == "synthetic_query"]
        ledger_purpose_total_usd = round(sum(float(r.get("usd") or 0) for r in ledger_purpose_rows), 6)

        # frozen_run_usd: the real cost of the 113 generation calls that produced
        # the frozen file, taken from the ledger rows following the calibration
        # sample (the calibration's own rows are excluded) -- distinct from
        # kept_questions_usd, which only reflects the 73 chunks whose questions
        # survived `_looks_like_question` filtering.
        calibration_rows = (generation_run.get("calibration") or {}).get("rows", [])
        calibration_chunk_ids = {r["chunk_id"] for r in calibration_rows}
        frozen_run_rows = [
            r for r in ledger_purpose_rows if r.get("chunk_id") not in calibration_chunk_ids
        ][-n_chunks_attempted:] if n_chunks_attempted else []
        frozen_run_usd = round(sum(float(r.get("usd") or 0) for r in frozen_run_rows), 6)

        synthetic = {
            "generator": first_row.get("generator"),
            "generator_version": first_row.get("generator_version"),
            "prompt_hash": first_row.get("prompt_hash"),
            "model": first_row.get("model"),
            "n_chunks_attempted": n_chunks_attempted,
            "n_chunks_kept": n_chunks_kept,
            "kept_filter": "dealpoint.rag_lab.synthetic._looks_like_question",
            "n_queries": synth_eval["n_queries"],
            "file_sha256": _sha256_file(SYNTHETIC_DEV_QUERIES_PATH),
            "results": synth_eval["results"],
            "verdict": synth_eval["verdict"],
            "calibration": generation_run.get("calibration"),
            "estimate": generation_run.get("estimate"),
            "kept_questions_usd": kept_questions_usd,
            "frozen_run_usd": frozen_run_usd,
            "ledger_purpose_total_usd": ledger_purpose_total_usd,
            "cost_note": (
                "Three distinct cost figures, none interchangeable: kept_questions_usd "
                f"({kept_questions_usd}) sums only the usd recorded against the "
                f"{n_chunks_kept} chunks whose questions survived the keep filter; "
                f"frozen_run_usd ({frozen_run_usd}) is the real cost of the {n_chunks_attempted} "
                "generation calls in the frozen run (kept and dropped chunks alike); "
                f"ledger_purpose_total_usd ({ledger_purpose_total_usd}) sums every "
                "purpose=synthetic_query ledger row across all attempts, including "
                "superseded ones from earlier interrupted runs."
            ),
            "n_rows_missing_usd": n_rows_missing_usd,
        }

    report = build_report(
        per_config,
        versions=versions,
        n_cases=len(all_cases),
        dataset_version=dataset_version,
        framework_versions=framework_versions,
        synthetic=synthetic,
    )

    cases_by_id = {c["case_id"]: c for c in all_cases}
    examples: list[dict] = []
    for d in report["disagreements"]:
        examples.extend(worked_examples([d], cases_by_id, n=1))
        if len(examples) >= 2:
            break
    report["worked_examples"] = examples

    LI_RAG_EVAL_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LI_RAG_EVAL_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    with open(LI_RAG_EVAL_MD_PATH, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))

    print(f"Wrote {LI_RAG_EVAL_JSON_PATH} and {LI_RAG_EVAL_MD_PATH}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
