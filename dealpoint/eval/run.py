"""The eval runner: an arm x model over a case set (spec §4).

`python -m dealpoint.eval.run --set dev|test|counterfactual --arm A|B --model <id>
                              [--limit N] [--cases id,id] [--fake] [--out-dir DIR]`

Writes one canonical result row per case (finding, execution record, every
score) plus a summary JSON, to
`data/results/{set}_{arm}_{model_slug}_{index_version}_{git_sha7}.jsonl`.
Works with no network except OpenRouter, and with Braintrust entirely absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from dealpoint.config import RESULTS_DIR
from dealpoint.corpus.chunks import Chunk, chunk_document
from dealpoint.corpus.chunks import chunk_version as _chunk_version
from dealpoint.eval.cases import (
    git_sha7,
    load_case_set,
    resolve_document_id,
    resolve_question,
    slugify_model,
)
from dealpoint.eval.scorers import majority_baseline, score_case, summarise

if TYPE_CHECKING:
    from dealpoint.corpus.retrievers import Retriever

MILESTONE_ABSOLUTE_ENV = "DEALPOINT_MILESTONE_ABSOLUTE_USD"
MILESTONE_SPEND_START_ENV = "DEALPOINT_MILESTONE_SPEND_START_USD"


class MilestoneSpendCapError(RuntimeError):
    """Raised when a sweep would push this milestone's own new spend over its
    per-milestone absolute (spec §5, "Milestone spend guard in the runner").
    """


def _assert_within_milestone_absolute(est_usd: float) -> None:
    """Refuse to start a sweep when (ledger realized - start) + estimate > absolute.

    Reads `DEALPOINT_MILESTONE_ABSOLUTE_USD` and `DEALPOINT_MILESTONE_SPEND_START_USD`
    from the environment -- set by the factory at build time; both may be
    absent outside the factory (e.g. a bare `just eval` invocation), in
    which case this guard is a no-op (the global `assert_within_cap` still
    applies). This is in addition to, never instead of, the operator's
    absolute `MAX_OPENROUTER_SPEND_USD` cap.
    """
    import os

    absolute_raw = os.environ.get(MILESTONE_ABSOLUTE_ENV, "").strip()
    start_raw = os.environ.get(MILESTONE_SPEND_START_ENV, "").strip()
    if not absolute_raw or not start_raw:
        return
    try:
        absolute = float(absolute_raw)
        start = float(start_raw)
    except ValueError:
        return
    from dealpoint.eval.spend import realized_usd

    realized = realized_usd()
    new_spend = realized - start
    projected = new_spend + est_usd
    if projected > absolute:
        raise MilestoneSpendCapError(
            f"projected new milestone spend ${projected:.4f} (already spent ${new_spend:.4f} "
            f"since baseline ${start:.4f} + estimate ${est_usd:.4f}) exceeds this milestone's "
            f"absolute ${absolute:.2f} ({MILESTONE_ABSOLUTE_ENV})"
        )


class OfflineChunkRetriever:
    """Deterministic, offline stand-in for the dense retriever.

    Arm A always calls `retriever.search(...)`, so `--fake` runs would
    otherwise require a built Qdrant index. This returns the first `k`
    section-bounded chunks for the document, computed straight from the
    canonical text -- no index, no embeddings, no network.
    """

    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]:
        from dealpoint.corpus.document import load_document

        doc = load_document(agreement_id)
        chunks = chunk_document(doc)
        return chunks[:k]


def _build_client(fake: bool, milestone_tag: str):
    if fake:
        from dealpoint.llm.client import FakeClient, ScriptedTurn

        return FakeClient(
            script=[
                ScriptedTurn(content=None, tool_calls=None),
                ScriptedTurn(
                    content=json.dumps(
                        {
                            "answer": "ABSTAIN",
                            "evidence": [],
                            "rationale": "Fake client scripted response for the eval runner.",
                        }
                    )
                ),
            ]
            * 200  # generous: enough turns for both arms across a --limit sweep
        )
    from dealpoint.llm.client import OpenRouterClient

    return OpenRouterClient(milestone_tag=milestone_tag)


# Qdrant local mode allows exactly one open client per index path (spec §5
# wiring note: "Construct the retriever once per sweep, not per case"). This
# module-level cache goes further -- once per *process* -- because a second
# `QdrantClient(path=...)` opened while the first is still alive (even
# transiently, before Python's GC gets around to collecting the first one)
# raises `RuntimeError: already accessed by another instance`. Caching by a
# key that only depends on the *retriever config* (never on `arm`) means
# repeated calls across arms/sweeps in one process reuse the same client.
_RETRIEVER_CACHE: dict[str, Retriever] = {}


def _build_retriever(fake: bool, arm: str) -> Retriever:
    """Build (or reuse) the retriever for one arm (spec §5, wiring notes).

    Arms A/B use the plain dense/offline retriever (unchanged since M2).
    Arms C/D use the frozen arm-C retriever (`dealpoint.config.ARM_C_RETRIEVER`).

    M4.1 runner fix: the plain-dense path (`LazyRetriever`) and the hybrid
    arm-C path each used to construct their OWN `DenseRetriever`, i.e. their
    own `QdrantClient(path=INDEX_DIR)`. Qdrant local mode allows exactly one
    open client per path, so a process that runs arm B (dense) and then arm
    C/D (hybrid) -- or the reverse -- in the same run raised `RuntimeError:
    ... already accessed by another instance` on the second construction,
    surfacing as an `EXECUTION_FAILED` on the very first call of every case
    in that leg. One shared `DenseRetriever` instance, cached under a
    `_dense_impl` key, backs both paths now.
    """
    if fake:
        return OfflineChunkRetriever()

    dense_impl_key = "_dense_impl"
    dense_impl = _RETRIEVER_CACHE.get(dense_impl_key)
    if dense_impl is None:
        from dealpoint.corpus.retrievers import DenseRetriever

        dense_impl = DenseRetriever()
        _RETRIEVER_CACHE[dense_impl_key] = dense_impl

    if arm in ("C", "D"):
        key = "arm_c"
        if key in _RETRIEVER_CACHE:
            return _RETRIEVER_CACHE[key]
        from dealpoint.config import ARM_C_RETRIEVER
        from dealpoint.corpus.retrievers import BM25Retriever, RetrieverConfig, build_retriever

        retriever = build_retriever(
            RetrieverConfig(**ARM_C_RETRIEVER), dense=dense_impl, sparse=BM25Retriever()
        )
        _RETRIEVER_CACHE[key] = retriever
        return retriever

    key = "dense"
    if key in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[key]
    _RETRIEVER_CACHE[key] = dense_impl
    return dense_impl


def _select_cases(case_set: list[dict], cases_arg: str | None, limit: int | None) -> list[dict]:
    if cases_arg:
        wanted = [c.strip() for c in cases_arg.split(",") if c.strip()]
        by_id = {c["case_id"]: c for c in case_set}
        selected = [by_id[cid] for cid in wanted if cid in by_id]
    else:
        selected = sorted(case_set, key=lambda c: c["case_id"])
    if limit is not None:
        selected = selected[:limit]
    return selected


def result_stem(case_set_name: str, arm: str, model: str, index_version: str, sha7: str) -> str:
    return f"{case_set_name}_{arm}_{slugify_model(model)}_{index_version}_{sha7}"


def run_eval_set(
    case_set: str,
    arm: str,
    model: str,
    limit: int | None = None,
    cases: str | None = None,
    case_rows: list[dict] | None = None,
    fake: bool = False,
    out_dir: Path | None = None,
    milestone_tag: str = "m2",
) -> dict:
    """Run one arm x model over (a slice of) one case set. Returns the summary dict.

    Writes the result JSONL and summary JSON under `out_dir` (default
    `data/results/`). Every case is wrapped in its own `try/except` so one
    crashing case does not throw away the whole sweep.

    `case_set` is normally one of `"dev"`/`"test"`/`"counterfactual"`, loaded
    from its JSONL and (optionally) filtered by `cases`. When `case_rows` is
    given instead (the frozen `test_subset_v1` case, which spans both `test`
    and `counterfactual`), those rows are used directly, in the order given,
    and `case_set` is used only as a label for the output filename -- each
    row's own `case_set` field (from its source JSONL) still drives scoring
    (spec deliverable 4/6).
    """
    from dealpoint.agent.loop import run_agent
    from dealpoint.agent.pipeline import run_pipeline
    from dealpoint.agent.schema import ExecutionRecord, Usage
    from dealpoint.corpus.document import load_document
    from dealpoint.corpus.retrievers import index_version as compute_index_version

    if case_rows is not None:
        selected = list(case_rows)
        if limit is not None:
            selected = selected[:limit]
    else:
        all_cases = load_case_set(case_set)
        selected = _select_cases(all_cases, cases, limit)

    if not fake and len(selected) > 10:
        from dealpoint.eval.spend import assert_within_cap, per_case_usd

        per_case, _basis = per_case_usd(arm, model)
        est_usd = per_case * len(selected)
        assert_within_cap(est_usd)
        _assert_within_milestone_absolute(est_usd)

    client = _build_client(fake, milestone_tag)
    retriever = _build_retriever(fake, arm)

    index_version = "offline" if fake else compute_index_version()
    sha7 = git_sha7()
    chunk_ver = _chunk_version()

    out_dir = out_dir if out_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result_stem(case_set, arm, model, index_version, sha7)
    results_path = out_dir / f"{stem}.jsonl"
    summary_path = out_dir / f"{stem}_summary.json"

    started_at = datetime.now(UTC).isoformat()
    rows: list[dict] = []
    status_counts: dict[str, int] = {}

    for case in selected:
        if hasattr(client, "context"):
            # `git_sha7` rides along as the stable per-run discriminator that
            # lets dealpoint.eval.spend.per_case_usd group repeated
            # executions of the same case_id correctly (spec deliverable 6).
            client.context = {  # type: ignore[attr-defined]
                "arm": arm,
                "case_id": case["case_id"],
                "case_set": case_set,
                "git_sha7": sha7,
            }

        try:
            document_id = resolve_document_id(case)
            doc = load_document(document_id)
            question = resolve_question(case)
            if arm == "A":
                finding, record = run_pipeline(
                    case, doc, retriever, client, question, model, index_version
                )
            else:
                from dealpoint.config import ARMS

                skill_blk = None
                if ARMS[arm]["skill"]:
                    from dealpoint.agent.skill import skill_block as _skill_block

                    skill_blk = _skill_block(question.id)
                finding, record = run_agent(
                    case,
                    doc,
                    retriever,
                    client,
                    question,
                    model,
                    index_version,
                    arm=arm,
                    skill_block=skill_blk,
                )
        except Exception as exc:  # noqa: BLE001 - one bad case must not kill the sweep
            from dealpoint.agent._common import describe_exception

            finding = None
            doc = None
            record = ExecutionRecord(
                status="EXECUTION_FAILED",
                failure_reason="tool_error",
                trajectory=[],
                usage=Usage(),
                wall_ms=0,
                model=model,
                arm=arm,
                case_id=case.get("case_id", ""),
                index_version=index_version,
                chunk_version=chunk_ver,
                failure_detail=describe_exception(exc),
            )

        scores = score_case(case, finding, record, doc)
        status_counts[record.status] = status_counts.get(record.status, 0) + 1

        from dealpoint.eval.scorers import skill_adherence_detail

        row = {
            "case_id": case["case_id"],
            "case_set": case.get("case_set", case_set),
            "arm": arm,
            "model": model,
            "question_id": case["question_id"],
            "index_version": index_version,
            "chunk_version": chunk_ver,
            "git_sha7": sha7,
            "finding": finding.model_dump() if finding is not None else None,
            "record": record.model_dump(),
            "scores": scores,
            # Per-rule adherence detail: metadata, not a Braintrust score
            # (spec §3 -- the 6-score budget is fixed; `skill_adherence`
            # itself is the only score name).
            "skill_rules": skill_adherence_detail(case, finding, record, doc),
            "usd": record.usage.cost_usd,
        }
        rows.append(row)

    with open(results_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            fh.write("\n")

    ended_at = datetime.now(UTC).isoformat()
    realized = round(sum(r["usd"] for r in rows), 6)

    est_usd: float | None = None
    est_basis: str | None = None
    if not fake:
        from dealpoint.eval.spend import per_case_usd

        per_case, est_basis = per_case_usd(arm, model)
        est_usd = round(per_case * len(selected), 6)

    summary = {
        "set": case_set,
        "arm": arm,
        "model": model,
        "index_version": index_version,
        "chunk_version": chunk_ver,
        "git_sha7": sha7,
        "n_cases": len(rows),
        "started_at": started_at,
        "ended_at": ended_at,
        "scores": summarise(rows),
        "majority_baseline": majority_baseline(selected),
        "realized_usd": realized,
        "est_usd": est_usd,
        "est_basis": est_basis,
        "status_counts": status_counts,
    }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

    summary["results_path"] = str(results_path)
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dealpoint.eval.run")
    parser.add_argument("--set", choices=["dev", "test", "counterfactual"])
    parser.add_argument("--arm", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cases", default=None, help="comma-separated case ids")
    parser.add_argument(
        "--subset", default=None, help="e.g. test_subset_v1: run exactly this frozen subset"
    )
    parser.add_argument(
        "--tranche", type=int, default=None, choices=[1, 2],
        help="with --subset: run only tranche 1 or 2",
    )
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--milestone-tag", default="m2")
    args = parser.parse_args(argv)

    if not args.subset and not args.set:
        parser.error("either --set or --subset is required")

    case_rows = None
    case_set = args.set or "test"
    if args.subset:
        from dealpoint.eval.subset import load_subset_cases

        case_rows = load_subset_cases(args.subset, tranche=args.tranche)
        case_set = f"{args.subset}" + (f"_tranche{args.tranche}" if args.tranche else "")

    summary = run_eval_set(
        case_set=case_set,
        arm=args.arm,
        model=args.model,
        limit=args.limit,
        cases=args.cases,
        case_rows=case_rows,
        fake=args.fake,
        out_dir=args.out_dir,
        milestone_tag=args.milestone_tag,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
