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


def _build_retriever(fake: bool):
    if fake:
        return OfflineChunkRetriever()
    from dealpoint.corpus.retrievers import LazyRetriever

    return LazyRetriever()


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
    fake: bool = False,
    out_dir: Path | None = None,
    milestone_tag: str = "m2",
) -> dict:
    """Run one arm x model over (a slice of) one case set. Returns the summary dict.

    Writes the result JSONL and summary JSON under `out_dir` (default
    `data/results/`). Every case is wrapped in its own `try/except` so one
    crashing case does not throw away the whole sweep.
    """
    from dealpoint.agent.loop import run_agent
    from dealpoint.agent.pipeline import run_pipeline
    from dealpoint.agent.schema import ExecutionRecord, Usage
    from dealpoint.corpus.document import load_document
    from dealpoint.corpus.retrievers import index_version as compute_index_version

    all_cases = load_case_set(case_set)
    selected = _select_cases(all_cases, cases, limit)

    if not fake and len(selected) > 10:
        from dealpoint.eval.spend import assert_within_cap, per_case_usd

        per_case, _basis = per_case_usd(arm, model)
        assert_within_cap(per_case * len(selected))

    client = _build_client(fake, milestone_tag)
    retriever = _build_retriever(fake)

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
            run_fn = run_pipeline if arm == "A" else run_agent
            finding, record = run_fn(case, doc, retriever, client, question, model, index_version)
        except Exception as exc:  # noqa: BLE001 - one bad case must not kill the sweep
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
            )
            _ = exc  # recorded via failure_reason; not re-raised

        scores = score_case(case, finding, record, doc)
        status_counts[record.status] = status_counts.get(record.status, 0) + 1

        row = {
            "case_id": case["case_id"],
            "case_set": case_set,
            "arm": arm,
            "model": model,
            "question_id": case["question_id"],
            "index_version": index_version,
            "chunk_version": chunk_ver,
            "git_sha7": sha7,
            "finding": finding.model_dump() if finding is not None else None,
            "record": record.model_dump(),
            "scores": scores,
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
    parser.add_argument("--set", required=True, choices=["dev", "test", "counterfactual"])
    parser.add_argument("--arm", required=True, choices=["A", "B"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cases", default=None, help="comma-separated case ids")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = run_eval_set(
        case_set=args.set,
        arm=args.arm,
        model=args.model,
        limit=args.limit,
        cases=args.cases,
        fake=args.fake,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
