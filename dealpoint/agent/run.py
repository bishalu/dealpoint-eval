"""CLI: `python -m dealpoint.agent.run --case <case_id> --arm A|B --model <id> [--fake]`.

Prints `{"finding": ..., "record": ...}` as JSON to stdout. Resolves the case
from dev/test/counterfactual JSONL by `case_id`. Exit non-zero on
`EXECUTION_FAILED` so the CLI is usable in a shell chain.
"""

from __future__ import annotations

import argparse
import json
import sys

from dealpoint.config import DEFAULT_MODEL
from dealpoint.eval.cases import find_case as _find_case
from dealpoint.eval.cases import resolve_document_id as _resolve_document_id
from dealpoint.eval.cases import resolve_question as _resolve_question


def _build_client(fake: bool):
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
                            "rationale": "Fake client scripted response for CLI smoke check.",
                        }
                    )
                ),
            ]
        )
    from dealpoint.llm.client import OpenRouterClient

    return OpenRouterClient()


def _build_retriever(arm: str):
    from dealpoint.corpus.retrievers import LazyRetriever

    if arm in ("C", "D"):
        from dealpoint.config import ARM_C_RETRIEVER
        from dealpoint.corpus.retrievers import (
            BM25Retriever,
            DenseRetriever,
            RetrieverConfig,
            build_retriever,
        )

        return build_retriever(
            RetrieverConfig(**ARM_C_RETRIEVER), dense=DenseRetriever(), sparse=BM25Retriever()
        )
    return LazyRetriever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dealpoint.agent.run")
    parser.add_argument("--case", required=True, help="case_id, e.g. contract_0__q01")
    parser.add_argument("--arm", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fake", action="store_true", help="use FakeClient (no network, no key)")
    args = parser.parse_args(argv)

    from dealpoint.corpus.document import load_document

    case = _find_case(args.case)
    document_id = _resolve_document_id(case)
    doc = load_document(document_id)
    question = _resolve_question(case)
    client = _build_client(args.fake)
    retriever = _build_retriever(args.arm)

    if args.arm == "A":
        from dealpoint.agent.pipeline import run_pipeline

        finding, record = run_pipeline(case, doc, retriever, client, question, args.model)
    else:
        from dealpoint.agent.loop import run_agent
        from dealpoint.config import ARMS

        skill_blk = None
        if ARMS[args.arm]["skill"]:
            from dealpoint.agent.skill import skill_block as _skill_block

            skill_blk = _skill_block(question.id)
        finding, record = run_agent(
            case, doc, retriever, client, question, args.model, arm=args.arm, skill_block=skill_blk
        )

    payload = {
        "finding": finding.model_dump() if finding is not None else None,
        "record": record.model_dump(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 1 if record.status == "EXECUTION_FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
