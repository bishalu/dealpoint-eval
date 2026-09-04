"""CLI: `python -m dealpoint.agent.run --case <case_id> --arm A|B --model <id> [--fake]`.

Prints `{"finding": ..., "record": ...}` as JSON to stdout. Resolves the case
from dev/test/counterfactual JSONL by `case_id`. Exit non-zero on
`EXECUTION_FAILED` so the CLI is usable in a shell chain.
"""

from __future__ import annotations

import argparse
import json
import sys

from dealpoint.config import (
    COUNTERFACTUAL_JSONL_PATH,
    DEFAULT_MODEL,
    DEV_JSONL_PATH,
    TEST_JSONL_PATH,
)
from dealpoint.data.questions import OUT_OF_SCOPE_QUESTION_BY_ID, QUESTION_BY_ID, QuestionSpec


def _find_case(case_id: str) -> dict:
    for path in (DEV_JSONL_PATH, TEST_JSONL_PATH, COUNTERFACTUAL_JSONL_PATH):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("case_id") == case_id:
                    return row
    raise KeyError(f"case_id {case_id!r} not found in dev/test/counterfactual JSONL")


def _resolve_document_id(case: dict) -> str:
    """The document a case is asked against: the redacted variant itself for a
    redacted counterfactual case, otherwise the base agreement.
    """
    if case.get("kind") == "redacted":
        return case["case_id"]
    return case["agreement_id"]


def _resolve_question(case: dict) -> QuestionSpec:
    question_id = case["question_id"]
    if question_id in QUESTION_BY_ID:
        return QUESTION_BY_ID[question_id]
    # out-of-scope counterfactual case: no fixed option list.
    oos = OUT_OF_SCOPE_QUESTION_BY_ID.get(question_id)
    text = case.get("question_text") or (oos.text if oos else question_id)
    return QuestionSpec(
        id=question_id,
        maud_question=text,
        text_type="out-of-scope",
        category="out-of-scope",
        gloss=text,
        options=(),
        canonical_query=text,
        reasoning_type="out-of-scope",
        required_evidence=None,
    )


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


def _build_retriever():
    from dealpoint.corpus.retrievers import LazyRetriever

    return LazyRetriever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dealpoint.agent.run")
    parser.add_argument("--case", required=True, help="case_id, e.g. contract_0__q01")
    parser.add_argument("--arm", required=True, choices=["A", "B"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fake", action="store_true", help="use FakeClient (no network, no key)")
    args = parser.parse_args(argv)

    from dealpoint.corpus.document import load_document

    case = _find_case(args.case)
    document_id = _resolve_document_id(case)
    doc = load_document(document_id)
    question = _resolve_question(case)
    client = _build_client(args.fake)
    retriever = _build_retriever()

    if args.arm == "A":
        from dealpoint.agent.pipeline import run_pipeline

        finding, record = run_pipeline(case, doc, retriever, client, question, args.model)
    else:
        from dealpoint.agent.loop import run_agent

        finding, record = run_agent(case, doc, retriever, client, question, args.model)

    payload = {
        "finding": finding.model_dump() if finding is not None else None,
        "record": record.model_dump(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 1 if record.status == "EXECUTION_FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
