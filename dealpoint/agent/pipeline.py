"""Arm A: the RAG baseline. No loop, no tools: dense top-5 chunks -> one call -> finding.

Same question spec block, same JSON schema, same validation/retry rules, same
execution record shape as arm B -- the only difference is that retrieval
happens once, deterministically, before the single LLM call, so the A->B
contrast isolates agency (brief §2.2).
"""

from __future__ import annotations

import time

from dealpoint.agent._common import build_record, call_with_retries
from dealpoint.agent.prompts import question_spec_block, system_prompt
from dealpoint.agent.schema import (
    ExecutionRecord,
    Finding,
    TrajectoryStep,
    finding_json_schema,
    validate_finding_json,
)
from dealpoint.config import (
    API_MAX_RETRIES,
    LLM_TEMPERATURE,
    MAX_TOKENS_FINAL,
    MAX_TOOL_RESULT_CHARS,
    RETRIEVER_DEFAULT_K,
    SCHEMA_MAX_RETRIES,
)
from dealpoint.corpus.document import Document
from dealpoint.corpus.retrievers import Retriever
from dealpoint.llm.client import build_system_message

ARM = "A"


def _render_context(doc: Document, chunks) -> str:
    blocks = []
    for c in chunks:
        text = c.text
        truncated = len(text) > MAX_TOOL_RESULT_CHARS
        if truncated:
            text = text[:MAX_TOOL_RESULT_CHARS]
        suffix = " \u2026[truncated]" if truncated else ""
        blocks.append(f"[section {c.section_ref or '(none)'} | chars {c.start}-{c.end}] {text}{suffix}")
    return "\n\n".join(blocks) if blocks else "(no chunks retrieved)"


def run_pipeline(
    case: dict,
    doc: Document,
    retriever: Retriever,
    client,
    question,
    model: str,
    index_version: str | None = None,
) -> tuple[Finding | None, ExecutionRecord]:
    """Arm A: one implicit retrieval (dense top-5) + one LLM call + finding."""
    wall_start = time.monotonic()
    case_id = case.get("case_id", "")

    chunks = retriever.search(doc.document_id, question.canonical_query, k=RETRIEVER_DEFAULT_K)
    trajectory = [
        TrajectoryStep(
            tool="search_agreement",
            args={"query": question.canonical_query, "k": RETRIEVER_DEFAULT_K},
            result_ref=None,
            chunk_ids=[c.chunk_id for c in chunks],
            char_ranges=[(c.start, c.end) for c in chunks],
            t_ms=0,
        )
    ]

    static_prefix = (
        system_prompt()
        + "\n\n"
        + question_spec_block(question)
        + "\n\nRetrieved passages from this agreement:\n"
        + _render_context(doc, chunks)
    )
    messages = [
        build_system_message(static_prefix),
        {"role": "user", "content": "Give your final answer based only on the passages above."},
    ]

    total_input = total_output = 0
    total_cost = 0.0
    total_cached = total_cache_write = 0

    def _fail(status: str, failure_reason: str | None) -> tuple[None, ExecutionRecord]:
        record = build_record(
            status=status,  # type: ignore[arg-type]
            failure_reason=failure_reason,  # type: ignore[arg-type]
            trajectory=trajectory,
            input_tokens=total_input,
            output_tokens=total_output,
            cost_usd=total_cost,
            cached_tokens=total_cached,
            cache_write_tokens=total_cache_write,
            tool_calls=0,
            wall_start=wall_start,
            model=model,
            arm=ARM,
            case_id=case_id,
            index_version=index_version,
        )
        return None, record

    schema = finding_json_schema(question)
    finding: Finding | None = None
    err: str | None = None
    final_messages = list(messages)
    for attempt in range(SCHEMA_MAX_RETRIES + 1):
        if attempt > 0:
            final_messages = final_messages + [
                {
                    "role": "user",
                    "content": (
                        f"Your previous answer was invalid: {err}. Respond again with ONLY a "
                        "JSON object matching the required schema."
                    ),
                }
            ]
        try:
            result = call_with_retries(
                client,
                API_MAX_RETRIES,
                messages=final_messages,
                model=model,
                response_format=schema,
                max_tokens=MAX_TOKENS_FINAL,
                temperature=LLM_TEMPERATURE,
            )
        except Exception:  # noqa: BLE001 - any client failure -> EXECUTION_FAILED
            return _fail("EXECUTION_FAILED", "api_error")

        total_input += result.input_tokens
        total_output += result.output_tokens
        total_cost += result.cost_usd
        total_cached += result.cached_tokens
        total_cache_write += result.cache_write_tokens

        finding, err = validate_finding_json(result.content, question)
        if finding is not None:
            break

    if finding is None:
        return _fail("EXECUTION_FAILED", "schema_invalid_after_retry")

    status = "ABSTAINED" if finding.answer == "ABSTAIN" else "ANSWERED"
    record = build_record(
        status=status,  # type: ignore[arg-type]
        failure_reason=None,
        trajectory=trajectory,
        input_tokens=total_input,
        output_tokens=total_output,
        cost_usd=total_cost,
        cached_tokens=total_cached,
        cache_write_tokens=total_cache_write,
        tool_calls=0,
        wall_start=wall_start,
        model=model,
        arm=ARM,
        case_id=case_id,
        index_version=index_version,
    )
    return finding, record
