"""Arm B: the bounded, native-tool-calling agent loop (spec §1.3, §4.4).

One agent, three tools, hard cap `MAX_TOOL_CALLS` (8) tool calls -> `CAP_HIT`.
Structured final output via a dedicated finalization call using
`response_format`; schema-invalid -> one retry (feeding the validation error
back) -> `EXECUTION_FAILED`; API errors -> `API_MAX_RETRIES` retries with
backoff -> `EXECUTION_FAILED`.
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
from dealpoint.agent.tools import (
    ToolResult,
    get_section,
    lookup_defined_term,
    search_agreement,
    tool_schemas,
)
from dealpoint.config import (
    API_MAX_RETRIES,
    LLM_TEMPERATURE,
    MAX_TOKENS_FINAL,
    MAX_TOKENS_TOOL_TURN,
    MAX_TOOL_CALLS,
    RETRIEVER_DEFAULT_K,
    SCHEMA_MAX_RETRIES,
)
from dealpoint.corpus.document import Document
from dealpoint.corpus.retrievers import Retriever
from dealpoint.llm.client import build_system_message

ARM = "B"


def _dispatch_tool(name: str, args: dict, doc: Document, retriever: Retriever) -> ToolResult:
    try:
        if name == "search_agreement":
            query = args.get("query", "")
            k = int(args.get("k", RETRIEVER_DEFAULT_K) or RETRIEVER_DEFAULT_K)
            return search_agreement(doc, retriever, query, k=k)
        if name == "get_section":
            return get_section(doc, args.get("section_ref", ""))
        if name == "lookup_defined_term":
            return lookup_defined_term(doc, args.get("term", ""))
    except Exception as exc:  # noqa: BLE001 - a tool must never crash the loop
        return ToolResult(text=f"tool {name!r} raised an error: {exc}")
    return ToolResult(text=f"unknown tool {name!r}")


def run_agent(
    case: dict,
    doc: Document,
    retriever: Retriever,
    client,
    question,
    model: str,
    index_version: str | None = None,
    *,
    arm: str = "B",
    skill_block: str | None = None,
) -> tuple[Finding | None, ExecutionRecord]:
    """Run the bounded agent loop for one (document, question) case.

    `case` is the case-JSONL row (only `case_id` is used here); `question`
    is a `dealpoint.data.questions.QuestionSpec`. `arm` is stamped into the
    execution record (default `"B"` preserves pre-M4 behaviour). `skill_block`
    (arm D only, composed by the caller -- never by this module's imports
    from `prompts.py`, which stays skill-free) is appended to the static
    prefix when given.
    """
    wall_start = time.monotonic()
    case_id = case.get("case_id", "")

    static_prefix = system_prompt() + "\n\n" + question_spec_block(question)
    if skill_block:
        static_prefix += "\n\n" + skill_block
    messages: list[dict] = [
        build_system_message(static_prefix),
        {
            "role": "user",
            "content": (
                "Begin. Use the available tools to locate the operative clause(s) in this "
                "agreement, then give your final answer."
            ),
        },
    ]

    trajectory: list[TrajectoryStep] = []
    total_input = total_output = 0
    total_cost = 0.0
    total_cached = total_cache_write = 0
    tool_call_count = 0
    schemas = tool_schemas()

    def _record(result) -> None:
        nonlocal total_input, total_output, total_cost, total_cached, total_cache_write
        total_input += result.input_tokens
        total_output += result.output_tokens
        total_cost += result.cost_usd
        total_cached += result.cached_tokens
        total_cache_write += result.cache_write_tokens

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
            tool_calls=tool_call_count,
            wall_start=wall_start,
            model=model,
            arm=arm,
            case_id=case_id,
            index_version=index_version,
        )
        return None, record

    cap_hit = False
    while not cap_hit:
        try:
            result = call_with_retries(
                client,
                API_MAX_RETRIES,
                messages=messages,
                model=model,
                tools=schemas,
                max_tokens=MAX_TOKENS_TOOL_TURN,
                temperature=LLM_TEMPERATURE,
            )
        except Exception:  # noqa: BLE001 - any client failure -> EXECUTION_FAILED
            return _fail("EXECUTION_FAILED", "api_error")

        _record(result)

        if not result.tool_calls:
            break  # model stopped calling tools; proceed to finalization

        messages.append(
            {
                "role": "assistant",
                "content": result.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": _dump_args(tc.arguments)},
                    }
                    for tc in result.tool_calls
                ],
            }
        )
        for tc in result.tool_calls:
            if tool_call_count >= MAX_TOOL_CALLS:
                cap_hit = True
                break
            tool_call_count += 1
            t0 = time.monotonic()
            tool_result = _dispatch_tool(tc.name, tc.arguments, doc, retriever)
            t_ms = int((time.monotonic() - t0) * 1000)
            trajectory.append(
                TrajectoryStep(
                    tool=tc.name,
                    args=tc.arguments,
                    result_ref=None,
                    chunk_ids=list(tool_result.chunk_ids),
                    char_ranges=list(tool_result.char_ranges),
                    t_ms=t_ms,
                )
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": tool_result.text}
            )

    if cap_hit:
        return _fail("CAP_HIT", "cap_hit")

    # --- finalization: dedicated structured-answer call, schema retry ----
    finding: Finding | None = None
    err: str | None = None
    schema = finding_json_schema(question)
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

        _record(result)
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
        tool_calls=tool_call_count,
        wall_start=wall_start,
        model=model,
        arm=arm,
        case_id=case_id,
        index_version=index_version,
    )
    return finding, record


def _dump_args(args: dict) -> str:
    import json

    return json.dumps(args, ensure_ascii=False)
