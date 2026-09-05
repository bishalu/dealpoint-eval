"""Private helpers shared by `loop.py` (arm B) and `pipeline.py` (arm A).

Not part of the milestone spec's named module list -- kept here so the retry
wrapper and execution-record construction are not duplicated between the two
call sites.
"""

from __future__ import annotations

import random
import time

from dealpoint.agent.schema import ExecutionRecord, FailureReason, Status, TrajectoryStep, Usage
from dealpoint.config import API_RETRY_BASE_DELAY_S, FAILURE_DETAIL_MAX_CHARS
from dealpoint.corpus.chunks import chunk_version as _chunk_version


def describe_exception(exc: BaseException) -> str:
    """`"<ExcClass>: <first N chars of the message>"` (spec deliverable 1).

    For an `openai.APIStatusError` (or any exception exposing `status_code`
    and/or a `body`/`response` with a provider error `code`/`type`), the
    HTTP status and provider error code/type are prefixed onto the message
    when present -- that is the field expected to actually name causes like
    the Haiku arm-D failure. Every attribute access is guarded so a missing
    attribute can never itself raise.
    """
    cls_name = type(exc).__name__
    message = str(exc)

    prefix_parts: list[str] = []
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        prefix_parts.append(f"status={status_code}")
    code = getattr(exc, "code", None)
    if code:
        prefix_parts.append(f"code={code}")
    err_type = getattr(exc, "type", None)
    if err_type:
        prefix_parts.append(f"type={err_type}")

    if prefix_parts:
        message = f"[{' '.join(prefix_parts)}] {message}"

    detail = f"{cls_name}: {message}"
    return detail[:FAILURE_DETAIL_MAX_CHARS]


def _extract_retry_after(exc: BaseException) -> float | None:
    """Seconds to wait, from a `retry_after` attribute or a `Retry-After`
    response header, if either is present and parseable. Never raises."""
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            header_val = headers.get("Retry-After")
        except AttributeError:
            header_val = None
        if header_val is not None:
            try:
                return float(header_val)
            except (TypeError, ValueError):
                pass
    return None


def call_with_response_format_downgrade(
    client,
    max_retries: int,
    *,
    messages: list[dict],
    model: str,
    question,
    max_tokens: int,
    temperature: float,
    base_delay_s: float | None = None,
):
    """Call the model with `response_format_for(model, question)`; on a
    `response_format`-shaped 400 (Anthropic/OpenRouter rejecting `json_schema`
    on some non-Anthropic path, or the like), retry exactly once with
    `{"type": "json_object"}` (spec deliverable 3: "use json_schema where
    supported, fall back to json_object ... otherwise").

    Returns `(result, downgraded, downgrade_detail)`. `downgrade_detail` is a
    human-readable note naming the error that triggered the downgrade -- the
    caller surfaces it (as `failure_detail` metadata, win or lose) so the
    provider limitation is visible rather than silently absorbed.
    """
    from dealpoint.llm.client import is_response_format_error, response_format_for

    schema = response_format_for(model, question)
    try:
        result = call_with_retries(
            client,
            max_retries,
            base_delay_s=base_delay_s,
            messages=messages,
            model=model,
            response_format=schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result, False, None
    except Exception as exc:
        if not is_response_format_error(exc):
            raise
        downgrade_detail = f"response_format downgraded to json_object after: {describe_exception(exc)}"
        result = call_with_retries(
            client,
            max_retries,
            base_delay_s=base_delay_s,
            messages=messages,
            model=model,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result, True, downgrade_detail


def call_with_retries(client, max_retries: int, *, base_delay_s: float | None = None, **kwargs):
    """Call `client.chat(**kwargs)`, retrying retryable API errors with backoff.

    Schema-invalid content is NOT a retryable-exception case -- that retry
    policy lives in the caller (exactly one retry, feeding back the error).

    Backoff is real exponential backoff with jitter, based at
    `base_delay_s` (default `dealpoint.config.API_RETRY_BASE_DELAY_S`),
    honouring the exception's `retry_after` (seconds) when present -- e.g.
    `Retry-After` surfaced by the client. Tests pass a small `base_delay_s`
    (or the `gate_m1` API-error test's own conftest fixture lowers the
    config default) to stay sub-second offline.
    """
    from dealpoint.llm.client import retryable_exceptions

    base = base_delay_s if base_delay_s is not None else API_RETRY_BASE_DELAY_S
    exc_types = retryable_exceptions()
    last_exc: Exception = RuntimeError("call_with_retries: no attempts were made")
    for attempt in range(max_retries + 1):
        try:
            return client.chat(**kwargs)
        except exc_types as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            retry_after = _extract_retry_after(exc)
            if retry_after is not None:
                delay = retry_after
            else:
                delay = base * (2**attempt)
            delay *= 1.0 + random.uniform(0, 0.1)
            time.sleep(delay)
    raise last_exc  # pragma: no cover - loop always returns or raises


def build_record(
    *,
    status: Status,
    failure_reason: FailureReason | None,
    trajectory: list[TrajectoryStep],
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    cached_tokens: int,
    cache_write_tokens: int,
    tool_calls: int,
    wall_start: float,
    model: str,
    arm: str,
    case_id: str,
    index_version: str | None = None,
    chunk_version: str | None = None,
    failure_detail: str | None = None,
    raw_final_text: str | None = None,
    finish_reasons: list[str] | None = None,
) -> ExecutionRecord:
    wall_ms = int((time.monotonic() - wall_start) * 1000)
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        wall_ms=wall_ms,
        tool_calls=tool_calls,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    return ExecutionRecord(
        status=status,
        failure_reason=failure_reason,
        trajectory=trajectory,
        usage=usage,
        wall_ms=wall_ms,
        model=model,
        arm=arm,
        case_id=case_id,
        index_version=index_version,
        chunk_version=chunk_version if chunk_version is not None else _chunk_version(),
        failure_detail=failure_detail,
        raw_final_text=raw_final_text,
        finish_reasons=list(finish_reasons) if finish_reasons is not None else [],
    )
