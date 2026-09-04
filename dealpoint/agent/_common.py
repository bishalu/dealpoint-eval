"""Private helpers shared by `loop.py` (arm B) and `pipeline.py` (arm A).

Not part of the milestone spec's named module list -- kept here so the retry
wrapper and execution-record construction are not duplicated between the two
call sites.
"""

from __future__ import annotations

import time

from dealpoint.agent.schema import ExecutionRecord, FailureReason, Status, TrajectoryStep, Usage
from dealpoint.corpus.chunks import chunk_version as _chunk_version


def call_with_retries(client, max_retries: int, **kwargs):
    """Call `client.chat(**kwargs)`, retrying retryable API errors with backoff.

    Schema-invalid content is NOT a retryable-exception case -- that retry
    policy lives in the caller (exactly one retry, feeding back the error).
    Backoff is deliberately small so the offline `gate_m1` API-error test
    stays fast; it is still exponential, just at a scale a unit test can
    afford.
    """
    from dealpoint.llm.client import retryable_exceptions

    exc_types = retryable_exceptions()
    last_exc: Exception = RuntimeError("call_with_retries: no attempts were made")
    for attempt in range(max_retries + 1):
        try:
            return client.chat(**kwargs)
        except exc_types as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            time.sleep(min(0.05 * (2**attempt), 2.0))
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
    )
