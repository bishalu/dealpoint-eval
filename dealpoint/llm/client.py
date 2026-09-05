"""OpenRouter LLM client (native tool calling) + `FakeClient` for offline tests.

Both the real client and the fake live here per the milestone spec. Every
metered call passes `extra_body={"usage": {"include": True}}` so realised USD
is captured in `usage.cost`, and appends one line to the spend ledger
(`data/results/spend_ledger.jsonl`) with the exact key `usd` — the M4-M6
unattended spend guard (`adws/adw_modules/spend.py::realized_usd`) sums that
field.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

from dealpoint.config import MILESTONE_TAG, OPENROUTER_BASE_URL, SPEND_LEDGER_PATH


class ApiError(Exception):
    """Generic, provider-agnostic API error, retryable by the agent loop.

    Used both by `FakeClient` to script an infra failure and, indirectly,
    real API failures: `retryable_exceptions()` below returns this plus any
    retryable `openai` exception classes available in the environment.
    """


def retryable_exceptions() -> tuple[type[Exception], ...]:
    """Exception types the agent loop should retry (connection/timeout/429/5xx).

    Deliberately excludes `openai.BadRequestError` (400) and other 4xx client
    errors -- a malformed request will never succeed on retry -- by naming
    only the specific retryable subclasses rather than the shared
    `APIStatusError` base (which `BadRequestError` also inherits from).
    """
    try:
        import openai

        return (
            ApiError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
    except ImportError:
        return (ApiError,)


def api_exceptions() -> tuple[type[Exception], ...]:
    """Every exception the agent loop should classify as `failure_reason="api_error"`.

    A superset of `retryable_exceptions()`: includes non-retryable API errors
    (e.g. a 400) too, since those are still an infra/API failure, just one
    `call_with_retries` will not retry them (it re-raises immediately).
    """
    try:
        import openai

        return (ApiError, openai.OpenAIError)
    except ImportError:
        return (ApiError,)


# M4.1 (spec deliverable 3): `response_format` capability per model, pinned.
# Both models this project currently sends metered traffic to are verified
# (2026-09-04/05) to support OpenRouter's `json_schema` response_format;
# the table exists so a future non-Anthropic/non-GLM provider can be pinned
# to the `json_object` fallback without touching call sites. Unknown models
# default to `True` (try the strict schema first) -- the 400-message
# downgrade in the agent loops is the safety net for a provider that
# silently disagrees with this table.
SUPPORTS_JSON_SCHEMA: dict[str, bool] = {
    "anthropic/claude-haiku-4.5": True,
    "z-ai/glm-5.3-flash": True,
}
DEFAULT_SUPPORTS_JSON_SCHEMA = True

# A 400 whose message mentions one of these is a `response_format`-shape
# rejection, not a generic bad request -- worth exactly one downgrade retry
# to `{"type": "json_object"}` (spec deliverable 3).
RESPONSE_FORMAT_ERROR_MARKERS: tuple[str, ...] = (
    "response_format",
    "json_schema",
    "structured output",
    "structured_outputs",
)


def response_format_for(model: str, question) -> dict:
    """`finding_json_schema(question)` when `model` is pinned (or defaulted)
    to support it, else `{"type": "json_object"}` -- the `json_schema` payload
    itself is byte-identical to before (spec deliverable 3).
    """
    from dealpoint.agent.schema import finding_json_schema

    if SUPPORTS_JSON_SCHEMA.get(model, DEFAULT_SUPPORTS_JSON_SCHEMA):
        return finding_json_schema(question)
    return {"type": "json_object"}


def is_response_format_error(exc: BaseException) -> bool:
    """Whether `exc` looks like a provider rejecting the `response_format`
    shape (a 400 naming it), rather than an unrelated bad request."""
    status_code = getattr(exc, "status_code", None)
    if status_code is not None and status_code != 400:
        return False
    message = str(exc).lower()
    return any(marker in message for marker in RESPONSE_FORMAT_ERROR_MARKERS)


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ChatResult:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    usd_missing: bool = False


def _maybe_wrap_openai(client):
    """Apply `braintrust.wrap_openai` when braintrust is importable and a key is
    loadable; otherwise return `client` unchanged.

    Imported lazily so the offline suite and pyright stay clean when
    braintrust is absent (same rule this codebase applies to `openai`).
    """
    try:
        from dealpoint.eval.braintrust_adapter import braintrust_available

        if not braintrust_available():
            return client
        import braintrust

        return braintrust.wrap_openai(client)
    except ImportError:
        return client


def build_system_message(static_prefix: str) -> dict:
    """System message with `cache_control` on the last (only) static content block.

    The static prefix is system prompt + question spec + tool schemas,
    concatenated by the caller. Sent as a content-block list (required for
    `cache_control` to attach) rather than a bare string.
    """
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": static_prefix,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }


def _append_ledger_row(
    ledger_path,
    model: str,
    milestone_tag: str,
    input_tokens: int,
    output_tokens: int,
    usd: float,
    cached_tokens: int,
    cache_write_tokens: int,
    usd_missing: bool,
    context: dict | None = None,
) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "milestone_tag": milestone_tag,
        "model": model,
        "calls": 1,
        "usd": round(usd, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
    }
    if usd_missing:
        row["usd_missing"] = True
    if context:
        row.update(context)
    with open(ledger_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
        fh.write("\n")
        fh.flush()


class OpenRouterClient:
    """`openai` SDK against OpenRouter, native tool calling, `usage.include`."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        milestone_tag: str = MILESTONE_TAG,
        ledger_path=SPEND_LEDGER_PATH,
    ) -> None:
        import openai

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        # No internal retry: retry policy (3x with backoff, API errors only) is
        # owned by the agent loop (dealpoint.agent.loop), which is what the
        # execution record's failure_reason must reflect.
        raw_client = openai.OpenAI(api_key=key, base_url=base_url, max_retries=0)
        # `wrap_openai` (brief §2.7, spec deliverable 3) when braintrust is
        # importable AND a key is loadable -- otherwise the plain client, so
        # offline behaviour (incl. this constructor's own contract) never
        # changes based on an optional dependency being installed.
        self._client = _maybe_wrap_openai(raw_client)
        self._openai = openai
        self._milestone_tag = milestone_tag
        self._ledger_path = ledger_path
        # Merged into every ledger row this client writes (dealpoint.eval.run
        # sets this per case: {"arm": ..., "case_id": ..., "case_set": ...}),
        # so cost can later be attributed to (arm, model, case) without
        # rewriting the rows M1 already wrote (spec §3.2).
        self.context: dict[str, str] = {}

    def chat(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        max_tokens: int = 300,
        temperature: float = 0,
        extra_body: dict | None = None,
    ) -> ChatResult:
        # extra_body (added M5, spec deliverable 3): merged on top of the
        # mandatory usage.include flag -- lets a caller (the M5 judge runner)
        # send provider-specific fields such as reasoning.enabled=False
        # (reasoning-model judges otherwise spend their whole max_tokens
        # budget on hidden reasoning tokens and never emit the JSON body)
        # without changing any other call site's behaviour.
        merged_extra_body: dict = {"usage": {"include": True}}
        if extra_body:
            merged_extra_body.update(extra_body)
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": merged_extra_body,
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)

        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = getattr(usage, "cost", None)
        usd_missing = cost is None
        cost_usd = float(cost) if cost is not None else 0.0

        cached_tokens = 0
        cache_write_tokens = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached_tokens = getattr(details, "cached_tokens", 0) or 0
            cache_write_tokens = getattr(details, "cache_write_tokens", 0) or 0

        _append_ledger_row(
            self._ledger_path,
            model=model,
            milestone_tag=self._milestone_tag,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=cost_usd,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            usd_missing=usd_missing,
            context=self.context,
        )

        tool_calls: list[ToolCallRequest] = []
        for tc in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(ToolCallRequest(id=tc.id, name=tc.function.name, arguments=arguments))

        return ChatResult(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            usd_missing=usd_missing,
        )


@dataclass
class ScriptedToolCall:
    name: str
    arguments: dict
    id: str = ""


@dataclass
class ScriptedTurn:
    """One scripted response for `FakeClient.chat`.

    Exactly one of `tool_calls`, `content`, or `error` should be set.
    """

    tool_calls: list[ScriptedToolCall] | None = None
    content: str | None = None
    error: Exception | None = None
    input_tokens: int = 10
    output_tokens: int = 10


class FakeClient:
    """Offline stand-in for `OpenRouterClient`. Never touches the network or the ledger.

    Constructed with a scripted list of `ScriptedTurn`s, replayed in order as
    `.chat(...)` is called. Records every `messages` list it was given so
    tests can assert on the prompt sent.
    """

    def __init__(self, script: list[ScriptedTurn]) -> None:
        self._script = list(script)
        self._cursor = 0
        self.calls: list[dict] = []

    def chat(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        max_tokens: int = 300,
        temperature: float = 0,
        extra_body: dict | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "tools": tools,
                "response_format": response_format,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "extra_body": extra_body,
            }
        )
        if self._cursor >= len(self._script):
            raise IndexError("FakeClient script exhausted -- test scripted too few turns")
        turn = self._script[self._cursor]
        self._cursor += 1

        if turn.error is not None:
            raise turn.error

        tool_calls: list[ToolCallRequest] = []
        finish_reason = "stop"
        if turn.tool_calls:
            finish_reason = "tool_calls"
            for i, tc in enumerate(turn.tool_calls):
                call_id = tc.id or f"call_{self._cursor}_{i}"
                tool_calls.append(ToolCallRequest(id=call_id, name=tc.name, arguments=tc.arguments))

        return ChatResult(
            content=turn.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
            cost_usd=0.0,
            cached_tokens=0,
            cache_write_tokens=0,
            usd_missing=False,
        )
