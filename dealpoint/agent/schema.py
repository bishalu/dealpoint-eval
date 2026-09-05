"""Public finding schema + internal execution record (spec §1.2, brief §1.2).

Two schemas live here, deliberately kept separate:

- `Finding`: the public output contract (answer, evidence, rationale) —
  question-agnostic; the `answer in options ∪ {"ABSTAIN"}` check happens in
  `loop.py`/`pipeline.py`, where the question spec is in hand.
- `ExecutionRecord`: never exposed as an answer; keeps operational failure
  (CAP_HIT, EXECUTION_FAILED, ...) out of legal/accuracy metrics.
"""

from __future__ import annotations

import json
import re as _re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from dealpoint.config import EVIDENCE_MAX_ITEMS, RATIONALE_MAX_WORDS
from dealpoint.data.canonical import canonicalise

Status = Literal["ANSWERED", "ABSTAINED", "CAP_HIT", "EXECUTION_FAILED"]
FailureReason = Literal[
    "schema_invalid_after_retry",
    "api_error",
    "tool_error",
    "cap_hit",
]


class Evidence(BaseModel):
    section_ref: str
    quote: str


class Finding(BaseModel):
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    rationale: str

    @field_validator("rationale")
    @classmethod
    def _rationale_word_limit(cls, v: str) -> str:
        if len(v.split()) > RATIONALE_MAX_WORDS:
            raise ValueError(f"rationale exceeds {RATIONALE_MAX_WORDS} words")
        return v

    @model_validator(mode="after")
    def _evidence_count(self) -> Finding:
        if self.answer == "ABSTAIN":
            if self.evidence:
                raise ValueError("evidence must be empty when answer is ABSTAIN")
        else:
            if not (1 <= len(self.evidence) <= EVIDENCE_MAX_ITEMS):
                raise ValueError(
                    f"evidence must have 1..{EVIDENCE_MAX_ITEMS} items unless answer is ABSTAIN"
                )
        return self


class TrajectoryStep(BaseModel):
    tool: str
    args: dict
    result_ref: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    char_ranges: list[tuple[int, int]] = Field(default_factory=list)
    t_ms: int = 0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    wall_ms: int = 0
    tool_calls: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0


class ExecutionRecord(BaseModel):
    status: Status
    failure_reason: FailureReason | None = None
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    wall_ms: int = 0
    model: str = ""
    arm: str = ""
    case_id: str = ""
    index_version: str | None = None
    chunk_version: str | None = None
    # M4.1 (spec deliverable 1): evidence on every failure. Metadata only --
    # never added to SCORE_FIELD_NAMES / the Braintrust six-score budget.
    failure_detail: str | None = None
    raw_final_text: str | None = None
    finish_reasons: list[str] = Field(default_factory=list)


def extract_json_object(raw: str | None) -> str | None:
    """Tolerant extraction of a JSON object from a model's raw text response
    (spec deliverable 2). Rules, applied in order, first hit wins:

      1. `raw.strip()` already parses as a JSON object.
      2. A fenced code block (```json ... ``` or ``` ... ```) whose body parses.
      3. The first balanced `{ ... }` span (string/escape aware), trying each
         `{` in turn if an earlier one does not yield a parseable object.

    Returns `None` if nothing parses -- in particular, truncated/unbalanced
    JSON (no closing brace) is never "repaired", it just fails.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None

    def _parses_as_object(candidate: str) -> bool:
        try:
            return isinstance(json.loads(candidate), dict)
        except json.JSONDecodeError:
            return False

    if _parses_as_object(text):
        return text

    for match in _re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, flags=_re.DOTALL):
        body = match.group(1).strip()
        if body and _parses_as_object(body):
            return body

    for i, ch in enumerate(text):
        if ch != "{":
            continue
        span = _find_balanced_brace(text, i)
        if span is not None and _parses_as_object(span):
            return span

    return None


def _find_balanced_brace(text: str, start: int) -> str | None:
    """From `text[start]` (which must be `"{"`), scan to the matching `"}"`,
    respecting string literals and backslash escapes. Returns the span
    `text[start:end+1]` or `None` if the braces never balance (truncated).
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def match_option(answer: str, options: Sequence[str]) -> str | None:
    """Normalising match of a model's `answer` string against `options` (spec
    deliverable 2): maps curly quotes/whitespace/case/wrapping-quote variants
    back to the EXACT option string, or to the literal `"ABSTAIN"`.

    No fuzzy/prefix matching -- a near-miss is still a failure (returns `None`).
    """

    def _norm(s: str) -> str:
        s = canonicalise(s).casefold().strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1].strip()
        return s

    normalised = _norm(answer)
    if normalised == _norm("ABSTAIN"):
        return "ABSTAIN"
    for option in options:
        if _norm(option) == normalised:
            return option
    return None


def validate_finding_json(raw_content: str | None, question) -> tuple[Finding | None, str | None]:
    """Parse and validate a model's raw JSON content against `question`.

    Returns `(finding, None)` on success or `(None, error_message)` on any of:
    unparseable JSON (tolerant extraction attempted first, spec deliverable
    2), Pydantic validation failure (including the rationale word limit and
    evidence-count rules), or `answer` not in this question's options
    \u222a {"ABSTAIN"} (also via the normalising `match_option`, spec
    deliverable 2 -- on a match, `finding.answer` is rewritten to the exact
    option string so `answer_correct`'s exact-string comparison is unaffected).
    """
    if not raw_content or not raw_content.strip():
        return None, "empty response content"
    extracted = extract_json_object(raw_content)
    if extracted is None:
        return None, "no JSON object found in response"
    try:
        payload = json.loads(extracted)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    try:
        finding = Finding.model_validate(payload)
    except ValidationError as exc:
        return None, f"schema validation failed: {exc}"
    allowed = set(question.options) | {"ABSTAIN"}
    if finding.answer in allowed:
        return finding, None
    matched = match_option(finding.answer, question.options)
    if matched is not None:
        finding = finding.model_copy(update={"answer": matched})
        return finding, None
    return None, f"answer {finding.answer!r} not in allowed options {sorted(allowed)!r}"


def finding_json_schema(question) -> dict:
    """Strict JSON schema for `response_format`, with `answer` bound to this
    question's exact option strings plus "ABSTAIN".
    """
    options = list(question.options) + ["ABSTAIN"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "finding",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "answer": {"type": "string", "enum": options},
                    "evidence": {
                        "type": "array",
                        "maxItems": EVIDENCE_MAX_ITEMS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "section_ref": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["section_ref", "quote"],
                        },
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["answer", "evidence", "rationale"],
            },
        },
    }
