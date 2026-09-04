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
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from dealpoint.config import EVIDENCE_MAX_ITEMS, RATIONALE_MAX_WORDS

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


def validate_finding_json(raw_content: str | None, question) -> tuple[Finding | None, str | None]:
    """Parse and validate a model's raw JSON content against `question`.

    Returns `(finding, None)` on success or `(None, error_message)` on any of:
    unparseable JSON, Pydantic validation failure (including the rationale
    word limit and evidence-count rules), or `answer` not in this question's
    options \u222a {"ABSTAIN"}.
    """
    if not raw_content or not raw_content.strip():
        return None, "empty response content"
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    try:
        finding = Finding.model_validate(payload)
    except ValidationError as exc:
        return None, f"schema validation failed: {exc}"
    allowed = set(question.options) | {"ABSTAIN"}
    if finding.answer not in allowed:
        return None, f"answer {finding.answer!r} not in allowed options {sorted(allowed)!r}"
    return finding, None


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
