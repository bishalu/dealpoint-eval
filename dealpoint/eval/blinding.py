"""Blinded judge packets: arm/model/identity stripped, gold span only for evidence (spec §4).

`build_packet` constructs a packet from an **allowlist** of top-level keys --
never by deleting keys from the input row -- so a new field added to the
result-row schema in a later milestone cannot silently leak into what a
judge (or a human scorer) sees. `render_packet_text` is the single function
that turns a packet into the text a reader sees; both `judge_run.py`'s
prompt and `calibration.py`'s `packets.md` call it, so "the same blinded
packet is what humans score" (spec deliverable 2) is a fact about the code,
not a claim about it.

Scoping decision (deliberate, not an oversight): the text scrub below is
applied ONLY to model-generated free text -- `finding.rationale` and each
trajectory step's `args` string values. It is never applied to
`retrieved_text` or `finding.evidence[].quote`, both of which are verbatim
merger-agreement text. A merger agreement may legitimately name a party
called "Google" or a fund called "Apex Capital"; scrubbing evidence text
would corrupt the very thing the evidence-sufficiency dimension measures.
"""

from __future__ import annotations

import hashlib
import re

from dealpoint.config import JUDGE_MAX_RETRIEVED_CHARS
from dealpoint.corpus.document import Document
from dealpoint.corpus.document import defined_term as _defined_term
from dealpoint.corpus.document import get_section as _get_section
from dealpoint.eval.cases import resolve_question
from dealpoint.eval.rubric import rubric_version

# The complete set of top-level packet keys (spec §4.1). `build_packet`
# constructs exactly this set -- the blinding test asserts equality against
# it, so a stray key added to the row schema can never leak in unnoticed.
PACKET_ALLOWLIST: tuple[str, ...] = (
    "packet_id",
    "question_text",
    "options",
    "status",
    "trajectory",
    "finding",
    "gold_span",
    "gold_span_note",
    "rubric_version",
)

NO_GOLD_SPAN_NOTE = "No expert-annotated span is recorded for this case."

# Vendor/family tokens scrubbed from model-generated free text (spec §4.3).
_VENDOR_TOKENS: tuple[str, ...] = (
    "anthropic", "claude", "haiku", "sonnet", "opus",
    "openai", "gpt",
    "google", "gemini",
    "deepseek",
    "qwen", "alibaba",
    "mistral",
    "nvidia", "nemotron",
    "bytedance",
    "llama", "meta-llama",
    "grok", "x-ai", "xai",
    "z-ai", "zhipu", "glm",
    "minimax",
    "moonshot", "kimi",
    "xiaomi", "mimo",
    "nova", "amazon",
    "seed-2",
)

_VENDOR_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(t) for t in _VENDOR_TOKENS) + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_ARM_RE = re.compile(r"\barm\s+[A-D]\b", re.IGNORECASE)


def scrub_text(text: str) -> str:
    """Redact vendor/family tokens and `arm [A-D]` mentions from model-generated text.

    Never applied to retrieved contract text or quoted evidence (see module
    docstring).
    """
    text = _VENDOR_RE.sub("[REDACTED-MODEL]", text)
    text = _ARM_RE.sub("[REDACTED-ARM]", text)
    return text


def _scrub_args(args: dict) -> dict:
    out: dict = {}
    for key, value in (args or {}).items():
        out[key] = scrub_text(value) if isinstance(value, str) else value
    return out


def _enclosing_section_ref(doc: Document, offset: int) -> str:
    best = None
    for section in doc.sections:
        if section.start <= offset and (best is None or section.start > best.start):
            best = section
    if best is None:
        return ""
    return best.ref


def _truncate(text: str, limit: int = JUDGE_MAX_RETRIEVED_CHARS) -> str:
    return text[:limit]


def _packet_id(case_id: str, variant_id: str) -> str:
    return hashlib.sha256(f"{case_id}|{variant_id}".encode()).hexdigest()[:12]


def _build_trajectory_step(step: dict, doc: Document, index: int) -> dict:
    tool = step.get("tool", "")
    args = _scrub_args(step.get("args") or {})
    char_ranges = step.get("char_ranges") or []

    retrieved_text: list[str] | None
    section_refs: list[str]

    if tool == "lookup_defined_term":
        term = (step.get("args") or {}).get("term", "")
        dt = _defined_term(doc, term) if term else None
        if dt is None:
            retrieved_text = None
            section_refs = []
        else:
            retrieved_text = [_truncate(dt.text)]
            section_refs = [dt.section_ref]
    elif tool == "get_section":
        section_ref_arg = (step.get("args") or {}).get("section_ref", "")
        section = _get_section(doc, section_ref_arg) if section_ref_arg else None
        if section is not None:
            retrieved_text = [_truncate(doc.text[section.start : section.end])]
            section_refs = [section.ref]
        elif char_ranges:
            start, end = char_ranges[0]
            retrieved_text = [_truncate(doc.text[start:end])]
            section_refs = [_enclosing_section_ref(doc, start)]
        else:
            retrieved_text = None
            section_refs = []
    else:
        # search_agreement (or any future multi-range tool): one truncated
        # item + one section ref per char_range.
        if char_ranges:
            retrieved_text = [_truncate(doc.text[start:end]) for start, end in char_ranges]
            section_refs = [_enclosing_section_ref(doc, start) for start, end in char_ranges]
        else:
            retrieved_text = []
            section_refs = []

    return {
        "step": index + 1,
        "tool": tool,
        "args": args,
        "section_refs": section_refs,
        "retrieved_text": retrieved_text,
    }


def _build_finding(row: dict) -> dict | None:
    finding = row.get("finding")
    if not finding:
        return None
    rationale = scrub_text(finding.get("rationale") or "")
    evidence = [
        {"section_ref": e.get("section_ref", ""), "quote": e.get("quote", "")}
        for e in (finding.get("evidence") or [])
    ]
    return {"answer": finding.get("answer"), "evidence": evidence, "rationale": rationale}


def _build_gold_span(case: dict, doc: Document) -> tuple[str | None, str | None]:
    spans = case.get("gold_spans") or []
    if not spans:
        return None, NO_GOLD_SPAN_NOTE
    text = " ".join(doc.text[span["start"] : span["end"]] for span in spans)
    return text, None


def build_packet(row: dict, case: dict, doc: Document, *, variant_id: str) -> dict:
    """Build one blinded judge/human packet from a result row + its case + document.

    `row` is one line of a `data/results/*.jsonl` file (arm/model already
    present but never copied into the packet). `case` is the case's row
    from `test.jsonl`/`counterfactual.jsonl` (used only for `gold_spans`).
    """
    case_id = case["case_id"]
    question = resolve_question(case)
    record = row.get("record") or {}
    trajectory_in = record.get("trajectory") or []
    trajectory = [_build_trajectory_step(step, doc, i) for i, step in enumerate(trajectory_in)]
    gold_span, gold_span_note = _build_gold_span(case, doc)

    packet = {
        "packet_id": _packet_id(case_id, variant_id),
        "question_text": question.gloss,
        "options": list(question.options),
        "status": record.get("status"),
        "trajectory": trajectory,
        "finding": _build_finding(row),
        "gold_span": gold_span,
        "gold_span_note": gold_span_note,
        "rubric_version": rubric_version(),
    }
    assert set(packet.keys()) == set(PACKET_ALLOWLIST)
    return packet


def render_packet_text(packet: dict) -> str:
    """The single rendering of a packet's content -- used verbatim as both the
    judge's user-message body and the human `packets.md` section body (spec
    deliverable 2: "the same blinded packet is what humans score").
    """
    lines: list[str] = []
    lines.append(f"## Question\n\n{packet['question_text']}")
    lines.append("")
    if packet["options"]:
        lines.append("## Options\n\n" + "\n".join(f"- {opt}" for opt in packet["options"]))
    else:
        lines.append("## Options\n\n(free-form; no fixed option list for this question)")
    lines.append("")
    lines.append(f"## Status\n\n{packet['status']}")
    lines.append("")

    lines.append("## Trajectory")
    lines.append("")
    if not packet["trajectory"]:
        lines.append("(no tool calls were made)")
    for step in packet["trajectory"]:
        lines.append(f"### Step {step['step']}: `{step['tool']}`")
        lines.append("")
        lines.append(f"args: `{step['args']}`")
        lines.append("")
        if step["section_refs"]:
            lines.append(f"section refs: {', '.join(r or '(none)' for r in step['section_refs'])}")
            lines.append("")
        if step["retrieved_text"] is None:
            lines.append("(tool result text not recorded)")
        elif not step["retrieved_text"]:
            lines.append("(no retrieved text)")
        else:
            for i, text in enumerate(step["retrieved_text"]):
                lines.append(f"> {text}")
                if i < len(step["retrieved_text"]) - 1:
                    lines.append("")
        lines.append("")

    lines.append("## Finding")
    lines.append("")
    finding = packet["finding"]
    if finding is None:
        lines.append("The system produced no finding for this case.")
    else:
        lines.append(f"answer: {finding['answer']}")
        lines.append("")
        lines.append("evidence:")
        for e in finding["evidence"]:
            lines.append(f"- ({e['section_ref']}) {e['quote']}")
        lines.append("")
        lines.append(f"rationale: {finding['rationale']}")
    lines.append("")

    lines.append("## Gold span (supplied for the evidence-sufficiency dimension ONLY)")
    lines.append("")
    if packet["gold_span"] is not None:
        lines.append(packet["gold_span"])
    else:
        lines.append(packet["gold_span_note"] or NO_GOLD_SPAN_NOTE)
    lines.append("")

    return "\n".join(lines)
