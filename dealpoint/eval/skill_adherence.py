"""Deterministic skill-adherence rules: Appendix B, one `Rule` per number (spec deliverable 2).

Each rule is a pair of pure predicates -- `applicable(case, finding, record, doc)` and
`satisfied(case, finding, record, doc)` -- evaluated over the trajectory/finding/case
labels only, never a model judgement. Computed for every arm (arm A's smaller tool
surface just means fewer rules are *applicable* to it -- see the honesty note in
`evaluate`'s docstring).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from dealpoint.agent.schema import ExecutionRecord, Finding
from dealpoint.corpus.document import Document, _normalise_ref, _refs_match
from dealpoint.eval.scorers import (
    _term_matches,
    locate_quote,
    overlap_chars,
    parse_required_evidence,
)

Predicate = Callable[..., bool]

_CROSS_REF_RE = re.compile(r"\bSection\s+\d+\.\d+\b|\bArticle\s+[IVXL]+\b", re.IGNORECASE)
_MID_LIST_ENDINGS = (";", ",", " and", " or")


@dataclass(frozen=True)
class Rule:
    id: int
    name: str
    applicable: Predicate
    satisfied: Predicate


def _canonical_text(doc: Document | None) -> str:
    return doc.text if doc is not None else ""


def _step_text(step, doc: Document | None) -> list[str]:
    text = _canonical_text(doc)
    return [text[s:e] for (s, e) in step.char_ranges if text]


def _all_retrieved_texts(record: ExecutionRecord, doc: Document | None) -> list[str]:
    texts: list[str] = []
    for step in record.trajectory:
        texts.extend(_step_text(step, doc))
    return texts


def _extract_cross_refs(text: str) -> list[str]:
    return [m.group(0) for m in _CROSS_REF_RE.finditer(text)]


def _ref_target(raw: str) -> tuple[str, object] | None:
    # "Section 6.3" -> "6.3"; "Article VI" -> "VI"
    parts = raw.split(None, 1)
    bare = parts[1] if len(parts) == 2 else raw
    return _normalise_ref(bare)


# --- rule 1: >= 1 search_agreement step -------------------------------------


def _r1_applicable(case, finding, record: ExecutionRecord, doc) -> bool:
    return True


def _r1_satisfied(case, finding, record: ExecutionRecord, doc) -> bool:
    return any(step.tool == "search_agreement" for step in record.trajectory)


# --- rule 2: defined-term lookup for required_evidence ----------------------


def _r2_applicable(case, finding, record: ExecutionRecord, doc) -> bool:
    return case.get("required_evidence") is not None


def _r2_satisfied(case, finding, record: ExecutionRecord, doc) -> bool:
    candidates = parse_required_evidence(case.get("required_evidence"))
    if not candidates:
        return False
    for step in record.trajectory:
        if step.tool != "lookup_defined_term":
            continue
        term_arg = str(step.args.get("term", ""))
        if any(_term_matches(candidate, term_arg) for candidate in candidates):
            return True
    return False


# --- rule 3: follow a cross-reference in retrieved text ----------------------


def _r3_applicable(case, finding, record: ExecutionRecord, doc) -> bool:
    for text in _all_retrieved_texts(record, doc):
        if _extract_cross_refs(text):
            return True
    return False


def _r3_satisfied(case, finding, record: ExecutionRecord, doc) -> bool:
    refs_seen_by_step: list[list[str]] = [
        _extract_cross_refs(t) for t in [
            "\n".join(_step_text(step, doc)) for step in record.trajectory
        ]
    ]
    for i, refs in enumerate(refs_seen_by_step):
        if not refs:
            continue
        targets = [_ref_target(r) for r in refs]
        for step in record.trajectory[i + 1 :]:
            if step.tool != "get_section":
                continue
            got = _normalise_ref(str(step.args.get("section_ref", "")))
            if any(_refs_match(got, t) for t in targets):
                return True
    return False


# --- rule 4: read a carve-out list to the end --------------------------------


def _mid_list_chunks(record: ExecutionRecord, doc) -> list[tuple[int, int]]:
    text = _canonical_text(doc)
    out: list[tuple[int, int]] = []
    if not text:
        return out
    for step in record.trajectory:
        for s, e in step.char_ranges:
            stripped = text[s:e].rstrip()
            if stripped and stripped.endswith(_MID_LIST_ENDINGS):
                out.append((s, e))
    return out


def _r4_applicable(case, finding, record: ExecutionRecord, doc) -> bool:
    return len(_mid_list_chunks(record, doc)) > 0


def _r4_satisfied(case, finding, record: ExecutionRecord, doc) -> bool:
    mid_list_ends = [e for (_s, e) in _mid_list_chunks(record, doc)]
    if not mid_list_ends:
        return False
    all_ranges = [cr for step in record.trajectory for cr in step.char_ranges]
    for end in mid_list_ends:
        for s, _e in all_ranges:
            if s != end and abs(s - end) <= 200 and s >= end - 10:
                return True
    return False


# --- rule 5: citation discipline ---------------------------------------------


def _r5_applicable(case, finding: Finding | None, record: ExecutionRecord, doc) -> bool:
    return finding is not None and bool(finding.evidence)


def _r5_satisfied(case, finding: Finding | None, record: ExecutionRecord, doc) -> bool:
    if finding is None or not finding.evidence:
        return False
    if len(finding.evidence) > 3:
        return False
    text = _canonical_text(doc)
    all_ranges = [cr for step in record.trajectory for cr in step.char_ranges]
    for ev in finding.evidence:
        located = locate_quote(ev.quote, text)
        if not located:
            return False
        if not any(overlap_chars(loc, cr) > 0 for loc in located for cr in all_ranges):
            return False
    return True


# --- rule 6: >= 2 differently-phrased searches before abstaining ------------


def _normalise_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip()).casefold()


def _r6_applicable(case, finding, record: ExecutionRecord, doc) -> bool:
    return record.status == "ABSTAINED"


def _r6_satisfied(case, finding, record: ExecutionRecord, doc) -> bool:
    queries = {
        _normalise_query(str(step.args.get("query", "")))
        for step in record.trajectory
        if step.tool == "search_agreement"
    }
    queries.discard("")
    return len(queries) >= 2


# --- rule 7: abstain only when genuinely absent ------------------------------


def _r7_applicable(case, finding, record: ExecutionRecord, doc) -> bool:
    return record.status == "ABSTAINED"


def _r7_satisfied(case, finding, record: ExecutionRecord, doc) -> bool:
    return case.get("case_set") == "counterfactual"


# --- rule 8: rationale discipline --------------------------------------------


def _r8_applicable(case, finding: Finding | None, record: ExecutionRecord, doc) -> bool:
    return finding is not None


def _r8_satisfied(case, finding: Finding | None, record: ExecutionRecord, doc) -> bool:
    if finding is None:
        return False
    if len(finding.rationale.split()) > 80:
        return False
    rationale_cf = finding.rationale.casefold()
    if finding.answer.casefold() in rationale_cf:
        return True
    return bool(_CROSS_REF_RE.search(finding.rationale)) or bool(
        re.search(r"\b\d+\.\d+\b", finding.rationale)
    )


RULES: tuple[Rule, ...] = (
    Rule(1, "search_before_answer", _r1_applicable, _r1_satisfied),
    Rule(2, "defined_term_lookup", _r2_applicable, _r2_satisfied),
    Rule(3, "follow_cross_reference", _r3_applicable, _r3_satisfied),
    Rule(4, "read_carveouts_to_end", _r4_applicable, _r4_satisfied),
    Rule(5, "citation_discipline", _r5_applicable, _r5_satisfied),
    Rule(6, "search_before_abstain", _r6_applicable, _r6_satisfied),
    Rule(7, "abstain_only_when_absent", _r7_applicable, _r7_satisfied),
    Rule(8, "rationale_discipline", _r8_applicable, _r8_satisfied),
)


def evaluate(
    case: dict, finding: Finding | None, record: ExecutionRecord, doc: Document | None
) -> dict:
    """Score every Appendix-B rule for one case's execution.

    Returns `{"score": n_satisfied / n_applicable (or None if 0 applicable),
    "n_applicable": int, "n_satisfied": int, "rules": {"1": "satisfied" |
    "violated" | "n/a", ...}}`.

    Honesty note: rules 2/3/4/6 need tools arm A does not have (it makes one
    implicit retrieval, never calls `lookup_defined_term`/`get_section`, and
    never abstains via a multi-search loop), so arm A's applicable set is
    structurally smaller than arms B/C/D's -- an A-vs-D adherence *ratio*
    comparison is not like-for-like; report `n_applicable`/`n_satisfied` per
    rule per arm alongside the ratio (spec §3 honesty note).
    """
    n_applicable = 0
    n_satisfied = 0
    rules: dict[str, str] = {}
    for rule in RULES:
        if rule.applicable(case, finding, record, doc):
            n_applicable += 1
            ok = rule.satisfied(case, finding, record, doc)
            if ok:
                n_satisfied += 1
            rules[str(rule.id)] = "satisfied" if ok else "violated"
        else:
            rules[str(rule.id)] = "n/a"
    score = (n_satisfied / n_applicable) if n_applicable else None
    return {
        "score": score,
        "n_applicable": n_applicable,
        "n_satisfied": n_satisfied,
        "rules": rules,
    }
