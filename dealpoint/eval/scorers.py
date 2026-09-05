"""Deterministic scorers over (case, finding, execution record, canonical text).

Pure functions, no I/O, no globals (spec §2). Every scorer shares the same
signature — ``(case, finding, record, canonical_text) -> bool | int | float |
None`` — even the ones that only look at ``record``, so callers (the runner,
the Braintrust adapter, tests) can iterate a name -> function table uniformly.
``required_evidence_met`` is the one exception: it takes an extra, optional
``doc`` keyword because rule 2 (evidence "by coverage") needs the `Document`
to resolve a defined-term block.

``None`` means *not applicable to this case* and must never be coerced to
``False``/``0`` — an ``answer_correct`` of ``None`` on a ``CAP_HIT`` case must
not count as a wrong answer in any aggregate.
"""

from __future__ import annotations

import re

from dealpoint.agent.schema import ExecutionRecord, Finding
from dealpoint.config import MIN_GOLD_OVERLAP_CHARS
from dealpoint.corpus.document import Document
from dealpoint.data.canonical import normalise_quote

Range = tuple[int, int]


# --- geometry helpers (private, unit-tested through the scorers) -----------


def overlap_chars(a: Range, b: Range) -> int:
    """Character overlap between two half-open [start, end) ranges."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def gold_ranges(case: dict) -> list[Range]:
    return [(span["start"], span["end"]) for span in case.get("gold_spans") or []]


def locate_quote(quote: str, canonical_text: str) -> list[Range]:
    """Every occurrence of `normalise_quote(quote)` in `canonical_text`.

    Empty list means the quote is not verbatim. A quote that normalises to
    the empty string never "locates" (an unbounded `str.find` loop over ""
    would otherwise match everywhere).
    """
    normalized = normalise_quote(quote)
    if not normalized:
        return []
    spans: list[Range] = []
    start = 0
    while True:
        idx = canonical_text.find(normalized, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(normalized)))
        start = idx + 1
    return spans


def _any_gold_overlap(span: Range, golds: list[Range]) -> bool:
    return any(overlap_chars(span, g) >= MIN_GOLD_OVERLAP_CHARS for g in golds)


# --- required_evidence_met: deterministic term parsing ----------------------

_OR_SPLIT_RE = re.compile(r"\(or |\)")
_DEFINED_TERM_PREFIX = "defined term "


def parse_required_evidence(spec: str | None) -> tuple[str, ...]:
    """Extract candidate defined-term strings from a `required_evidence` prose value.

    Rule: strip a leading `"defined term "`, strip quotes, split on `"(or "` /
    `")"`, strip each part, drop empties.
    """
    if spec is None:
        return ()
    s = spec
    s = s.removeprefix(_DEFINED_TERM_PREFIX)
    s = s.replace('"', "").replace("'", "")
    parts = _OR_SPLIT_RE.split(s)
    return tuple(p.strip() for p in parts if p.strip())


def _term_matches(candidate: str, term_arg: str) -> bool:
    c = candidate.strip().casefold()
    t = term_arg.strip().casefold()
    if not c or not t:
        return False
    return c == t or c in t


# --- scorers -----------------------------------------------------------


def gold_seen(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> bool | None:
    golds = gold_ranges(case)
    if not golds:
        return None
    for step in record.trajectory:
        for cr in step.char_ranges:
            if _any_gold_overlap(cr, golds):
                return True
    return False


def gold_first_rank(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> int | None:
    golds = gold_ranges(case)
    if not golds:
        return None
    first_search = next((s for s in record.trajectory if s.tool == "search_agreement"), None)
    if first_search is None:
        return None
    for i, cr in enumerate(first_search.char_ranges, start=1):
        if _any_gold_overlap(cr, golds):
            return i
    return None


def answer_correct(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> bool | None:
    if record.status != "ANSWERED" or finding is None:
        return None
    return finding.answer == case["gold_answer"]


def citation_verbatim(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> bool | None:
    if finding is None or not finding.evidence:
        return None
    return all(len(locate_quote(ev.quote, canonical_text)) > 0 for ev in finding.evidence)


def citation_gold_overlap(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> bool | None:
    golds = gold_ranges(case)
    if not golds or finding is None:
        return None
    for ev in finding.evidence:
        for span in locate_quote(ev.quote, canonical_text):
            if _any_gold_overlap(span, golds):
                return True
    return False


def fabrication(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> bool | None:
    if finding is None or not finding.evidence:
        return None
    return any(len(locate_quote(ev.quote, canonical_text)) == 0 for ev in finding.evidence)


def grounded_accuracy(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> bool | None:
    ac = answer_correct(case, finding, record, canonical_text)
    cgo = citation_gold_overlap(case, finding, record, canonical_text)
    if ac is None or cgo is None:
        return None
    return bool(ac) and bool(cgo)


def abstain_correct(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> bool:
    if case.get("case_set") == "counterfactual":
        return record.status == "ABSTAINED"
    return record.status != "ABSTAINED"


def redacted_fabrication(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> bool | None:
    if case.get("kind") != "redacted":
        return None
    return fabrication(case, finding, record, canonical_text)


def required_evidence_met(
    case: dict,
    finding: Finding | None,
    record: ExecutionRecord,
    canonical_text: str,
    doc: Document | None = None,
) -> bool | None:
    spec = case.get("required_evidence")
    candidates = parse_required_evidence(spec)
    if spec is None or not candidates:
        return None

    for step in record.trajectory:
        if step.tool == "lookup_defined_term":
            term_arg = str(step.args.get("term", ""))
            if any(_term_matches(candidate, term_arg) for candidate in candidates):
                return True

    if doc is not None:
        from dealpoint.corpus.document import defined_term

        for candidate in candidates:
            dt = defined_term(doc, candidate)
            if dt is None:
                continue
            block = (dt.start, dt.end)
            for step in record.trajectory:
                for cr in step.char_ranges:
                    if overlap_chars(cr, block) >= MIN_GOLD_OVERLAP_CHARS:
                        return True

    return False


def tool_calls(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> int:
    return record.usage.tool_calls


def cap_hit(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> bool:
    return record.status == "CAP_HIT"


def execution_failed(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> bool:
    return record.status == "EXECUTION_FAILED"


def input_tokens(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> int:
    return record.usage.input_tokens


def output_tokens(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> int:
    return record.usage.output_tokens


def usd(case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str) -> float:
    return record.usage.cost_usd


def wall_ms(
    case: dict, finding: Finding | None, record: ExecutionRecord, canonical_text: str
) -> int:
    return record.wall_ms


def skill_adherence(
    case: dict,
    finding: Finding | None,
    record: ExecutionRecord,
    canonical_text: str,
    doc: Document | None = None,
) -> float | None:
    """Skill-following score: satisfied / applicable over the Appendix-B rules.

    Computed for ALL arms (spec deliverable 2), delegating to
    `dealpoint.eval.skill_adherence.evaluate`. `None` when no rule is
    applicable to this case's execution.
    """
    from dealpoint.eval.skill_adherence import evaluate as _evaluate_skill_rules

    return _evaluate_skill_rules(case, finding, record, doc)["score"]


def skill_adherence_detail(
    case: dict, finding: Finding | None, record: ExecutionRecord, doc: Document | None = None
) -> dict:
    """The full per-rule breakdown (`n_applicable`, `n_satisfied`, `rules`) --

    metadata, not a Braintrust score (spec §3: "a separate `skill_rules` key").
    """
    from dealpoint.eval.skill_adherence import evaluate as _evaluate_skill_rules

    return _evaluate_skill_rules(case, finding, record, doc)


SCORE_FIELD_NAMES: tuple[str, ...] = (
    "gold_seen",
    "gold_first_rank",
    "answer_correct",
    "citation_verbatim",
    "citation_gold_overlap",
    "fabrication",
    "grounded_accuracy",
    "abstain_correct",
    "redacted_fabrication",
    "required_evidence_met",
    "tool_calls",
    "cap_hit",
    "execution_failed",
    "input_tokens",
    "output_tokens",
    "usd",
    "wall_ms",
    "skill_adherence",
)


def score_case(
    case: dict, finding: Finding | None, record: ExecutionRecord, doc: Document | None
) -> dict:
    """Every score field, by name. `None` values are kept, never dropped."""
    canonical_text = doc.text if doc is not None else ""
    return {
        "gold_seen": gold_seen(case, finding, record, canonical_text),
        "gold_first_rank": gold_first_rank(case, finding, record, canonical_text),
        "answer_correct": answer_correct(case, finding, record, canonical_text),
        "citation_verbatim": citation_verbatim(case, finding, record, canonical_text),
        "citation_gold_overlap": citation_gold_overlap(case, finding, record, canonical_text),
        "fabrication": fabrication(case, finding, record, canonical_text),
        "grounded_accuracy": grounded_accuracy(case, finding, record, canonical_text),
        "abstain_correct": abstain_correct(case, finding, record, canonical_text),
        "redacted_fabrication": redacted_fabrication(case, finding, record, canonical_text),
        "required_evidence_met": required_evidence_met(case, finding, record, canonical_text, doc=doc),
        "tool_calls": tool_calls(case, finding, record, canonical_text),
        "cap_hit": cap_hit(case, finding, record, canonical_text),
        "execution_failed": execution_failed(case, finding, record, canonical_text),
        "input_tokens": input_tokens(case, finding, record, canonical_text),
        "output_tokens": output_tokens(case, finding, record, canonical_text),
        "usd": usd(case, finding, record, canonical_text),
        "wall_ms": wall_ms(case, finding, record, canonical_text),
        "skill_adherence": skill_adherence(case, finding, record, canonical_text, doc=doc),
    }


def majority_baseline(cases: list[dict]) -> dict:
    """Per-question and overall accuracy of predicting `majority_answer`.

    Cases with `majority_answer is None` (the 10 out-of-scope counterfactual
    cases) are skipped entirely, per spec §2.4.
    """
    per_question: dict[str, list[bool]] = {}
    overall: list[bool] = []
    for case in cases:
        majority = case.get("majority_answer")
        if majority is None:
            continue
        correct = majority == case["gold_answer"]
        per_question.setdefault(case["question_id"], []).append(correct)
        overall.append(correct)

    result: dict = {
        qid: (sum(values) / len(values) if values else None)
        for qid, values in per_question.items()
    }
    result["overall"] = (sum(overall) / len(overall)) if overall else None
    return result


def summarise(rows: list[dict]) -> dict:
    """Mean over non-`None` values for every score field, plus derived rates.

    `rows` are result rows shaped like the runner's output: each carries a
    `"scores"` dict (score name -> value), a `"case_set"` and a `"record"`
    dict with a `"status"` key. Also derives the brief's §2.4 metric names:
    `abstain_recall` (mean ABSTAINED over counterfactual rows) and
    `false_abstain` (mean ABSTAINED over dev/test rows) — the spec's
    per-case `abstain_correct` field is the canonical implementation; these
    two are aggregate names the brief also expects to exist.
    """
    score_names: set[str] = set()
    for row in rows:
        score_names.update((row.get("scores") or {}).keys())

    result: dict = {}
    for name in sorted(score_names):
        values = [
            row["scores"][name]
            for row in rows
            if (row.get("scores") or {}).get(name) is not None
        ]
        numeric_values = [float(v) for v in values if isinstance(v, bool | int | float)]
        result[name] = {
            "mean": (sum(numeric_values) / len(numeric_values)) if numeric_values else None,
            "n": len(numeric_values),
        }

    counterfactual_abstained = [
        row["record"]["status"] == "ABSTAINED" for row in rows if row.get("case_set") == "counterfactual"
    ]
    dev_test_abstained = [
        row["record"]["status"] == "ABSTAINED"
        for row in rows
        if row.get("case_set") in ("dev", "test")
    ]
    result["abstain_recall"] = {
        "mean": (
            sum(counterfactual_abstained) / len(counterfactual_abstained)
            if counterfactual_abstained
            else None
        ),
        "n": len(counterfactual_abstained),
    }
    result["false_abstain"] = {
        "mean": (sum(dev_test_abstained) / len(dev_test_abstained)) if dev_test_abstained else None,
        "n": len(dev_test_abstained),
    }
    return result
