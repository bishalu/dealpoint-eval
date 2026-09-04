"""Build the dev/test/counterfactual case JSONL files (spec §3.8).

Case sets:
  - dev.jsonl / test.jsonl: one row per (agreement, question) with a usable
    gold span, for the 20 selected agreements (5 dev / 15 test by agreement).
  - counterfactual.jsonl: 30 redacted-span cases + 10 out-of-scope cases,
    all with gold_answer == "ABSTAIN".

Determinism: every file is sorted by case_id and written with
``json.dumps(obj, sort_keys=True, ensure_ascii=False)`` + ``\\n`` endings, so a
rebuild is byte-identical to the committed files (spec Gate 7).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from dealpoint.config import (
    CANONICAL_DIR,
    COUNTERFACTUAL_JSONL_PATH,
    DEV_JSONL_PATH,
    EXCLUDED_PATH,
    N_REDACTED,
    SECTIONS_DIR,
    SEED,
    TEST_JSONL_PATH,
)
from dealpoint.data.align import AlignResult
from dealpoint.data.labels import Label, is_null_answer
from dealpoint.data.questions import OUT_OF_SCOPE_QUESTIONS, QUESTION_BY_ID, QUESTION_SPEC
from dealpoint.data.sections import parse_sections, parser_coverage
from dealpoint.data.select import AgreementData, SelectionResult, compute_question_alignments


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: r["case_id"])
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows_sorted:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            fh.write("\n")


def _build_case_row(
    agreement_id: str, question_id: str, label: Label, result: AlignResult, case_set: str
) -> dict:
    spec = QUESTION_BY_ID[question_id]
    return {
        "case_id": f"{agreement_id}__{question_id}",
        "agreement_id": agreement_id,
        "question_id": question_id,
        "gold_answer": label.answer,
        "gold_spans": [{"start": s, "end": e} for s, e in result.ranges],
        "span_type": label.text_type,
        "majority_answer": spec.options[0],
        "required_evidence": spec.required_evidence,
        "case_set": case_set,
    }


def build_dev_test_cases(
    labels: dict[tuple[str, str, str], Label],
    selection: SelectionResult,
    agreement_data: dict[str, AgreementData],
    usable_by_id: dict[str, dict[str, tuple[Label, AlignResult]]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (dev_rows, test_rows, excluded_rows) for the 20 selected agreements."""
    dev_rows: list[dict] = []
    test_rows: list[dict] = []
    excluded_rows: list[dict] = []

    for agreement_id, case_set, rows in (
        *((a, "dev", dev_rows) for a in selection.dev),
        *((a, "test", test_rows) for a in selection.test),
    ):
        data = agreement_data[agreement_id]
        usable = usable_by_id[agreement_id]
        all_aligned = compute_question_alignments(data, labels)
        for q in QUESTION_SPEC:
            if q.id in usable:
                label, result = usable[q.id]
                rows.append(_build_case_row(agreement_id, q.id, label, result, case_set))
                continue
            # Determine why this (agreement, question) was excluded.
            key = (agreement_id, q.maud_question, "<NONE>")
            label = labels.get(key)
            if label is None:
                reason = "no label row"
            elif is_null_answer(label.answer):
                reason = "null gold answer"
            else:
                _, result = all_aligned[q.id]
                if result.n_aligned < 1:
                    reason = "no aligned fragment"
                else:
                    reason = (
                        f"covered_fraction {result.covered_fraction:.3f} below MIN_FRAGMENT_COVER"
                    )
            excluded_rows.append(
                {"agreement_id": agreement_id, "question_id": q.id, "reason": reason}
            )

    return dev_rows, test_rows, excluded_rows


def _allocate_quota(qids: list[str], total: int) -> dict[str, int]:
    """Split `total` as evenly as possible across `qids`, deterministic remainder order."""
    base = total // len(qids)
    remainder = total % len(qids)
    quota = {qid: base for qid in qids}
    for qid in sorted(qids)[:remainder]:
        quota[qid] += 1
    return quota


def _redact_ranges(canonical: str, ranges: list[tuple[int, int]]) -> str:
    """Delete the given (start, end) ranges from canonical text, no marker left behind."""
    merged = sorted(ranges)
    out_parts = []
    cursor = 0
    for start, end in merged:
        out_parts.append(canonical[cursor:start])
        cursor = end
    out_parts.append(canonical[cursor:])
    return "".join(out_parts)


@dataclass(frozen=True)
class RedactedDocument:
    agreement_id: str
    question_id: str
    doc_id: str
    canonical: str
    redacted_ranges: list[tuple[int, int]]


def build_redacted_cases(
    test_rows: list[dict],
    agreement_data: dict[str, AgreementData],
    seed: int = SEED,
) -> tuple[list[dict], list[RedactedDocument]]:
    """Seeded stratified pick of N_REDACTED cases from `test_rows`, one derived doc each."""
    by_question: dict[str, list[dict]] = {}
    for row in test_rows:
        by_question.setdefault(row["question_id"], []).append(row)

    qids = sorted(by_question.keys())
    quota = _allocate_quota(qids, N_REDACTED)

    rng = random.Random(seed)
    picked: list[dict] = []
    for qid in qids:
        candidates = sorted(by_question[qid], key=lambda r: r["case_id"])
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        picked.extend(shuffled[: quota[qid]])

    picked.sort(key=lambda r: r["case_id"])

    rows: list[dict] = []
    documents: list[RedactedDocument] = []
    for source_case in picked:
        agreement_id = source_case["agreement_id"]
        question_id = source_case["question_id"]
        data = agreement_data[agreement_id]
        ranges = [(span["start"], span["end"]) for span in source_case["gold_spans"]]
        redacted_canonical = _redact_ranges(data.canonical, ranges)
        doc_id = f"{agreement_id}__redacted_{question_id}"
        documents.append(
            RedactedDocument(
                agreement_id=agreement_id,
                question_id=question_id,
                doc_id=doc_id,
                canonical=redacted_canonical,
                redacted_ranges=ranges,
            )
        )
        rows.append(
            {
                "case_id": doc_id,
                "agreement_id": agreement_id,
                "question_id": question_id,
                "gold_answer": "ABSTAIN",
                "gold_spans": [],
                "span_type": source_case["span_type"],
                "majority_answer": source_case["majority_answer"],
                "required_evidence": source_case["required_evidence"],
                "case_set": "counterfactual",
                "kind": "redacted",
                "source_case_id": source_case["case_id"],
                "redacted_ranges": [{"start": s, "end": e} for s, e in ranges],
            }
        )
    return rows, documents


def build_out_of_scope_cases(
    test_agreement_ids: list[str], seed: int = SEED
) -> list[dict]:
    """Assign each OUT_OF_SCOPE_QUESTIONS entry to a test agreement, deterministically."""
    sorted_agreements = sorted(test_agreement_ids, key=lambda s: int(s.split("_")[1]))
    rows: list[dict] = []
    for idx, question_text in enumerate(OUT_OF_SCOPE_QUESTIONS):
        agreement_id = sorted_agreements[idx % len(sorted_agreements)]
        case_id = f"{agreement_id}__oos{idx:02d}"
        rows.append(
            {
                "case_id": case_id,
                "agreement_id": agreement_id,
                "question_id": f"oos{idx:02d}",
                "question_text": question_text,
                "gold_answer": "ABSTAIN",
                "gold_spans": [],
                "span_type": None,
                "majority_answer": None,
                "required_evidence": None,
                "case_set": "counterfactual",
                "kind": "out_of_scope",
            }
        )
    return rows


def write_redacted_documents(documents: list[RedactedDocument]) -> None:
    """Persist each redacted document's own canonical text and section map."""
    for doc in documents:
        canon_path = CANONICAL_DIR / f"{doc.doc_id}.txt"
        canon_path.parent.mkdir(parents=True, exist_ok=True)
        canon_path.write_text(doc.canonical, encoding="utf-8")

        sections = parse_sections(doc.canonical)
        coverage = parser_coverage(doc.canonical, sections)
        sections_path = SECTIONS_DIR / f"{doc.doc_id}.json"
        sections_path.parent.mkdir(parents=True, exist_ok=True)
        sections_payload = {
            "n_sections": len(sections),
            "parser_coverage": coverage,
            "sections": [
                {"ref": s.ref, "title": s.title, "start": s.start, "end": s.end}
                for s in sections
            ],
        }
        sections_path.write_text(
            json.dumps(sections_payload, sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_case_files(
    dev_rows: list[dict],
    test_rows: list[dict],
    counterfactual_rows: list[dict],
    excluded_rows: list[dict],
) -> None:
    _write_jsonl(DEV_JSONL_PATH, dev_rows)
    _write_jsonl(TEST_JSONL_PATH, test_rows)
    _write_jsonl(COUNTERFACTUAL_JSONL_PATH, counterfactual_rows)

    EXCLUDED_PATH.parent.mkdir(parents=True, exist_ok=True)
    excluded_sorted = sorted(
        excluded_rows, key=lambda r: (r["agreement_id"], r["question_id"])
    )
    with open(EXCLUDED_PATH, "w", encoding="utf-8") as fh:
        json.dump(excluded_sorted, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")
