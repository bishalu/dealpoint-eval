"""Build the dev/test/counterfactual case JSONL files (spec §3.8, M0.1 §Problem B).

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
    OOS_COLLISION_WINDOW,
    OOS_COLLISIONS_REPORT_PATH,
    SECTIONS_DIR,
    SEED,
    TEST_JSONL_PATH,
)
from dealpoint.data.align import AlignResult
from dealpoint.data.labels import Label, is_null_answer
from dealpoint.data.questions import (
    OUT_OF_SCOPE_QUESTION_SPEC,
    OUT_OF_SCOPE_SPARES,
    QUESTION_BY_ID,
    QUESTION_SPEC,
    OutOfScopeQuestion,
    question_collides,
    term_match_offsets,
)
from dealpoint.data.sections import (
    PARSER_VERSION,
    assert_sections_dir_not_stale,
    compute_structural_metrics,
    parse_sections,
)
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
    """Return (dev_rows, test_rows, excluded_rows) for the 20 selected agreements.

    Refuses to run if `data/derived/sections/` holds files built by a
    different `parser_version` (specs/milestones/m0_1.md §A.5).
    """
    assert_sections_dir_not_stale(SECTIONS_DIR)

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


def _collision_rows_for(
    question: OutOfScopeQuestion,
    sorted_agreements: list[str],
    canonicals: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    for agreement_id in sorted_agreements:
        canonical = canonicals[agreement_id]
        collides = question_collides(canonical, question.key_terms, OOS_COLLISION_WINDOW)
        row: dict = {
            "question_id": question.id,
            "question_text": question.text,
            "agreement_id": agreement_id,
            "collides": collides,
        }
        if collides:
            row["matched_offsets"] = term_match_offsets(canonical, question.key_terms)
        rows.append(row)
    return rows


def _assign_agreement(
    question: OutOfScopeQuestion,
    sorted_agreements: list[str],
    preferred_idx: int,
    canonicals: dict[str, str],
) -> str | None:
    """First non-colliding agreement, scanning from `preferred_idx` (round-robin), wrapping.

    Returns None if `question` collides on every agreement in `sorted_agreements`.
    """
    n = len(sorted_agreements)
    for step in range(n):
        candidate = sorted_agreements[(preferred_idx + step) % n]
        if not question_collides(canonicals[candidate], question.key_terms, OOS_COLLISION_WINDOW):
            return candidate
    return None


def build_out_of_scope_cases(
    test_agreement_ids: list[str],
    agreement_data: dict[str, AgreementData],
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Assign each out-of-scope question to a non-colliding test agreement.

    Returns (rows, collision_report_rows, substitutions). For each question,
    every (question, test-agreement) pair is collision-checked and recorded
    (specs/milestones/m0_1.md Problem B: "Write the result to
    data/reports/m0_1_oos_collisions.json"). A question that collides on
    every test agreement is replaced by the next unused spare, in order, and
    the substitution is recorded.
    """
    sorted_agreements = sorted(test_agreement_ids, key=lambda s: int(s.split("_")[1]))
    canonicals = {a: agreement_data[a].canonical for a in sorted_agreements}

    collision_rows: list[dict] = []
    substitutions: list[dict] = []
    rows: list[dict] = []

    spares_remaining = list(OUT_OF_SCOPE_SPARES)

    for idx, question in enumerate(OUT_OF_SCOPE_QUESTION_SPEC):
        collision_rows.extend(_collision_rows_for(question, sorted_agreements, canonicals))

        preferred_idx = idx % len(sorted_agreements)
        chosen_question = question
        assigned = _assign_agreement(question, sorted_agreements, preferred_idx, canonicals)

        if assigned is None:
            if not spares_remaining:
                raise RuntimeError(
                    f"{question.id} collides on every test agreement and no spares remain"
                )
            spare = spares_remaining.pop(0)
            substitutions.append(
                {
                    "original_id": question.id,
                    "original_text": question.text,
                    "replacement_id": spare.id,
                    "replacement_text": spare.text,
                    "reason": "collided on every test agreement",
                }
            )
            collision_rows.extend(_collision_rows_for(spare, sorted_agreements, canonicals))
            assigned = _assign_agreement(spare, sorted_agreements, preferred_idx, canonicals)
            if assigned is None:
                raise RuntimeError(f"spare {spare.id} also collides on every test agreement")
            chosen_question = spare

        slot_id = f"oos{idx:02d}"
        rows.append(
            {
                "case_id": f"{assigned}__{slot_id}",
                "agreement_id": assigned,
                "question_id": slot_id,
                "question_text": chosen_question.text,
                "gold_answer": "ABSTAIN",
                "gold_spans": [],
                "span_type": None,
                "majority_answer": None,
                "required_evidence": None,
                "case_set": "counterfactual",
                "kind": "out_of_scope",
            }
        )

    return rows, collision_rows, substitutions


def write_collision_report(collision_rows: list[dict], substitutions: list[dict]) -> None:
    payload = {
        "collisions": sorted(
            collision_rows, key=lambda r: (r["question_id"], r["agreement_id"])
        ),
        "substitutions": substitutions,
        "collision_window": OOS_COLLISION_WINDOW,
    }
    OOS_COLLISIONS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OOS_COLLISIONS_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_redacted_documents(documents: list[RedactedDocument]) -> None:
    """Persist each redacted document's own canonical text and section map."""
    for doc in documents:
        canon_path = CANONICAL_DIR / f"{doc.doc_id}.txt"
        canon_path.parent.mkdir(parents=True, exist_ok=True)
        canon_path.write_text(doc.canonical, encoding="utf-8")

        sections = parse_sections(doc.canonical)
        metrics = compute_structural_metrics(doc.canonical, sections)
        sections_path = SECTIONS_DIR / f"{doc.doc_id}.json"
        sections_path.parent.mkdir(parents=True, exist_ok=True)
        sections_payload = {
            "n_sections": metrics.n_sections,
            "max_section_chars": metrics.max_section_chars,
            "structural_coverage": metrics.structural_coverage,
            "giant_single_section": metrics.giant_single_section,
            "parser_version": PARSER_VERSION,
            "sections": [
                {"ref": s.ref, "title": s.title, "start": s.start, "end": s.end}
                for s in sections
            ],
        }
        sections_path.write_text(
            json.dumps(sections_payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
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
