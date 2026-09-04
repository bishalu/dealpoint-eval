"""Build the M0.1 report artifacts (specs/milestones/m0_1.md §A.7).

Three files, all deterministic (no timestamps, no absolute paths,
``sort_keys=True``, trailing newline) so they participate in the
byte-for-byte rebuild determinism gate:

  - ``data/reports/m0_1_parser_report.json``
  - ``data/reports/parser_version.txt``
  - ``data/reports/dataset_version.txt``
"""

from __future__ import annotations

import hashlib
import json
import statistics

from dealpoint.config import (
    ALIGNMENT_PATH,
    COUNTERFACTUAL_JSONL_PATH,
    DATASET_VERSION_TXT_PATH,
    DEV_JSONL_PATH,
    GIANT_SECTION_FRACTION,
    MIN_STRUCTURAL_COVERAGE,
    PARSER_REPORT_PATH,
    PARSER_VERSION_TXT_PATH,
    SECTIONS_DIR,
    TEST_JSONL_PATH,
)
from dealpoint.data.labels import Label
from dealpoint.data.sections import PARSER_VERSION, Section, gold_span_section_breakdown
from dealpoint.data.select import (
    AgreementData,
    SelectionResult,
    agreement_gold_span_rate,
    compute_question_alignments,
)


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"min": None, "p10": None, "p25": None, "median": None, "p75": None, "p90": None, "max": None}
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
        return ordered[idx]

    buckets = [0] * 10
    for v in ordered:
        bucket = min(9, int(v * 10))
        buckets[bucket] += 1
    histogram = {f"{i / 10:.1f}-{(i + 1) / 10:.1f}": buckets[i] for i in range(10)}

    return {
        "min": ordered[0],
        "p10": pct(0.10),
        "p25": pct(0.25),
        "median": statistics.median(ordered),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "max": ordered[-1],
        "histogram": histogram,
    }


def build_parser_report(
    labels: dict[tuple[str, str, str], Label],
    selection: SelectionResult,
    agreement_data: dict[str, AgreementData],
) -> dict:
    """Assemble the full per-agreement + corpus-wide parser report payload."""
    per_agreement: dict[str, dict] = {}
    coverages: list[float] = []
    giant_ids: list[str] = []
    eligible_ids: list[str] = []

    rejection_reason_counts: dict[str, int] = {}
    for reason in selection.rejected.values():
        # bucket by the reason's leading word so e.g. every distinct
        # "structural_coverage 0.xxx < 0.8" string still rolls up.
        bucket = reason.split(" ", 1)[0]
        rejection_reason_counts[bucket] = rejection_reason_counts.get(bucket, 0) + 1

    eligible_set = set(selection.selected) | {
        aid
        for aid, reason in selection.rejected.items()
        if reason == "eligible but not selected (greedy diversity pass stopped at 20)"
    }

    for agreement_id in sorted(agreement_data.keys(), key=lambda s: int(s.split("_")[1])):
        data = agreement_data[agreement_id]
        rate = agreement_gold_span_rate(data, labels)
        per_agreement[agreement_id] = {
            "structural_coverage": data.structural_coverage,
            "n_sections": data.n_sections,
            "max_section_chars": data.max_section_chars,
            "giant_single_section": data.giant_single_section,
            "gold_span_section_rate": rate,
            "parser_version": PARSER_VERSION,
        }
        coverages.append(data.structural_coverage)
        if data.giant_single_section:
            giant_ids.append(agreement_id)
        if agreement_id in eligible_set:
            eligible_ids.append(agreement_id)

    # Corpus-wide gold_span_section_rate: recompute hits/total directly, via
    # the same causal breakdown reported per-cause below, so agreements with
    # zero aligned spans don't skew a simple mean of rates. gold_span_section_
    # rate is report-only (spec Definition of Done, amended 2026-09-04): this
    # report carries the rate, the miss breakdown by cause, and the
    # arithmetic ceiling implied by the body definition, but no test may
    # threshold it.
    corpus_breakdown = {
        "hits": 0,
        "before_body_start": 0,
        "after_signature_block": 0,
        "giant_section": 0,
        "unrecognised_gap": 0,
        "total": 0,
    }
    for data in agreement_data.values():
        aligned = compute_question_alignments(data, labels)
        ranges: list[tuple[int, int]] = []
        for _label, result in aligned.values():
            ranges.extend(result.ranges)
        if not ranges:
            continue
        body = (data.body_start, data.body_end)
        breakdown = gold_span_section_breakdown(data.sections, body, ranges)
        for key in corpus_breakdown:
            corpus_breakdown[key] += breakdown[key]

    corpus_hits = corpus_breakdown["hits"]
    corpus_total = corpus_breakdown["total"]
    corpus_gold_span_section_rate = (corpus_hits / corpus_total) if corpus_total else None
    # The arithmetic ceiling implied by the body definition alone: the best
    # `gold_span_section_rate` could ever be is bounded by how many gold
    # spans even fall inside the body window (before_body_start and
    # after_signature_block spans are excluded by the body definition
    # itself and can never become hits, no matter how good the section
    # recognition gets). hits + giant_section + unrecognised_gap are all
    # "in body" misses/hits; only those two miss causes are, in principle,
    # reclaimable by better section recognition, so their sum with hits is
    # the ceiling.
    corpus_ceiling_numerator = (
        corpus_hits + corpus_breakdown["giant_section"] + corpus_breakdown["unrecognised_gap"]
    )
    gold_span_section_rate_ceiling = (
        (corpus_ceiling_numerator / corpus_total) if corpus_total else None
    )

    report = {
        "parser_version": PARSER_VERSION,
        "n_agreements": len(agreement_data),
        "eligible_count": len(eligible_ids),
        "eligible_ids": sorted(eligible_ids, key=lambda s: int(s.split("_")[1])),
        "min_structural_coverage_threshold": MIN_STRUCTURAL_COVERAGE,
        "giant_section_fraction_threshold": GIANT_SECTION_FRACTION,
        "structural_coverage_distribution": _distribution(coverages),
        "giant_section_failures": sorted(giant_ids, key=lambda s: int(s.split("_")[1])),
        "corpus_gold_span_section_rate": corpus_gold_span_section_rate,
        "corpus_gold_span_hits": corpus_hits,
        "corpus_gold_span_total": corpus_total,
        "corpus_gold_span_miss_breakdown": {
            "before_body_start": corpus_breakdown["before_body_start"],
            "unrecognised_gap": corpus_breakdown["unrecognised_gap"],
            "giant_section": corpus_breakdown["giant_section"],
            "after_signature_block": corpus_breakdown["after_signature_block"],
        },
        "gold_span_section_rate_ceiling": gold_span_section_rate_ceiling,
        "rejection_reason_counts": rejection_reason_counts,
        "per_agreement": per_agreement,
        "selected": sorted(selection.selected, key=lambda s: int(s.split("_")[1])),
        "dev": selection.dev,
        "test": selection.test,
    }
    return report


def write_parser_report(report: dict) -> None:
    PARSER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARSER_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_parser_version_txt() -> None:
    PARSER_VERSION_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARSER_VERSION_TXT_PATH, "w", encoding="utf-8") as fh:
        fh.write(PARSER_VERSION + "\n")


def recompute_corpus_gold_span_section_rate_from_disk() -> float | None:
    """Fresh recomputation of the corpus `gold_span_section_rate`, from disk only.

    Reads `data/derived/sections/{id}.json` (per-contract only -- skips
    redacted-document sections files, which are keyed by `{id}__redacted_*`
    and are not part of the 152-contract corpus) and
    `data/derived/alignment.jsonl`, and recomputes the rate exactly as
    `build_parser_report` does, but independently of any in-memory
    `AgreementData`. This is what the Definition of Done means by "equals a
    fresh recomputation from the derived sections + alignment files" --
    proof that the reported number is not just whatever happened to be
    computed once and never checked again.
    """
    sections_by_agreement: dict[str, tuple[list[Section], tuple[int, int]]] = {}
    for path in sorted(SECTIONS_DIR.glob("contract_*.json")):
        agreement_id = path.stem
        if "__" in agreement_id:
            continue  # a redacted document's sections file, not a base contract
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        sections = [
            Section(ref=s["ref"], title=s["title"], start=s["start"], end=s["end"])
            for s in payload["sections"]
        ]
        sections_by_agreement[agreement_id] = (sections, (payload["body_start"], payload["body_end"]))

    ranges_by_agreement: dict[str, list[tuple[int, int]]] = {}
    if ALIGNMENT_PATH.exists():
        with open(ALIGNMENT_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ranges = [(r["start"], r["end"]) for r in row["ranges"]]
                ranges_by_agreement.setdefault(row["agreement_id"], []).extend(ranges)

    hits = 0
    total = 0
    for agreement_id, (sections, body) in sections_by_agreement.items():
        ranges = ranges_by_agreement.get(agreement_id, [])
        if not ranges:
            continue
        breakdown = gold_span_section_breakdown(sections, body, ranges)
        hits += breakdown["hits"]
        total += breakdown["total"]
    return (hits / total) if total else None


def write_dataset_version_txt() -> None:
    """sha256 over dev.jsonl + test.jsonl + counterfactual.jsonl, in that fixed order.

    A different concatenation order silently changes the value -- this order
    (dev, test, counterfactual) matches the spec's listing in §A.7.
    """
    hasher = hashlib.sha256()
    for path in (DEV_JSONL_PATH, TEST_JSONL_PATH, COUNTERFACTUAL_JSONL_PATH):
        hasher.update(path.read_bytes())
    digest = hasher.hexdigest()[:12]
    DATASET_VERSION_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_VERSION_TXT_PATH, "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")
