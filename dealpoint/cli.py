"""`python -m dealpoint.cli data [--force-download] [--skip-download]`

Runs the full M0/M0.1 data-foundation chain end to end and prints a summary
table:

  download -> canonicalise -> sections -> labels -> align -> select -> cases
  -> reports

Idempotent: re-running must not change any committed file (spec Gate 7 /
Definition of Done #4). This command always rebuilds derived sections files
from raw source rather than reusing whatever is on disk -- it is the thing
that repairs a `parser_version` mismatch, so it must never trip the
stale-state guard itself (specs/milestones/m0_1.md §A.5).
"""

from __future__ import annotations

import argparse
import json
import sys

from dealpoint.config import (
    AGREEMENT_NAMES_PATH,
    CANONICAL_DIR,
    DERIVED_DIR,
    N_CONTRACTS,
    SECTIONS_DIR,
    SELECTION_PATH,
)
from dealpoint.data.cases import (
    build_dev_test_cases,
    build_out_of_scope_cases,
    build_redacted_cases,
    write_case_files,
    write_collision_report,
    write_redacted_documents,
)
from dealpoint.data.download import ensure_dataset
from dealpoint.data.identity import parse_agreement_title_and_parties
from dealpoint.data.labels import load_labels
from dealpoint.data.questions import QUESTION_SPEC
from dealpoint.data.reports import (
    build_parser_report,
    write_dataset_version_txt,
    write_parser_report,
    write_parser_version_txt,
)
from dealpoint.data.sections import PARSER_VERSION
from dealpoint.data.select import (
    AgreementData,
    compute_question_alignments,
    list_all_agreement_ids,
    load_agreement,
    select_agreements,
)


def _rebuild_agreement_data() -> dict[str, AgreementData]:
    """Recompute canonical text + section map for all N_CONTRACTS from raw source.

    Always recomputed, never read back from `data/derived/sections/` -- this
    is the rebuild step that repairs a stale `parser_version` on disk
    (specs/milestones/m0_1.md §A.5).
    """
    agreement_data: dict[str, AgreementData] = {}
    for agreement_id in list_all_agreement_ids():
        agreement_data[agreement_id] = load_agreement(agreement_id)
    return agreement_data


def _clear_derived_dirs() -> None:
    """Wipe canonical/ and sections/ before a full rebuild.

    `data/derived/sections/` holds both per-contract files and per-redacted-
    document files (written later by `write_redacted_documents`); clearing it
    up front guarantees no stale file from a previous parser_version (or a
    previous, now-dropped redacted document) survives to trip the
    stale-state guard this same run relies on (specs/milestones/m0_1.md
    section A.5).
    """
    for directory in (CANONICAL_DIR, SECTIONS_DIR):
        if directory.is_dir():
            for path in directory.glob("*"):
                if path.is_file():
                    path.unlink()
        directory.mkdir(parents=True, exist_ok=True)


def _persist_derived(agreement_data: dict[str, AgreementData]) -> None:
    """Write each agreement's canonical text and full derived sections payload.

    The sections payload includes the section list and body bounds (not just
    the summary metrics) so that `gold_span_section_rate` -- report-only per
    the amended spec -- can be freshly recomputed directly from these
    committed derived files plus `data/derived/alignment.jsonl`, without
    re-parsing raw contract text (specs/milestones/m0_1.md Definition of
    Done: "a test asserts the field is present and equals a fresh
    recomputation").
    """
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for agreement_id, data in agreement_data.items():
        canon_path = CANONICAL_DIR / f"{agreement_id}.txt"
        canon_path.write_text(data.canonical, encoding="utf-8")

        payload = {
            "n_sections": data.n_sections,
            "max_section_chars": data.max_section_chars,
            "structural_coverage": data.structural_coverage,
            "giant_single_section": data.giant_single_section,
            "body_start": data.body_start,
            "body_end": data.body_end,
            "parser_version": PARSER_VERSION,
            "sections": [
                {"ref": s.ref, "title": s.title, "start": s.start, "end": s.end}
                for s in data.sections
            ],
        }
        sections_path = SECTIONS_DIR / f"{agreement_id}.json"
        sections_path.write_text(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _write_alignment_cache(labels, agreement_data: dict[str, AgreementData]) -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for agreement_id in sorted(agreement_data.keys(), key=lambda s: int(s.split("_")[1])):
        data = agreement_data[agreement_id]
        aligned = compute_question_alignments(data, labels)
        for question_id, (label, result) in aligned.items():
            rows.append(
                {
                    "agreement_id": agreement_id,
                    "question_id": question_id,
                    "text_type": label.text_type,
                    "n_fragments": result.n_fragments,
                    "n_aligned": result.n_aligned,
                    "covered_fraction": result.covered_fraction,
                    "modes": result.modes,
                    "ranges": [{"start": s, "end": e} for s, e in result.ranges],
                }
            )
    rows.sort(key=lambda r: (r["agreement_id"], r["question_id"]))
    alignment_path = DERIVED_DIR / "alignment.jsonl"
    with open(alignment_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            fh.write("\n")


def _write_agreement_names(selection, agreement_data: dict[str, AgreementData]) -> None:
    """Flag any selected agreement the regex-parse misses for hand review (spec C2).

    `data/agreement_names.json` is a hand-checked, committed override file for
    the small number of the 20 *selected* agreements whose title/parties the
    tolerant regex in `identity.py` cannot recover. Agreements the regex
    already handles are resolved on the fly and never written here. This
    function never overwrites an existing (hand-checked) entry; it only adds
    a `needs_hand_review` placeholder for a selected agreement when the regex
    genuinely misses and no entry exists yet.
    """
    existing: dict[str, dict[str, str]] = {}
    if AGREEMENT_NAMES_PATH.exists():
        with open(AGREEMENT_NAMES_PATH, encoding="utf-8") as fh:
            existing = json.load(fh)

    for agreement_id in selection.selected:
        if agreement_id in existing:
            continue
        data = agreement_data[agreement_id]
        if parse_agreement_title_and_parties(data.canonical) is not None:
            continue  # regex handled it; no override needed
        existing[agreement_id] = {"title": "", "parties": "", "needs_hand_review": "true"}

    # Always write the file (even if empty) so it exists as a committed,
    # inspectable artifact per spec, independent of whether any override
    # was needed this run.
    AGREEMENT_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AGREEMENT_NAMES_PATH, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")


def _write_selection(selection) -> None:
    payload = {
        "selected": sorted(selection.selected, key=lambda s: int(s.split("_")[1])),
        "dev": selection.dev,
        "test": selection.test,
        "rejected": {k: selection.rejected[k] for k in sorted(selection.rejected)},
        "parser_version": PARSER_VERSION,
    }
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")


def _compute_global_alignment_stats(labels, agreement_data: dict[str, AgreementData]) -> dict:
    """Fragment- and case-level alignment rates over all 12 questions x 152 contracts."""
    per_question = {q.id: {"frag": 0, "frag_aligned": 0, "cases": 0, "any": 0, "all_": 0} for q in QUESTION_SPEC}
    total_frag = 0
    total_frag_aligned = 0
    total_cases = 0
    total_any = 0
    total_all = 0

    for data in agreement_data.values():
        aligned = compute_question_alignments(data, labels)
        for question_id, (label, result) in aligned.items():
            if result.n_fragments == 0:
                continue
            stat = per_question[question_id]
            stat["frag"] += result.n_fragments
            stat["frag_aligned"] += result.n_aligned
            stat["cases"] += 1
            has_any = result.n_aligned >= 1
            has_all = result.n_aligned == result.n_fragments
            stat["any"] += int(has_any)
            stat["all_"] += int(has_all)

            total_frag += result.n_fragments
            total_frag_aligned += result.n_aligned
            total_cases += 1
            total_any += int(has_any)
            total_all += int(has_all)

    per_question_rates = {
        qid: {
            "fragment_rate": (s["frag_aligned"] / s["frag"]) if s["frag"] else None,
            "case_any_rate": (s["any"] / s["cases"]) if s["cases"] else None,
            "case_all_rate": (s["all_"] / s["cases"]) if s["cases"] else None,
            "n_cases": s["cases"],
        }
        for qid, s in per_question.items()
    }

    return {
        "fragment_rate": (total_frag_aligned / total_frag) if total_frag else None,
        "case_any_rate": (total_any / total_cases) if total_cases else None,
        "case_all_rate": (total_all / total_cases) if total_cases else None,
        "n_cases": total_cases,
        "per_question": per_question_rates,
    }


def _print_summary(
    alignment_stats: dict,
    parser_report: dict,
    selection,
    agreement_data: dict[str, AgreementData],
    dev_rows,
    test_rows,
    counterfactual_rows,
) -> None:
    print()
    print("=" * 78)
    print("Deal-Point Eval — data build summary")
    print("=" * 78)
    print()
    print(f"Contracts processed: {len(agreement_data)} (expected {N_CONTRACTS})")
    print(f"parser_version: {PARSER_VERSION}")
    print()
    print("Alignment rates (all 12 questions x all contracts with non-null answers):")
    print(
        f"  fragment-level:        {alignment_stats['fragment_rate']:.4f}  "
        f"(n={alignment_stats['n_cases']} cases)"
    )
    print(f"  case >=1 aligned:      {alignment_stats['case_any_rate']:.4f}")
    print(f"  case all aligned:      {alignment_stats['case_all_rate']:.4f}")
    print()
    print("  per-question:")
    print(f"  {'id':<5}{'n':>6}{'fragment':>12}{'case>=1':>12}{'case-all':>12}")
    for q in QUESTION_SPEC:
        s = alignment_stats["per_question"][q.id]
        frag = f"{s['fragment_rate']:.3f}" if s["fragment_rate"] is not None else "n/a"
        any_r = f"{s['case_any_rate']:.3f}" if s["case_any_rate"] is not None else "n/a"
        all_r = f"{s['case_all_rate']:.3f}" if s["case_all_rate"] is not None else "n/a"
        print(f"  {q.id:<5}{s['n_cases']:>6}{frag:>12}{any_r:>12}{all_r:>12}")
    print()
    print(
        f"Structural coverage: eligible={parser_report['eligible_count']}/"
        f"{parser_report['n_agreements']}  "
        f"median={parser_report['structural_coverage_distribution']['median']:.3f}  "
        f"corpus gold_span_section_rate="
        f"{parser_report['corpus_gold_span_section_rate']:.4f}"
        if parser_report["corpus_gold_span_section_rate"] is not None
        else "Structural coverage: (no gold spans)"
    )
    print()
    print(f"Agreements selected: {len(selection.selected)} (5 dev / 15 test)")
    print()
    print("  structural_coverage per selected agreement:")
    for agreement_id in sorted(selection.selected, key=lambda s: int(s.split("_")[1])):
        cov = agreement_data[agreement_id].structural_coverage
        split = "dev" if agreement_id in selection.dev else "test"
        print(f"    {agreement_id:<14} {split:<5} coverage={cov:.4f}")
    print()
    print("Case counts:")
    print(f"  dev.jsonl:            {len(dev_rows)}")
    print(f"  test.jsonl:           {len(test_rows)}")
    print(f"  counterfactual.jsonl: {len(counterfactual_rows)}")
    print()


def run_data_pipeline(force_download: bool = False, skip_download: bool = False) -> None:
    ensure_dataset(force_download=force_download, skip_download=skip_download)

    labels = load_labels()
    agreement_data = _rebuild_agreement_data()

    # Wipe then persist derived sections/canonical fresh, at the current
    # parser_version, so the stale-state guard in select_agreements/
    # build_dev_test_cases never fires against this run's own output (and no
    # now-dropped redacted-document sections file lingers).
    _clear_derived_dirs()
    _persist_derived(agreement_data)

    selection, _agreement_data, usable_by_id = select_agreements(
        labels, agreement_ids=list(agreement_data.keys()), agreement_data=agreement_data
    )

    _write_alignment_cache(labels, agreement_data)
    _write_agreement_names(selection, agreement_data)
    _write_selection(selection)

    dev_rows, test_rows, excluded_rows = build_dev_test_cases(
        labels, selection, agreement_data, usable_by_id
    )
    redacted_rows, redacted_docs = build_redacted_cases(test_rows, agreement_data)
    oos_rows, collision_rows, substitutions = build_out_of_scope_cases(
        selection.test, agreement_data
    )
    counterfactual_rows = redacted_rows + oos_rows

    write_case_files(dev_rows, test_rows, counterfactual_rows, excluded_rows)
    write_redacted_documents(redacted_docs)
    write_collision_report(collision_rows, substitutions)

    parser_report = build_parser_report(labels, selection, agreement_data)
    write_parser_report(parser_report)
    write_parser_version_txt()
    write_dataset_version_txt()

    alignment_stats = _compute_global_alignment_stats(labels, agreement_data)
    _print_summary(
        alignment_stats, parser_report, selection, agreement_data, dev_rows, test_rows, counterfactual_rows
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dealpoint.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="run the full data-foundation pipeline")
    data_parser.add_argument("--force-download", action="store_true")
    data_parser.add_argument("--skip-download", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "data":
        run_data_pipeline(force_download=args.force_download, skip_download=args.skip_download)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
