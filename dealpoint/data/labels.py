"""MAUD gold labels: load and union the three CSVs (spec C1).

The three CSVs (`MAUD_train.csv`, `MAUD_dev.csv`, `MAUD_test.csv`) partition
*items*, not contracts: all 152 contracts appear in all three files, and only
the union of `data_type == "main"` rows across all three gives the complete
set of 20,623 gold labels. MAUD's own train/dev/test split is irrelevant to
this project; our split is by agreement (see `select.py`).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from dealpoint.config import CSV_PATHS, N_CONTRACTS, N_MAIN_ROWS

# Gold span text fields reach 16,768 chars; the default csv field size limit
# (128 KiB on most platforms, but as low as 131072 bytes) is not guaranteed
# to be enough headroom, so raise it generously.
csv.field_size_limit(10**9)

_NULL_ANSWER_STRINGS = frozenset({"", "None", "<NONE>", "nan"})


@dataclass(frozen=True)
class Label:
    contract_name: str
    question: str
    subquestion: str
    text: str
    answer: str | None
    label: str
    text_type: str
    category: str


def is_null_answer(answer: str | None) -> bool:
    """True if `answer` is a null sentinel: empty, the literal 'None', '<NONE>', or 'nan'.

    Spec C9: q10 (and others) use the literal string 'None' as their null
    sentinel in the CSV, not an empty string, so this must be checked
    explicitly rather than only testing for falsy/empty values.
    """
    if answer is None:
        return True
    return answer in _NULL_ANSWER_STRINGS


def _load_csv_main_rows(path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader if row["data_type"] == "main"]


def load_labels() -> dict[tuple[str, str, str], Label]:
    """Load and union all three MAUD CSVs, keyed by (contract_name, question, subquestion).

    Raises AssertionError if the union does not match the expected integrity
    facts measured during recon: exactly N_MAIN_ROWS rows, N_CONTRACTS
    contracts, and every key unique.
    """
    rows: list[dict[str, str]] = []
    for path in CSV_PATHS.values():
        rows.extend(_load_csv_main_rows(path))

    labels: dict[tuple[str, str, str], Label] = {}
    for row in rows:
        key = (row["contract_name"], row["question"], row["subquestion"])
        if key in labels:
            raise AssertionError(f"duplicate label key across MAUD CSVs: {key!r}")
        labels[key] = Label(
            contract_name=row["contract_name"],
            question=row["question"],
            subquestion=row["subquestion"],
            text=row["text"],
            answer=row["answer"],
            label=row["label"],
            text_type=row["text_type"],
            category=row["category"],
        )

    if len(labels) != N_MAIN_ROWS:
        raise AssertionError(f"expected {N_MAIN_ROWS} main rows, got {len(labels)}")
    n_contracts = len({lbl.contract_name for lbl in labels.values()})
    if n_contracts != N_CONTRACTS:
        raise AssertionError(f"expected {N_CONTRACTS} contracts, got {n_contracts}")

    return labels
