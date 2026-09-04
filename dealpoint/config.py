"""Repo-relative paths, seeds and thresholds shared across dealpoint modules.

All paths are derived from ``REPO_ROOT``, resolved from ``__file__``, never
from the process's current working directory.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SEED = 42

# --- Milestone 0 dataset facts (verified 2026-09-03, see specs/§0.1) -------
ZENODO_URL = "https://zenodo.org/records/7500064/files/maud_v1.zip?download=1"
ZIP_MD5 = "aad11e3f68cd4f3e0067a15b923a19ce"
ZIP_BYTES = 29_153_514
N_CONTRACTS = 152
N_MAIN_ROWS = 20_623  # union of all 3 CSVs, data_type == "main"

# --- Alignment / section / selection thresholds ----------------------------
MIN_FRAGMENT_CHARS = 20
FUZZY_CUTOFF = 90
MIN_FRAGMENT_COVER = 0.50
MIN_PARSER_COVERAGE = 0.80
MAX_SECTION_CHARS = 20_000
N_AGREEMENTS = 20
N_DEV = 5
N_TEST = 15
N_REDACTED = 30
N_OUT_OF_SCOPE = 10

# --- Paths -------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_ZIP_PATH = RAW_DIR / "maud_v1.zip"
# The MAUD zip extracts to a `data/` subdirectory inside data/raw/.
RAW_DATA_DIR = RAW_DIR / "data"
CONTRACTS_DIR = RAW_DATA_DIR / "contracts"
CSV_PATHS = {
    "train": RAW_DATA_DIR / "MAUD_train.csv",
    "dev": RAW_DATA_DIR / "MAUD_dev.csv",
    "test": RAW_DATA_DIR / "MAUD_test.csv",
}

DERIVED_DIR = DATA_DIR / "derived"
CANONICAL_DIR = DERIVED_DIR / "canonical"
SECTIONS_DIR = DERIVED_DIR / "sections"
ALIGNMENT_PATH = DERIVED_DIR / "alignment.jsonl"

EVAL_DIR = DATA_DIR / "eval"
SELECTION_PATH = EVAL_DIR / "selection.json"
EXCLUDED_PATH = EVAL_DIR / "excluded.json"
DEV_JSONL_PATH = EVAL_DIR / "dev.jsonl"
TEST_JSONL_PATH = EVAL_DIR / "test.jsonl"
COUNTERFACTUAL_JSONL_PATH = EVAL_DIR / "counterfactual.jsonl"

AGREEMENT_NAMES_PATH = DATA_DIR / "agreement_names.json"

REPORTS_DIR = DATA_DIR / "reports"
