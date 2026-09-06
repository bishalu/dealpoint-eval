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
MIN_STRUCTURAL_COVERAGE = 0.80

# Fraction of body characters above which a single section is "giant" and
# therefore excluded from eligibility and from gold_span_section_rate
# (specs/milestones/m0_1.md §A.3).
GIANT_SECTION_FRACTION = 0.40

# A dense run of heading-like matches (table of contents, defined-terms
# index, exhibit list, ...) is only treated as TOC-like if it *starts*
# within this fraction of the document. Measured during M0.1 recon: contracts
# routinely have several such runs in sequence before the real numbering
# starts, and 8% comfortably covers all of them without eating real body
# text in the (rare) short-front-matter contracts (see plan §0.2/§2.3).
TOC_SCAN_FRACTION = 0.08

# Section numbering is accepted as "advancing" if it increases by up to this
# many steps in one hop, not just by exactly 1. Measured during M0.1 recon:
# strict +1 loses materially more real headings (~4,000 plausible forward
# skips across the corpus, 77 contracts with >20 skips) than a small window
# lets through as false positives, while still rejecting genuine
# cross-references, which are almost never a small forward skip that also
# matches the article number (see plan §0.4).
SECTION_ADVANCE_WINDOW = 3

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
PARSER_REPORT_PATH = REPORTS_DIR / "m0_1_parser_report.json"
OOS_COLLISIONS_REPORT_PATH = REPORTS_DIR / "m0_1_oos_collisions.json"
PARSER_VERSION_TXT_PATH = REPORTS_DIR / "parser_version.txt"
DATASET_VERSION_TXT_PATH = REPORTS_DIR / "dataset_version.txt"

# Eligible-agreement floor asserted by the gate_m0 test; the target ("~120")
# is reported, not gated (specs/milestones/m0_1.md §A.6).
MIN_ELIGIBLE_COUNT = 100

# Sliding-window size (canonical chars) for the out-of-scope collision check
# (specs/milestones/m0_1.md Problem B).
OOS_COLLISION_WINDOW = 400

# --- Milestone 1: chunking, index, agent ----------------------------------
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = DATA_DIR / "results"
SPEND_LEDGER_PATH = RESULTS_DIR / "spend_ledger.jsonl"
INDEX_VERSION_TXT_PATH = REPORTS_DIR / "index_version.txt"
VERSIONS_JSON_PATH = REPORTS_DIR / "versions.json"
CHUNK_REPORT_PATH = REPORTS_DIR / "chunk_report.json"

# Chunking. The brief (§3.4) specifies "~400 tokens / 50 overlap". Tokens are
# not a framework-free unit, so the parameters are stored in CHARS at the
# corpus's measured chars/token ratio (see data/reports/chunk_report.json);
# the ratio is measured once with the bge tokenizer and recorded, never guessed.
# Measured chars_per_token_ratio_mean = 4.796 (data/reports/chunk_report.json,
# produced by dealpoint/corpus/build_index.py's _chunk_report over the bge
# tokenizer) -- the params below are retuned to that measured ratio, not the
# earlier ~4.0 assumption.
CHUNK_TARGET_CHARS = 1900  # ~400 tokens at the measured 4.796 chars/token
CHUNK_OVERLAP_CHARS = 240  # ~50 tokens at the measured 4.796 chars/token
CHUNK_ALGO_VERSION = "1"  # bump to invalidate chunk_version deliberately

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QDRANT_COLLECTION = "dealpoint_chunks"
RETRIEVER_DEFAULT_K = 5

MAX_TOOL_CALLS = 8  # brief §1.3 hard cap -> CAP_HIT -- NEVER raised (M4.1 spec deliverable 3)
MAX_TOOL_RESULT_CHARS = 1500  # per tool result, section_ref preserved
# M4.1 request-shape repair (specs/milestones/m4_1.md deliverable 3): the v1
# ledger showed every one of the 21 GLM arm-A schema failures dying at
# output_tokens == 1200 == 2 x the old MAX_TOKENS_FINAL (600) -- raised
# uniformly for every arm and model, never a per-arm value.
# MAX_TOKENS_TOOL_TURN stays at 300: the M4.1 probes (data/reports/
# m4_1_probes.json) did not show tool-turn truncation losing tool calls or
# forcing a premature finalization at either probed model, so it is left
# alone per the spec's explicit "if the probes do not show it hurting, leave
# it at 300 and say so" instruction.
MAX_TOKENS_TOOL_TURN = 300
MAX_TOKENS_FINAL = 1200
MAX_DEFINITION_CHARS = 6000  # ceiling on a returned defined-term block
LLM_TEMPERATURE = 0
API_MAX_RETRIES = 3  # API errors -> 3 retries w/ backoff
SCHEMA_MAX_RETRIES = 1  # schema-invalid -> exactly one retry
RATIONALE_MAX_WORDS = 80
EVIDENCE_MAX_ITEMS = 3

# M4.1: evidence-capture truncation limits (spec deliverable 1).
FAILURE_DETAIL_MAX_CHARS = 300
RAW_FINAL_TEXT_MAX_CHARS = 1500

# M4.1: real exponential backoff base (spec deliverable 3, request shape --
# the old 0.05 * 2^n backoff was not a real backoff). Tests monkeypatch this
# down via tests/conftest.py so the offline suite stays fast.
API_RETRY_BASE_DELAY_S = 1.0

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MILESTONE_TAG = "m1"

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
M1_SMOKE_CASE_IDS = ("contract_0__q01", "contract_0__q06")  # direct + defined-term
M1_SMOKE_MAX_USD = 0.15

# --- Milestone 2: eval harness (scorers, spend accounting, Braintrust) -----
MIN_GOLD_OVERLAP_CHARS = 50
EVAL_MILESTONE_TAG = "m2"

# 3 dev cases: a direct question, a defined-term question with non-null
# required_evidence, and a gold-"No" question (exercises "correct No is not
# an abstention" on a real metered run, not only in a unit test). All three
# are present in dev.jsonl (verified in the M2 plan).
M2_SMOKE_CASE_IDS = ("contract_0__q01", "contract_0__q06", "contract_0__q12")
M2_SMOKE_MAX_USD = 0.10  # 3 cases at the measured ~$0.026/case, with slack
# specs/milestones/m2.md "Budget scaling": the $0.50 figure is the allocation
# guide (target, reported) -- the enforced limit is double that, per the
# spec's explicit "Absolute per-milestone limit: double the guide" clause.
M1_M2_TARGET_USD = 0.50  # the M1+M2 allocation guide -- reported, not gated
M1_M2_MAX_USD = 1.00  # absolute per-milestone limit (2x guide) -- enforced

# Measured from M1's ledger (68 calls over its smoke runs): the calls/case
# ratio used by dealpoint.eval.spend's "ledger:tokens x price" estimate basis
# when the ledger carries no case_id to compute a true per-case mean from.
EST_CALLS_PER_CASE = 4

# --- Milestone 3: retrieval tournament -------------------------------------
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # fastembed cross-encoder, 88 MB, CPU
RRF_K = 60  # standard reciprocal-rank-fusion constant
HYBRID_FETCH_K = 20  # per-leg depth before fusion (brief §2.3: "hybrid top-20")
RERANK_FETCH_K = 20  # candidates re-scored by the cross-encoder
TOURNAMENT_K = 10  # ranked-list depth the metrics are computed over
TOURNAMENT_QUERIES_PATH = EVAL_DIR / "tournament_queries.json"
TOURNAMENT_JSON_PATH = REPORTS_DIR / "tournament.json"
TOURNAMENT_MD_PATH = REPORTS_DIR / "tournament.md"
TOURNAMENT_MILESTONE_TAG = "m3"

# The cheap workhorse for every dev-loop/smoke/exploratory metered call from M3 on
# (engineer's instruction, 2026-09-04). Verified on OpenRouter 2026-09-04:
# prompt $0.075/M, completion $0.25/M; supports tools, tool_choice, response_format,
# structured_outputs. M3 itself is LLM-free.
WORKHORSE_MODEL = "z-ai/glm-5.3-flash"

# Arm C, frozen by the M3 tournament (dealpoint/eval/tournament.py), full dev
# set (58 cases), git_sha7=2afdce5. Winner by the tournament's rule (highest
# hit\u00405 on canonical queries; ties -> MRR canonical, then hit\u00405 on maud,
# then lower wall-clock, then name): `hybrid_rrf`, canonical hit\u00405=0.9138 vs
# dense hit\u00405=0.8103 (margin +0.1034); it also beat `hybrid_rrf_rerank`
# (same 0.9138 hit\u00405) on MRR (0.7319 vs 0.7271). See data/reports/
# tournament.json for the full run. NOTE (brief-vs-measurement, reported not
# resolved): the brief names arm C `agent-hybrid-rerank`, but the tournament
# winner is hybrid RRF *without* rerank -- the measurement wins over the
# label, per the milestone spec's explicit instruction not to hand-pick a
# hybrid+rerank config to make the name true.
ARM_C_RETRIEVER: dict = {
    "fetch_k": HYBRID_FETCH_K,
    "kind": "hybrid_rrf",
    "multi_query": False,
    "name": "hybrid_rrf",
    "rerank_model": None,
    "rrf_k": RRF_K,
}

# --- Milestone 4: four-arm experiment on the frozen test set --------------
FOUR_ARM_MILESTONE_TAG = "m4"
M4_TARGET_USD = 1.50   # the M4 allocation guide -- reported, not gated
M4_MAX_USD = 3.00      # absolute per-milestone limit (2x guide) -- enforced
M4_ENVELOPE_USD = 4.00 # the whole unattended M1-M6 OpenRouter envelope

SKILL_DIR = REPO_ROOT / "skills" / "ma-deal-point-review"
SKILL_PLAYBOOKS_DIR = SKILL_DIR / "playbooks"

# Arm B's own retriever config (plain dense, same shape as arm C's config
# dict so "arm differs by exactly one key" is checkable -- spec deliverable 3).
DENSE_RETRIEVER: dict = {
    "kind": "dense",
    "name": "dense",
    "fetch_k": RETRIEVER_DEFAULT_K,
    "rerank_model": None,
    "multi_query": False,
    "rrf_k": RRF_K,
}

# The four arms, brief §2.2. Exactly three keys per arm so "differs from its
# predecessor by exactly one config key" is mechanically checkable: A->B
# differs in `loop`, B->C in `retriever`, C->D in `skill`.
ARMS: dict[str, dict] = {
    "A": {"loop": "pipeline", "retriever": DENSE_RETRIEVER, "skill": False},
    "B": {"loop": "agent", "retriever": DENSE_RETRIEVER, "skill": False},
    "C": {"loop": "agent", "retriever": ARM_C_RETRIEVER, "skill": False},
    "D": {"loop": "agent", "retriever": ARM_C_RETRIEVER, "skill": True},
}
ARM_ORDER: tuple[str, ...] = ("A", "B", "C", "D")

TEST_SUBSET_V1_PATH = EVAL_DIR / "test_subset_v1.json"
FOUR_ARM_JSON_PATH = REPORTS_DIR / "four_arm.json"
FOUR_ARM_MD_PATH = REPORTS_DIR / "four_arm.md"
README_PATH = REPO_ROOT / "README.md"

# --- Milestone 4.1: execution-reliability repair and four-arm v2 re-run ----
M4_1_MILESTONE_TAG = "m4_1"
M4_1_TARGET_USD = 1.00   # the M4.1 allocation guide -- reported, not gated
M4_1_MAX_USD = 1.50      # absolute per-milestone limit -- enforced

# v1 artefacts preserved before any v2 write (spec deliverable 4).
FOUR_ARM_V1_JSON_PATH = REPORTS_DIR / "four_arm_v1_execution_defects.json"
FOUR_ARM_V1_MD_PATH = REPORTS_DIR / "four_arm_v1_execution_defects.md"
FOUR_ARM_MANIFEST_V1_PATH = REPORTS_DIR / "four_arm_manifest_v1.json"

M4_1_PROBES_JSON_PATH = REPORTS_DIR / "m4_1_probes.json"

# --- Milestone 5: calibrated multi-judge evaluation ------------------------
JUDGES_MILESTONE_TAG = "m5"
M5_TARGET_USD = 0.60   # the M5 allocation guide -- reported, not gated
M5_MAX_USD = 1.20      # absolute per-milestone limit (2x guide) -- enforced

RUBRICS_PATH = REPO_ROOT / "eval" / "judges" / "rubrics.md"
JUDGED_SUBSET_PATH = EVAL_DIR / "judged_subset.json"
JUDGE_SCORES_PATH = EVAL_DIR / "judge_scores.jsonl"
CALIBRATION_DIR = EVAL_DIR / "calibration"
JUDGES_JSON_PATH = REPORTS_DIR / "judges.json"
JUDGES_MD_PATH = REPORTS_DIR / "judges.md"
JUDGE_SLATE_PATH = REPORTS_DIR / "judge_slate.json"
JUDGE_MAX_RETRIEVED_CHARS = 600
JUDGE_MAX_TOKENS = 400

# --- Milestone 6: model cost/quality Pareto -------------------------------
PARETO_MILESTONE_TAG = "m6"
M6_TARGET_USD = 1.00     # the M6 allocation guide -- reported, not gated
M6_ENVELOPE_USD = 4.00   # the whole unattended M1-M6 envelope -- never exceeded
# The runner's own stop-floor: no NEW model is started once the projected
# total ledger would cross this. Leaves headroom of the envelope untouched
# for a re-run of the judging pass or a report regeneration.
M6_STOP_USD = 3.90
PARETO_JSON_PATH = REPORTS_DIR / "pareto.json"
PARETO_MD_PATH = REPORTS_DIR / "pareto.md"
PARETO_SVG_PATH = REPORTS_DIR / "pareto.svg"
PARETO_SLATE_PATH = REPORTS_DIR / "pareto_slate.json"
PARETO_MANIFEST_PATH = REPORTS_DIR / "pareto_manifest.json"
PARETO_PROBE_N_CASES = 2

# --- Milestone 7a: framework roles (LlamaIndex, DeepEval, Braintrust sync) --
M7A_MILESTONE_TAG = "m7a"
M7A_TARGET_USD = 1.00     # soft target -- reported, not gated
M7A_MAX_USD = 2.50        # absolute per-milestone limit -- enforced
GLOBAL_ENVELOPE_USD = 6.00
RAG_SYNTH_MODEL_ENV = "RAG_SYNTH_MODEL"
DEEPEVAL_MODEL_ENV = "DEEPEVAL_MODEL"
LI_RAG_EVAL_JSON_PATH = REPORTS_DIR / "li_rag_eval.json"
LI_RAG_EVAL_MD_PATH = REPORTS_DIR / "li_rag_eval.md"
SYNTHETIC_DEV_QUERIES_PATH = EVAL_DIR / "synthetic_dev_queries.jsonl"
DEEPEVAL_CROSSCHECK_JSON_PATH = REPORTS_DIR / "deepeval_crosscheck.json"
DEEPEVAL_CROSSCHECK_MD_PATH = REPORTS_DIR / "deepeval_crosscheck.md"
BTQL_INVESTIGATIONS_PATH = REPORTS_DIR / "btql_investigations.json"
REPRESENTATIVE_CASES_PATH = REPORTS_DIR / "representative_cases.json"
BRAINTRUST_SYNC_PATH = REPORTS_DIR / "braintrust_sync.json"
FRAMEWORK_VERSIONS_PATH = REPORTS_DIR / "framework_versions.json"
M7A_DISK_GUARD_PATH = REPORTS_DIR / "m7a_disk_guard.json"
M7A_REVIEW_SET_N = 12
M7A_MAX_SCORES_PER_CASE = 12
M7A_MAX_SUBSET_FOR_WIDE_SCORES = 60
M7A_DISK_FLOOR_BYTES = 500_000_000

