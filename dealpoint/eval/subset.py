"""The frozen discriminative subset, `data/eval/test_subset_v1.json` (spec deliverable 4).

Chosen before any test case runs and without looking at any model output --
the only case-level signal used is `gold_answer != majority_answer`, a MAUD
label fact recorded in `test.jsonl`/`counterfactual.jsonl` at M0. The rule
and the seed are written into the JSON itself; `generate_subset()` is pure
and deterministic, so a `gate_m4` test can regenerate it byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json

from dealpoint.config import (
    COUNTERFACTUAL_JSONL_PATH,
    DATASET_VERSION_TXT_PATH,
    EVAL_DIR,
    JUDGED_SUBSET_PATH,
    SEED,
    TEST_JSONL_PATH,
    TEST_SUBSET_V1_PATH,
)

# Reasoning types where arms should differ come first (brief Appendix
# B / spec §4.1): defined-term, cross-ref, carve-out questions, then the rest.
QUESTION_PRIORITY: tuple[str, ...] = (
    "q05", "q06", "q07", "q08", "q09", "q10", "q11", "q12",
    "q01", "q02", "q03", "q04",
)

N_REDACTED = 6
N_OUT_OF_SCOPE = 2

RULE_TEXT = (
    "1. Question priority order: reasoning types where arms should differ come first "
    "(QuestionSpec.reasoning_type) -- defined-term, cross-ref, carve-out -> "
    "q05, q06, q07, q08, q09, q10, q11, q12; then the rest -> q01, q02, q03, q04. "
    "Within a group, ascending question id.\n"
    "2. 2 test cases per question (24) -- from data/eval/test.jsonl, prefer cases where "
    "gold_answer != majority_answer (the majority baseline already gets these wrong, so "
    "arm differences are visible rather than masked by a degenerate question). Order "
    "within the pool by sha256(f'{SEED}:{case_id}') ascending; take the first 2. SEED = 42 "
    "from dealpoint.config.\n"
    "3. 8 counterfactual cases -- 6 kind == 'redacted', at most one per question, taking "
    "questions in priority order (-> q05, q06, q07, q08, q09, q10: six distinct questions, "
    ">= 4); within a question, the same seeded hash order. Plus 2 kind == 'out_of_scope', "
    "first two by seeded hash order of their case ids.\n"
    "4. Total 32.\n"
    "Tranches: tranche_1 (18 cases) = the rank-1 case of each of the 12 questions "
    "(priority order) + 4 redacted (q05, q06, q07, q08) + 2 out-of-scope. tranche_2 (14 "
    "cases) = the rank-2 case of each of the 12 questions + the remaining 2 redacted "
    "(q09, q10). case_ids is tranche_1 + tranche_2 in that order, so 'the largest "
    "whole-question prefix' is literal.\n"
    "Fallback (recorded if invoked, not by default): if the ledger's measured cost/case "
    "says 32 cases does not fit 4 arms at the constant model, reduce to 1 case per "
    "question + 6 counterfactual (18) by the same rule."
)


def _load_jsonl(path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _seeded_key(case_id: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()


def _dataset_version() -> str:
    if DATASET_VERSION_TXT_PATH.exists():
        return DATASET_VERSION_TXT_PATH.read_text(encoding="utf-8").strip()
    return ""


def generate_subset(seed: int = SEED) -> dict:
    """Pure, deterministic construction of the frozen subset payload.

    Reads only `data/eval/test.jsonl` and `data/eval/counterfactual.jsonl`
    (label data, never model output) plus the recorded `dataset_version`.
    """
    test_rows = _load_jsonl(TEST_JSONL_PATH)
    cf_rows = _load_jsonl(COUNTERFACTUAL_JSONL_PATH)

    by_question: dict[str, list[dict]] = {}
    for row in test_rows:
        by_question.setdefault(row["question_id"], []).append(row)

    rank1_by_q: dict[str, str] = {}
    rank2_by_q: dict[str, str] = {}
    for qid in QUESTION_PRIORITY:
        rows = by_question.get(qid, [])
        preferred = [r for r in rows if r.get("gold_answer") != r.get("majority_answer")]
        pool = preferred if len(preferred) >= 2 else rows
        pool_sorted = sorted(pool, key=lambda r: _seeded_key(r["case_id"], seed))
        top2 = pool_sorted[:2]
        if len(top2) < 2:
            raise ValueError(f"question {qid!r} has fewer than 2 candidate test cases")
        rank1_by_q[qid] = top2[0]["case_id"]
        rank2_by_q[qid] = top2[1]["case_id"]

    redacted_by_question: dict[str, list[dict]] = {}
    for row in cf_rows:
        if row.get("kind") == "redacted":
            redacted_by_question.setdefault(row["question_id"], []).append(row)

    redacted_by_selected_q: dict[str, str] = {}
    for qid in QUESTION_PRIORITY:
        if len(redacted_by_selected_q) >= N_REDACTED:
            break
        rows = redacted_by_question.get(qid, [])
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: _seeded_key(r["case_id"], seed))
        redacted_by_selected_q[qid] = rows_sorted[0]["case_id"]

    if len(redacted_by_selected_q) < N_REDACTED:
        raise ValueError(
            f"only {len(redacted_by_selected_q)} distinct questions have a redacted case; "
            f"need {N_REDACTED}"
        )

    oos_rows = [r for r in cf_rows if r.get("kind") == "out_of_scope"]
    oos_sorted = sorted(oos_rows, key=lambda r: _seeded_key(r["case_id"], seed))
    oos_selected = [r["case_id"] for r in oos_sorted[:N_OUT_OF_SCOPE]]
    if len(oos_selected) < N_OUT_OF_SCOPE:
        raise ValueError(f"only {len(oos_selected)} out-of-scope cases available")

    # tranche_1: rank-1 of each question (priority order) + first 4 redacted
    # questions in priority order (q05, q06, q07, q08) + both oos.
    redacted_t1_questions = [q for q in QUESTION_PRIORITY if q in redacted_by_selected_q][:4]
    redacted_t2_questions = [
        q for q in QUESTION_PRIORITY if q in redacted_by_selected_q
    ][4:N_REDACTED]

    tranche_1 = (
        [rank1_by_q[qid] for qid in QUESTION_PRIORITY]
        + [redacted_by_selected_q[qid] for qid in redacted_t1_questions]
        + oos_selected
    )
    tranche_2 = [rank2_by_q[qid] for qid in QUESTION_PRIORITY] + [
        redacted_by_selected_q[qid] for qid in redacted_t2_questions
    ]

    case_ids = tranche_1 + tranche_2
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate case id in generated subset")

    subset_hash = hashlib.sha256(
        json.dumps(case_ids, sort_keys=False).encode("utf-8")
    ).hexdigest()[:12]

    return {
        "case_ids": case_ids,
        "dataset_version": _dataset_version(),
        "generated_from": {
            "test": "data/eval/test.jsonl",
            "counterfactual": "data/eval/counterfactual.jsonl",
        },
        "n_cases": len(case_ids),
        "question_priority": list(QUESTION_PRIORITY),
        "rule": RULE_TEXT,
        "seed": seed,
        "subset_hash": subset_hash,
        "tranche_1": tranche_1,
        "tranche_2": tranche_2,
        "version": "v1",
    }


def write_subset(path=TEST_SUBSET_V1_PATH, seed: int = SEED) -> dict:
    payload = generate_subset(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    return payload


def _subset_path(name: str):
    if name == "test_subset_v1":
        return TEST_SUBSET_V1_PATH
    return EVAL_DIR / f"{name}.json"


def load_subset_payload(name: str = "test_subset_v1") -> dict:
    path = _subset_path(name)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_subset_case_ids(name: str = "test_subset_v1", tranche: int | None = None) -> list[str]:
    payload = load_subset_payload(name)
    if tranche == 1:
        return list(payload["tranche_1"])
    if tranche == 2:
        return list(payload["tranche_2"])
    return list(payload["case_ids"])


def load_subset_cases(name: str = "test_subset_v1", tranche: int | None = None) -> list[dict]:
    """The frozen subset's case rows, in file order, resolved across test/counterfactual."""
    from dealpoint.eval.cases import find_case

    case_ids = load_subset_case_ids(name, tranche=tranche)
    return [find_case(cid) for cid in case_ids]


# --- M5: the judged subset (spec deliverable 4) -----------------------------

JUDGED_SUBSET_RULE_TEXT = (
    "1. Universe: the 18 case_ids of test_subset_v1.json tranche_1 -- the only tranche for which "
    "both M5 Haiku variants have completed traces on disk (data/reports/four_arm_manifest.json, "
    "git_sha7 e3ee9cc). tranche_1 is exactly 1 test case per question (12) + 4 redacted + 2 "
    "out_of_scope.\n"
    "2. All 18 are judged. Recorded factory-latitude decision (D1): the spec's 12-case "
    "one-per-question default cannot satisfy the same spec's floor that counterfactuals stay in; "
    "tranche_1 is the smallest set that satisfies both, and the 6 extra cases cost ~$0.05.\n"
    "3. Ordering (deterministic, seeded, fixed BEFORE any judge call, computed only from the frozen "
    "M4.1 v2 agent results -- never from any judge output): ascending by "
    "(disagreement_class, sha256(f'{SEED}:{case_id}')), where disagreement_class is 0 when arm A "
    "@ anthropic/claude-haiku-4.5 and arm D @ anthropic/claude-haiku-4.5 produced different "
    "answer strings -- a missing finding counting as the distinct value '<no finding>' -- and 1 "
    "otherwise. SEED = 42 from dealpoint.config. The ordering exists so that any later "
    "budget-forced trim takes a deterministic prefix; it does not exclude anything here.\n"
    "4. Variants judged: A @ anthropic/claude-haiku-4.5 and D @ anthropic/claude-haiku-4.5 (spec "
    "deliverable 4), plus D @ z-ai/glm-5.3-flash (spec, 'Judge trio': judge the default and GLM "
    "in M5). 18 cases x 3 variants = 54 traces; 54 x 3 judges = 162 calls."
)

# The three M5-judged (variant_id, arm, model) legs. Their result-file paths
# are resolved from the frozen `four_arm_manifest.json` by
# `generate_judged_subset`, not hard-coded here, so a git_sha7 change is
# visible rather than silently stale.
JUDGED_VARIANT_LEGS: tuple[tuple[str, str, str], ...] = (
    ("A@haiku", "A", "anthropic/claude-haiku-4.5"),
    ("D@haiku", "D", "anthropic/claude-haiku-4.5"),
    ("D@glm", "D", "z-ai/glm-5.3-flash"),
)


def _read_results_jsonl(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["case_id"]] = row
    return rows


def _resolve_variant_paths(manifest: list[dict]) -> dict[str, str]:
    """variant_id -> results_path, resolved from the frozen four_arm_manifest.json."""
    resolved: dict[str, str] = {}
    for variant_id, arm, model in JUDGED_VARIANT_LEGS:
        matches = [e for e in manifest if e.get("arm") == arm and e.get("model") == model]
        if not matches:
            raise ValueError(f"no manifest entry for arm={arm!r} model={model!r}")
        # Prefer the tranche_1-scoped entry when more than one exists (the
        # Haiku replication legs are tranche_1-only; the GLM headline leg
        # covers the full 32-case subset and is used for all 18 cases here).
        preferred = next((e for e in matches if e.get("tranche") == "tranche_1"), matches[-1])
        resolved[variant_id] = preferred["results_path"]
    return resolved


def generate_judged_subset(seed: int = SEED, manifest_path=None) -> dict:
    """Pure, deterministic construction of the M5 judged-subset payload.

    Universe is tranche_1 of the already-frozen `test_subset_v1.json` (18
    cases); ranking is computed only from the frozen M4.1 v2 Haiku A/D result
    rows on disk (never from any judge output). See `JUDGED_SUBSET_RULE_TEXT`
    for the full rule, written verbatim into the output payload.
    """
    from dealpoint.eval.four_arm_sweep import MANIFEST_PATH

    manifest_path = manifest_path or MANIFEST_PATH
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    variant_paths = _resolve_variant_paths(manifest)
    case_ids = load_subset_case_ids("test_subset_v1", tranche=1)

    a_rows = _read_results_jsonl(variant_paths["A@haiku"])
    d_rows = _read_results_jsonl(variant_paths["D@haiku"])

    def _answer(rows: dict[str, dict], case_id: str) -> str:
        row = rows.get(case_id)
        finding = row.get("finding") if row else None
        return finding["answer"] if finding else "<no finding>"

    disagreement_class: dict[str, int] = {}
    for cid in case_ids:
        disagreement_class[cid] = 0 if _answer(a_rows, cid) != _answer(d_rows, cid) else 1

    ranked = sorted(case_ids, key=lambda cid: (disagreement_class[cid], _seeded_key(cid, seed)))
    rank = {cid: i for i, cid in enumerate(ranked)}

    variants = [
        {"variant_id": vid, "arm": arm, "model": model, "results_path": variant_paths[vid]}
        for vid, arm, model in JUDGED_VARIANT_LEGS
    ]

    subset_hash = hashlib.sha256(
        json.dumps(ranked, sort_keys=False).encode("utf-8")
    ).hexdigest()[:12]

    n_cases = len(ranked)
    n_traces = n_cases * len(variants)

    return {
        "case_ids": ranked,
        "rank": rank,
        "disagreement_class": disagreement_class,
        "variants": variants,
        "n_cases": n_cases,
        "n_traces": n_traces,
        "n_judge_calls": n_traces * 3,
        "seed": seed,
        "rule": JUDGED_SUBSET_RULE_TEXT,
        "subset_hash": subset_hash,
        "source_subset": "test_subset_v1",
        "source_tranche": 1,
        "dataset_version": _dataset_version(),
        "version": "v1",
    }


def write_judged_subset(path=JUDGED_SUBSET_PATH, seed: int = SEED) -> dict:
    payload = generate_judged_subset(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    payload = write_subset()
    print(json.dumps({"n_cases": payload["n_cases"], "subset_hash": payload["subset_hash"]}))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
