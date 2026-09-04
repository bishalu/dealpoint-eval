"""Case resolution shared by the agent CLI and the eval runner (spec §1).

`find_case`, `resolve_document_id` and `resolve_question` were originally
private to `dealpoint/agent/run.py`; they live here now as the single,
public implementation so a redacted case is resolved identically (against
its own redacted document) everywhere it is run. `dealpoint/agent/run.py`
imports these instead of forking the logic.
"""

from __future__ import annotations

import json
import subprocess

from dealpoint.config import (
    COUNTERFACTUAL_JSONL_PATH,
    DEV_JSONL_PATH,
    TEST_JSONL_PATH,
)
from dealpoint.data.questions import OUT_OF_SCOPE_QUESTION_BY_ID, QUESTION_BY_ID, QuestionSpec

CASE_SET_PATHS = {
    "dev": DEV_JSONL_PATH,
    "test": TEST_JSONL_PATH,
    "counterfactual": COUNTERFACTUAL_JSONL_PATH,
}


def load_case_set(name: str) -> list[dict]:
    """Load one case set (`"dev"`, `"test"` or `"counterfactual"`) as a list of rows.

    Rows are returned in file order. Raises `KeyError` for an unknown set name
    and `FileNotFoundError` (naming `just data`) if the JSONL is absent.
    """
    if name not in CASE_SET_PATHS:
        raise KeyError(f"unknown case set {name!r}; expected one of {sorted(CASE_SET_PATHS)}")
    path = CASE_SET_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `just data` to build it.")
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_case(case_id: str) -> dict:
    """Locate one case row by `case_id` across dev/test/counterfactual JSONL."""
    for path in (DEV_JSONL_PATH, TEST_JSONL_PATH, COUNTERFACTUAL_JSONL_PATH):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("case_id") == case_id:
                    return row
    raise KeyError(f"case_id {case_id!r} not found in dev/test/counterfactual JSONL")


def resolve_document_id(case: dict) -> str:
    """The document a case is asked against: the redacted variant itself for a
    redacted counterfactual case, otherwise the base agreement.
    """
    if case.get("kind") == "redacted":
        return case["case_id"]
    return case["agreement_id"]


def resolve_question(case: dict) -> QuestionSpec:
    question_id = case["question_id"]
    if question_id in QUESTION_BY_ID:
        return QUESTION_BY_ID[question_id]
    # out-of-scope counterfactual case: no fixed option list.
    oos = OUT_OF_SCOPE_QUESTION_BY_ID.get(question_id)
    text = case.get("question_text") or (oos.text if oos else question_id)
    return QuestionSpec(
        id=question_id,
        maud_question=text,
        text_type="out-of-scope",
        category="out-of-scope",
        gloss=text,
        options=(),
        canonical_query=text,
        reasoning_type="out-of-scope",
        required_evidence=None,
    )


def git_sha7() -> str:
    """Short (7-char) git sha of HEAD, or `"nogit"` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            if sha:
                return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return "nogit"


def slugify_model(model: str) -> str:
    """Model id -> filesystem/experiment-name-safe slug (`/` -> `_`)."""
    return model.replace("/", "_")
