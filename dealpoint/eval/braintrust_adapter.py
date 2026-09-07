"""Braintrust adapter: `braintrust.Eval()` replaying already-computed result
rows (spec §5).

`braintrust` is imported lazily inside every function, never at module top
level, so pyright and the offline test suite stay clean when the package or
its API key is absent -- the same rule this codebase already applies to
`openai`/`qdrant`.

Exactly six scores are logged per case (`SCORE_NAMES`) -- the free tier
meters scores, and the milestone spec caps this deliberately. Everything
else computed by `dealpoint.eval.scorers.score_case` goes into per-case
metadata instead.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from dealpoint.config import REPORTS_DIR, VERSIONS_JSON_PATH
from dealpoint.eval.cases import slugify_model

# The Braintrust project name. Override with BRAINTRUST_PROJECT to point every sync, cockpit and
# showroom command at another org/project (the demo org), without touching this repo's config.
PROJECT = os.environ.get("BRAINTRUST_PROJECT", "dealpoint-eval")

# Exactly six scores per case (spec §5 / deliverable 3). `skill_adherence`
# returns None until M4 but must already be in this list from M2 onward.
SCORE_NAMES: tuple[str, ...] = (
    "grounded_accuracy",
    "answer_correct",
    "citation_gold_overlap",
    "citation_verbatim",
    "abstain_correct",
    "skill_adherence",
)

METADATA_KEYS: tuple[str, ...] = (
    "arm",
    "model",
    "index_version",
    "skill_version",
    "git_sha",
    "case_set",
)

BRAINTRUST_RUNS_PATH = REPORTS_DIR / "braintrust_runs.json"

# --- per-org report paths -------------------------------------------------------
# Braintrust API keys are org-bound, so the org is a property of the active key, never of
# this repo's config. Anything that records what was written to an org (the score ledger,
# the showroom manifest) lives under data/reports/orgs/<org-slug>/ so a fresh org starts
# empty and never inherits or overwrites another org's state. BRAINTRUST_LEDGER_FILE stays
# as an explicit override only.
ORG_API_URL = "https://api.braintrust.dev/v1/organization"
_ORG_SLUG_CACHE: dict[str, str] = {}


def org_slug_from_name(name: str) -> str:
    """Deterministic, filesystem-safe slug: 'Vibeset Technologies' -> 'vibeset-technologies'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "org"


def org_slug(key: str | None = None) -> str:
    """Slug of the org the active API key belongs to; 'local' when no key is active.

    The active key is the one exported to `BRAINTRUST_API_KEY` (every live entry point exports
    the key it resolved, which is also what the `braintrust` SDK reads). Dry runs without a key
    resolve to 'local' and never touch the network. One GET per key per process, cached.
    """
    key = key or os.environ.get("BRAINTRUST_API_KEY")
    if not key:
        return "local"
    if key not in _ORG_SLUG_CACHE:
        import requests

        r = requests.get(ORG_API_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        orgs = r.json().get("objects", [])
        if len(orgs) != 1:
            raise RuntimeError(f"expected the API key to be bound to exactly one org, found {len(orgs)}")
        _ORG_SLUG_CACHE[key] = org_slug_from_name(orgs[0]["name"])
    return _ORG_SLUG_CACHE[key]


def org_report_path(name: str, override_env: str | None = None) -> Path:
    """`data/reports/orgs/<org-slug>/<name>`, unless `override_env` names a set environment variable."""
    override = os.environ.get(override_env) if override_env else None
    if override:
        return Path(override)
    return REPORTS_DIR / "orgs" / org_slug() / name


def experiment_name(arm: str, model: str, index_version: str, git_sha7: str) -> str:
    return f"{arm}-{slugify_model(model)}-{index_version}-{git_sha7}"


def _skill_version() -> str | None:
    """`skill_version` from `data/reports/versions.json` when present, else `None`.

    Skill logic is M4 work; the key is None until then (spec §5).
    """
    if not VERSIONS_JSON_PATH.exists():
        return None
    try:
        payload = json.loads(VERSIONS_JSON_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload.get("skill_version")


def experiment_metadata(
    arm: str, model: str, index_version: str, skill_version: str | None, git_sha: str, case_set: str
) -> dict:
    return {
        "arm": arm,
        "model": model,
        "index_version": index_version,
        "skill_version": skill_version,
        "git_sha": git_sha,
        "case_set": case_set,
    }


def load_braintrust_key() -> str | None:
    """`BRAINTRUST_API_KEY`: env, then `.env.braintrust`, then `.braintrust.json`."""
    key = os.environ.get("BRAINTRUST_API_KEY")
    if key:
        return key

    from dealpoint.config import REPO_ROOT

    env_path = REPO_ROOT / os.environ.get("BRAINTRUST_ENV_FILE", ".env.braintrust")
    if env_path.exists():
        from dotenv import dotenv_values

        key = dotenv_values(str(env_path)).get("BRAINTRUST_API_KEY")
        if key:
            return key

    json_path = REPO_ROOT / ".braintrust.json"
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        key = payload.get("BRAINTRUST_API_KEY")
        if key:
            return key

    return None


def braintrust_available() -> bool:
    """True iff `braintrust` is importable AND a key is loadable."""
    if load_braintrust_key() is None:
        return False
    try:
        import braintrust  # noqa: F401
    except ImportError:
        return False
    return True


def _make_scorer(name: str):
    """A Braintrust scorer function reading `name` out of the stored `output["scores"]`."""

    def scorer(input, output, expected=None, metadata=None):
        return (output or {}).get("scores", {}).get(name)

    scorer.__name__ = name
    return scorer


def _row_to_eval_case(row: dict) -> dict:
    """One persisted result row -> a Braintrust `EvalCase`-shaped dict.

    Braintrust's `task` only receives the case's `input`, so the full scores
    dict travels there (as `input["scores"]`) and `_replay_task` hands it
    straight back as `output` -- a pure replay, never a model call. Every
    score not in `SCORE_NAMES` goes into `metadata` instead of `scores`, per
    the six-score budget (spec §5).
    """
    scores = row.get("scores", {})
    extra_metadata = {k: v for k, v in scores.items() if k not in SCORE_NAMES}
    extra_metadata.update(
        {
            "case_id": row["case_id"],
            "question_id": row.get("question_id"),
            "usd": row.get("usd"),
            "chunk_version": row.get("chunk_version"),
        }
    )
    return {
        "input": {"scores": scores, "finding": row.get("finding"), "record": row.get("record")},
        "expected": None,
        "metadata": extra_metadata,
    }


def _replay_task(input):
    """The `task` Braintrust calls per case: a pure replay, never a model call.

    The scores are already computed and stored in `input` (see
    `_row_to_eval_case`); this just hands it back as `output` so the scorer
    functions (`_make_scorer`) can read from it.
    """
    return input


def run_eval(
    rows: list[dict],
    *,
    arm: str,
    model: str,
    case_set: str,
    no_send_logs: bool = True,
    project: str = PROJECT,
) -> dict:
    """Replay already-computed result `rows` through `braintrust.Eval()`.

    Never re-invokes the agent: the `task` returns each row's stored output
    verbatim, and the six scorers just read fields out of it. On a
    successful, actually-sent run (`no_send_logs=False`), appends a record
    to `data/reports/braintrust_runs.json`.
    """
    import braintrust

    if not rows:
        raise ValueError("run_eval requires at least one result row")

    index_version = rows[0].get("index_version") or "unknown"
    git_sha7 = rows[0].get("git_sha7") or "nogit"
    skill_version = _skill_version()

    exp_name = experiment_name(arm, model, index_version, git_sha7)
    metadata = experiment_metadata(arm, model, index_version, skill_version, git_sha7, case_set)

    eval_cases = [_row_to_eval_case(row) for row in rows]
    scorers = [_make_scorer(name) for name in SCORE_NAMES]

    result = braintrust.Eval(
        name=project,
        data=eval_cases,  # type: ignore[arg-type]
        task=_replay_task,
        scores=scorers,
        experiment_name=exp_name,
        metadata=metadata,
        no_send_logs=no_send_logs,
    )

    if not no_send_logs:
        summary = getattr(result, "summary", None)
        url = getattr(summary, "experiment_url", None) if summary is not None else None
        _record_braintrust_run(
            {
                "experiment_name": exp_name,
                "project": project,
                "case_set": case_set,
                "arm": arm,
                "model": model,
                "index_version": index_version,
                "git_sha": git_sha7,
                "n_cases": len(rows),
                "ts": datetime.now(UTC).isoformat(),
                "url": url,
            }
        )

    return {"experiment_name": exp_name, "metadata": metadata, "result": result}


def _record_braintrust_run(entry: dict, path: Path = BRAINTRUST_RUNS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- M5: judged-subset Braintrust path (spec deliverable 6) ----------------

# The four M5 judge-aggregate scores, kept strictly separate from the M2
# six-score budget (`SCORE_NAMES`, never touched by this section).
JUDGE_SCORE_NAMES: tuple[str, ...] = (
    "judge_reasoning",
    "judge_evidence",
    "judge_trajectory",
    "judge_professional",
)

_JUDGE_DIMENSIONS: tuple[str, ...] = ("reasoning", "evidence", "trajectory", "professional")


def _normalise_1_5_to_0_1(value: float | None) -> float | None:
    """Map a 1-5 mean-of-judges score to Braintrust's conventional 0-1 range."""
    if value is None:
        return None
    return (value - 1) / 4


def _judge_scorer(dim: str):
    def scorer(input, output, expected=None, metadata=None):
        return (output or {}).get("judge_scores_0_1", {}).get(dim)

    scorer.__name__ = f"judge_{dim}"
    return scorer


def _row_to_judge_eval_case(row: dict) -> dict:
    """One judged-subset row -> a Braintrust `EvalCase`-shaped dict.

    `row` carries the mean-of-judges per dimension (1-5, `dimension_means`)
    and, separately, each judge's own per-dimension score (`per_judge_scores`,
    `{judge_model: {dim: value_or_None}}`) -- the latter goes ONLY into
    per-case metadata, never into the four logged scores (spec: "per-judge
    scores in metadata").
    """
    dimension_means = row.get("dimension_means") or {}
    judge_scores_0_1 = {dim: _normalise_1_5_to_0_1(dimension_means.get(dim)) for dim in _JUDGE_DIMENSIONS}
    metadata = {
        "packet_id": row.get("packet_id"),
        "case_id": row.get("case_id"),
        "question_id": row.get("question_id"),
        "rubric_version": row.get("rubric_version"),
        "status": row.get("status"),
        "grounded_accuracy": row.get("grounded_accuracy"),
        "dimension_means_1_5": dimension_means,
        "per_judge_scores": row.get("per_judge_scores") or {},
    }
    return {
        "input": {"judge_scores_0_1": judge_scores_0_1},
        "expected": None,
        "metadata": metadata,
    }


def _replay_judge_task(input):
    return input


def run_judge_eval(
    rows: list[dict],
    *,
    arm: str,
    model: str,
    case_set: str,
    no_send_logs: bool = True,
    project: str = PROJECT,
) -> dict:
    """Replay already-computed M5 judge-aggregate `rows` through `braintrust.Eval()`,
    one experiment per judged variant (spec deliverable 6).

    Never re-invokes a judge model: each row's `dimension_means` (1-5,
    mean-of-judges) is normalised to 0-1 and logged as the four
    `JUDGE_SCORE_NAMES` scores; each judge's own score is metadata-only.
    Offline-safe: gated on `braintrust_available()` by the caller, default
    `no_send_logs=True`; only a real send (`no_send_logs=False`) appends to
    `data/reports/braintrust_runs.json`.
    """
    import braintrust

    if not rows:
        raise ValueError("run_judge_eval requires at least one row")

    index_version = rows[0].get("index_version") or "unknown"
    git_sha7 = rows[0].get("git_sha7") or "nogit"

    exp_name = f"judged-{experiment_name(arm, model, index_version, git_sha7)}"
    metadata = experiment_metadata(arm, model, index_version, _skill_version(), git_sha7, case_set)

    eval_cases = [_row_to_judge_eval_case(row) for row in rows]
    scorers = [_judge_scorer(dim) for dim in _JUDGE_DIMENSIONS]

    result = braintrust.Eval(
        name=project,
        data=eval_cases,  # type: ignore[arg-type]
        task=_replay_judge_task,
        scores=scorers,
        experiment_name=exp_name,
        metadata=metadata,
        no_send_logs=no_send_logs,
    )

    if not no_send_logs:
        summary = getattr(result, "summary", None)
        url = getattr(summary, "experiment_url", None) if summary is not None else None
        _record_braintrust_run(
            {
                "experiment_name": exp_name,
                "project": project,
                "case_set": case_set,
                "arm": arm,
                "model": model,
                "index_version": index_version,
                "git_sha": git_sha7,
                "n_cases": len(rows),
                "ts": datetime.now(UTC).isoformat(),
                "url": url,
            }
        )

    return {"experiment_name": exp_name, "metadata": metadata, "result": result}


def push_datasets(sets: tuple[str, ...] = ("dev", "test", "counterfactual"), version: str = "v1") -> dict:
    """Push the three case sets as Braintrust datasets `maud-dealpoint-{set}-{version}`.

    Network, unmetered (a dataset push, not a scored eval). Idempotent
    enough to re-run: `init_dataset` + `insert` upserts by row content.
    """
    import braintrust

    from dealpoint.eval.cases import load_case_set

    pushed = {}
    for set_name in sets:
        rows = load_case_set(set_name)
        dataset = braintrust.init_dataset(project=PROJECT, name=f"maud-dealpoint-{set_name}-{version}")
        for row in rows:
            dataset.insert(input=row["case_id"], expected=row.get("gold_answer"), metadata=row)
        dataset.flush()
        pushed[set_name] = len(rows)
    return pushed


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dealpoint.eval.braintrust_adapter")
    parser.add_argument("--push-datasets", action="store_true")
    args = parser.parse_args(argv)

    if args.push_datasets:
        result = push_datasets()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
