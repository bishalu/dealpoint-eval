"""The M2 metered smoke: the *only* module in M2 that may call OpenRouter
(spec §6).

`python -m dealpoint.eval.smoke` (no flag): if the canonical smoke result
file already exists at the current git_sha7 + index_version, prints it and
exits 0 without any model call. Otherwise runs the 3 `M2_SMOKE_CASE_IDS` dev
cases through `dealpoint.eval.run` (arm B, `DEFAULT_MODEL`), scores offline,
writes results, and additionally invokes the Braintrust adapter with
`no_send_logs=True` so that path is genuinely exercised at zero score cost.

`python -m dealpoint.eval.smoke --send`: loads the *persisted* result rows
and sends them once, for real, to Braintrust (`no_send_logs=False`). Refuses
with a non-zero exit if the result file is missing -- that refusal is what
prevents a second metered run.
"""

from __future__ import annotations

import argparse
import json
import sys

from dealpoint.config import (
    DEFAULT_MODEL,
    M1_M2_MAX_USD,
    M1_M2_TARGET_USD,
    M2_SMOKE_CASE_IDS,
    M2_SMOKE_MAX_USD,
    RESULTS_DIR,
)
from dealpoint.eval.cases import git_sha7


class SmokeError(RuntimeError):
    """Raised when the smoke cannot proceed safely (e.g. `--send` with no result file)."""


def _current_index_version() -> str:
    from dealpoint.corpus.retrievers import index_version

    return index_version()


def _smoke_stem(index_version: str, sha7: str) -> str:
    from dealpoint.eval.run import result_stem

    return result_stem("dev", "B", DEFAULT_MODEL, index_version, sha7)


def _existing_smoke_paths() -> tuple[str, str] | None:
    """(results_path, summary_path) for the current git_sha7 + index_version, if present."""
    sha7 = git_sha7()
    index_version = _current_index_version()
    stem = _smoke_stem(index_version, sha7)
    results_path = RESULTS_DIR / f"{stem}.jsonl"
    summary_path = RESULTS_DIR / f"{stem}_summary.json"
    if results_path.exists() and summary_path.exists():
        return str(results_path), str(summary_path)
    return None


def _ledger_total() -> float:
    from dealpoint.eval.spend import realized_usd

    return realized_usd()


def run_smoke() -> dict:
    """Run (or reuse) the 3-case smoke and return its summary dict."""
    existing = _existing_smoke_paths()
    if existing is not None:
        _, summary_path = existing
        with open(summary_path, encoding="utf-8") as fh:
            return json.load(fh)

    from dealpoint.eval.run import run_eval_set

    ledger_before = _ledger_total()

    summary = run_eval_set(
        case_set="dev",
        arm="B",
        model=DEFAULT_MODEL,
        cases=",".join(M2_SMOKE_CASE_IDS),
        fake=False,
        out_dir=RESULTS_DIR,
        milestone_tag="m2",
    )

    ledger_after = _ledger_total()
    delta = round(ledger_after - ledger_before, 6)
    if delta <= 0:
        raise SmokeError(
            f"smoke ran but the ledger gained no usd (delta=${delta:.6f}) -- the client "
            f"may not have been metered as expected"
        )
    if delta > M2_SMOKE_MAX_USD:
        raise SmokeError(
            f"smoke cost ${delta:.4f} exceeds the ${M2_SMOKE_MAX_USD} ceiling -- an "
            f"expensive smoke is a defect, not a threshold to raise"
        )
    # spec's "Budget scaling": the $0.50 figure is the allocation guide
    # (target) -- reported here, never gated. Only the absolute (2x) limit
    # is enforced by an assertion.
    if ledger_after > M1_M2_TARGET_USD:
        print(
            f"note: M1+M2 realised total ${ledger_after:.4f} exceeds the "
            f"${M1_M2_TARGET_USD} allocation guide (target); still within the "
            f"${M1_M2_MAX_USD} absolute limit",
            file=sys.stderr,
        )
    if ledger_after > M1_M2_MAX_USD:
        raise SmokeError(
            f"M1+M2 realised total ${ledger_after:.4f} exceeds the ${M1_M2_MAX_USD} "
            f"absolute per-milestone limit"
        )

    # Exercise the Braintrust adapter path at zero score cost (no_send_logs=True).
    from dealpoint.eval.braintrust_adapter import braintrust_available, run_eval

    if braintrust_available():
        results_path = summary["results_path"]
        rows = []
        with open(results_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        run_eval(rows, arm="B", model=DEFAULT_MODEL, case_set="dev", no_send_logs=True)

    return summary


def send_smoke() -> dict:
    """Load the persisted smoke result rows and send them once, for real."""
    existing = _existing_smoke_paths()
    if existing is None:
        raise SmokeError(
            "no persisted smoke result file for the current git_sha7 + index_version -- "
            "run `just smoke` first (without --send) to produce one. --send never re-runs "
            "the agent, to guarantee at most one metered smoke."
        )
    results_path, _summary_path = existing

    from dealpoint.eval.braintrust_adapter import run_eval

    rows = []
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    result = run_eval(rows, arm="B", model=DEFAULT_MODEL, case_set="dev", no_send_logs=False)
    return {"experiment_name": result["experiment_name"], "metadata": result["metadata"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dealpoint.eval.smoke")
    parser.add_argument(
        "--send",
        action="store_true",
        help="send the persisted smoke results to Braintrust once (never re-runs the agent)",
    )
    args = parser.parse_args(argv)

    try:
        if args.send:
            result = send_smoke()
        else:
            result = run_smoke()
    except SmokeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
