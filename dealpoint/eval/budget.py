"""`python -m dealpoint.eval.budget <four_arm|judges|pareto> [--sample]`.

Prints `estimate(sweep)` as JSON to stdout and exits 0. Nothing else may go
to stdout: `adws/adw_modules/spend.py::estimate` slices from the first `{`
to the last `}` of the process's stdout, so any diagnostic goes to stderr.

`--sample` is built and unit-tested (with `FakeClient`) but must never be
invoked metered during M2 (see the M2 plan §0.1) -- it runs 3 dev cases for
real before adding `sample_est_usd`/`sample_realized_usd` to the estimate.
"""

from __future__ import annotations

import argparse
import json
import sys

from dealpoint.eval.spend import SWEEP_DEFS, estimate, realized_usd


def run_sample(sweep_name: str) -> dict:
    """Run 3 dev cases (arm B, DEFAULT_MODEL) for real, then compare cost.

    Adds `sample_est_usd` (this run's own pre-run estimate for 3 cases) and
    `sample_realized_usd` (the ledger delta actually spent) to the sweep
    estimate. Callers that want to avoid spending money must not call this
    -- use `estimate()` directly.
    """
    from dealpoint.config import DEFAULT_MODEL
    from dealpoint.eval.run import run_eval_set
    from dealpoint.eval.spend import per_case_usd

    before = realized_usd()
    per_case, _basis = per_case_usd("B", DEFAULT_MODEL)
    sample_est = per_case * 3
    run_eval_set(case_set="dev", arm="B", model=DEFAULT_MODEL, limit=3, fake=False)
    after = realized_usd()

    data = estimate(sweep_name)
    data["sample_est_usd"] = round(sample_est, 6)
    data["sample_realized_usd"] = round(after - before, 6)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dealpoint.eval.budget")
    parser.add_argument("sweep", choices=sorted(SWEEP_DEFS))
    parser.add_argument(
        "--sample",
        action="store_true",
        help="run 3 dev cases for real first, then report sample_est_usd/sample_realized_usd "
        "(SPENDS MONEY -- never invoke this in an automated gate)",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse already printed usage (incl. valid sweep names) to stderr.
        return exc.code if isinstance(exc.code, int) else 2

    if args.sample:
        data = run_sample(args.sweep)
    else:
        data = estimate(args.sweep)

    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
