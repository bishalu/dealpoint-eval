"""M6 Pareto slate verification: live availability + one tool-calling probe per
candidate, on 2 dev cases, arm D (spec deliverable 1).

Two audiences: `verify_slate` (metered -- one 2-dev-case arm-D probe per
candidate actually being verified this run) and the pure-ish
`check_availability`/the ranking tables below (offline, no spend, driving
`tests/test_pareto_slate.py` with a `FakeClient`).

A model that fails tool calling on the probe is recorded with its failure
rate and excluded -- never silently swapped for another (spec deliverable 1,
verbatim). No model outside `PARETO_CANDIDATES` is ever contacted by
`probe_candidate` -- enforced in code, not just by convention.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from dealpoint.config import PARETO_SLATE_PATH
from dealpoint.eval.spend import (
    PARETO_ANCHOR_MODELS,
    PARETO_NEW_MODELS,
    PARETO_STRETCH_MODELS,
    fetch_prices,
)

# The same two dev case ids M1 already uses as its smoke (direct + defined-
# term -- the defined-term question is the one that actually forces
# lookup_defined_term, so it is the one that tests tool calling). Fixed here
# and used for EVERY candidate, so results are comparable across models.
PROBE_CASE_IDS: tuple[str, ...] = ("contract_0__q01", "contract_0__q06")

# Reused, never re-run (spec deliverable 1 + "Cheap workhorse model" clause):
# arm-D results for both are already on disk from M4/M4.1.
PARETO_REUSED: tuple[dict, ...] = (
    {
        "model": "anthropic/claude-haiku-4.5",
        "family": "Anthropic",
        "role": "default (already run in M4/M4.1; reused, not re-run)",
        "reuse": True,
    },
    {
        "model": "z-ai/glm-5.3-flash",
        "family": "Zhipu",
        "role": "cheap workhorse (mandatory candidate; reused from M4.1's replication leg)",
        "reuse": True,
    },
)

_FAMILY_BY_MODEL: dict[str, str] = {
    "deepseek/deepseek-v4-flash": "DeepSeek",
    "qwen/qwen3.7-flash": "Alibaba",
    "google/gemini-3.1-flash-lite": "Google",
    "openai/gpt-5.6-luna-pro": "OpenAI",
    "meta-llama/llama-4-maverick": "Meta",
    "xiaomi/mimo-v2.5": "Xiaomi",
    "minimax/minimax-m2.5": "MiniMax",
    "moonshotai/kimi-k2.5": "Moonshot",
    "anthropic/claude-sonnet-5": "Anthropic",
    "x-ai/grok-4.3": "xAI",
}

_ROLE_BY_MODEL: dict[str, str] = {
    "deepseek/deepseek-v4-flash": "open-weight",
    "qwen/qwen3.7-flash": "cheapest credible",
    "google/gemini-3.1-flash-lite": "cross-family flash",
    "openai/gpt-5.6-luna-pro": "strong cheap OpenAI",
    "meta-llama/llama-4-maverick": "open-weight, large ctx",
    "xiaomi/mimo-v2.5": "breadth of cheap tier (stretch)",
    "minimax/minimax-m2.5": "breadth of cheap tier (stretch)",
    "moonshotai/kimi-k2.5": "breadth of cheap tier (stretch)",
    "anthropic/claude-sonnet-5": "stronger same-family ceiling (anchor, last, headroom only)",
    "x-ai/grok-4.3": "second ceiling anchor (last, headroom only)",
}

# Every model this milestone may ever contact, ranked: the 5 core new
# candidates, then the 3 stretch candidates, then the 2 ceiling anchors.
# Haiku/GLM are NOT here -- they are reused, never probed or swept again.
PARETO_CANDIDATES: tuple[dict, ...] = tuple(
    {
        "model": model,
        "family": _FAMILY_BY_MODEL[model],
        "role": _ROLE_BY_MODEL[model],
        "rank": i + 1,
    }
    for i, model in enumerate((*PARETO_NEW_MODELS, *PARETO_STRETCH_MODELS, *PARETO_ANCHOR_MODELS))
)

# Recorded explicitly, with reasons (spec deliverable 1: "Opus is out of
# budget and is listed in the report as 'not run (budget)'").
PARETO_NOT_RUN: tuple[dict, ...] = (
    {
        "model": "anthropic/claude-opus-5",
        "reason": (
            "not run (budget) -- spec deliverable 1 names Opus as out of budget for this "
            "milestone; at $5/M in, $25/M out it would cost more alone than this milestone's "
            "entire remaining envelope."
        ),
    },
)


def _candidate_models() -> set[str]:
    return {c["model"] for c in PARETO_CANDIDATES}


def check_availability(model: str, prices: dict) -> dict:
    """No-spend availability + price check against an already-fetched price table."""
    price = prices.get(model)
    if price is None:
        return {
            "available": False,
            "excluded": True,
            "reason": "not in the OpenRouter model listing",
        }
    return {
        "available": True,
        "prompt_usd_per_token": price["prompt"],
        "completion_usd_per_token": price["completion"],
    }


def _read_jsonl(path) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def probe_candidate(
    client,
    model: str,
    *,
    case_ids: tuple[str, ...] = PROBE_CASE_IDS,
    milestone_tag: str = "m6",
    fake: bool = False,
) -> dict:
    """One arm-D probe of `model` on `case_ids` (default: the 2 fixed dev cases).

    Raises `ValueError` if `model` is not in `PARETO_CANDIDATES` -- no model
    outside the slate is ever contacted (operator instruction, 2026-09-05;
    router endpoints such as `openrouter/auto`/`openrouter/fusion` are never
    in this list and so can never be reached this way).

    Calls `assert_within_cap`/`_assert_within_milestone_absolute` itself
    before running: `run_eval_set` only guards sweeps of more than 10 cases,
    so a 2-case probe would otherwise skip the spend gate entirely.

    `fake=True` (offline `gate_m6` tests only): skips the spend guard and
    drives `run_eval_set` with `fake=True` (an `OfflineChunkRetriever`, no
    index/network), so a scripted `FakeClient` exercises this exact code
    path with no spend and no built index.
    """
    if model not in _candidate_models():
        raise ValueError(
            f"{model!r} is not in dealpoint.eval.pareto_slate.PARETO_CANDIDATES; "
            "refusing to probe a model outside the M6 slate"
        )

    from dealpoint.eval.cases import find_case, slugify_model
    from dealpoint.eval.run import _assert_within_milestone_absolute, run_eval_set
    from dealpoint.eval.spend import assert_within_cap, per_case_usd

    case_rows = [find_case(cid) for cid in case_ids]

    if not fake:
        per_case, _basis = per_case_usd("D", model)
        est_usd = per_case * len(case_rows)
        assert_within_cap(est_usd)
        _assert_within_milestone_absolute(est_usd)

    summary = run_eval_set(
        case_set=f"pareto_probe_{slugify_model(model)}",
        arm="D",
        model=model,
        case_rows=case_rows,
        milestone_tag=milestone_tag,
        extra_context={"purpose": "probe", "probe_model": model},
        client=client,
        fake=fake,
    )

    rows = _read_jsonl(summary["results_path"])
    n_cases = len(rows)
    n_tool_calls = sum(int((r.get("scores") or {}).get("tool_calls") or 0) for r in rows)
    n_with_tool_call = sum(
        1 for r in rows if int((r.get("scores") or {}).get("tool_calls") or 0) > 0
    )
    n_execution_failed = sum(
        1 for r in rows if (r.get("record") or {}).get("status") == "EXECUTION_FAILED"
    )
    n_valid_finding = sum(1 for r in rows if r.get("finding") is not None)
    failure_details = [
        (r.get("record") or {}).get("failure_detail")
        for r in rows
        if (r.get("record") or {}).get("status") == "EXECUTION_FAILED"
    ]
    failure_rate = (n_execution_failed / n_cases) if n_cases else 1.0
    ok = failure_rate == 0 and n_with_tool_call == n_cases

    return {
        "model": model,
        "n_cases": n_cases,
        "n_tool_calls": n_tool_calls,
        "n_with_tool_call": n_with_tool_call,
        "n_execution_failed": n_execution_failed,
        "failure_rate": failure_rate,
        "n_valid_finding": n_valid_finding,
        "failure_details": [d for d in failure_details if d],
        "ok": ok,
        "results_path": summary["results_path"],
        "probe_realized_usd": summary["realized_usd"],
    }


def verify_slate(
    client,
    *,
    models: tuple[str, ...] | None = None,
    prices: dict | None = None,
    case_ids: tuple[str, ...] | None = None,
    milestone_tag: str = "m6",
    fake: bool = False,
) -> dict:
    """Verify tool-calling for `models` (default `PARETO_NEW_MODELS`) on 2 dev
    cases each, arm D.

    Availability + price (no spend) is checked for EVERY ranked candidate
    (core, stretch and anchor); only the models named in `models` are
    actually probed. A model with no pricing entry is recorded
    `available: false, excluded: true` and never contacted. A model that
    fails tool calling (any EXECUTION_FAILED, or any case with no tool call)
    is recorded with its `failure_rate` and `excluded: true` -- the
    surviving list is simply shorter, never back-filled with a substitute.

    `prices=None` -> `fetch_prices()` (live/pinned); `prices=<dict>` -> basis
    `"provided"`, which is what drives this function fully offline with a
    `FakeClient` in `tests/test_pareto_slate.py`.
    """
    models = tuple(models) if models is not None else PARETO_NEW_MODELS
    case_ids = case_ids or PROBE_CASE_IDS

    if prices is None:
        live_prices, price_basis = fetch_prices()
    else:
        live_prices, price_basis = prices, "provided"

    candidates_out: list[dict] = []
    probe_spend = 0.0

    for cand in PARETO_CANDIDATES:
        model = cand["model"]
        avail = check_availability(model, live_prices)
        entry = {**cand, **avail}

        if model not in models:
            entry["probed"] = False
            candidates_out.append(entry)
            continue

        if not avail.get("available"):
            entry["probed"] = False
            entry["ok"] = False
            entry["excluded"] = True
            candidates_out.append(entry)
            continue

        probe = probe_candidate(
            client, model, case_ids=case_ids, milestone_tag=milestone_tag, fake=fake
        )
        entry.update(probe)
        entry["probed"] = True
        entry["excluded"] = not probe["ok"]
        probe_spend += probe.get("probe_realized_usd", 0.0)
        candidates_out.append(entry)

    survivors = [c["model"] for c in candidates_out if c.get("probed") and c.get("ok")]

    payload = {
        "verified_at": datetime.now(UTC).isoformat(),
        "price_basis": price_basis,
        "probe_case_ids": list(case_ids),
        "probe_n_cases": len(case_ids),
        "candidates": candidates_out,
        "survivors": survivors,
        "reused": list(PARETO_REUSED),
        "not_run": list(PARETO_NOT_RUN),
        "probe_spend_usd": round(probe_spend, 6),
    }
    try:
        from dealpoint.eval.cases import git_sha7

        payload["git_sha7"] = git_sha7()
    except Exception:  # noqa: BLE001 - never let git metadata block the payload
        payload["git_sha7"] = None
    return payload


def write_pareto_slate(payload: dict, path=PARETO_SLATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dealpoint.eval.pareto_slate")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="availability + pricing only ($0.00); writes the payload with probed=false everywhere",
    )
    args = parser.parse_args(argv)

    prices, price_basis = fetch_prices()

    if args.dry_run:
        candidates_out = []
        for cand in PARETO_CANDIDATES:
            avail = check_availability(cand["model"], prices)
            candidates_out.append({**cand, **avail, "probed": False, "probe": None})
        payload = {
            "verified_at": datetime.now(UTC).isoformat(),
            "price_basis": price_basis,
            "probe_case_ids": list(PROBE_CASE_IDS),
            "probe_n_cases": 0,
            "candidates": candidates_out,
            "survivors": [],
            "reused": list(PARETO_REUSED),
            "not_run": list(PARETO_NOT_RUN),
            "probe_spend_usd": 0.0,
            "dry_run": True,
        }
    else:
        from dealpoint.llm.client import OpenRouterClient

        client = OpenRouterClient(milestone_tag="m6")
        payload = verify_slate(client)

    write_pareto_slate(payload)
    print(
        json.dumps(
            {"n_candidates": len(payload["candidates"]), "survivors": payload["survivors"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
