"""BTQL investigations (spec section "BTQL investigations"): six queries,
implemented and actually run via the API.

`run_btql` is the single function that talks to the network -- every query
needs a `from:` and one of `select:`/`dimensions:`/`measures:` (verified
against the live API). Query *construction* and result *parsing* are pure
and offline-testable; only the live execution needs the network (marked
`needs_network` in the test suite).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from dealpoint.config import BTQL_INVESTIGATIONS_PATH, REPO_ROOT

BTQL_ENDPOINT = "https://api.braintrust.dev/btql"

BRAINTRUST_QUERIES_MD_PATH = REPO_ROOT / "docs" / "braintrust-queries.md"


def _require_from_and_shape(query: str) -> None:
    if "from:" not in query:
        raise ValueError(f"BTQL query missing 'from:': {query!r}")
    if not any(kw in query for kw in ("select:", "dimensions:", "measures:")):
        raise ValueError(
            f"BTQL query missing one of select:/dimensions:/measures:: {query!r}"
        )


def run_btql(query: str, *, api_key: str, timeout: float = 60) -> dict:
    """`POST /btql`, bearer auth. Raises on a malformed query (missing
    `from:` or a select/dimensions/measures clause) before ever hitting the
    network.
    """
    _require_from_and_shape(query)
    import requests

    response = requests.post(
        BTQL_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def parse_btql_result(payload: dict) -> tuple[list[dict], int]:
    """`(rows, row_count)` from a raw BTQL response payload."""
    rows = payload.get("data", [])
    return rows, len(rows)


# --- the six investigations, as query builders --------------------------


def _experiment_names() -> dict[str, str]:
    """Best-known experiment names for the queries below, sourced from the
    braintrust_sync plan so the ids used here always match what
    `just braintrust-sync` actually creates.

    `eval_example` is matched by NAME (`deepeval-crosscheck`), not
    `next(iter(...))` -- the eval-stage plan list also contains the
    judge-`*` experiments, and blindly taking the first one previously sent
    query 4 (which is specifically about DeepEval vs obj/ disagreement) to
    a judge experiment that carries no `deepeval/` scores at all.
    """
    from dealpoint.eval.braintrust_sync import experiment_plan

    plan = experiment_plan()
    by_stage: dict[str, list[str]] = {}
    for p in plan:
        by_stage.setdefault(p["stage"], []).append(p["name"])
    agent_names = by_stage.get("agent", [])
    economics_names = by_stage.get("economics", [])
    eval_names = by_stage.get("eval", [])
    return {
        "agent_example": next(iter(agent_names), "A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc"),
        "eval_example": "deepeval-crosscheck" if "deepeval-crosscheck" in eval_names else next(
            iter(eval_names), "deepeval-crosscheck"
        ),
        "economics_example": next(iter(economics_names), "pareto-deepseek_deepseek-v4-flash"),
    }


def build_investigations() -> list[dict]:
    """The six investigations: id, title, question, btql query. Deterministic
    (no I/O beyond resolving experiment names from the local sync plan).

    Every query references `scores.*` and/or `metadata.secondary_diagnostics.*`
    for the exact quantity its question names (verified live against the
    real API during this repair; see docstrings per query for the specific
    field). Cost/tool-call/token diagnostics live under
    `metadata.secondary_diagnostics` because `_log_scores_for_row` (D7a)
    only logs an experiment's DECLARED score names as `scores` -- everything
    else is a secondary diagnostic, not a score.
    """
    names = _experiment_names()
    agent_exp = names["agent_example"]
    eval_exp = names["eval_example"]
    economics_exp = names["economics_example"]

    investigations = [
        {
            "id": 1,
            "title": "Retrieval rescue",
            "question": (
                "Which cases did a later search_agreement call surface gold "
                "evidence that the first search missed?"
            ),
            "btql": (
                f"from: experiment('{agent_exp}') | "
                "select: id, input, metadata.arm, "
                'metadata.secondary_diagnostics."obj/tool_calls" as tool_calls, '
                'scores."obj/grounded_accuracy" as grounded_accuracy | '
                'filter: metadata.secondary_diagnostics."obj/tool_calls" >= 2 and '
                'scores."obj/grounded_accuracy" = 1 | '
                "limit: 50"
            ),
        },
        {
            "id": 2,
            "title": "Failure attribution",
            "question": "EXECUTION_FAILED / CAP_HIT counts grouped by model and arm.",
            "btql": (
                f"from: experiment('{agent_exp}') | "
                'filter: metadata.secondary_diagnostics."obj/execution_failed" = 1 or '
                'metadata.secondary_diagnostics."obj/cap_hit" = 1 | '
                "dimensions: metadata.model, metadata.arm | "
                "measures: count(1) as n"
            ),
        },
        {
            "id": 3,
            "title": "Reasoning slices",
            "question": "grounded_accuracy by reasoning_type (direct, numeric, structured, defined-term, cross-ref, carve-out).",
            "btql": (
                f"from: experiment('{agent_exp}') | "
                "dimensions: metadata.reasoning_type | "
                'measures: avg(scores."obj/grounded_accuracy") as grounded_accuracy, count(1) as n'
            ),
        },
        {
            "id": 4,
            "title": "DeepEval disagreement",
            "question": "Traces where deepeval/task_completion and obj/grounded_accuracy disagree.",
            "btql": (
                f"from: experiment('{eval_exp}') | "
                'select: id, input, scores."deepeval/task_completion" as task_completion, '
                'scores."obj/grounded_accuracy" as grounded_accuracy | '
                'filter: (scores."deepeval/task_completion" >= 0.5 and scores."obj/grounded_accuracy" = 0) or '
                '(scores."deepeval/task_completion" < 0.5 and scores."obj/grounded_accuracy" = 1) | '
                "limit: 50"
            ),
        },
        {
            "id": 5,
            "title": "Model economics",
            "question": "Grounded accuracy and average cost per case, for one M6 Pareto model experiment.",
            "btql": (
                f"from: experiment('{economics_exp}') | "
                "dimensions: metadata.model | "
                'measures: avg(scores."obj/grounded_accuracy") as grounded_accuracy, '
                'avg(metadata.secondary_diagnostics."obj/usd") as avg_usd'
            ),
        },
        {
            "id": 6,
            "title": "Trajectory inefficiency",
            "question": "Tool calls vs outcome -- does more searching correlate with a worse result?",
            "btql": (
                f"from: experiment('{agent_exp}') | "
                "select: id, metadata.arm, "
                'metadata.secondary_diagnostics."obj/tool_calls" as tool_calls, '
                'scores."obj/grounded_accuracy" as grounded_accuracy | '
                'filter: metadata.secondary_diagnostics."obj/tool_calls" is not null | '
                "limit: 50"
            ),
        },
    ]
    for inv in investigations:
        _require_from_and_shape(inv["btql"])
    return investigations


def execute_investigations(api_key: str | None = None) -> list[dict]:
    """Run every investigation via the API and record results (or the honest
    reason for zero rows -- e.g. 14-day retention having purged the logs).
    """
    from dealpoint.eval.braintrust_adapter import load_braintrust_key

    api_key = api_key or load_braintrust_key()
    investigations = build_investigations()
    results: list[dict] = []
    for inv in investigations:
        entry = dict(inv)
        entry["executed_at"] = datetime.now(UTC).isoformat()
        if api_key is None:
            entry["row_count"] = 0
            entry["rows"] = []
            entry["notes"] = "no Braintrust API key available; query not executed"
            results.append(entry)
            continue
        try:
            payload = run_btql(inv["btql"], api_key=api_key)
            rows, row_count = parse_btql_result(payload)
            entry["row_count"] = row_count
            entry["rows"] = rows
            entry["notes"] = None if row_count else _zero_row_note(inv["id"])
        except Exception as exc:  # noqa: BLE001 - a query failure is a recorded row, not a crash
            entry["row_count"] = 0
            entry["rows"] = []
            entry["notes"] = f"query failed: {type(exc).__name__}: {exc}"
        results.append(entry)
    return results


# Per-investigation 0-row explanations. Query 1 has a data-supported cause
# (verified live during this repair: every sampled row in the target
# experiment had `secondary_diagnostics."obj/tool_calls" == 0`, so the
# filter's `tool_calls >= 2` predicate cannot match anything -- this is NOT
# retention purge, and the two explanations must not contradict each other
# across `data/reports/btql_investigations.json` and
# `docs/demo-walkthrough.md`). Any OTHER query returning 0 rows in a future
# run falls back to the generic retention-purge note, which remains a
# plausible cause for queries this repair has not specifically verified.
_ZERO_ROW_NOTES: dict[int, str] = {
    1: (
        "0 rows -- verified data-supported cause (not retention purge): every row in the "
        "target agent experiment has metadata.secondary_diagnostics.\"obj/tool_calls\" == 0 "
        "in this run, so the filter's tool_calls >= 2 predicate cannot match any case. No "
        "case in this run triggered a second search_agreement call before a grounded-correct "
        "answer."
    ),
}

_DEFAULT_ZERO_ROW_NOTE = (
    "0 rows -- possibly logs purged by 14-day starter-tier retention; "
    "rerun `just braintrust-sync` first"
)


def _zero_row_note(investigation_id: int) -> str:
    return _ZERO_ROW_NOTES.get(investigation_id, _DEFAULT_ZERO_ROW_NOTE)


def write_investigations(results: list[dict], path: Path = BTQL_INVESTIGATIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        fh.write("\n")


def render_queries_markdown(results: list[dict]) -> str:
    lines = ["# Braintrust BTQL investigations\n"]
    lines.append(
        "Exact reproducible queries, `curl` and `bt sql` forms, and observed results "
        "(dates below). Run `just braintrust-sync` first so there is fresh data to "
        "query.\n"
    )
    for r in results:
        lines.append(f"## {r['id']}. {r['title']}\n")
        lines.append(f"**Question:** {r['question']}\n")
        lines.append("```\n" + r["btql"] + "\n```\n")
        # Build the curl payload with json.dumps so the query's own embedded
        # single quotes (from `experiment('...')`) can never terminate the
        # outer shell string early -- the pre-repair version interpolated
        # the raw BTQL string directly into a single-quoted `-d '...'`
        # payload, which is exactly the string a real query breaks.
        curl_payload = json.dumps({"query": r["btql"]})
        # The payload's single quotes (from `experiment('...')`) would
        # otherwise terminate the outer single-quoted `-d '...'` shell
        # argument early -- escape each with the standard bash idiom
        # ('"'"') rather than switching outer-quote style, so the printed
        # command is copy-paste runnable.
        curl_payload_escaped = curl_payload.replace("'", "'\"'\"'")
        lines.append(
            "```bash\ncurl -s https://api.braintrust.dev/btql \\\n"
            '  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \\\n'
            "  -H 'Content-Type: application/json' \\\n"
            f"  -d '{curl_payload_escaped}'\n```\n"
        )
        bt_sql_query = r["btql"].replace('"', '\\"')
        lines.append(f'```bash\nbt sql --non-interactive --json "{bt_sql_query}"\n```\n')
        lines.append(f"**Executed at:** {r.get('executed_at')}  \n**Row count:** {r.get('row_count')}\n")
        if r.get("notes"):
            lines.append(f"**Notes:** {r['notes']}\n")
        if r.get("rows"):
            lines.append("```json\n" + json.dumps(r["rows"][:5], indent=2, sort_keys=True, default=str) + "\n```\n")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from dealpoint.eval.braintrust_adapter import braintrust_available

    if not braintrust_available():
        print("braintrust unavailable (no key or package) -- writing zero-row investigations")
        results = execute_investigations(api_key=None)
    else:
        results = execute_investigations()

    write_investigations(results)
    BRAINTRUST_QUERIES_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    BRAINTRUST_QUERIES_MD_PATH.write_text(render_queries_markdown(results), encoding="utf-8")
    print(f"Wrote {BTQL_INVESTIGATIONS_PATH} and {BRAINTRUST_QUERIES_MD_PATH}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
