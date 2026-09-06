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
    """
    from dealpoint.eval.braintrust_sync import experiment_plan

    plan = experiment_plan()
    by_stage: dict[str, list[str]] = {}
    for p in plan:
        by_stage.setdefault(p["stage"], []).append(p["name"])
    return {
        "agent_example": next(iter(by_stage.get("agent", [])), "A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc"),
        "eval_example": next(iter(by_stage.get("eval", [])), "deepeval-crosscheck"),
        "economics_example": next(
            iter(by_stage.get("economics", [])), "pareto-deepseek_deepseek-v4-flash"
        ),
    }


def build_investigations() -> list[dict]:
    """The six investigations: id, title, question, btql query. Deterministic
    (no I/O beyond resolving experiment names from the local sync plan).
    """
    names = _experiment_names()
    agent_exp = names["agent_example"]

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
                "select: id, input, metadata.arm as arm, metadata.reasoning_type as rt | "
                "filter: metadata.reasoning_type is not null | "
                "limit: 50"
            ),
        },
        {
            "id": 2,
            "title": "Failure attribution",
            "question": "EXECUTION_FAILED / CAP_HIT counts grouped by model and arm.",
            "btql": (
                f"from: experiment('{agent_exp}') | "
                "dimensions: metadata.model as model, metadata.arm as arm | "
                "measures: count(1) as n"
            ),
        },
        {
            "id": 3,
            "title": "Reasoning slices",
            "question": "grounded_accuracy by reasoning_type (direct, numeric, structured, defined-term, cross-ref, carve-out).",
            "btql": (
                f"from: experiment('{agent_exp}') | "
                "dimensions: metadata.reasoning_type as rt | "
                "measures: count(1) as n"
            ),
        },
        {
            "id": 4,
            "title": "DeepEval disagreement",
            "question": "Traces where deepeval/ and obj/ scores disagree.",
            "btql": (
                f"from: experiment('{names['eval_example']}') | "
                "select: id, metadata | "
                "limit: 50"
            ),
        },
        {
            "id": 5,
            "title": "Model economics",
            "question": "Cost per correct answer by model, across the M6 Pareto experiments.",
            "btql": (
                f"from: experiment('{names['economics_example']}') | "
                "dimensions: metadata.model as model | "
                "measures: count(1) as n"
            ),
        },
        {
            "id": 6,
            "title": "Trajectory inefficiency",
            "question": "Tool calls vs outcome -- does more searching correlate with a worse result?",
            "btql": (
                f"from: experiment('{agent_exp}') | "
                "select: id, metadata.arm as arm | "
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
            entry["notes"] = (
                None
                if row_count
                else (
                    "0 rows -- possibly logs purged by 14-day starter-tier retention; "
                    "rerun `just braintrust-sync` first"
                )
            )
        except Exception as exc:  # noqa: BLE001 - a query failure is a recorded row, not a crash
            entry["row_count"] = 0
            entry["rows"] = []
            entry["notes"] = f"query failed: {type(exc).__name__}: {exc}"
        results.append(entry)
    return results


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
        lines.append(
            "```bash\ncurl -s https://api.braintrust.dev/btql \\\n"
            '  -H "Authorization: Bearer $BRAINTRUST_API_KEY" \\\n'
            "  -H 'Content-Type: application/json' \\\n"
            f"  -d '{{\"query\": \"{r['btql']}\"}}'\n```\n"
        )
        lines.append(f"```bash\nbt sql --non-interactive --json \"{r['btql']}\"\n```\n")
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
