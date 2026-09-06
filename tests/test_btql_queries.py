"""M7a gate: the six BTQL investigations -- query construction and result
parsing, fully offline. Live execution is `needs_network` (spec section
"BTQL investigations").
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gate_m7


def test_all_six_queries_present_with_required_clauses():
    from dealpoint.eval.btql import build_investigations

    investigations = build_investigations()
    assert len(investigations) == 6
    ids = {inv["id"] for inv in investigations}
    assert ids == {1, 2, 3, 4, 5, 6}
    for inv in investigations:
        q = inv["btql"]
        assert "from:" in q
        assert any(kw in q for kw in ("select:", "dimensions:", "measures:"))
        assert inv["title"]
        assert inv["question"]


def test_run_btql_rejects_query_without_from():
    from dealpoint.eval.btql import run_btql

    with pytest.raises(ValueError):
        run_btql("select: id", api_key="fake")


def test_run_btql_rejects_query_without_select_dimensions_or_measures():
    from dealpoint.eval.btql import run_btql

    with pytest.raises(ValueError):
        run_btql("from: experiment('x')", api_key="fake")


def test_parse_btql_result_against_recorded_fixture():
    from dealpoint.eval.btql import parse_btql_result

    fixture = {
        "data": [{"arm": "A", "n": 32}, {"arm": "D", "n": 32}],
        "schema": {"type": "array"},
    }
    rows, count = parse_btql_result(fixture)
    assert count == 2
    assert rows[0]["arm"] == "A"


def test_parse_btql_result_zero_rows():
    from dealpoint.eval.btql import parse_btql_result

    rows, count = parse_btql_result({"data": []})
    assert rows == []
    assert count == 0


def test_each_query_references_the_quantity_its_question_names():
    """A placeholder query (e.g. `count(1)` with no reference to the field
    its question asks about) must fail this test -- see the pre-repair
    D8 audit, where none of the six queries referenced `scores.*` at all.
    """
    from dealpoint.eval.btql import build_investigations

    investigations = build_investigations()
    required_substring_by_id = {
        1: "obj/grounded_accuracy",
        2: "obj/execution_failed",
        3: "obj/grounded_accuracy",
        4: "deepeval/task_completion",
        5: "obj/grounded_accuracy",
        6: "obj/tool_calls",
    }
    for inv in investigations:
        required = required_substring_by_id[inv["id"]]
        assert required in inv["btql"], f"query {inv['id']} does not reference {required!r}"


def test_query_4_targets_deepeval_crosscheck_not_a_judge_experiment():
    from dealpoint.eval.btql import build_investigations

    investigations = build_investigations()
    query_4 = next(inv for inv in investigations if inv["id"] == 4)
    assert "deepeval-crosscheck" in query_4["btql"]


def test_curl_rendering_is_shell_syntax_valid():
    """The printed curl command must actually be runnable bash -- the
    pre-repair version embedded the BTQL string's own single quotes
    (from `experiment('...')`) inside an outer single-quoted `-d '...'`
    payload, which a shell would terminate early.
    """
    import subprocess

    from dealpoint.eval.btql import render_queries_markdown

    fake_results = [
        {
            "id": 1,
            "title": "x",
            "question": "y",
            "btql": "from: experiment('a-b-c') | select: id, scores.\"obj/x\" as x | limit: 5",
            "executed_at": "2026-01-01T00:00:00+00:00",
            "row_count": 0,
            "notes": None,
            "rows": [],
        }
    ]
    markdown = render_queries_markdown(fake_results)
    lines = markdown.splitlines()
    curl_line = next(line for line in lines if line.startswith("curl -s"))
    result = subprocess.run(["bash", "-n", "-c", curl_line], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"curl line is not valid bash syntax: {result.stderr}"


@pytest.mark.needs_network
def test_execute_investigations_live():
    from dealpoint.eval.btql import execute_investigations

    results = execute_investigations()
    assert len(results) == 6
    for r in results:
        assert "row_count" in r
