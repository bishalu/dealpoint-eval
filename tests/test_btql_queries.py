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


@pytest.mark.needs_network
def test_execute_investigations_live():
    from dealpoint.eval.btql import execute_investigations

    results = execute_investigations()
    assert len(results) == 6
    for r in results:
        assert "row_count" in r
