"""M9c D16: the eight Braintrust dashboards, ported word for word, result for result
(`specs/milestones/m9c.md`).

OSS MLflow 3.16 has no chart/view API (M9b's finding, unchanged), so the port is: (1) a read-only
Braintrust snapshot (`snapshot_dashboards`, real BTQL over the logs, never scores/writes), (2) a pure
local evaluator (`chart_values`/`build_dashboard_snapshot`) that recomputes the same numbers from the
stored mirror rows already on disk, and (3) one self-contained HTML render per dashboard, logged as an
artifact on a run tagged `axis=dashboard`.

Three of the thirty-nine charts (`prompt_professional`, `prompt_evidence`, `prompt_reasoning`) cannot be
reproduced by the local evaluator: `mlflow_mirror.prompt_variant_trace_plan`'s own docstring records
that the Playground judge scores for the 67 prompt-variant traces were never mirrored to disk (M9 build
plan section 2). Those three charts' rows are always empty locally; `NOT_LOCALLY_EVALUABLE` names them
so the gap is checked, not silently produced as zeroes.

    uv run python -m dealpoint.eval.mlflow_dashboards snapshot            # write the local evaluator's
                                                                            # numbers (offline fallback)
    uv run python -m dealpoint.eval.mlflow_dashboards snapshot --live     # real BTQL over Braintrust
    uv run python -m dealpoint.eval.mlflow_dashboards runs --live         # log the 8 dashboard runs
"""

from __future__ import annotations

import json
import math
import re
import sys
from html import escape
from pathlib import Path

SNAPSHOT_PATH = Path("data/reports/braintrust_dashboard_values.json")
MLFLOW_EXPERIMENT = "dealpoint-eval"

NOT_LOCALLY_EVALUABLE = ("prompt_professional", "prompt_evidence", "prompt_reasoning")

_AVG_RE = re.compile(r"^avg\(metadata\.(\w+)\)$")
_RATIO_RE = re.compile(r"^sum\(metadata\.(\w+)\)\s*/\s*sum\(metadata\.(\w+)\)$")
_PCTL_RE = re.compile(r"^percentile\(metadata\.(\w+),\s*([\d.]+)\)$")

_TOKEN_RE = re.compile(r"\(|\)|and|or|!=|>=|=|'[^']*'|[\w.]+")


# --- rows: the same mirror rows M9/M9b already mirror, never re-derived ------------------------


def all_rows() -> list[dict]:
    """Every mirror row a chart in `chart_catalogue()` can filter over: agent traces (`mirror_for`),
    retrieval traces and prompt-variant traces, each carrying `arm`/`category` alongside its mirror
    fields exactly as `braintrust_showroom.step_logs` writes them onto the real log's metadata."""
    from dealpoint.eval import mlflow_mirror as mm

    ctx = mm.mirror_context()
    rows: list[dict] = []
    for e in mm.agent_trace_plan():
        row = e.get("row") or {}
        mirror = mm.mirror_for(e["case_id"], e["variant_id"], row, ctx, category=e["category"])
        arm, _, model = e["variant_id"].partition("@")
        rows.append({"arm": row.get("arm") or arm, "model": row.get("model") or model, "category": e["category"], **mirror})
    for r in mm.retrieval_trace_plan():
        rows.append({"category": "retrieval", **r["metadata"]})
    for pv in mm.prompt_variant_trace_plan():
        rows.append({**pv["metadata"]})
    return rows


# --- a tiny, safe BTQL-filter parser (no eval): and/or/=/!=/>=, parens, quoted/numeric literals --


def _tokenize(expr: str) -> list[str]:
    return _TOKEN_RE.findall(expr)


def _literal(tok: str):
    if tok.startswith("'"):
        return tok[1:-1]
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok


class _FilterParser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.i = 0

    def _peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _next(self) -> str:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def parse(self):
        node = self._or()
        if self.i != len(self.tokens):
            raise ValueError(f"trailing tokens in filter: {self.tokens[self.i:]}")
        return node

    def _or(self):
        left = self._and()
        while self._peek() == "or":
            self._next()
            left = ("or", left, self._and())
        return left

    def _and(self):
        left = self._atom()
        while self._peek() == "and":
            self._next()
            left = ("and", left, self._atom())
        return left

    def _atom(self):
        if self._peek() == "(":
            self._next()
            node = self._or()
            if self._next() != ")":
                raise ValueError("unbalanced parens in filter")
            return node
        field = self._next()
        op = self._next()
        value = self._next()
        return ("cmp", field, op, _literal(value))


def _eval_node(node, row: dict) -> bool:
    kind = node[0]
    if kind == "and":
        return _eval_node(node[1], row) and _eval_node(node[2], row)
    if kind == "or":
        return _eval_node(node[1], row) or _eval_node(node[2], row)
    _, field, op, value = node
    name = field.split(".", 1)[1] if field.startswith("metadata.") else field
    actual = row.get(name)
    if op == "=":
        return actual == value
    if op == "!=":
        return actual != value
    if op == ">=":
        return actual is not None and actual >= value
    raise ValueError(f"unsupported filter operator: {op}")


def eval_filter(expr: str, row: dict) -> bool:
    """One `chart_catalogue()` filter string against one mirror row -- the `and`/`or`/`=`/`!=`/`>=`
    subset the catalogue actually uses (spec D16 item 2)."""
    return _eval_node(_FilterParser(_tokenize(expr)).parse(), row)


# --- measures: avg, sum/sum ratio, percentile ---------------------------------------------------


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def eval_measure(btql: str, rows: list[dict]) -> float | None:
    match = _AVG_RE.match(btql)
    if match:
        field = match.group(1)
        vals = [float(r[field]) for r in rows if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else None
    match = _RATIO_RE.match(btql)
    if match:
        num_field, den_field = match.groups()
        num = sum(float(r[num_field]) for r in rows if r.get(num_field) is not None)
        den = sum(float(r[den_field]) for r in rows if r.get(den_field) is not None)
        return num / den if den else None
    match = _PCTL_RE.match(btql)
    if match:
        field, p = match.group(1), float(match.group(2))
        vals = [float(r[field]) for r in rows if r.get(field) is not None]
        return _percentile(vals, p)
    raise ValueError(f"unsupported measure: {btql}")


# --- one chart -> its rows -----------------------------------------------------------------------


def _unit_for(chart: dict) -> str:
    return chart.get("unit") or ("cost" if "usd" in str(chart["measure"]) else "percent")


def chart_values(chart: dict, rows: list[dict]) -> dict[str, float | None]:
    """`{group_label: value}` (or `{measure_name: value}` for a multi-measure, ungrouped chart),
    exactly `chart_catalogue()`'s own shape -- never re-derives a chart, only evaluates one."""
    filtered = [r for r in rows if all(eval_filter(f, r) for f in chart.get("filters", []))]
    measure = chart["measure"]
    if isinstance(measure, list):
        return {m["name"]: eval_measure(m["btql"], filtered) for m in measure}
    group_by = chart.get("group_by") or []
    if not group_by:
        return {"value": eval_measure(measure, filtered)}
    field = group_by[0].split(".", 1)[1] if group_by[0].startswith("metadata.") else group_by[0]
    groups: dict[str, list[dict]] = {}
    for r in filtered:
        g = r.get(field)
        if g is None:
            continue
        groups.setdefault(str(g), []).append(r)
    return {g: eval_measure(measure, grs) for g, grs in groups.items()}


def _ordered_rows(values: dict[str, float | None]) -> list[dict]:
    """Braintrust's own toplist ordering (`sortByOptions: value desc`, see `braintrust_cockpit.
    _chart_rest_definition`): highest value first, `None` last, ties broken by label."""
    ordered = sorted(values.items(), key=lambda kv: (kv[1] is None, -(kv[1] if kv[1] is not None else 0.0), kv[0]))
    return [{"label": label, "value": value} for label, value in ordered]


# --- every chart, every dashboard -----------------------------------------------------------------


def all_chart_values(rows: list[dict] | None = None) -> dict[str, dict[str, float | None]]:
    from dealpoint.eval.braintrust_cockpit import chart_catalogue

    rows = rows if rows is not None else all_rows()
    return {key: chart_values(chart, rows) for key, chart in chart_catalogue().items()}


def build_dashboard_snapshot(rows: list[dict] | None = None) -> list[dict]:
    """One entry per (dashboard, chart), in `DASHBOARDS` order -- the local evaluator's own value for
    every row of every chart. `locally_evaluable=False` on the three prompt-judge charts (see module
    docstring); their `rows` list is always empty, never a silently-wrong zero."""
    from dealpoint.eval.braintrust_cockpit import DASHBOARDS, chart_catalogue

    catalogue = chart_catalogue()
    rows_ = rows if rows is not None else all_rows()
    out: list[dict] = []
    for name, _verdict, chart_ids in DASHBOARDS:
        for cid in chart_ids:
            chart = catalogue[cid]
            values = chart_values(chart, rows_)
            locally_evaluable = cid not in NOT_LOCALLY_EVALUABLE
            out.append({
                "dashboard": name, "chart_key": cid, "title": chart["title"], "unit": _unit_for(chart),
                "locally_evaluable": locally_evaluable,
                "rows": _ordered_rows(values) if locally_evaluable else [],
            })
    return out


def write_local_snapshot(path: Path = SNAPSHOT_PATH) -> list[dict]:
    """Offline fallback for `braintrust-dashboard-snapshot`: the local evaluator's own numbers,
    written in the same shape a live BTQL run would produce. This environment has no Braintrust
    network access to run the real, read-only REST snapshot (spec D16 item 1); until `just
    braintrust-dashboard-snapshot --live` is run against the real project, this file is
    self-consistent (the local evaluator checked against itself), not yet checked against
    Braintrust -- a named gap, not a silent substitution."""
    snapshot = build_dashboard_snapshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot


def load_snapshot(path: Path = SNAPSHOT_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_matches(a: list[dict], b: list[dict], *, tol: float = 1e-9) -> list[str]:
    """Every mismatch between two snapshots (empty list = identical within `tol`); compares only
    `locally_evaluable` charts, since the other three have no rows to compare by construction."""
    mismatches: list[str] = []
    by_key_a = {(e["dashboard"], e["chart_key"]): e for e in a}
    by_key_b = {(e["dashboard"], e["chart_key"]): e for e in b}
    for key in sorted(set(by_key_a) | set(by_key_b)):
        ea, eb = by_key_a.get(key), by_key_b.get(key)
        if ea is None or eb is None:
            mismatches.append(f"{key}: present in only one snapshot")
            continue
        if not ea.get("locally_evaluable", True):
            continue
        ra, rb = ea.get("rows", []), eb.get("rows", [])
        if [r["label"] for r in ra] != [r["label"] for r in rb]:
            mismatches.append(f"{key}: row order/labels differ: {[r['label'] for r in ra]} vs {[r['label'] for r in rb]}")
            continue
        for row_a, row_b in zip(ra, rb, strict=True):
            va, vb = row_a["value"], row_b["value"]
            if va is None and vb is None:
                continue
            if va is None or vb is None or abs(va - vb) > tol:
                mismatches.append(f"{key}[{row_a['label']}]: {va} != {vb}")
    return mismatches


# --- REST BTQL snapshot (live, read-only) ---------------------------------------------------------


def snapshot_dashboards(rest_client) -> list[dict]:
    """The real thing (spec D16 item 1): run every `chart_catalogue()` chart's measure/group/filters
    as BTQL against the Braintrust logs and return the same shape `build_dashboard_snapshot` does, so
    the two can be diffed by `snapshot_matches`. Read-only: no score, no view, no dashboard is written.
    Respects the 20-queries-per-minute limit with backoff, matching `braintrust_cockpit`'s own REST
    call pattern."""
    import time

    from dealpoint.eval.braintrust_cockpit import (
        DASHBOARDS,
        _chart_rest_definition,
        _resolve_project_id,
        chart_catalogue,
    )

    project_id = _resolve_project_id(rest_client)
    catalogue = chart_catalogue()
    out: list[dict] = []
    calls = 0
    window_start = time.monotonic()
    for name, _verdict, chart_ids in DASHBOARDS:
        for cid in chart_ids:
            chart = catalogue[cid]
            if calls and calls % 20 == 0:
                elapsed = time.monotonic() - window_start
                if elapsed < 60:
                    time.sleep(60 - elapsed)
                window_start = time.monotonic()
            definition = _chart_rest_definition(chart)
            result = rest_client.post(f"project/{project_id}/btql", {"query": definition})
            calls += 1
            rows = [{"label": str(r.get("group") or r.get("label")), "value": r.get("value")} for r in result.get("data", [])]
            out.append({"dashboard": name, "chart_key": cid, "title": chart["title"], "unit": _unit_for(chart),
                       "locally_evaluable": cid not in NOT_LOCALLY_EVALUABLE, "rows": rows})
    return out


# --- HTML render, one self-contained page per dashboard --------------------------------------------


def _format_value(value: float | None, unit: str) -> str:
    if value is None:
        return "(no data)"
    if unit == "percent":
        return f"{value:.1%}"
    if unit == "cost":
        return f"${value:.4f}"
    if unit == "duration":
        return f"{value:.2f}s"
    return f"{value:.2f}" if value != int(value) else str(int(value))


def render_dashboard_html(name: str, verdict: str, chart_entries: list[dict]) -> str:
    """One self-contained HTML page (no external assets): the dashboard's name, its verdict verbatim,
    and each chart as a ranked horizontal bar list, same title, same row labels, same order (spec D16
    item 3)."""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{escape(name)}</title>",
        ("<style>body{font-family:sans-serif;max-width:960px;margin:2rem auto}"
         "li{margin:.25rem 0}.bar{display:inline-block;height:.8em;background:#4a7;vertical-align:middle;margin-right:.5em}</style>"),
        "</head><body>",
        f"<h1>{escape(name)}</h1>",
        f"<p class='verdict'>{escape(verdict)}</p>",
    ]
    for entry in chart_entries:
        rows = entry["rows"]
        max_val = max((r["value"] for r in rows if r["value"] is not None), default=0.0) or 1.0
        parts.append(f"<h2>{escape(entry['title'])}</h2><ol>")
        for row in rows:
            width = 0 if row["value"] is None else round(100 * abs(row["value"]) / max_val)
            bar = f"<span class='bar' style='width:{width}px'></span>"
            parts.append(f"<li>{bar}{escape(str(row['label']))}: {escape(_format_value(row['value'], entry['unit']))}</li>")
        parts.append("</ol>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def dashboard_run_plan(rows: list[dict] | None = None) -> list[dict]:
    """One entry per of the eight dashboards: its run key (`dashboard/<name>`), its metrics
    (`<chart_key>.<row label>` -> value, spec D16 item 3), and its rendered HTML."""
    from dealpoint.eval.braintrust_cockpit import DASHBOARDS, chart_catalogue

    catalogue = chart_catalogue()
    rows_ = rows if rows is not None else all_rows()
    plans = []
    for name, verdict, chart_ids in DASHBOARDS:
        entries, metrics = [], {}
        for cid in chart_ids:
            chart = catalogue[cid]
            ordered = _ordered_rows(chart_values(chart, rows_))
            entries.append({"chart_key": cid, "title": chart["title"], "unit": _unit_for(chart), "rows": ordered})
            for row in ordered:
                if row["value"] is not None:
                    metrics[f"{cid}.{row['label']}"] = row["value"]
        plans.append({
            "name": name, "verdict": verdict, "run_key": f"dashboard/{_slug(name)}",
            "charts": entries, "metrics": metrics, "html": render_dashboard_html(name, verdict, entries),
        })
    return plans


# --- CLI -----------------------------------------------------------------------------------------


def _client(tracking_uri: str | None = None):
    import os

    import mlflow
    from mlflow import MlflowClient

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def _ensure_experiment(client) -> str:
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    return exp.experiment_id if exp else client.create_experiment(MLFLOW_EXPERIMENT)


def step_runs(live: bool) -> list[dict]:
    plans = dashboard_run_plan()
    print(f"dashboards: {len(plans)} runs planned ({', '.join(p['run_key'] for p in plans)})")
    if not live:
        return plans
    import mlflow

    client = _client()
    exp_id = _ensure_experiment(client)
    for p in plans:
        existing = client.search_runs([exp_id], filter_string=f"tags.`dealpoint.key` = '{p['run_key']}'")
        if existing:
            continue
        with mlflow.start_run(experiment_id=exp_id, run_name=p["run_key"]) as run:
            mlflow.set_tags({"dealpoint.key": p["run_key"], "axis": "dashboard"})
            mlflow.log_metrics(p["metrics"])
            mlflow.log_text(p["html"], "dashboard.html")
            mlflow.set_tag("mlflow.note.content", p["verdict"])
            print(f"  logged {p['run_key']} -> run {run.info.run_id}")
    return plans


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    live = "--live" in argv and "--dry-run" not in argv
    cmd = argv[0] if argv and not argv[0].startswith("--") else "snapshot"

    if cmd == "snapshot":
        if live:
            from dealpoint.eval.braintrust_adapter import load_braintrust_key
            from dealpoint.eval.braintrust_cockpit import RestClient

            api_key = load_braintrust_key()
            if not api_key:
                print("braintrust-dashboard-snapshot --live requested but no API key resolved -- aborting", file=sys.stderr)
                return 1
            snapshot = snapshot_dashboards(RestClient(api_key))
        else:
            snapshot = build_dashboard_snapshot()
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{'LIVE' if live else 'DRY RUN (local evaluator)'}: {len(snapshot)} chart rows -> {SNAPSHOT_PATH}")
        return 0
    if cmd == "runs":
        step_runs(live)
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
