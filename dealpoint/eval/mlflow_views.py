"""The eight Braintrust dashboards as native MLflow chart views (saved experiment views).

Why this exists. `mlflow_dashboards.py` (M9c D16) ports each dashboard as a rendered HTML artifact and
as metrics on one run per dashboard. That is a picture of the charts, not charts. MLflow's open-source
server has no chart or dashboard API, but its experiment page persists *saved views* server-side: an
experiment tag `mlflow.sharedViewState.<id>` holding the runs-table search facets and the chart cards of
the compare-runs Chart view (verified in the 3.16 UI bundle: tag prefix, `viewStateShareKey` URL param,
the saved-view envelope `{name, createdAt, updatedAt, state}`, the BAR card fields). Writing those tags
from code gives every Braintrust chart a real, interactive MLflow bar chart with the same title, the
same row labels, the same values, opened by URL or from the experiment's views dropdown.

How Braintrust and MLflow disagree about data, and the mapping that follows from it:

- A Braintrust dashboard chart is a query over one log table: a measure, a `group_by` (system, model,
  system@model, retriever, prompt variant) or a list of named measures (the judges), a filter, sorted
  as a toplist. Each chart chooses its own grouping, so one dashboard mixes row sets freely.
- An MLflow bar chart is one metric across the runs currently in the runs table. Every chart in a view
  shares the same run set, and a run without the metric shows as an empty row.
- So: a Braintrust group becomes an MLflow **run** (run name = the row label), a chart's measure
  becomes a **metric** on those runs (metric key = the chart key, chart title = the card's display
  name), a Braintrust dashboard becomes **one saved view per row set it uses** (its filter selects the
  runs of that row set; its cards are that dashboard's charts over that row set, in dashboard order),
  and the dashboard's verdict plus the list of its views goes in the experiment description. Runs are
  shared by row set, so a chart that appears on two dashboards (e.g. `sys_safe` on the overview and
  on "Which system?") is one metric on the same runs, and both views show it.
- Toplist order: Braintrust sorts every chart by its own value; an MLflow view has one sort order for
  its runs table, and bars follow it. Each view sorts by its first chart, descending. Every chart
  still shows every value; only the bar order of the later charts can differ from Braintrust's.
- Units: Braintrust formats a percent measure as a percent; MLflow shows raw numbers. Percent charts are
  logged as 0 to 100 and the card title says "(%)"; dollars, seconds and counts are logged as they are.

Values come from the committed Braintrust snapshot (`data/reports/braintrust_dashboard_values.json`,
M9c D16 item 1), which the local evaluator reproduces to 1e-9, so the port is result for result.
Zero model calls, zero Braintrust writes. Dry run by default; `--live` writes to the tracking server;
idempotent (runs keyed by tag `dealpoint.key`, views overwritten in place).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

from dealpoint.eval.mlflow_dashboards import SNAPSHOT_PATH, load_snapshot

VIEWS_EXPERIMENT = "dealpoint-dashboards"
MAIN_EXPERIMENT = "dealpoint-eval"
SHARED_VIEW_TAG_PREFIX = "mlflow.sharedViewState."
MANIFEST_PATH = Path("data/reports/mlflow_views_manifest.json")
PUBLIC_URL_ENV = "MLFLOW_PUBLIC_URL"
DEFAULT_PUBLIC_URL = "https://balpad.exe.xyz:5000"

# How the row set of a chart is named in a view title. Braintrust's group_by fields carry a display
# name (`GROUP_DISPLAY`); the judges' charts have no group_by (they are lists of named measures), so
# their row set is named by what the labels are.
_DIMENSIONS = frozenset({"evidence", "reasoning", "professional", "trajectory"})
METRIC_KEY_RE = re.compile(r"^[/\w.\- :]*$")
# The overview dashboard keeps only the views over these row sets (see `view_plan`).
OVERVIEW_KEEP = frozenset({"system@model"})


def _rowset_name(chart: dict, labels: list[str]) -> str:
    from dealpoint.eval.braintrust_cockpit import GROUP_DISPLAY

    group_by = chart.get("group_by") or []
    if group_by:
        return GROUP_DISPLAY.get(group_by[0]) or group_by[0].split(".")[-1]
    if set(labels) <= _DIMENSIONS:
        return "dimension"
    return "judge"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _scale(unit: str) -> float:
    return 100.0 if unit == "percent" else 1.0


def _card_title(title: str, unit: str) -> str:
    return f"{title} (%)" if unit == "percent" else title


def view_plan(snapshot: list[dict] | None = None) -> dict:
    """The whole port as data: `rowsets` (row-set id -> labels, runs and their metrics), `views` (one
    per (dashboard, row set) in dashboard order: title, slug, filter, chart cards) and `dashboards`
    (name, verdict, its views). Pure; the only inputs are the snapshot and the cockpit's `DASHBOARDS`
    and `chart_catalogue()`."""
    from dealpoint.eval.braintrust_cockpit import DASHBOARDS, chart_catalogue

    catalogue = chart_catalogue()
    snap = snapshot if snapshot is not None else load_snapshot()
    snap_by = {(e["dashboard"], e["chart_key"]): e for e in snap}

    rowsets: dict[str, dict] = {}
    dashboards: list[dict] = []
    views: list[dict] = []
    for d_index, (name, verdict, chart_keys) in enumerate(DASHBOARDS, start=1):
        dash_views: list[dict] = []
        for cid in chart_keys:
            entry = snap_by[(name, cid)]
            chart = catalogue[cid]
            labels = [str(r["label"]) for r in entry["rows"]]
            unit = entry["unit"]
            rs_name = _rowset_name(chart, labels)
            rs_id = f"{rs_name}/{'|'.join(sorted(labels))}"
            rs = rowsets.setdefault(rs_id, {
                "id": rs_id, "name": rs_name, "slug": None, "labels": [], "metrics": {}, "charts": {},
            })
            for lab in labels:
                if lab not in rs["labels"]:
                    rs["labels"].append(lab)
            scale = _scale(unit)
            values = {str(r["label"]): (None if r["value"] is None else r["value"] * scale) for r in entry["rows"]}
            prior = rs["metrics"].get(cid)
            if prior is not None and any(abs((prior.get(k) or 0) - (v or 0)) > 1e-9 for k, v in values.items()):
                raise ValueError(f"chart {cid} has different values on two dashboards for row set {rs_id}")
            rs["metrics"][cid] = values
            rs["charts"][cid] = {"title": _card_title(entry["title"], unit), "unit": unit}

            view = next((v for v in dash_views if v["rowset"] == rs_id), None)
            if view is None:
                view = {
                    "dashboard": name, "rowset": rs_id, "rowset_name": rs_name, "chart_keys": [],
                    "title": None, "slug": None, "order": (d_index, len(dash_views) + 1),
                }
                dash_views.append(view)
            if cid not in view["chart_keys"]:
                view["chart_keys"].append(cid)
        views.extend(dash_views)
        dashboards.append({"name": name, "verdict": verdict, "views": [], "elsewhere": []})

    # The overview is Braintrust's "one screen": the headline chart of each question dashboard over
    # seven different row sets. MLflow cannot put seven row sets on one screen. An overview view whose
    # charts all appear in a question's view on the same runs is dropped and the note points there;
    # the accuracy point (every system@model, same 18 cases) and any overview-only chart keep a view.
    overview_name = DASHBOARDS[0][0]
    others = [v for v in views if v["dashboard"] != overview_name]
    for v in [v for v in views if v["dashboard"] == overview_name]:
        if v["rowset_name"] in OVERVIEW_KEEP:
            continue
        home = next((o for o in others if o["rowset"] == v["rowset"] and set(v["chart_keys"]) <= set(o["chart_keys"])), None)
        if home is None:
            continue
        dashboards[0]["elsewhere"].append({"rowset": v["rowset"], "chart_keys": list(v["chart_keys"]), "home": home})
        views.remove(v)

    counter: dict[str, int] = {}
    for v in views:
        d_index = v["order"][0]
        i = counter[v["dashboard"]] = counter.get(v["dashboard"], 0) + 1
        n = len(rowsets[v["rowset"]]["labels"])
        v["title"] = f"{d_index}. {v['dashboard']} · by {v['rowset_name']} ({n} rows)"
        v["slug"] = _slug(f"{d_index}-{v['dashboard']}-by-{v['rowset_name']}-{i}")
    for d in dashboards:
        d["views"] = [v["slug"] for v in views if v["dashboard"] == d["name"]]
        for e in d["elsewhere"]:
            e["view"] = e.pop("home")["slug"]

    # Row-set slugs (unique, short) and the run plan.
    seen: dict[str, int] = {}
    for rs in rowsets.values():
        base = _slug(rs["name"])
        seen[base] = seen.get(base, 0) + 1
        rs["slug"] = base if seen[base] == 1 else f"{base}-{seen[base]}"
    for rs in rowsets.values():
        rs["runs"] = [{
            "run_name": lab,
            "key": f"dashboard-row/{rs['slug']}/{lab}",
            "tags": {"dealpoint.key": f"dashboard-row/{rs['slug']}/{lab}", "dealpoint.rowset": rs["slug"],
                     "dealpoint.row": lab, "axis": "dashboard-row"},
            "metrics": {cid: vals[lab] for cid, vals in rs["metrics"].items() if vals.get(lab) is not None},
        } for lab in rs["labels"]]
    for v in views:
        v["filter"] = f"tags.`dealpoint.rowset` = '{rowsets[v['rowset']]['slug']}'"
    return {"rowsets": rowsets, "views": views, "dashboards": dashboards}


def _card(chart_key: str, title: str, section_id: str, n_runs: int) -> dict:
    """A BAR card as the 3.16 UI serializes it (`RunsChartsBarCardConfig`)."""
    return {
        "uuid": str(uuid.uuid4()), "type": "BAR", "metricKey": chart_key, "selectedMetricKeys": None,
        "datasetName": None, "dataAccessKey": None, "runsCountToCompare": max(10, n_runs),
        "metricSectionId": section_id, "deleted": False, "isGenerated": False, "displayName": title,
    }


def view_state(view: dict, rowset: dict) -> dict:
    """The flat state the UI reads back: search facets (filter + sort) and the UI state (cards and
    sections). Keys outside the two allow-lists are ignored by the reader; `runsPinned`/`runsHidden`
    are stripped on read, which is why the filter, not hiding, selects the runs."""
    section_id = str(uuid.uuid4())
    n = len(rowset["labels"])
    cards = [_card(cid, rowset["charts"][cid]["title"], section_id, n) for cid in view["chart_keys"]]
    first = view["chart_keys"][0]
    return {
        "searchFilter": view["filter"],
        "orderByKey": f"metrics.`{first}`",
        "orderByAsc": False,
        "startTime": "ALL",
        "lifecycleFilter": "Active",
        "modelVersionFilter": "All Runs",
        "datasetsFilter": [],
        "compareRunCharts": cards,
        "compareRunSections": [{
            "uuid": section_id, "name": view["dashboard"], "display": True, "isReordered": False,
            "deleted": False, "isGenerated": False,
        }],
        "runsHiddenMode": "SHOW_ALL",
        "hideEmptyCharts": True,
        "viewMaximized": False,
        "runListHidden": False,
        "isAccordionReordered": False,
        "useGroupedValuesInCharts": True,
        "groupBy": None,
        "groupsExpanded": {},
        "chartsSearchFilter": "",
    }


def saved_view_tag(view: dict, rowset: dict, *, now_ms: int | None = None) -> tuple[str, str]:
    """(tag key, tag value): the saved-view envelope the UI lists in its views dropdown and loads
    through `?viewStateShareKey=<slug>`."""
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    envelope = {"name": view["title"], "createdAt": now, "updatedAt": now, "state": json.dumps(view_state(view, rowset))}
    return SHARED_VIEW_TAG_PREFIX + view["slug"], json.dumps(envelope)


def view_url(base_url: str, experiment_id: str, slug: str) -> str:
    return f"{base_url.rstrip('/')}/#/experiments/{experiment_id}?viewStateShareKey={slug}&compareRunsMode=CHART"


def experiment_note(plan: dict, *, base_url: str, experiment_id: str) -> str:
    """The Overview tab: what this experiment is, the mapping, one section per Braintrust dashboard
    with its verdict verbatim and links to its views."""
    lines = [
        "# The eight Braintrust dashboards as MLflow chart views",
        "",
        ("Every chart of every DealPoint dashboard in Braintrust, as a native MLflow bar chart with the same "
        "title, the same rows and the same values (Braintrust snapshot `data/reports/braintrust_dashboard_values.json`; "
        "percent charts are logged 0 to 100). One run per row (the runs table is the set of bars), one metric per chart, "
        "one saved view per dashboard section: pick a view from the views dropdown or open its link, and the Chart "
        "tab shows that section's charts in dashboard order. Bars follow the table's sort (the first chart, descending); "
        "the values are exact, the order of the later charts may differ from Braintrust's toplists. "
        "Built by `just mlflow-views --live`; zero model calls."),
        "",
        "| Braintrust dashboard | MLflow views |",
        "|---|---|",
    ]
    views_by_slug = {v["slug"]: v for v in plan["views"]}
    for d in plan["dashboards"]:
        links = ", ".join(f"[{views_by_slug[s]['title']}]({view_url(base_url, experiment_id, s)})" for s in d["views"])
        lines.append(f"| {d['name']} | {links} |")
    lines.append("")
    for d in plan["dashboards"]:
        lines.append(f"## {d['name']}")
        lines.append("")
        lines.append(d["verdict"])
        lines.append("")
        for s in d["views"]:
            v = views_by_slug[s]
            charts = ", ".join(v["chart_keys"])
            lines.append(f"- [{v['title']}]({view_url(base_url, experiment_id, s)}): {charts}")
        for e in d.get("elsewhere", []):
            if e.get("view"):
                home = views_by_slug[e["view"]]
                lines.append(f"- {', '.join(e['chart_keys'])}: the same chart on the same runs in "
                             f"[{home['title']}]({view_url(base_url, experiment_id, home['slug'])})")
        lines.append("")
    return "\n".join(lines)


# --- live -----------------------------------------------------------------------------------------


def _client(tracking_uri: str | None = None):
    import mlflow
    from mlflow import MlflowClient

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def _ensure_experiment(client, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    return exp.experiment_id if exp else client.create_experiment(name)


def apply(plan: dict, *, live: bool, client=None, base_url: str | None = None, refresh: bool = False) -> dict:
    """Create the experiment, one run per row (skipped when its `dealpoint.key` exists unless
    `refresh`), the saved-view tags (always overwritten: same slug, same URL) and the note. Returns
    the manifest."""
    base = base_url or os.environ.get(PUBLIC_URL_ENV, DEFAULT_PUBLIC_URL)
    n_runs = sum(len(rs["runs"]) for rs in plan["rowsets"].values())
    n_cards = sum(len(v["chart_keys"]) for v in plan["views"])
    print(f"views: {len(plan['views'])} saved views, {n_cards} chart cards, {len(plan['rowsets'])} row sets, {n_runs} runs")
    for v in plan["views"]:
        print(f"  - {v['title']}  [{v['slug']}]  charts={len(v['chart_keys'])}  filter={v['filter']}")
    manifest = {"experiment": VIEWS_EXPERIMENT, "experiment_id": None, "views": [], "runs": {}, "base_url": base}
    if not live:
        return manifest

    import mlflow

    client = client or _client()
    exp_id = _ensure_experiment(client, VIEWS_EXPERIMENT)
    manifest["experiment_id"] = exp_id
    created = skipped = 0
    for rs in plan["rowsets"].values():
        for run in rs["runs"]:
            existing = client.search_runs([exp_id], filter_string=f"tags.`dealpoint.key` = '{run['key']}'")
            if existing and not refresh:
                manifest["runs"][run["key"]] = existing[0].info.run_id
                skipped += 1
                continue
            if existing:
                run_id = existing[0].info.run_id
                for k, val in run["metrics"].items():
                    client.log_metric(run_id, k, val)
                manifest["runs"][run["key"]] = run_id
                created += 1
                continue
            with mlflow.start_run(experiment_id=exp_id, run_name=run["run_name"]) as active:
                mlflow.set_tags(run["tags"])
                mlflow.log_metrics(run["metrics"])
                manifest["runs"][run["key"]] = active.info.run_id
            created += 1
    for v in plan["views"]:
        key, value = saved_view_tag(v, plan["rowsets"][v["rowset"]])
        client.set_experiment_tag(exp_id, key, value)
        manifest["views"].append({"slug": v["slug"], "title": v["title"], "dashboard": v["dashboard"],
                                  "charts": v["chart_keys"], "url": view_url(base, exp_id, v["slug"])})
    client.set_experiment_tag(exp_id, "mlflow.note.content", experiment_note(plan, base_url=base, experiment_id=exp_id))
    print(f"  runs: {created} written, {skipped} already present; {len(plan['views'])} views set on experiment {exp_id}")

    # One line on the main experiment's Overview pointing here.
    main_exp = client.get_experiment_by_name(MAIN_EXPERIMENT)
    if main_exp is not None:
        note = main_exp.tags.get("mlflow.note.content", "")
        marker = "**Dashboards as chart views**"
        line = (f"\n\n{marker}: every Braintrust chart as a native MLflow bar chart, in experiment "
                f"`{VIEWS_EXPERIMENT}` ({base.rstrip('/')}/#/experiments/{exp_id}); pick a view from its views dropdown.")
        if marker not in note:
            client.set_experiment_tag(main_exp.experiment_id, "mlflow.note.content", note + line)
    return manifest


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    live = "--live" in argv
    refresh = "--refresh" in argv
    plan = view_plan()
    manifest = apply(plan, live=live, refresh=refresh)
    if live:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"manifest -> {MANIFEST_PATH}")
        for v in manifest["views"]:
            print(f"  {v['title']}: {v['url']}")
    else:
        print(f"DRY RUN: nothing written (snapshot {SNAPSHOT_PATH}); add --live to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
