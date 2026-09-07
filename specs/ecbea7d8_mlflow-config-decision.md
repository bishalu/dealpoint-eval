# M9b build plan — the config decision on MLflow's own strengths

**Spec (read-only):** `specs/milestones/m9b.md`
**Requirements (read-only, wins on product decisions):** `specs/grilled-product-brief.md`
**Prior art this builds on:** `specs/milestones/m9.md`, `dealpoint/eval/mlflow_mirror.py`, `docs/mlflow-tour.md`, `docs/demo-tour.md` stop 9.
**Budget:** $0. No OpenRouter call anywhere. `needs_model=False` on the milestone entry.
**Done:** every item under §9 holds on disk; `uv run pytest -m "gate_m9b and not needs_network" -q` green; full offline suite, `uv run ruff check .`, `uv run pyright` green.

---

## 0. What the planner already measured (do not re-derive; verify)

Everything below was computed by running the existing pure functions against the stored rows on
2026-09-07. The builder should reproduce these numbers from code, not paste them as literals into
production code — but the tests may assert them, and the tour must report them.

### 0.1 The two pools, and how to select their rows

`mlflow_mirror.agent_trace_plan()` returns 265 entries, each `{case_id, variant_id, category, row, run_name}`.
`mlflow_mirror.mirror_for(case_id, variant_id, row, ctx, category=...)` returns the metadata mirror,
which carries `variant_label` (e.g. `D@gemini-3.1-flash-lite`), `model_label`, `comparable`, `case_pool`.

| pool | selection predicate over `agent_trace_plan()` | result |
|---|---|---|
| `judged-18` | `category == "judged"` and `mirror["comparable"] == 1` | 6 configs × 18 rows = 108 |
| `test-32` | `category in ("agent", "judged")` and `mirror["comparable"] == 1` and `mirror["model_label"] == "glm"` | 4 configs × 32 rows = 128 |

Measured membership (exact, verified):

```
judged-18: A@haiku 18, D@haiku 18, D@glm 18, D@deepseek-v4-flash 18, D@qwen3.7-flash 18, D@gemini-3.1-flash-lite 18
test-32  : A@glm 32, B@glm 32, C@glm 32, D@glm 32
```

**Why `test-32` must union `agent` and `judged`.** `agent_run_log_plan()` excludes (case, variant) pairs
already planned as `judged`, so the raw `agent` category holds only 14 D@glm rows; the other 18 live under
`judged`. The union restores the full 32. This is exactly the filter the Braintrust showroom already uses
for the same chart — see `data/reports/mlflow_views.json`:
`metadata.comparable = 1 and metadata.model_label = 'glm' and (metadata.category = 'agent' or metadata.category = 'judged')`.
D@glm therefore legitimately appears in both pools (18 rows in `judged-18`, 32 rows in `test-32`) with
different metrics. That is not a bug; the parent run name is what keeps the comparison inside one pool.

`comparable == 1` is the M9 rule that drops representative picks, live replays and the partial
`D@openai/gpt-5.6-luna-pro` trace. Keep it. No cross-pool comparison anywhere.

### 0.2 Measured child metrics (the numbers D7 must reproduce)

`judged-18`:

| config | n | safe_acc | precision | net | usd/case | p50 s | p90 s | corr/$ | cap | correct | misleading | fabrication | verbatim | tool calls | silent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A@haiku | 18 | 0.6667 | 0.4545 | 0.2222 | 0.003356 | 3.71 | 5.47 | 165.55 | 0.0000 | 0.5556 | 0.3333 | 0.0000 | 1.0000 | 0.000 | 0.1111 |
| D@deepseek-v4-flash | 18 | 0.8889 | 0.7143 | 0.1667 | 0.002358 | 17.80 | 25.58 | 117.80 | 0.5000 | 0.2778 | 0.1111 | 0.2857 | 0.7143 | 6.722 | 0.6111 |
| D@gemini-3.1-flash-lite | 18 | 0.8333 | 0.6667 | 0.3333 | 0.005134 | 11.14 | 18.41 | 97.39 | 0.2778 | 0.5000 | 0.1667 | 0.0000 | 1.0000 | 4.389 | 0.3333 |
| D@glm | 18 | 0.8333 | 0.6250 | 0.1667 | 0.002811 | 24.60 | 52.66 | 118.57 | 0.2778 | 0.3333 | 0.1667 | 0.3750 | 0.6250 | 5.222 | 0.5000 |
| D@haiku | 18 | 0.8333 | 0.6250 | 0.1111 | 0.045796 | 18.20 | 63.15 | 6.07 | 0.5000 | 0.2778 | 0.1667 | 0.1250 | 0.8750 | 6.444 | 0.5556 |
| D@qwen3.7-flash | 18 | 0.8333 | 0.5714 | 0.0556 | 0.001117 | 14.76 | 17.04 | 198.98 | 0.5000 | 0.2222 | 0.1667 | 0.5714 | 0.4286 | 6.444 | 0.6111 |

`test-32`:

| config | n | safe_acc | precision | net | usd/case | p50 s | p90 s | corr/$ | cap | correct | misleading | fabrication | verbatim | tool calls | silent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A@glm | 32 | 0.7812 | 0.6500 | 0.2812 | 0.000990 | 10.79 | 23.60 | 504.95 | 0.0000 | 0.5000 | 0.2188 | 0.0000 | 1.0000 | 0.000 | 0.2812 |
| B@glm | 32 | 0.7500 | 0.5556 | 0.0625 | 0.002429 | 24.30 | 56.11 | 128.63 | 0.4375 | 0.3125 | 0.2500 | 0.1111 | 0.8889 | 5.375 | 0.4375 |
| C@glm | 32 | 0.7812 | 0.6667 | 0.2500 | 0.002405 | 22.12 | 42.06 | 194.91 | 0.3125 | 0.4688 | 0.2188 | 0.1905 | 0.8095 | 5.031 | 0.3125 |
| D@glm | 32 | 0.7188 | 0.5500 | 0.0938 | 0.002775 | 34.00 | 54.58 | 135.12 | 0.1562 | 0.3750 | 0.2812 | 0.3500 | 0.6500 | 4.812 | 0.3438 |

These agree with `docs/demo-tour.md` stop 9 (D@gemini safe 83% / precision 67% / net 33% / verbatim 100%;
A@haiku precision 45%; the GLM ladder net A 28%, C 25%, D 9%, B 6%). Reproducing them is the point.

### 0.3 **Spec/data conflict — report it, do not resolve it silently**

`specs/milestones/m9b.md` §2 says the Pareto gate must show that the frontier
"names D@gemini and D@qwen on safe-vs-dollars for `judged-18`". Applying the M6 rule
(`pareto_report.pareto_frontier`, minimise x, maximise y) to the measured numbers gives:

```
judged-18  usd_per_case vs safe_accuracy          -> D@qwen3.7-flash, D@deepseek-v4-flash
judged-18  wall_s_p50   vs precision_when_answering-> A@haiku, D@gemini-3.1-flash-lite, D@deepseek-v4-flash
judged-18  usd_per_case vs net_accuracy           -> D@qwen3.7-flash, D@deepseek-v4-flash, A@haiku, D@gemini-3.1-flash-lite
test-32    usd_per_case vs safe_accuracy          -> A@glm
test-32    wall_s_p50   vs precision_when_answering-> A@glm, C@glm
test-32    usd_per_case vs net_accuracy           -> A@glm
```

D@gemini is **dominated** on safe-vs-dollars: D@deepseek is both cheaper ($0.002358 vs $0.005134) and
safer (0.8889 vs 0.8333). D@gemini and D@qwen do appear together, but on the **net-accuracy-vs-dollars**
frontier — which is the tie-breaker axis the tour's verdict actually turns on.

**Instruction:** the M6 rule is the authority. Implement the frontier by calling
`pareto_report.pareto_frontier` unchanged, assert the frontier it actually produces, and record the
difference explicitly:

- a `spec_differences` list inside each `pareto.json` artifact (same convention as
  `data/reports/pareto.json`'s `brief_differences`), with the spec's claim, the measured frontier, and the
  axis on which D@gemini and D@qwen *do* share a frontier;
- one sentence in `docs/mlflow-tour.md` stop 9;
- a line in the builder's report to the operator.

Do **not** edit `specs/milestones/m9b.md`. Do **not** bend the frontier rule, the metric definitions or the
pool selection to make the spec sentence come true.

### 0.4 Measured answers to the three D7 searches

Run over the decision run tree only (see §4 on scoping):

| search | filter | configurations returned |
|---|---|---|
| safe and cheap | `metrics.safe_accuracy >= 0.8 and metrics.usd_per_case <= 0.003` | `D@deepseek-v4-flash`, `D@glm`, `D@qwen3.7-flash` (all `judged-18`) |
| right and fast | `metrics.precision_when_answering >= 0.65 and metrics.wall_s_p50 <= 15` | `D@gemini-3.1-flash-lite` (`judged-18`), `A@glm` (`test-32`) |
| agents that never invent a clause | `params.loop = 'agent' and metrics.fabrication_rate = 0` | `D@gemini-3.1-flash-lite` only |

Note on the second search: `A@glm`'s precision is exactly `0.65` (13/20; the Python float and the literal
`0.65` are the same double, so `>= 0.65` holds). Keep the boundary, and say in the tour that A@glm sits
exactly on it. Note also that A@glm is a *pipeline*, not an agent — the search is honest about that.

### 0.5 Environment facts, verified

- MLflow **3.16.0** installed; extra already declared (`[project.optional-dependencies] mlflow = ["mlflow>=3.16"]`).
- `mlflow.genai.evaluate(data, scorers, predict_fn=None, model_id=None)` — `predict_fn=None` is the default,
  so a zero-model-call row-level evaluation is native.
- `mlflow.genai.scorer(func=None, *, name=None, description=None, aggregations=None, pass_if=None)`.
- `MlflowClient` has `log_artifact`, `log_artifacts`, `create_model_version`, `set_model_version_tag`,
  `set_registered_model_alias`, `search_model_versions`. **No chart/view/dashboard API** at 3.16 — already
  documented as a gap in `docs/mlflow-tour.md`; D7's "saved chart views" therefore land as an exported JSON
  plus documented clicks, exactly as M9's `views` step did.
- `adws/adw_modules/milestones.py` **already registers `m9b`** (`spec_path="specs/milestones/m9b.md"`,
  `gate_marker="gate_m9b"`, `needs_model` omitted → defaults to `False` per
  `adws/adw_modules/data_types.py:481`). Spec §5 is done; verify, change nothing.
- `gate_m9b` is **not** yet in `pyproject.toml`'s `markers` list. It must be added.

---

## 1. Files to touch

| file | action |
|---|---|
| `dealpoint/eval/mlflow_decision.py` | **new** — the whole D7/D8/D9 implementation and its CLI |
| `tests/test_mlflow_decision.py` | **new** — the `gate_m9b` suite |
| `docs/mlflow-tour.md` | rewrite stop 9, add the cookbook section (D10) |
| `justfile` | add the `mlflow-decision` recipe |
| `pyproject.toml` | register the `gate_m9b` marker |
| `data/reports/mlflow_decision_views.json` | generated by `--live` (committed) |
| `data/reports/mlflow_decision_manifest.json` | generated by `--live` (committed) |

Do not touch: `specs/grilled-product-brief.md`, `specs/milestones/m9b.md`, `dealpoint/eval/mlflow_mirror.py`
(read from it, do not modify), `dealpoint/eval/pareto_report.py` (call it, do not modify), anything under
`dealpoint/agent/`, any frozen artifact, anything Braintrust.

---

## 2. `dealpoint/eval/mlflow_decision.py` — structure

Follow `mlflow_mirror.py`'s conventions exactly: `from __future__ import annotations`; module docstring
naming the spec; `mlflow` imported **lazily inside functions** so every planning function and the whole
offline suite work without the extra; pure planning functions above the client plumbing; `step_<name>(client,
live, manifest)` functions; a hand-rolled `main(argv)` (no argparse — `mlflow_mirror.main` parses `argv` by
hand and `--dry-run` wins over `--live`); metrics logged with `client.log_metric`, params with
`client.log_param`, descriptions with `client.set_tag(run_id, "mlflow.note.content", text)`.

### 2.1 Module constants

```python
MLFLOW_EXPERIMENT = "dealpoint-eval"          # same experiment as M9
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
MANIFEST_PATH = Path("data/reports/mlflow_decision_manifest.json")
VIEWS_PATH = Path("data/reports/mlflow_decision_views.json")
REGISTERED_MODEL = "dealpoint-agent"          # M9's model, extended
STEPS = ("tree", "pareto", "views", "registry", "evaluate")
DECISION_TAG = "dealpoint.decision"           # "parent" | "config" | "evaluation"
POOLS = ("judged-18", "test-32")
DECISION_METRICS = (  # exactly the 15 the spec names, in this order
    "safe_accuracy", "precision_when_answering", "net_accuracy", "correct_outcome_rate",
    "misleading_rate", "silent_failure_rate", "cap_hit_rate", "fabrication_rate",
    "verbatim_quote_rate", "usd_per_case", "wall_s_p50", "wall_s_p90",
    "tool_calls_per_case", "correct_per_dollar", "usd_per_correct",
)
REGISTRY_ALIASES = {
    "champion":   ("judged-18", "D@gemini-3.1-flash-lite"),
    "baseline":   ("judged-18", "A@haiku"),
    "cost-floor": ("judged-18", "D@qwen3.7-flash"),
    "safest":     ("judged-18", "D@deepseek-v4-flash"),
}
PARETO_PAIRS = (  # (x_key, y_key) — pareto_frontier minimises x, maximises y
    ("usd_per_case", "safe_accuracy"),
    ("wall_s_p50", "precision_when_answering"),
    ("usd_per_case", "net_accuracy"),
)
```

`_manifest_path()` / `_views_path()` accessor functions so tests can monkeypatch the module attributes,
mirroring `mlflow_mirror`.

### 2.2 Pure planning: `decision_pools()`

```python
def decision_pools() -> dict[str, dict[str, list[dict]]]:
    """{pool_name: {config_label: [agent_trace_plan entries]}} for the two comparable pools."""
```

- Call `mlflow_mirror.agent_trace_plan()` and `mlflow_mirror.mirror_context()` once.
- For each entry compute `mirror_for(...)` once and keep it alongside the entry (avoid recomputing;
  `mirror_for` does per-row case lookups and this runs over 265 rows several times otherwise).
- Apply the two predicates from §0.1. Group by `mirror["variant_label"]`.
- Assert nothing here; the counts are asserted in tests.

### 2.3 Pure planning: `config_metrics(entries, ctx) -> dict[str, float]`

Call `mlflow_mirror._run_metrics_for_agent_rows(entries, ctx)` — this is the function the M9 runs already
use, so every number is by construction the Braintrust mirror's number. Then project it onto the fifteen:

```
safe_accuracy            = m["safe_accuracy"]
precision_when_answering = m["precision_when_answering"]
net_accuracy             = m["net_accuracy"]
correct_outcome_rate     = m["correct_all"]
misleading_rate          = m["misleading_rate"]
silent_failure_rate      = m["silent_failure_rate"]
cap_hit_rate             = m["cap_hit_rate"]
fabrication_rate         = m["fabrication_rate"]
verbatim_quote_rate      = m["cite_verbatim_answered"]      # mean over answered rows
usd_per_case             = m["usd_per_case"]
wall_s_p50               = m["wall_s_p50"]
wall_s_p90               = m["wall_s_p90"]
tool_calls_per_case      = m["tool_calls"]
correct_per_dollar       = m["correct_per_dollar"]
usd_per_correct          = 1.0 / m["correct_per_dollar"]     # 0 correct -> omit? see below
```

`usd_per_correct` is the only genuinely new arithmetic. Compute it as
`sum(usd) / sum(correct_all)` over the pool's rows (identical to `1 / correct_per_dollar`, but derive it
from the sums so a zero-correct configuration is handled by the same guard `_run_metrics_for_agent_rows`
uses). Every configuration in both pools has at least one correct outcome, so no `None` arises in practice —
but do not add a fallback branch for a case that cannot occur; if the divisor is zero, let the metric be
absent, which the "exactly 15 metrics" test will catch loudly.

Return exactly `DECISION_METRICS` keys, no more. The "15 metrics per child" gate assertion is
`set(run.data.metrics) == set(DECISION_METRICS)`.

### 2.4 Pure planning: `decision_run_plan() -> list[dict]`

One entry per parent and per child:

```python
{"role": "parent", "pool": "judged-18", "key": "decision/judged-18",
 "name": "decision/judged-18", "tags": {...}, "description": <pool verdict>}
{"role": "config", "pool": "judged-18", "config": "D@gemini-3.1-flash-lite",
 "key": "decision/judged-18/D@gemini-3.1-flash-lite",
 "name": "decision/judged-18/D@gemini-3.1-flash-lite",
 "params": {...}, "tags": {...}, "metrics": {...}, "description": <verdict line>}
```

- **Child params:** `arm`, `loop`, `retriever`, `skill`, `model`, `system_label`.
  - `arm` = the `A`/`B`/`C`/`D` prefix of the config label.
  - `loop`, `retriever`, `skill` from `braintrust_showroom.arm_parameter_sets()` keyed by arm — reuse
    `mlflow_mirror._arm_params_by_arm()`. Map its `agent_loop` field to the param name `loop`
    (`"agent"` for B/C/D, `"pipeline"` for A) so the D7 search `params.loop = 'agent'` works. Verify the
    stored value spelling before writing the test; if `arm_parameter_sets()` yields a boolean or a different
    token, normalise to `"agent"` / `"pipeline"` in one place and say so in a one-line comment.
  - `model` = the mirror's `model_label` (short name: `gemini-3.1-flash-lite`, `glm`, ...).
  - `system_label` = the mirror's `system_label` (`SYSTEM_LABEL[arm]`).
- **Child tags:** the factorial schema, i.e. the same keys `mlflow_mirror.mirror_run_plan()` sets
  (`axis`, `varies`, `holds`, `arm`, `loop`, `retriever`, `skill`, `model`, `cases`) obtained by calling
  `braintrust_showroom.classify_experiment` on the M9 run name that corresponds to this configuration, plus
  `dealpoint.key` and `dealpoint.decision = "config"` and `dealpoint.pool = <pool>`.
  To find the M9 run name for a configuration, reuse the same mapping M9's registry uses: iterate
  `mlflow_mirror.mirror_run_plan()`, classify each, and key by `f"{arm}@{model}"` — this is exactly
  `mlflow_mirror.registry_plan()`'s body, so call `registry_plan()` and build
  `{variant_id: run_name}` from it rather than duplicating the loop. Note `registry_plan()` keys are the
  same `arm@short-model` labels the mirror produces, so they join directly to `variant_label`.
  For `test-32` the four GLM configs join to `A/B/C/D-z-ai_glm-5.3-flash-...`; for `judged-18` A@haiku and
  D@haiku join to the two `*claude-haiku-4.5*` runs and the rest to the `pareto-*`/agent runs. Assert the
  join is total (every config resolves) in a test; if any config has no M9 run name, fall back to
  classifying nothing and emit only the arm/loop/retriever/skill/model tags — but prefer to fix the join.
- **Parent tags:** `dealpoint.key`, `dealpoint.decision = "parent"`, `dealpoint.pool`, `comparable = "1"`,
  `cases = "18"` / `"32"`, `n_configs`.
- **Descriptions:** the child description is that configuration's line in the verdicts. Build it from the
  verdict text already in the repo, never by hand:
  `mlflow_mirror._axis_to_dashboard_text()` returns `{axis: verdict paragraph}` from
  `braintrust_cockpit.DASHBOARDS`. Use the `"model"` verdict for `judged-18` children and the `"system"`
  verdict for `test-32` children, prefixed by one generated sentence naming the configuration, its pool,
  its safe accuracy, precision, net accuracy and $/case (formatted from the metrics that are about to be
  logged, so a number can never drift from the metric beside it). The parent description is the pool's
  verdict paragraph plus the scoring definitions sentence from `docs/demo-tour.md` stop 9.

### 2.5 `step_tree(client, live, manifest)`

Dry run: print `2 parents, 10 children (6 + 4), 15 metrics per child` and populate `manifest["tree"]`.

Live:
1. `exp_id = _ensure_experiment(client)` — reuse `mlflow_mirror._ensure_experiment` (import it; do not
   copy it).
2. For each parent: `_find_run(client, exp_id, key)` (reuse `mlflow_mirror._find_run`, which searches
   ``tags.`dealpoint.key` = '<key>'``); create with `client.create_run(exp_id, run_name=..., tags=...)` if
   absent, else update tags. Record `parent_run_id`.
3. For each child: same idempotent find/create, with `"mlflow.parentRunId": parent_run_id` in the tag dict
   so the UI renders the nested tree, plus `mlflow.runName`. Log the 15 metrics and the 6 params, set
   `mlflow.note.content`.
4. Track `created_this_run` (parents + children) in the manifest — this is what the second-`--live`
   idempotency test asserts is `0`.

### 2.6 `step_pareto(client, live, manifest)` — artifacts on each parent

Three artifacts per parent (the gate counts exactly 3):

**`pareto.json`** — a dict:
```python
{"pool": "judged-18",
 "rule": <pareto_report.pareto_frontier's docstring rule, one sentence, verbatim from data/reports/pareto.json's "frontier"."rule" phrasing adapted to the axes>,
 "frontiers": [{"x": "usd_per_case", "y": "safe_accuracy",
                "points": [{"config": ..., "x": ..., "y": ..., "frontier": bool,
                            "frontier_note": ..., "tied_with": ...}],
                "frontier": ["D@qwen3.7-flash", "D@deepseek-v4-flash"]}, ...],
 "spec_differences": [ ... see §0.3 ... ]}
```
Build `points` by calling `pareto_report.pareto_frontier(pts, x_key=..., y_key=..., id_key="config")`
unchanged — it already minimises x and maximises y, which fits all three pairs — and joining its result
back to the (x, y) values.

**`pareto.svg`** — a new **pure, stdlib-only** function in this module:

```python
def pareto_svg(points: list[dict], *, x_key: str, y_key: str, title: str) -> str:
```

Deterministic: no timestamps, no random ids, coordinates rounded to 2 decimals, so two calls produce
byte-identical output. Model it on `pareto_report.render_svg` (`dealpoint/eval/pareto_report.py:555`) —
same 640×420 canvas, white background, axes with tick labels, frontier points filled and joined by a
`<polyline>` sorted by x, dominated points hollow, each point labelled with its config. Do **not** import
or generalise `render_svg`; it is bound to the M6 report dict shape. Write a small sibling. Draw all three
pairs into one SVG only if it stays readable — otherwise one `<g>` panel per pair stacked vertically;
either is fine, but the file must be one artifact named `pareto.svg`.
No new dependency. Escape any text that reaches the SVG with `xml.sax.saxutils.escape` (config labels
contain `@` and `.`, which are safe, but escape anyway — it is one call and it removes a whole class of
injection).

**`decision.md`** — the verdict paragraph from `docs/demo-tour.md` stop 9, **verbatim**. Read it out of the
file at build time rather than pasting it into Python: locate the `### Stop 9. What should we deploy?`
heading and take the paragraph beginning `The decision, written down:` through the end of that paragraph.
Add a helper `stop9_verdict(path: Path = Path("docs/demo-tour.md")) -> str` and a test that it is non-empty,
contains `D@gemini`, and appears character-for-character in `docs/demo-tour.md`. This keeps the two
documents from drifting.

Log all three with `client.log_artifact(parent_run_id, str(local_path))` after writing them into a
`tempfile.TemporaryDirectory()`. Artifacts land under the experiment's artifact location
(`data/mlflow/artifacts/` for the real store, the tmp dir for the gate) — gitignored, a few hundred KB.

Dry run prints `pareto: 2 parents x 3 artifacts (pareto.json, pareto.svg, decision.md)` and records
`manifest["pareto"] = {"artifacts_per_parent": 3, "frontiers": {pool: {f"{x}|{y}": [...ids...]}}}` —
the frontier ids are computed in the dry run too, because they are pure.

### 2.7 `step_views(client, live, manifest)` — saved chart views

MLflow 3.16 OSS exposes no chart-view or experiment-view API (verified in §0.5, and already stated as a gap
in `docs/mlflow-tour.md`). **Probe once and record the probe**, so the claim is evidence rather than
assertion: in `step_views`, check for a view-creation entry point (e.g.
`hasattr(client, "create_chart")`, `hasattr(client, "set_experiment_view")`, or a
`mlflow.tracking` view module) and write the boolean into the manifest as
`{"chart_api_available": False, "probed": [<names checked>]}`.

Write `data/reports/mlflow_decision_views.json` — a list, one entry per saved view, per parent:

```json
{"pool": "judged-18",
 "kind": "parallel_coordinates",
 "params": ["loop", "retriever", "skill", "model"],
 "metrics": ["safe_accuracy", "precision_when_answering", "usd_per_case", "wall_s_p50"],
 "title": "...",
 "clicks": "Experiment > Chart view > + > Parallel coordinates; params loop, retriever, skill, model; metrics ..."}
```

Four views per parent (8 entries total):
1. `parallel_coordinates`: params `loop, retriever, skill, model` → metrics
   `safe_accuracy, precision_when_answering, usd_per_case, wall_s_p50`.
2. `scatter`: x `usd_per_case`, y `safe_accuracy`, colour by `model`.
3. `scatter`: x `wall_s_p50`, y `precision_when_answering`.
4. `bar`: `correct_per_dollar` by run.

Each entry carries a `clicks` string; `docs/mlflow-tour.md` reproduces the three clicks for the ones the
tour walks. Also set the parent runs' `mlflow.note.content` to include the view recipe (already done in
`step_tree`; do not duplicate the text).

### 2.8 `step_registry(client, live, manifest)` — D8

Extend M9's `dealpoint-agent`. Ten new versions, one per decision child.

- `client.create_registered_model(REGISTERED_MODEL)` inside a `try/except` (M9 already does this; the model
  may exist). Match M9's existing `# noqa: BLE001, S110` comment style.
- **Idempotency by config hash:** tag `config_hash = <child dealpoint.key>` (e.g.
  `decision/judged-18/D@gemini-3.1-flash-lite`). Before creating, scan
  `client.search_model_versions("name = 'dealpoint-agent'")` for a version whose
  `tags.get("config_hash")` equals the key; reuse it if found. This is exactly M9's mechanism with a
  decision-scoped key, so M9's nine versions and M9b's ten never collide.
- `client.create_model_version(REGISTERED_MODEL, source=f"runs:/{child_run_id}", run_id=child_run_id,
  tags={"config_hash": key, "variant_id": config, "pool": pool}, description=<verdict line + pool>)`.
- Version tags carry the same metrics: `client.set_model_version_tag(REGISTERED_MODEL, version, k, str(v))`
  for each of the 15.
- Aliases: `client.set_registered_model_alias(REGISTERED_MODEL, alias, version)` for the four in
  `REGISTRY_ALIASES`, all pointing at `judged-18` children. This **re-points** `champion`, `baseline` and
  `cost-floor` from M9's versions to M9b's — that is the spec's intent ("the registry as the decision
  record"), it is idempotent, and it must be stated in one line of the tour so nobody is surprised that the
  alias moved. `safest` is new.

### 2.9 `step_evaluate(client, live, manifest)` — D9, zero model calls

Per configuration (10 of them):

```python
mlflow.genai.evaluate(data=frame, predict_fn=None, scorers=scorers)
```

- **`scorers`** = `list(mlflow_mirror.build_deterministic_scorers().values())` (the six, reused, not
  redefined) **plus three new pure `@mlflow.genai.scorer` functions** defined in this module:
  - `safe(outputs=None, metadata=None)` → `metadata["safe"]`
  - `misleading(outputs=None, metadata=None)` → `metadata["misleading"]`
  - `deployable(outputs=None, metadata=None)` → `int(bool(metadata["correct_all"]) or bool(metadata["silent_failure"]))`

  **Note the identity:** `metadata_mirror` defines `silent_failure = not correct_all and not misleading`
  and `safe = 1 - misleading`, so `correct or silent` ≡ `not misleading` ≡ `safe`. `deployable` is the same
  rule stated the way a lawyer states it. Assert the identity in a test (`deployable == safe` on every row)
  and say so in one line of the tour — restating it as an independent metric would be dishonest.
- **`data`** = a `pandas.DataFrame` built by a new pure function
  `evaluation_frame(entries, ctx) -> list[dict]` with columns `inputs`, `outputs`, `metadata` — one row per
  (case, variant), `metadata` being the full `mirror_for` dict merged with the stored row's `scores` (the
  deterministic scorer handlers from `braintrust_sync._make_scorer_handler` read `output=` and `metadata=`;
  match the shape `mlflow_mirror.build_deterministic_scorers` already feeds them). Build the frame from the
  stored rows, **not** from `mlflow.search_traces` — the gate's temporary store has no traces (M9's `traces`
  step is not run in this gate) and the spec forbids re-running anything. Say in the tour that the
  Evaluations tab compares the same (case, variant) rows the traces carry.
- **Run placement and counting:** `mlflow.genai.evaluate` opens its own run. Wrap each call in
  `with mlflow.start_run(run_name=f"eval/{pool}/{config}", nested=True, parent_run_id=parent_run_id)` if
  that reuses the active run (probe once); otherwise tag `result.run_id` afterwards. Either way, set
  `dealpoint.decision = "evaluation"` and `dealpoint.key = f"eval/{pool}/{config}"` so the counting
  assertions in the gate filter by tag and never mistake an evaluation run for a config child.
  Idempotency: skip the call when a run with that `dealpoint.key` already exists.
- Record `manifest["evaluate"] = {"configs": 10, "written_this_run": n}`.
- The row-level comparison the tour names is `contract_144__q05` and its twin
  `contract_39__redacted_q05` — verify both case ids are present in the `judged-18` frames for the
  configurations the tour compares, and assert it.

### 2.10 `main(argv)`

Copy `mlflow_mirror.main`'s shape: `--live`, `--dry-run` (wins), `--only <comma list>`; dry run is the
default and writes nothing; live writes `data/reports/mlflow_decision_manifest.json`. Reuse
`mlflow_mirror.mlflow_dir_bytes()` and `DISK_GUARD_BYTES` for the same disk guard before writing (return
exit code 2 on breach, as M9 does). Manifest keys: `mode`, `started_at`, `finished_at`, `experiment`,
`tracking_uri`, `mlflow_version`, plus one key per step.

**`_client()`**: reuse `mlflow_mirror._client(tracking_uri)` — do not write a second one.

---

## 3. `justfile` and `pyproject.toml`

```make
# M9b: the config decision run tree (nested runs, Pareto artifacts, registry versions). No model calls.
mlflow-decision *ARGS:
    uv run python -m dealpoint.eval.mlflow_decision "$@"
```

Place it directly after `mlflow-sync-dry-run`, matching the surrounding comment style.

`pyproject.toml` markers: add `"gate_m9b: milestone 9b gate",` after the `gate_m9` line.

---

## 4. Search scoping (important)

The decision runs live in the **same experiment** as M9's 33 runs, several of which also carry
`safe_accuracy` and `usd_per_case`. Every cookbook search and every test search must therefore be scoped:

```
tags.`dealpoint.decision` = 'config' and <the spec's predicate>
```

Write the scope prefix once as a module constant (`DECISION_SEARCH_SCOPE`) and use it in both the module
and the tour, so the cookbook a reader copies is the one the test runs. The tour prints the full string
including the scope.

---

## 5. `docs/mlflow-tour.md` — D10

Rewrite **stop 9** into the showcase. Keep everything else, and in particular keep every phrase the existing
`gate_m9` test `test_mlflow_tour_names_every_unique_feature_and_gap` looks for: the "What MLflow adds"
section must still contain `in-process`, `judge alignment`, `prompt optimization`, `model registry`,
`timestamps`; the "What only Braintrust has here" section must still contain `Topics`, `Patterns`, `Loop`,
`labeling`, `chart`, `online scoring`. Run the M9 gate after editing.

New stop 9 content, in this order:

1. **The two parent runs.** Open `decision/judged-18`; name the pool (every system@model on the same 18
   cases) and say plainly that no chart crosses pools, and why (`comparable = 1`).
2. **The nested run tree.** Six children under `judged-18`, four under `test-32`, params on one side,
   fifteen metrics on the other.
3. **Parallel coordinates** — `loop, retriever, skill, model → safe_accuracy, precision_when_answering,
   usd_per_case, wall_s_p50`, with the three clicks (no chart API in OSS 3.16). One line: *this is the thing
   a Braintrust monitor dashboard cannot do — four metrics and four params on one chart instead of one
   ranked bar list per metric.*
4. **The two scatters** — `usd_per_case` vs `safe_accuracy` coloured by model; `wall_s_p50` vs
   `precision_when_answering`. One line: *params become axes.*
5. **The Pareto artifact** — `pareto.json` and `pareto.svg` on the parent run, the M6 rule applied to three
   axis pairs. Report the measured frontiers from §0.3 **and** the §0.3 spec difference in one honest
   sentence: the spec expected D@gemini and D@qwen on safe-vs-dollars; measured, D@deepseek dominates
   D@gemini there (cheaper and safer), and D@gemini and D@qwen share the net-accuracy-vs-dollars frontier
   instead — which is the axis the deployment verdict turns on.
6. **Cookbook: three metric-filtered searches.** The three filter strings verbatim (scoped per §4) and the
   configurations each returns, from §0.4. One line: *search by metric — a Braintrust dashboard ranks, it
   does not filter.*
7. **The registry as the decision record.** Ten versions on `dealpoint-agent`, version tags carrying the
   same fifteen metrics, aliases `champion` = D@gemini, `baseline` = A@haiku, `cost-floor` = D@qwen,
   `safest` = D@deepseek; each version's description is its verdict line and the pool it was measured on;
   the version-comparison view is the deployment record. One line stating that M9b re-points M9's three
   aliases at the decision versions, and adds `safest`.
8. **Row-level comparison at zero model calls.** `mlflow.genai.evaluate(..., predict_fn=None)` with the six
   deterministic scorers plus `safe`, `misleading`, `deployable`; the Evaluations tab compares any two
   configurations on `contract_144__q05` and the redacted twin `contract_39__redacted_q05` — the MLflow
   form of the Braintrust Grid. State the `deployable ≡ safe` identity.
9. **"Not here."** One line naming what Braintrust has and MLflow does not for this question: the eight
   ranked single-metric dashboards over *logs* (with a verdict paragraph attached to the dashboard object),
   and online scoring. MLflow's run comparisons are run-level and rebuilt by hand from
   `data/reports/mlflow_decision_views.json` because OSS 3.16 has no chart/view API.

Do not change any number that `docs/demo-tour.md` states for the same object.

---

## 6. `tests/test_mlflow_decision.py` — the `gate_m9b` suite

Follow `tests/test_mlflow_mirror.py` exactly: module docstring naming the spec,
`mlflow = pytest.importorskip("mlflow")`, `pytestmark = pytest.mark.gate_m9b`, module-scoped
`tracking_uri` fixture on `tmp_path_factory` returning `f"sqlite:///{d}/m.db"`, module-scoped `client`
fixture calling `m._client(tracking_uri)`, a module-scoped `monkeypatch_module` fixture, and a module-scoped
`live_manifest` fixture that sets `os.environ["MLFLOW_TRACKING_URI"]`, redirects `VIEWS_PATH` and
`MANIFEST_PATH` into tmp dirs, and runs the five steps live. Real client, temporary SQLite store, no server.

Required tests (the gate, §2 of the spec):

**Dry-run counts**
1. `decision_pools()` returns 2 pools; `judged-18` has 6 configs × 18 rows; `test-32` has 4 configs × 32
   rows; every config label is one of the ten named in §0.1.
2. `decision_run_plan()` yields 2 parents and 10 children; every child's `metrics` dict has exactly the 15
   `DECISION_METRICS` keys.
3. Dry run writes no manifest and no views file (`main([])`, and `main(["--live", "--dry-run"])`).

**Live**
4. After `live_manifest`: `search_runs` with ``tags.`dealpoint.decision` = 'parent'`` returns 2;
   `= 'config'` returns 10; each child run has exactly 15 metrics and the 6 named params;
   each parent has 3 artifacts (`client.list_artifacts(parent_run_id)` names `pareto.json`, `pareto.svg`,
   `decision.md`); `search_model_versions` finds 10 versions tagged with a `decision/` config hash; the
   four aliases resolve (`client.get_model_version_by_alias`) to the expected configs.
5. Nesting: every child's `mlflow.parentRunId` tag equals its pool's parent run id.
6. Idempotency: run the five steps `--live` a second time into the same store; parent/child/version counts
   unchanged, alias targets unchanged, and `manifest["tree"]["created_this_run"] == 0`,
   `manifest["registry"]["created_this_run"] == 0`, `manifest["evaluate"]["written_this_run"] == 0`.

**Equality with the Braintrust mirror (spine checks)**
7. For every (pool, config), each of the 15 logged metrics equals `config_metrics(entries, ctx)` recomputed
   from `mlflow_mirror._run_metrics_for_agent_rows` over the same rows — i.e. the MLflow number *is* the
   mirror number, by identity not by coincidence. Use `pytest.approx` with a tight rel tolerance for the
   float round-trip through SQLite.
8. Named spine assertions, as literals, so a silent data change fails loudly:
   - `judged-18 / D@gemini-3.1-flash-lite` `safe_accuracy == pytest.approx(15/18)`
   - `judged-18 / A@haiku` `precision_when_answering == pytest.approx(5/11)`
   - `judged-18 / D@qwen3.7-flash` `correct_per_dollar == pytest.approx(198.98, rel=1e-3)`
   - `test-32 / A@glm` `net_accuracy == pytest.approx(0.28125)` and
     `test-32 / C@glm` `net_accuracy == pytest.approx(0.25)`, with `A > C`.

**Pareto**
9. `pareto_report.pareto_frontier` reproduces the M6 rule on the M6 data: load `data/reports/pareto.json`,
   feed its per-model `usd_per_case` / `grounded_accuracy` points back through `pareto_frontier`, and assert
   the frontier equals the recorded `frontier.models` (`["qwen/qwen3.7-flash", "deepseek/deepseek-v4-flash"]`).
   Follow the assertion style already in `tests/test_pareto_frontier.py`.
10. The `judged-18` safe-vs-dollars frontier is `{"D@qwen3.7-flash", "D@deepseek-v4-flash"}`, **and** a
    second assertion that D@gemini and D@qwen are both on the `usd_per_case` vs `net_accuracy` frontier.
    A comment above these two assertions records the §0.3 spec difference and points at the
    `spec_differences` block in the artifact.
11. `pareto_svg` is deterministic: two calls on the same points return identical strings; the output starts
    with `<svg` and ends with `</svg>`; it names every config label.
12. `pareto.json`'s `spec_differences` is non-empty and names both the spec's claim and the measured
    frontier.

**Searches**
13. The three filter strings from §0.4, scoped per §4, run against the temporary store via
    `client.search_runs([exp_id], filter_string=...)` and return exactly the named configuration sets.
    Build the filter strings from the module constants the tour quotes, so a change to one changes both.

**D9**
14. For every configuration, the evaluation run's aggregate for `safe` equals the child's `safe_accuracy`
    and for `misleading` equals the child's `misleading_rate` (`pytest.approx`).
15. `deployable == safe` on every row of both pools (pure, no MLflow needed).
16. The `judged-18` evaluation frames for the configurations the tour compares contain both
    `contract_144__q05` and `contract_39__redacted_q05`.

**Zero network**
17. Patch the OpenRouter client constructor and assert it is never constructed:
    `monkeypatch.setattr("dealpoint.llm.client.OpenRouterClient.__init__", _boom)` (and, belt and braces,
    `monkeypatch.setattr("openai.OpenAI", _boom)`), then run the full five-step `--live` sequence into a
    fresh temporary store. `dealpoint/llm/client.py:206` is `class OpenRouterClient`, whose `__init__`
    builds `openai.OpenAI(...)` at line 224 — that is the only construction site.
    Also assert `mlflow_decision` never calls `mlflow.genai.judges.make_judge` (the M9 module's judges are
    not used here): patch it to raise.

**Docs**
18. `docs/mlflow-tour.md` stop 9 names: `decision/judged-18`, `decision/test-32`, `parallel coordinates`,
    `pareto.json`, all three search filter strings, all four aliases, `deployable`,
    `contract_144__q05`, and the "not here" line. Assert the three filter strings appear character-for-character
    as the module builds them.
19. `stop9_verdict()` returns a non-empty paragraph that appears verbatim in `docs/demo-tour.md`.

Mark nothing `needs_network` or `needs_model` — the whole gate is offline and free.

---

## 7. Order of work

1. Read `dealpoint/eval/mlflow_mirror.py` end to end (it is the template) and
   `dealpoint/eval/pareto_report.py:40-135` and `:555-660`.
2. Write the pure layer: `decision_pools`, `config_metrics`, `decision_run_plan`, `pareto_svg`,
   `stop9_verdict`, `evaluation_frame`. Get tests 1, 2, 8, 9, 10, 11, 15, 19 green with no MLflow writes.
3. Add `gate_m9b` to `pyproject.toml`; add the justfile recipe.
4. Write `step_tree` and `step_pareto`; get tests 4, 5, 6 (partial), 7, 12 green.
5. Write `step_views`, `step_registry`, `step_evaluate`; finish 4, 6, 13, 14, 16, 17.
6. Rewrite `docs/mlflow-tour.md` stop 9 and the cookbook; get 18 green; **re-run `gate_m9`** to confirm the
   existing tour test still passes.
7. `just mlflow-decision` (dry run) → `--live` → `--live` again against the real store; commit
   `data/reports/mlflow_decision_views.json` and `data/reports/mlflow_decision_manifest.json`.
8. `uv run ruff check .`, `uv run pyright`, `uv run pytest -q` (full offline suite), then
   `uv run pytest -m "gate_m9b and not needs_network" -q`.

---

## 8. Constraints the builder must not break

- **$0.** No OpenRouter call. If any step appears to need one, stop and report; do not spend.
- **No re-derivation.** Every metric comes from `mlflow_mirror._run_metrics_for_agent_rows` over
  `mirror_for`. Do not reimplement safe accuracy, precision, net accuracy or any rate.
- **No cross-pool comparison.** `comparable = 1` everywhere; two parents, never one.
- **No new dependency.** The SVG is stdlib. No matplotlib, no numpy.
- **Additive only.** `mlflow_mirror.py` is imported, never edited. M9's 33 runs, 738 traces, 9 datasets,
  8 prompts and 9 model versions stay exactly as they are; the only M9 object M9b changes is the three
  registry aliases, deliberately, per D8.
- **Scoring definitions frozen.** Safe accuracy is the accuracy point, precision when answering its check,
  net accuracy the tie-breaker. No new definition.
- **Frozen artifacts untouched.** `data/reports/pareto.json` is read, never written.
- **Read-only specs.** `specs/grilled-product-brief.md` and `specs/milestones/m9b.md` are not edited.

---

## 9. Definition of done

- `dealpoint/eval/mlflow_decision.py` exists with the five steps, dry run by default, `--live`, `--only`,
  idempotent by `dealpoint.key`.
- `just mlflow-decision` dry run prints 2 parents / 10 children / 15 metrics per child / 3 artifacts per
  parent / 10 versions / 4 aliases; `--live` creates them; a second `--live` creates nothing new.
- Every child metric equals its Braintrust mirror aggregate for the same pool and configuration (test).
- `pareto.json` + `pareto.svg` + `decision.md` on both parents; the frontier is the M6 rule's output; the
  §0.3 spec difference is recorded in `spec_differences`, in the tour, and in the builder's report.
- `data/reports/mlflow_decision_views.json` holds 8 saved-view entries with their clicks;
  `data/reports/mlflow_decision_manifest.json` is committed.
- Ten `dealpoint-agent` versions carrying the 15 metrics as version tags; aliases `champion`, `baseline`,
  `cost-floor`, `safest` resolve to D@gemini, A@haiku, D@qwen, D@deepseek on `judged-18`.
- Per-configuration `mlflow.genai.evaluate(predict_fn=None)` runs with nine scorers; aggregates equal the
  child metrics; zero model calls, asserted by a patched-client test.
- `docs/mlflow-tour.md` stop 9 is the showcase, with the cookbook, the four MLflow-does-this lines and the
  "not here" line; the M9 tour test still passes.
- `uv run pytest -m "gate_m9b and not needs_network" -q` green; full offline suite green;
  `uv run ruff check .` and `uv run pyright` green.

## 10. What to report to the operator at the end

1. The §0.3 Pareto difference between the spec's expected frontier and the measured one, with both
   frontiers and the axis on which the spec's pair does appear.
2. That `deployable` is, by the mirror's own definitions, identical to `safe` — the spec asks for both, so
   both exist, and the test pins the identity.
3. That M9b re-points M9's `champion` / `baseline` / `cost-floor` aliases at the decision versions and adds
   `safest`.
4. That `A@glm` sits exactly on the `precision_when_answering >= 0.65` boundary of the second cookbook
   search, and that it is a pipeline, not an agent.
5. No difference found between `specs/grilled-product-brief.md` and this milestone's product decisions;
   §2.6's Pareto output (`data/reports/pareto.json` + image) is read, not rewritten, and §2.7's
   "surface, never the source of truth" rule holds — every number here comes from Git.
