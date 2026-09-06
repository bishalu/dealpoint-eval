# M7b — Braintrust cockpit and the multi-judge demo (reproducible layer)

**Spec (read-only):** `specs/milestones/m7b.md`
**Requirements authority (read-only):** `specs/grilled-product-brief.md`
**Resumes from:** M7a checkpoint at `d418929` / `ce4fc98`. Nothing from M1–M7a is rerun, retuned or re-judged.
**Budget:** $0.00 new OpenRouter spend. M7b makes **no model calls**. No new dependency groups.
**Gate:** `uv run pytest -m "gate_m7b and not needs_network" -q`, plus the full offline suite,
`uv run ruff check .` and `uv run pyright`.

---

## 0. Verified starting state (measured during planning — trust these, re-verify only if something looks wrong)

### 0.1 Existing code you will extend

| File | What it is |
|---|---|
| `dealpoint/eval/braintrust_sync.py` (1949 ln) | `just braintrust-sync`. Pure mapping functions + thin driver `sync(client, *, dry_run)`. Uses the **braintrust Python SDK** (lazy `import braintrust`), never REST. |
| `dealpoint/eval/braintrust_adapter.py` | `PROJECT = "dealpoint-eval"`, `load_braintrust_key()` (env → `.env.braintrust` → `.braintrust.json`), `braintrust_available()`. |
| `dealpoint/eval/btql.py` (316 ln) | `build_investigations() -> list[dict]` (the six queries, `{id,title,question,btql}`), `run_btql(query, *, api_key, timeout)` via lazy `import requests` against `https://api.braintrust.dev/btql`, `render_queries_markdown()` which **generates** `docs/braintrust-queries.md`. |
| `tests/test_braintrust_sync.py` (555 ln) | `pytestmark = pytest.mark.gate_m7`. Hand-written `FakeDataset` / `FakeExperiment` / `FakeSpan` / `FakeSpanWithChildren` / `FakeBraintrustClient` with real upsert-by-`id` semantics, so the idempotency test has teeth. |
| `tests/test_btql_queries.py` (127 ln) | The doc/code sync pattern to imitate for the view-filter test. |
| `tests/conftest.py` | Only `_fast_retry_backoff` (autouse) and `dataset_available` / `require_dataset`. **No HTTP fixtures** — add your fakes in the new test module. |

Key symbols in `braintrust_sync.py` (line numbers as of `d418929`):

- `SCORE_NAMESPACES` (:42) = `("obj", "li", "judge", "deepeval")`
- `_normalize_score_value(namespaced_name, value)` (:62) — rescales `judge/*` from 1–5 to 0–1, passes everything else through.
- `score_namespace(name)` (:78) — bare name → namespace; **names already containing `/` are returned unchanged**.
- `assert_score_budget(plan)` (:177) / `assert_actual_score_budget(plan, actual)` (:203); `_score_budget_limit` (:167) returns `M7A_MAX_SCORES_PER_CASE = 12` when `n_cases <= 60`.
- `_judge_score_row(case_id, arm, model, dims)` (:562) and `_eval_experiment_plans()` (:570) — build the six `judge-<variant_id>` plans, `score_names = ["judge/reasoning","judge/evidence","judge/trajectory","judge/professional"]`, one row per case.
- `_packet_id_for(case_id, variant_id)` (:656) → `dealpoint.eval.blinding._packet_id` — deterministic.
- `_mean_judge_dims_for_packet(judge_rows, packet_id)` (:662).
- `log_hierarchy(row, case, doc, *, judge_dims=None, deepeval_scores=None)` (:850) — builds the `case > agent > {search_agreement…, tools…, final_answer, scoring}` dict tree; the `scoring` node carries `provenance = {"obj": …, "judge": …, "deepeval": …}`.
- `_emit_span_tree(root_span_source, node, parent_span=None)` (:1680) — recursive emitter; scores outside `[0,1]` are moved to `metadata.provenance_excluded_from_scores`.
- `experiment_plan()` (:804), `review_set(n=12)` (:1037), `_DryRunClient` (:1413), `main(argv)` (:1915) with `--dry-run`.
- `LIVE_RUN_STATUS` (:1892) — records that the last live sync was blocked by the workspace quota.

`experiment_plan()` currently yields **29 experiments**: 8 `rag-*`, 9 agent (`<ARM>-<model>-<index_version>-<git_sha7>`), 7 eval (`judge-A@haiku`, `judge-D@haiku`, `judge-D@glm`, `judge-D@deepseek-v4-flash`, `judge-D@qwen3.7-flash`, `judge-D@gemini-3.1-flash-lite`, `deepeval-crosscheck`), 5 `pareto-*`.
Datasets: `dev` (58), `test` (167), `counterfactual` (40), `judged_calibration` (18), `synthetic_query` (106), plus `maud-dealpoint-review-set` (12).

### 0.2 Data on disk

- `data/eval/judged_subset.json` — `subset_hash = "5918ef10a7e6"`, 18 `case_ids`, `n_traces 108`, `n_judge_calls 324`, `variants` = 6 entries each `{arm, model, variant_id, results_path}`: `A@haiku`, `D@haiku`, `D@glm`, `D@deepseek-v4-flash`, `D@qwen3.7-flash`, `D@gemini-3.1-flash-lite`.
- `data/eval/judge_scores.jsonl` — **324 rows**, keys: `case_id, question_id, variant_id, packet_id, judge_family, judge_model, reasoning, evidence, trajectory, professional, notes, ok, failure_detail, rubric_version, input_tokens, output_tokens, usd, ts`.
  Families/models: `Mistral` → `mistralai/mistral-small-3.2-24b-instruct`, `NVIDIA` → `nvidia/nemotron-3-super-120b-a12b`, `ByteDance` → `bytedance-seed/seed-2.0-mini`.
  **There is no separate "raw judge JSON" blob** — the parsed dimensions + `notes` in this row *are* the stored judge output. `usd` is the realised call cost.
- `data/eval/calibration/packets.jsonl` — **108 blinded packets**, keys `packet_id, question_text, options, status, trajectory, finding, gold_span, gold_span_note, rubric_version`. **No `case_id`, no `variant_id`** — this is literally what each judge saw. This is the §2 "blinded packet".
- `data/eval/calibration/variant_key.json` — `packet_id -> {arm, case_id, model, variant_id}`. The de-blinding map; **never** push it into a blinded surface.
- `data/eval/calibration/human_scores.jsonl` — **24 rows**, `{packet_id, scorer:"lawyer_1", scored_at, reasoning, evidence, trajectory, professional, notes}`; exactly 4 per variant across all six variants. Plus `human_scores.schema.json`, `human_scores.provenance.json`, `form.md`.
- `data/reports/judges.json` — `rubric_version "cfda9f8cc401"`, `subset_hash "5918ef10a7e6"`; the only place Spearman / weighted kappa / pairwise agreement are computed.
- `data/results/spend_ledger.jsonl` — 4779 rows, field name is **`milestone_tag`**; existing tags `m1 m2 m4 m4_1 m5 m6 m7a`. No `m7b`.
- `data/reports/framework_versions.json` — flat `{name: version}` + `git_sha7`.
- `eval/judges/rubrics.md` — hash `cfda9f8cc401`, the frozen anchors.

### 0.3 The hero case — already computed, use this as the expected value

Applying the §1 rule to the artifacts on disk:

- Candidates after (a) *strict* disagreement on `obj/grounded_accuracy` between `A@haiku` and `D@haiku`
  (**both values non-null and different**; a `null` is an abstention/no-gold outcome, not a disagreement):
  `contract_32__q04` (False → True) and `contract_32__q06` (False → True).
- Max pairwise judge spread over the two hero variants' traces:
  `contract_32__q04` → **4** (`evidence` on `A@haiku`: NVIDIA 5, Mistral 2, ByteDance 1);
  `contract_32__q06` → 1.
- **Winner: `contract_32__q04`.** No tie-break needed.

Supporting numbers (from `data/eval/judge_scores.jsonl`):

| variant | packet_id | Mistral | NVIDIA | ByteDance |
|---|---|---|---|---|
| `A@haiku` | `ced644da8ee5` | 3/2/3/3 | 4/5/4/4 | 2/1/4/3 |
| `D@haiku` | `ca69a6dab336` | 5/5/4/5 | 5/5/4/5 | 5/5/4/4 |

(order: reasoning/evidence/trajectory/professional). Narrative: arm A fails the gold-span check and the
judges cannot agree how bad it is; arm D fixes it and the judges converge. That is the demo.

**Neither hero packet has a human score** (`ced644da8ee5`, `ca69a6dab336` are not in `human_scores.jsonl`),
and **the hero case is not in the M7a `review_set(12)`** (only 5 of those 12 packets are human-scored).
This is a real gap, not a bug. §1 stop 5 and §5 chart 3 must state it plainly; the cockpit session (§7 step 1)
is what closes it.

**If the code's own computation disagrees with `contract_32__q04`, the code wins** — record what the code
computes in the manifest and say so in the report. Do not hardcode the answer; hardcode the *rule*.

### 0.4 Environment (already set up by the operator — do not touch)

- `pi --version` → `0.84.4`; `~/.pi/agent/mcp.json` exists; `pi extension list` shows `[pi-mcp]` resolving
  Braintrust view/chart schemas, so the extension is installed and the MCP tools are live.
- `BRAINTRUST_API_KEY` resolves (`load_braintrust_key()` returns a key; `braintrust_available()` is True).
- **Known blocker:** the last live sync (2026-09-06 19:41 UTC) failed on the workspace quota
  `num_scores_calendar_months` 11016/11000. Expect live writes to still fail this calendar month.
  Live failure is a **recorded outcome**, never a gate failure. See §W11.
- `requests` 2.34.2 is importable in the venv (used lazily by `btql.py` and `spend.py`). Use it; do not add it
  to `pyproject.toml` dependencies and do not add `jinja2`.

---

## 1. Design decisions (settled — build to these, do not re-litigate)

**D1. Three code artifacts, one command.**

- `dealpoint/eval/braintrust_cockpit.py` — **new**. Owns everything persistent that M7b adds via **REST**:
  the REST client, the seven views, the dashboard, topics config, the pattern, review flags, the
  human-scoring probe, the human-score push, the manifest, and `main(argv)`.
- `dealpoint/eval/braintrust_sync.py` — **extended** for §2 (judge child spans) and the `human/` namespace,
  because that is where the span tree and the score-row mapping already live and are tested. Splitting them
  would duplicate `log_hierarchy`.
- `dealpoint/eval/demo_walkthrough.py` — **new**. Generates `docs/demo-walkthrough.md` from
  `data/reports/demo_manifest.json` with a **plain Python formatter** (no jinja2 — it is not a dependency and
  §6 forbids adding one). Do not create `docs/templates/`.

`just braintrust-cockpit` runs `braintrust_cockpit.main([])`, which **first calls `braintrust_sync.sync(...)`**
(so the refined judge spans and the `human/*` scores land) and **then** creates the REST objects. This makes
the cockpit a superset of the sync and keeps "recreate everything from Git with one command" true.

**D2. Provenance is mechanical, not asserted in prose.** Every object recorded in the manifest carries
`"created_by": "<module>.<function>"`. A `gate_m7b` test walks the manifest and asserts every entry's
`created_by` names a function that actually exists (`getattr` on the imported module). That is how the
reviewer's "an object with no creating code fails review" check becomes a test rather than an opinion.

**D3. REST client seam.** In `braintrust_cockpit.py`:

```python
BRAINTRUST_API_BASE = "https://api.braintrust.dev"

class RestClient:
    """POST/GET/PATCH against the Braintrust REST API, bearer auth, lazy `import requests`."""
    def __init__(self, api_key: str, *, base: str = BRAINTRUST_API_BASE, timeout: float = 60) -> None: ...
    def request(self, method: str, path: str, *, json: dict | None = None,
                params: dict | None = None) -> dict: ...
```

Every REST call in the module goes through `client.request(...)` and nothing else — that single method is the
whole test seam. Two stand-ins:

- `_DryRunRestClient` (production offline path, mirroring `_DryRunClient`'s role and docstring style):
  records every `(method, path, json, params)` and returns plausible synthetic payloads with deterministic
  ids derived from `sha256(path + name)[:12]`, so `--dry-run` regenerates a complete, honest-shaped manifest
  with `"live": false` and every id prefixed `dryrun-`.
- `FakeRestClient` in `tests/test_braintrust_cockpit.py`: records calls and implements **name-based upsert**
  (a second `POST /v1/view` with the same `name` + `object_id` + `view_type` returns the existing id and does
  **not** append), exactly as `FakeExperiment` implements id-based upsert. Without that the idempotency test
  has no teeth.

**D4. Payloads are data, not string templates.** Views, charts, topics config and the pattern are
module-level dicts / functions returning dicts. Tests assert on the dicts, not on rendered text.

**D5. Everything offline-first.** `gate_m7b` never touches the network. Live behaviour is exercised by
`@pytest.mark.needs_network` tests that are excluded from the gate.

---

## 2. Work items

### W1 — `human/` becomes a first-class namespace (`dealpoint/eval/braintrust_sync.py`)

1. `SCORE_NAMESPACES` → `("obj", "li", "judge", "human", "deepeval")`.
2. `_normalize_score_value`: rescale `human/` on the same 1–5 → 0–1 rule as `judge/`. Change the guard to
   `if namespaced_name.startswith(("judge/", "human/")):` and update the docstring to say why (the human form
   uses the *identical* rubric, brief Appendix C).
3. `score_namespace` is unchanged — human scores are passed in **already namespaced** (`"human/reasoning"`),
   which the existing `if "/" in name: return name` branch handles. Add a one-line comment saying the bare
   dimension names are ambiguous between `judge/` and `human/`, which is why human scores arrive namespaced.
4. New helper `_human_scores_by_packet() -> dict[str, dict]` reading
   `data/eval/calibration/human_scores.jsonl` (via a new `HUMAN_SCORES_PATH` constant in `dealpoint/config.py`,
   `= CALIBRATION_DIR / "human_scores.jsonl"`), returning `{packet_id: {"human/reasoning": 4, ...}}`.
   Missing file → `{}`. **Absent entries contribute nothing; never log a zero.**
5. `_eval_experiment_plans()`: for each of the six `judge-<variant_id>` plans, merge the human dims into the
   row's `scores` when `human_scores` has that packet, and extend the plan's `score_names` with the four
   `human/*` names **only if at least one row in that plan carries them**. 4 + 4 = 8 ≤ the 12-score limit for
   an 18-case subset.
6. Also thread `packet_id` and `variant_id` into each judge row's metadata (`_judge_score_row` gains
   `packet_id` and `variant_id` parameters) — the cockpit needs them to resolve rows, and the walkthrough
   quotes them.

### W2 — Judge child spans in the replayed trace tree (§2) (`dealpoint/eval/braintrust_sync.py`)

1. New pure function:

```python
def judge_span_children(packet_id: str, judge_rows: list[dict], packet: dict | None) -> list[dict]:
    """One `judge/<family>` node per stored judge result plus one `judge/aggregate`
    node, in the `_emit_span_tree` node shape. Replay only -- no model call."""
```

   - Family → span name mapping is explicit and lowercase: `Mistral -> judge/mistral`,
     `NVIDIA -> judge/nvidia`, `ByteDance -> judge/bytedance`. Put it in a module constant
     `JUDGE_FAMILY_SPAN_NAMES: dict[str, str]` and derive the span name from it; an unknown family falls back
     to `f"judge/{family.lower()}"`.
   - Per-judge node: `input` = the blinded packet dict from `packets.jsonl` (the *whole* packet:
     `question_text, options, status, trajectory, finding, gold_span, gold_span_note`);
     `output` = `{"reasoning","evidence","trajectory","professional","notes"}` from the stored row;
     `metadata` = `{judge_model, judge_family, rubric_version, judge_price_usd, call_cost_usd, subset_hash,
     packet_id}`. `call_cost_usd` is the row's `usd`; `judge_price_usd` is the per-1M price for
     `judge_model` from `data/reports/judge_slate.json` (fall back to `None` + a
     `"judge_price_usd_source": "unavailable"` metadata key rather than inventing a number).
   - `judge/aggregate` node: `output` = mean-of-judges per dimension; `metadata` =
     `{"per_judge": {family: {dim: value}}, "rounding_rule": <M5 D5 text>, "n_judges": 3,
     "rubric_version", "subset_hash"}`. **Per-judge values are metadata, never scores.**
   - **Honesty branch (mandatory):** if `packets.jsonl` has no entry for the packet, or the stored judge row
     has `ok` false / missing dimensions, the node's metadata gets
     `{"replay_incomplete": true, "reason": "<what is missing>"}` and the input/output fields are omitted.
     Never synthesise packet text or judge JSON.
2. `_emit_span_tree` (:1680) currently only understands `case_id`, `span`, `finding`, `provenance`.
   Add handling for `input`, `output` and a plain `metadata` dict on a node (merge into `log_kwargs`,
   do not clobber existing metadata). Keep the existing `[0,1]` score guard untouched.
3. `log_hierarchy` gains `judge_children: list[dict] | None = None`; when given, they become the `scoring`
   node's `children`. The `scoring` node's own `provenance` block is unchanged.
4. In `sync()`'s replay section, for rows belonging to a **judged variant** (resolve via
   `_judged_variant_id_for(arm, model)`), compute `judge_span_children(...)` and pass them through. Replay the
   trace tree for **every judged variant's 18 cases** (108 trees) so the DoD's "for every judged variant"
   holds — these are span writes, not score writes, so the score budget is unaffected. Record
   `judge_span_traces: <n>` in the sync result dict alongside the existing `replayed_traces`.

### W3 — `braintrust_cockpit.py`: the REST core

Module docstring in the house style: what it owns, why REST rather than the SDK (views/dashboards/topics/
patterns have no SDK surface), and the M7a rule that **nothing persistent is created by an MCP tool call**.

```
RestClient                                  # D3
_DryRunRestClient                           # D3
resolve_project(client) -> dict             # GET /v1/project?project_name=dealpoint-eval
resolve_experiments(client, prefixes) -> dict[str, dict]
                                            # GET /v1/experiment?project_id=...; newest-wins by created
resolve_datasets(client) -> dict[str, dict]
upsert_view(client, *, project_id, spec) -> dict     # idempotent by (name, object_id, view_type)
upsert_dashboard(client, *, project_id, spec) -> dict
generate_permalink(client, *, object_type, object_id, ...) -> str
```

`upsert_view` is the idempotency workhorse: `GET /v1/view?object_id=...&object_type=project` (list), match on
`name` **and** `view_type`; if found → `PATCH /v1/view/{id}` with the new payload and return
`{"id": ..., "created": False}`; else `POST /v1/view` and return `{"id": ..., "created": True}`. Same shape
for the dashboard (`view_type: "monitor"`). The `created` flag is what the idempotency test asserts on.

Experiments are resolved **by name prefix, newest wins** (§6) — never created here.

### W4 — The seven saved views (§4)

`VIEW_SPECS: tuple[dict, ...]` — one dict per row of the §4 table, each carrying:
`name`, `view_type`, `caption` (the one-line description, which is also the walkthrough caption),
`columns` (where the table names them), `btql_source` (`None`, or the investigation id `1|2|4|6`), and
`btql` (**the verbatim string**, filled at build time from `btql.build_investigations()`).

```
Judged traces by variant   | experiments | tag stage=evaluation, name prefix judge-,
                             columns: variant, judge/* aggregates, obj/grounded_accuracy
Judge disagreement         | experiment (on the hero variant) | max pairwise judge spread >= 2 on any
                             dimension (from span metadata), sorted by spread desc
Retrieval rescue           | logs | btql_source 1
Failure attribution        | logs | btql_source 2
DeepEval vs judge disagreement | logs | btql_source 4
Trajectory inefficiency    | logs | btql_source 6
Review set (12)            | for_review_experiments (fall back to logs) | the 12 flagged packets
```

Build the specs through a function `view_specs() -> list[dict]` that pulls the BTQL strings from
`btql.build_investigations()` at call time, so drift is impossible by construction.

The §4 note on `stage=evaluation`: the existing sync tags eval-stage experiments `stage=eval`, not
`stage=evaluation`. **Use the tag the code actually writes (`stage=eval`)** and record the discrepancy in the
report's deviations list (§4 below). Do not retag M7a's experiments.

### W5 — The dashboard (§5)

`DASHBOARD_SPEC: dict` with `name = "DealPoint eval overview"`, `view_type = "monitor"` and five
`custom_charts` **in the spec's order**, each a dict (no string templates):

1. `obj/grounded_accuracy` by `metadata.arm`, grouped by `metadata.model`.
2. `judge/<dimension>` mean by variant — four series.
3. judge vs human per dimension on the review set. **Created unconditionally.** When no human score exists
   for a series, the chart still exists and its caption is `"pending human calibration"`.
4. `$/case` by model — `avg(metadata.secondary_diagnostics."obj/usd")`.
5. DeepEval vs `obj/` agreement rate.

Record the created view/dashboard ids in the manifest.

### W6 — Human-scoring path: probe once, record, wire the branch (§3)

```python
def probe_review_score_config(client) -> dict:
    """One attempt to create a project score config. Returns
    {"branch": "review"|"local", "http_status": int|None, "detail": str, "probed_at": iso}."""
```

- Attempt `POST /v1/project_score` for `human/reasoning` (categorical, five levels, the frozen rubric anchors
  from `eval/judges/rubrics.md` in `description`). 2xx → `branch: "review"`. 402/403/plan-limited/404 →
  `branch: "local"` with the API's own message verbatim in `detail`. Any transport error → `branch: "local"`,
  `detail` = the exception text. **Never guess; never retry into a different answer.**
- Persist the probe result to `data/reports/human_score_path.json` so the offline gate can assert which
  branch is wired without a network call. `--dry-run` reads this file and does **not** re-probe; if it is
  absent, `--dry-run` records `{"branch": "local", "detail": "probe not yet run offline"}`.
- **Branch `review`:** create the four scores `human/{reasoning,evidence,trajectory,professional}`; flag the
  12 review packets with `~__bt_review_lists: "PENDING"` and `~__bt_assignments` = the operator's user id
  (resolved by REST user lookup); add a `calibration-pull` justfile recipe +
  `pull_review_scores(client) -> int` that fetches scores back via BTQL into `human_scores.jsonl`
  **without changing its schema** and never overwriting an existing `(packet_id, scorer)` row.
- **Branch `local`:** no `calibration-pull` recipe, no review flags. `push_human_scores(client)` upserts the
  24 `human/<dimension>` scores onto the existing `judge-<variant>` experiment rows (W1 already puts them in
  the plan, so this is really "run the sync, then verify"). The walkthrough states in **one sentence** that
  review scores need a paid plan and that the local form is canonical anyway.
- Either way, **`data/reports/judges.json` remains the only place Spearman / weighted kappa / pairwise
  agreement are computed**, and the walkthrough quotes those numbers from it.

### W7 — Topics and one Pattern (§7 factory half)

- `topics_config() -> dict`: a preprocessor spec that renders a `case > agent` span as text
  (`question`, tool sequence, `final_answer`, `obj/grounded_accuracy`), one facet with the prompt
  *"what went wrong, or what made it succeed, in one sentence"*, and clustering enabled. Create it by REST.
  Record the resulting cluster names in the manifest under `topics.clusters`, and for each cluster record
  whether it maps to a `failure_attribution` cause from BTQL query 2 (`topics.maps_to_failure_attribution`).
  Offline/dry-run: `clusters: []` with `"note": "clusters are assigned server-side; empty until a live run"`.
- `pattern_spec() -> dict`: **one** pattern for the recurring trajectory-inefficiency behaviour from query 6,
  with a one-paragraph description and **≥ 3 supporting trace ids** taken from the recorded rows of query 6 in
  `data/reports/btql_investigations.json` (fall back to three case ids from the judged subset with the highest
  `obj/tool_calls`, and say so in the pattern's `evidence_source` field). Created by REST; record the pattern
  id in the manifest. Verify live with `mcp_braintrust_search_patterns`; record what you saw in the report.
- The builder **may** use `mcp_braintrust_test_preprocessor_on_trace` / `test_facet_on_trace` /
  `test_evaluator` to prototype these payloads. The object that survives must be the one this code creates.

### W8 — `data/reports/demo_manifest.json` (§6)

Written by `braintrust_cockpit.main`. New config constants `DEMO_MANIFEST_PATH`, `HUMAN_SCORE_PATH_PATH`,
`DEMO_WALKTHROUGH_PATH`, `COCKPIT_SESSION_PATH` in `dealpoint/config.py`. Shape:

```json
{
  "schema_version": 1,
  "project": {"name": "dealpoint-eval", "id": "...", "created_by": "..."},
  "experiments": {"<name>": {"id": "...", "resolved_by": "prefix:<p>", "created_by": "braintrust_sync.sync"}},
  "datasets":    {"<name>": {"id": "...", "created_by": "braintrust_sync.sync"}},
  "views":       {"<name>": {"id": "...", "view_type": "...", "caption": "...",
                             "btql": "...|null", "created_by": "braintrust_cockpit.upsert_view"}},
  "dashboard":   {"name": "DealPoint eval overview", "id": "...", "chart_ids": [...],
                  "created_by": "braintrust_cockpit.upsert_dashboard"},
  "hero_case":   {"case_id": "...", "rule": "<verbatim rule text>", "variants": ["A@haiku","D@haiku"],
                  "packet_ids": {...}, "max_pairwise_spread": 4, "spread_dimension": "evidence",
                  "candidates": [...], "human_scored": false},
  "review_set":  {"n": 12, "packet_ids": [...], "human_scored": 5},
  "human_score_path": {"branch": "...", "http_status": null, "detail": "...", "probed_at": "..."},
  "topics":  {"id": "...", "clusters": [...], "maps_to_failure_attribution": {...}, "created_by": "..."},
  "pattern": {"id": "...", "trace_ids": [...], "created_by": "..."},
  "permalinks": {"<walkthrough stop>": "<url>"},
  "cockpit_session": {"steps": []},
  "live": true,
  "git_sha7": "...", "rubric_version": "cfda9f8cc401", "subset_hash": "5918ef10a7e6",
  "synced_at": "..."
}
```

`HERO_CASE_RULE_TEXT` is a module constant, quoted verbatim into `hero_case.rule`, and states the
null-handling and tie-break exactly as §0.3 above.

`hero_case()` computes the pick from `judged_subset.json` + the two variants' `results_path` files +
`judge_scores.jsonl`. Deterministic, offline, no hardcoded case id.

### W9 — `demo_walkthrough.py` and the regenerated `docs/demo-walkthrough.md` (§6)

- `just demo-walkthrough` → `uv run python -m dealpoint.eval.demo_walkthrough`.
- Reads `demo_manifest.json` + `data/reports/{judges,four_arm,pareto,deepeval_crosscheck,btql_investigations}.json`.
- **Keeps the M7a section order** (Datasets, RAG Lab, Agent Systems, Logs trace, Scorers, Review,
  Eval of Evals, Loop/SQL, Debugger, Model Economics, Dashboard) and **keeps the role diagram verbatim**.
- **Replaces every `[M7b]` marker** (currently 6, at lines 7, 125, 159, 203, 228, 250) with either finished
  content or a `[cockpit session]` marker. **Zero `[M7b]` markers may remain.**
- Threads the hero case (`contract_32__q04`) through the six §1 stops, and states honestly that the hero case
  carries no human score yet.
- Numbers are quoted from the reports, never recomputed here.
- Footer: the "Reproduce this walkthrough" block, updated with the new recipes, plus one sentence saying
  every link above is regenerable from `demo_manifest.json`.
- **Hard cap 1,400 words excluding the appendix of links.** The M7a draft is 2,090 words — you are cutting,
  not padding. Emit the link appendix under a stable heading `## Appendix — links` so the word-count test can
  split on it deterministically.

### W10 — `docs/cockpit-session.md` and `just demo-manifest-record` (§7)

- `docs/cockpit-session.md`: a numbered procedure, exactly the five steps of §7, **written by the factory**.
  Each step names the object by its manifest key, gives the exact prompt or question to type, says what to
  save, and ends with the literal `just demo-manifest-record --step <n> --note "..."` line to run.
  State at the top that any Playground spend is the operator's, outside the ledger, recorded here.
- `demo-manifest-record` is a small `argparse` entry point in `braintrust_cockpit.py`
  (`main(["record", "--step", "2", "--note", "...", "--url", "..."])` or a dedicated
  `dealpoint.eval.demo_manifest_record` module — your call, keep it one file). It appends
  `{"step": n, "note": ..., "url": ..., "recorded_at": ...}` to `manifest["cockpit_session"]["steps"]`,
  replacing any existing entry with the same `step`, and rewrites the manifest with sorted keys.

### W11 — Live run, honestly recorded

Run `just braintrust-cockpit` **once live**. Two acceptable outcomes:

- **Success:** run it a **second** time and assert nothing new was created (every `upsert_*` returns
  `created: False`). Record the second run's diff in the report.
- **Blocked** (expected — the `num_scores_calendar_months` quota):
  record a `LIVE_RUN_STATUS`-style dict in the manifest under `live_run_status` with the API's verbatim error,
  the step it blocked at, and the resolution, exactly as `braintrust_sync.LIVE_RUN_STATUS` does. Set
  `manifest["live"] = false`. Regenerate the manifest with `--dry-run` so it is complete and shaped correctly.
  **This is not a gate failure.** State it in the report and in the walkthrough's environment-limitation note.

Never work around the quota by deleting existing Braintrust data.

### W12 — `framework_versions.json` gains the pi + extension versions

The one config-adjacent write M7b is allowed. Extend `dealpoint/eval/framework_versions.py` to record
`pi` (from `pi --version`, currently `0.84.4`) and `pi-mcp-extension` (from the extension's own manifest or
`pi extension list`; if it cannot be determined, record `null` plus a `"pi_mcp_extension_note"` explaining
why — do not guess a version). Both lookups must be exception-safe and must not make the module fail when
`pi` is absent. Run `just framework-versions` and commit the refreshed artifact **at the start of the run**.

---

## 3. Tests — `tests/test_braintrust_cockpit.py` (new, `pytestmark = pytest.mark.gate_m7b`)

Plus `gate_m7b` additions to `tests/test_braintrust_sync.py` (W1/W2) and a new
`tests/test_demo_walkthrough.py`. The `gate_m7b` marker is **already registered** in `pyproject.toml` — no
change needed there.

Required tests (one behaviour each, no placeholders):

1. `test_cockpit_is_idempotent_second_run_creates_nothing_new` — run `cockpit(FakeRestClient())` twice;
   every view/dashboard result in run 2 has `created is False`; the fake's `POST /v1/view` count equals the
   number of distinct view names, not twice it.
2. `test_seven_views_have_stable_names_and_captions` — exactly 7, names match §4 verbatim, every one has a
   non-empty caption.
3. `test_view_filters_equal_the_documented_btql` — for views 3–6, `spec["btql"]` is **identical** to the
   corresponding entry from `btql.build_investigations()` (ids 1, 2, 4, 6). Mirrors
   `test_btql_queries.py`'s doc/code-sync pattern.
4. `test_dashboard_has_five_charts_in_spec_order` — including that chart 3 exists even with no human scores
   and then carries the `pending human calibration` caption.
5. `test_hero_case_follows_the_recorded_rule` — recompute the rule from the artifacts in the test and assert
   the manifest's `hero_case.case_id` equals it; assert `max_pairwise_spread >= 2`; assert
   `hero_case.rule == HERO_CASE_RULE_TEXT`.
6. `test_judge_spans_three_children_plus_aggregate` — drive `log_hierarchy` + `_emit_span_tree` through the
   existing `FakeSpanWithChildren` for the hero packet; assert child span names are exactly
   `{judge/mistral, judge/nvidia, judge/bytedance, judge/aggregate}` under `scoring`.
7. `test_per_judge_scores_are_metadata_not_scores` — no `judge/aggregate` child logs a `scores` dict
   containing per-judge values; the per-judge numbers appear only under `metadata.per_judge`.
8. `test_judge_span_input_is_the_blinded_packet` — the per-judge span's `input` contains the packet's
   `question_text` and **does not** contain the arm, the model id, or the `variant_id`.
9. `test_missing_packet_marks_replay_incomplete` — with a packet removed, the span's metadata carries
   `replay_incomplete: True` and no invented input/output.
10. `test_human_scores_only_logged_where_present` — of the 18 rows in `judge-A@haiku`, exactly the
    human-scored packets carry `human/*`; the rest carry **no** `human/` key at all (not zero).
11. `test_score_budget_extended_for_human` — judged-variant plans declare at most 8 score names, all in
    `{judge/*, human/*}`; `assert_score_budget` and `assert_actual_score_budget` both pass; and a synthetic
    13-name plan still raises `ScoreBudgetError`.
12. `test_human_scores_are_rescaled_into_0_1` — a raw `5` becomes `1.0`, a raw `1` becomes `0.0`.
13. `test_manifest_is_complete` — every key from the §W8 shape is present and non-null (except where the
    recorded branch legitimately makes it null), `rubric_version == "cfda9f8cc401"`,
    `subset_hash == "5918ef10a7e6"`.
14. `test_every_manifest_object_names_creating_code` — D2's `created_by` walk.
15. `test_walkthrough_has_no_m7b_markers` — `"[M7b]" not in text`.
16. `test_walkthrough_word_cap` — words before `## Appendix — links` ≤ 1400.
17. `test_walkthrough_names_resolve_to_the_manifest` — every experiment/view/dataset name and every id-like
    token quoted in the doc appears in `demo_manifest.json`.
18. `test_walkthrough_numbers_match_the_reports` — the judge/human agreement figures quoted in the doc equal
    the values in `data/reports/judges.json`; the headline accuracy equals `four_arm.json`.
19. `test_human_score_path_recorded` — `data/reports/human_score_path.json` exists, `branch` is one of
    `{"review","local"}`, `detail` non-empty; and the branch actually wired matches it (branch `review` ⇒ a
    `calibration-pull` recipe exists in the justfile; branch `local` ⇒ it does not).
20. `test_ledger_unchanged_by_m7b` — no row in `data/results/spend_ledger.jsonl` has
    `milestone_tag == "m7b"`.
21. `test_frozen_artifacts_untouched` — `judged_subset.json` `subset_hash`, `versions.json`
    `rubric_version` and `dataset_version` are unchanged from the values in §0.2.
22. `test_topics_and_pattern_recorded` — manifest has a topics entry with a facet prompt and a pattern with
    ≥ 3 trace ids.
23. `test_demo_manifest_record_upserts_a_step` (tmp_path) — recording step 2 twice leaves one entry.
24. `@pytest.mark.needs_network test_cockpit_live_idempotent` — the live second-run assertion, excluded from
    the gate.

Keep every test's failure message specific enough to name the artifact that broke.

---

## 4. Deviations to report (do not silently resolve)

The report/envelope must list these. The brief wins on product decisions; the difference is *reported*.

1. **Judged scope.** Brief §2.5 specifies 40 cases × 6 variants = 240 traces and 30 hand-scored calibration
   traces. Reality on disk (frozen at M5, budget-scaled): **18 cases × 6 variants = 108 traces, 324 judge
   calls, 24 human scores**. Pre-existing and already documented in `docs/milestones/m5.md`; M7b inherits it
   and must not re-judge.
2. **`stage=evaluation` vs `stage=eval`.** §4's view filter names a tag the code does not write. Use the real
   tag `stage=eval`; report the mismatch.
3. **Hero case has no human score, and is not in the review set.** §1 stop 5 cannot show a human score for the
   hero until the cockpit session runs. Reported, shown as a gap in the walkthrough, closed by §7 step 1.
4. **Live writes may stay blocked** by the workspace `num_scores_calendar_months` quota (11016/11000 as of
   2026-09-06). Record, do not work around.
5. Anything else the build surfaces where `specs/milestones/m7b.md` and `specs/grilled-product-brief.md`
   disagree.

---

## 5. Justfile additions

```
# M7b: recreate every M7b Braintrust object (views, dashboard, topics, pattern, human scores); idempotent
braintrust-cockpit *ARGS:
    uv run python -m dealpoint.eval.braintrust_cockpit "$@"

# M7b: offline, zero-network regeneration of demo_manifest.json from current code
braintrust-cockpit-dry-run:
    uv run python -m dealpoint.eval.braintrust_cockpit --dry-run

# M7b: regenerate docs/demo-walkthrough.md from data/reports/demo_manifest.json
demo-walkthrough:
    uv run python -m dealpoint.eval.demo_walkthrough

# M7b: record one cockpit-session result: just demo-manifest-record --step 2 --note "..." --url "..."
demo-manifest-record *ARGS:
    uv run python -m dealpoint.eval.braintrust_cockpit record "$@"

# milestone 7b acceptance gates only, offline
gate-m7b:
    uv run pytest -m "gate_m7b and not needs_network" -q
```

(`calibration-pull` is added **only** under branch `review` — see W6.)

---

## 6. Verification (run all of these; judge by exit status)

```bash
just braintrust-cockpit-dry-run                 # manifest regenerates offline, zero network
just demo-walkthrough                           # doc regenerates from the manifest
uv run pytest -m "gate_m7b and not needs_network" -q
uv run pytest -m "not needs_network and not needs_model" -q
uv run ruff check .
uv run pyright
just framework-versions                         # pi + extension versions recorded
```

Then the live attempt (W11), then `just braintrust-cockpit-dry-run` once more so the committed manifest is
consistent with the committed code.

---

## 7. Guardrails

- **No model calls. No metered spend. No new dependency.** If you find yourself wanting `jinja2`, write a
  Python formatter instead.
- **Never edit** `specs/grilled-product-brief.md`, `specs/milestones/m5.md`, `specs/milestones/m7a.md`,
  `specs/milestones/m7b.md`, `~/.pi/**`, `~/.claude.json`, or `adws/adw_sssf_config/sssf.config.yaml`.
- **Never rerun** M1–M7a: no re-judging, no re-sweeping, no regenerating the frozen subset, the index, the
  rubric, `judge_scores.jsonl` or `human_scores.jsonl` content.
- **Never invent a Braintrust object by clicking.** MCP tools are for looking, testing and prototyping only;
  the surviving object is the one `braintrust_cockpit.py` creates.
- **Never invent data.** A missing packet, a failed judge call, an unresolvable id, an empty cluster list —
  each is recorded as what it is. Empty is honest; fabricated is a defect.
- Out of scope: the cockpit session itself, product UI, the MCP layer, new judges, new cases, new metered runs.
