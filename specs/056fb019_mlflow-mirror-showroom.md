# M9 build plan — the MLflow mirror of the Braintrust showroom

**Requirement document:** `specs/grilled-product-brief.md` (authority).
**Bounding spec:** `specs/milestones/m9.md` (read-only; do not edit either file).
**Milestone registration:** already present at `adws/adw_modules/milestones.py:72-73` — `m9`,
`gate_marker="gate_m9"`, `needs_model=True`, and `m9` is **not** in `DRAFT_MILESTONES`. Nothing to add there.

M9 is additive: nothing from M1–M8 is rerun, retuned or re-judged. Every MLflow object is built from
stored rows on disk through the existing pure functions. Braintrust is not read, not written, not
credentialed.

---

## 0. Environment facts, verified on 2026-09-07 against a real MLflow 3.16.0 install

These were measured, not assumed. Several correct the spec. **Trust this section over the spec's §0.3
where they differ**, and record the differences in the report (§11).

### 0.1 Confirmed working against a bare `sqlite:///<tmp>/m.db` tracking URI, with **no server running**

A full smoke of the M9 surface passed offline against SQLite alone:

| capability | verdict |
|---|---|
| `create_experiment`, experiment description via tag `mlflow.note.content` | works |
| `create_run` with tags/params/metrics, run description via tag `mlflow.note.content` | works |
| `search_runs(filter_string="tags.\`dealpoint.key\` = '...'")` | works (backtick-quote dotted tag keys) |
| traces with **original timestamps** and nested spans | works — see §0.2 |
| `mlflow.log_feedback` / `mlflow.log_expectation`, sources `CODE` / `LLM_JUDGE` / `HUMAN` | works, all four read back off `trace.info.assessments` |
| `mlflow.search_traces(filter_string="tags.\`dealpoint.key\` = '...'")` | works |
| `mlflow.genai.create_dataset` + `dataset.merge_records(...)` | works; **`merge_records` is content-idempotent** (merging the same records twice left the record count at 2) |
| `mlflow.genai.create_dataset` with a name that already exists | does **not** raise; returns the dataset |
| `mlflow.genai.register_prompt` text form and chat form (`[{"role":..,"content":..}]`), `set_prompt_alias` | works |
| `@mlflow.genai.scorer` functions, callable directly in-process | works |
| `mlflow.genai.evaluate(data=<search_traces pandas df>, scorers=[...])` with `predict_fn=None` | works; returns a run id and `{"<scorer>/mean": ...}` metrics |
| `create_registered_model` / `create_model_version` / `set_registered_model_alias` / version description | works |
| `mlflow.genai.make_judge(..., model="openai:/<model>", base_url=..., extra_headers=...)` | constructs fine with **no network call** |

### 0.2 Original timestamps: use the **client** API, not the module-level one

The spec says `mlflow.start_span(..., start_time_ns, end_time_ns)`. That is wrong — the module-level
`mlflow.start_span` has no timestamp parameters. The working API is on `MlflowClient`:

```python
root = client.start_trace(name, span_type=..., inputs=..., tags=..., experiment_id=exp_id,
                          start_time_ns=t0, run_id=run_id)          # t0 from the row's t_ms
child = client.start_span("tool.search", trace_id=root.trace_id, parent_id=root.span_id,
                          span_type="TOOL", inputs=..., start_time_ns=...)
client.end_span(root.trace_id, child.span_id, outputs=..., end_time_ns=...)
client.end_trace(root.trace_id, outputs=..., end_time_ns=...)
```

Verified round-trip: `trace.info.request_time` came back as the injected start (ms) and
`trace.info.execution_duration` as exactly the injected span (9000 ms). This is the real "MLflow adds
what Braintrust could not" claim in D5 — keep it, it is true.

**Traces are written asynchronously.** Immediately after `end_trace`, `mlflow.get_trace(...)` returned
`None` and `search_traces` returned 0. Call `mlflow.flush_trace_async_logging()` after each batch, and
before any count/read. Every test that counts traces must flush first. This is the single most likely
cause of a flaky gate — do it in the step function, not in the test.

`search_traces(experiment_ids=[...])` emits a `FutureWarning`; the non-deprecated spelling is
`locations=[exp_id]`. Use `locations=`. Note `search_traces` paginates — pass `max_results` large enough
(or page) when counting 700+ traces.

### 0.3 Two features need packages MLflow does not pull in

- **D3 judge alignment requires `dspy`.** `judge.align()` → `get_default_optimizer()` →
  `import mlflow.genai.judges.optimizers.memalign`, and that package's `__init__` eagerly imports the
  GEPA→DSPy chain, which raises `MlflowException("DSPy library is required but not installed")`.
  Importing the MemAlign submodule alone does **not** avoid it. Verified by direct import.
- **D4 GEPA requires `gepa>=0.0.26`** (`GepaPromptOptimizer` raises with that exact install hint).
  `MetaPromptOptimizer` needs no extra package; it is the fallback if `gepa` is unavailable.
- `GepaPromptOptimizer(reflection_model=..., max_metric_calls=100, ...)` — `max_metric_calls` is the
  natural cap knob for D4's budget.
- `mlflow.genai.optimize_prompts` signature is keyword-only:
  `optimize_prompts(*, predict_fn, train_data, prompt_uris, optimizer, scorers=None, aggregation=None)`.

**Decision:** two extras, so the offline gate stays light and D4's "skipped, not failed, when the extra
is not installed" language in the spec stays truthful.

```toml
mlflow = ["mlflow>=3.16"]
mlflow-optimize = ["dspy>=3.3", "gepa>=0.1"]
```

`uv lock` with the existing extras plus both new ones resolved cleanly (195 packages, `mlflow==3.16.0`);
no conflict with `braintrust`, `deepeval`, `rag-lab`, `api`. Verified in a scratch copy of `pyproject.toml`.

### 0.4 There is no chart-view / dashboard API in OSS MLflow 3.16

`MlflowClient` exposes no chart, view or dashboard method, and the tracking protos contain no chart
entity. The spec anticipated this. Take the documented branch: the `views` step writes
`data/reports/mlflow_views.json` and `docs/mlflow-tour.md` says "select tag `axis=<q>`, chart `<metric>`".
Do not invent an endpoint.

### 0.5 Write throughput — the gate test budget

Measured against SQLite: **~54 ms per 5-span trace** and **~5 ms per assessment**. A full live sync is
therefore ~40 s of traces + ~37 s of assessments ≈ 80 s, and an idempotency test that does it twice is
~3 minutes. SQLite grew to 1.3 MB per 50 traces → ~20 MB for the whole mirror, comfortably inside the
200 MB `data/mlflow/` guard. See §9 for how the tests are split so the offline suite stays fast.

---

## 1. Measured counts — the spec's numbers, corrected

Every number below was measured today by calling the repo's own loaders. **Assert the measured values.**
Where the spec claims something else, that is a spec/reality difference to *report*, not to force.

| thing | spec says | **measured** | how it was measured |
|---|---|---|---|
| result files | 32 | **32** ✓ | `ls data/results/*.jsonl \| wc -l` |
| `stored_row_index()` entries | (implied 266) | **426** | dual-keyed: every row is indexed under both `D@glm` and `D@z-ai/glm-5.3-flash`. It is a lookup index, **not** a trace count. Do not derive counts from `len()` of it. |
| judged plan traces | — | **108** | 6 variants × 18 cases from `_judged_subset()` |
| representative picks | 6 | **6** ✓ | `representative_cases()["selections"]` |
| sweep traces | — | **151** | `agent_run_log_plan(already)` where `already` = judged ∪ reps |
| **agent traces total** | 266 | **265** | 108 + 6 + 151 |
| retrieval traces | 406 | **406** ✓ | `retrieval_log_rows()` (348 tournament canonical rows, 58 `li_native_bm25` rows) |
| prompt-variant traces | 67 | **67** (keys only — see §2) | 67 `prompt:<variant>:<case>` keys in `data/reports/orgs/bishal-ai/braintrust_score_ledger.jsonl` |
| **root traces total** | 739 | **738** | 265 + 406 + 67 |
| `judge_packet_rows()` | 108 | **24** | it returns the lawyer-scored packets, not all packets |
| `playground_rows()` | 4 × 18 | **18** | one row per case; the four variants multiply it |
| blinded packets | 108 | **108** ✓ | `data/eval/calibration/packets.jsonl` |
| judge calls | 324 | **324** ✓ | `data/eval/judge_scores.jsonl`; 108 packets × 3 families; each line carries all four dims inline (`reasoning`, `evidence`, `trajectory`, `professional`) and a `judge_family` field → **1296 LLM_JUDGE feedbacks** |
| lawyer rows | 24 × 4 | **24 rows × 4 dims = 96 HUMAN feedbacks** ✓ | `data/eval/calibration/human_scores.jsonl` (fields `reasoning, evidence, trajectory, professional`, plus `scorer`, `scored_at`, `notes` for provenance) |
| DeepEval traces | 108 × 4 | **108 × 4** ✓ | `deepeval_crosscheck.json["per_trace_scores"]`; metrics `task_completion`, `tool_correctness`, `argument_correctness`, `step_efficiency` |
| tournament | 58 × 6 | **58 × 6 = 348** ✓ | `tournament.json["per_case"]`, `query_type == "canonical"` |
| LlamaIndex | 58 × 7 | **7 configs** ✓ | `li_rag_eval.json["retrievers"]` |
| case sets | 58 / 167 / 40 / 106 | **58 / 167 / 40 / 106** ✓ | `wc -l` on `data/eval/{dev,test,counterfactual,synthetic_dev_queries}.jsonl` |
| judged subset | 18 | **18 cases × 6 variants** ✓ | `judged_subset.json` |
| **Braintrust experiments** | 35 | **29** in `braintrust_sync.json["experiments"]` | + 4 `playground-arm-A-<variant>` from `playground_prompts()` = **33** |
| **Braintrust datasets** | 7 | **5** in `braintrust_sync.json["datasets"]` | `dev, test, counterfactual, judged_calibration, synthetic_query` |

**Consequence for D2's `runs` step:** build the run set from a single explicit, committed source —
`braintrust_sync.json["experiments"]` (29) plus the four `playground-arm-A-<variant>` names derived from
`playground_prompts()` — for **33 runs**, and assert 33. Do not hardcode 35. Put the derivation in one
pure function `mirror_run_plan()` so the test and the step agree by construction.

**Consequence for D2's `datasets` step:** the spec's nine names are the M9 deliverable regardless of how
many Braintrust has. Build **9** datasets: `maud-dealpoint-{dev,test,counterfactual,judged_calibration,
synthetic_query,review-set,playground-armA,judge-packets}` + `maud-dealpoint-retrieval-dev`.

**The spine and its twin** (used by the equality tests, per `docs/demo-tour.md`): spine
`contract_144__q05` at `D@glm`; redacted twin **`contract_39__redacted_q05`**.

---

## 2. The one genuine blocker, and its resolution

**The 67 prompt-variant rows' content is not on disk.** `prompt_variant_log_rows(events_by_variant)`
takes *live Braintrust events* as its argument — `step_promptlogs` fetches them via
`api.fetch_rows(...)`. The only thing on disk is the 67 ledger keys
(`{"experiment": "logs", "key": "prompt:base:contract_75__oos09", ...}`). The model outputs and the
Playground judge scores were never mirrored to `data/reports/`, contrary to spec §0.1's
"mirrored to `data/reports/orgs/bishal-ai/` manifest — 4 x 18 rows, judge scores"; that manifest holds
only `mode/started_at/project/project_id/org/env_file/ledger/mirror/finished_at`.

**Resolution (offline, deterministic, honest):**

1. Build the 67 prompt-variant traces from the ledger keys — they give the exact `(variant, case_id)`
   set — joined to `playground_rows()` for the input, gold answer and case metadata.
2. Log them with tags `category="prompt-variant"`, `prompt_variant=<variant>`, `case_id`,
   `system_label=SYSTEM_LABEL["A"]`, `model_label="glm"`, and **no output and no judge assessments**,
   because neither exists on disk.
3. Record the gap explicitly: a `prompt_variant_outputs: "absent"` note in
   `data/reports/mlflow_manifest.json`, and one line in `docs/mlflow-tour.md`'s gaps section: the arm-A
   Playground outputs and judge scores live only in Braintrust; MLflow mirrors the row set, not the text.
4. Do **not** add a Braintrust fetch path. Spec §4.3: "No Braintrust credentials are read."

This keeps the 67 count (which the tests assert) truthful and keeps D4's fifth-row comparison honest:
D4 scores the optimized prompt with the evidence judge and reports it against whatever the Git reports
already hold for the four hand-written variants, naming the asymmetry rather than papering over it.

---

## 3. Files to touch

**New**
- `dealpoint/eval/mlflow_mirror.py` — the whole D2 sync (steps, CLI, manifest).
- `dealpoint/eval/mlflow_judges.py` — D3 alignment + the `make_judge` constructors + the six
  `@mlflow.genai.scorer` wrappers. Split out so the offline scorer/judge tests do not import the sync.
- `docs/mlflow-tour.md` — D5.
- `tests/test_mlflow_mirror.py` — D6 (sync steps, counts, idempotency, spine equality).
- `tests/test_mlflow_judges.py` — D6 (scorers, judge construction, cap/`--live` refusal).
- `data/reports/mlflow_manifest.json`, `data/reports/mlflow_views.json` — committed artifacts.
- `data/reports/judge_alignment.json` — D3 artifact.

**Edited**
- `pyproject.toml` — the two extras (§0.3) and the `gate_m9` marker.
- `justfile` — five recipes (§8).
- `.gitignore` — add `data/mlflow/` next to the existing `data/index/` (line 30).
- `dealpoint/config.py` — paths/constants only if the house style requires them there (the M7a disk
  guard constants live there); otherwise keep M9 constants module-local.

**Never touched:** `specs/grilled-product-brief.md`, `specs/milestones/m9.md`, anything under
`dealpoint/agent/`, the benchmark, the rubric, any frozen artifact, any Braintrust module.

---

## 4. `dealpoint/eval/mlflow_mirror.py`

### 4.1 House style to copy

Copy `braintrust_showroom.py`'s shape exactly (read `main()` at `braintrust_showroom.py:1104` and
`step_promptlogs` at `:743` before writing):

- module-level `STEPS = ("datasets", "prompts", "runs", "traces", "assessments", "scorers",
  "evaluations", "registry", "views")` in run order;
- one `def step_<name>(client, live: bool, manifest: dict) -> None` per step, dispatched with
  `globals()[f"step_{name}"](client, live, manifest)`;
- `main(argv)` parses `--live`, `--only a,b`, `--limit N` by hand from `argv` (no argparse — the
  showroom does not use it), prints `LIVE`/`DRY RUN` on the first line, writes the manifest at the end
  when live, returns an int, and `if __name__ == "__main__": sys.exit(main())`;
- every step prints its counts in dry run and writes nothing.

**Lazy import.** `import mlflow` at module scope would break the offline suite for anyone without the
extra. Follow the `braintrust` pattern: import inside the function that needs it. Provide one helper:

```python
def _client(tracking_uri: str | None = None):
    import mlflow
    from mlflow import MlflowClient
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient()
```

Everything that computes *what* to write must be a pure function importable without `mlflow`, so the
dry-run count tests never need the extra.

### 4.2 Keys and idempotency

Deterministic key on every object, searched before writing:

| object | key | idempotency check |
|---|---|---|
| run | tag `dealpoint.key` = the Braintrust experiment name | `search_runs([exp_id], filter_string="tags.\`dealpoint.key\` = '<name>'")`; update tags/params/metrics in place, never create a second |
| trace | tag `dealpoint.key` = `f"{case_id}:{variant}:{category}"` (retrieval: `f"{case_id}:{retriever}:retrieval"`; prompt-variant: `f"{case_id}:{variant}:prompt-variant"`) | `search_traces(locations=[exp_id], filter_string=...)` after `flush_trace_async_logging()` |
| dataset | name | `create_dataset` is safe to call again; `merge_records` is content-idempotent (§0.1) |
| prompt | name + content hash | **`register_prompt` with an identical template still bumps the version** (measured: 1 → 2). You must read the existing versions, compare the template, and skip when unchanged. This is the one place where MLflow will silently break idempotency if you trust it. |
| model version | registered model name + `config_hash` tag | search versions, compare the tag |

Assessments: keyed by `(trace_id, assessment name)`. Read `trace.info.assessments` and skip names
already present, or the second `--live` doubles ~8000 rows.

### 4.3 The steps

**`datasets`** — 9 evaluation datasets (§1). Records as
`{"inputs": {...}, "expectations": {...}, "tags": {...}}`. Sources: `braintrust_sync.dataset_rows` for
the five case sets, `review_set()` for `review-set`, `playground_rows()` for `playground-armA`,
`judge_packet_rows()` for `judge-packets` (24 with lawyer expectations), and the 58 dev queries with
gold span ids as expectations for `retrieval-dev`.

**`prompts`** — 8 prompts, chat form where the Braintrust one is chat. Sources: `prompt_plan`,
`playground_prompts()` (the four arm-A variants), `judge_run._system_prompt`, `dealpoint/eval/rubric.py`,
`dealpoint/agent/prompts.py`, `skills/ma-deal-point-review`. Aliases: `baseline` → arm-A base,
`best` → cite-first, `judge-panel`, `rubric`, `skill`.

**`runs`** — 33 runs (§1) in experiment `dealpoint-eval`. Tags = `classify_experiment(name)`'s factorial
schema (`axis`, `varies`, `holds`, `arm`, `loop`, `retriever`, `skill`, `model`, `cases`) +
`dealpoint.key`. Params from `arm_parameter_sets()` + model. Metrics: every aggregate the Braintrust
dashboards chart — compute them from the same `metadata_mirror` fields the toplists average, so the
numbers match by construction:

| metric | from |
|---|---|
| `obj/<name>` for the six deterministic scorers | mean of the row's `scores[<name>]` over the run's pool |
| `safe_accuracy`, `net_accuracy`, `misleading_rate`, `silent_failure_rate` | mean of mirror `safe`, `net_accuracy`, `misleading`, `silent_failure` |
| `precision_when_answering` | `sum(correct_answered) / sum(answered)` |
| `cap_hit_rate`, `fabrication_rate` | mean of `cap_hit`, `fabrication` |
| `usd_per_case`, `wall_s_p50`, `wall_s_p90`, `correct_per_dollar` | mirror `usd`, `wall_s`, `sum(correct_all)/sum(usd)` |
| `judge/*`, `human/*`, `li/*`, `deepeval/*` | the mirror's per-judge and cross-check fields |

Run description (tag `mlflow.note.content`) = that run's line from the cockpit verdicts. The verdict
text lives in `braintrust_cockpit.DASHBOARDS` (`braintrust_cockpit.py:485`) — **reuse those strings
verbatim**, do not retype them, so D5's "no number that differs from `docs/demo-tour.md`" holds
automatically. Chart titles and measures live in `chart_catalogue()` (`:415`).

**`traces`** — 738 traces (§1), each under its run, with original timestamps from the row's `t_ms` /
`wall_ms` (§0.2). Reuse the span-tree plan `_log_one_tree` builds (`braintrust_showroom.py:615`); the
nesting and per-span inputs/outputs must match the Braintrust replay. Trace tags = `metadata_mirror(row)`
flattened to strings (MLflow tags are strings; keep the numeric values as assessments and metrics, and
the mirror as tags for filtering). Flush after each batch.

**`assessments`** — on each trace:
- expectations `gold_answer`, `gold_span` (source HUMAN, `source_id="gold"`);
- CODE feedback for each of the six `obj/*` scores, `source_id="dealpoint.eval.scorers"`;
- CODE feedback for the four DeepEval metrics, `source_id="deepeval"`;
- LLM_JUDGE feedback `judge/<family>/<dim>` for each of the 324 judge rows × 4 dims (1296 total), with
  the row's `notes` / `failure_detail` as `rationale` and `source_id` = the judge model;
- HUMAN feedback `human/<dim>` for the lawyer's 24 packets × 4 dims (96 total), with `scorer` and
  `scored_at` in `metadata`;
- retrieval traces: CODE `hit@5`, `hit@10`, `mrr` on the 348 scored rows, and `li/hit_rate`, `li/mrr`
  where `llamaindex_per_case()` has them.

**`scorers`** — the six deterministic scorers as `@mlflow.genai.scorer` functions calling
`dealpoint.eval.scorers` (they run in-process here; Braintrust could not run them server-side — say so
in the manifest and the tour), plus the four `make_judge` judges on OpenRouter
(`openai:/mistralai/mistral-small-3.2-24b-instruct`, `openai:/bytedance-seed/seed-2.0-mini`,
`base_url="https://openrouter.ai/api/v1"`), instructions = `judge_scorer_messages(dim)` rendered into
the `{{ inputs }} / {{ outputs }} / {{ expectations }}` template. Construction only — never called
offline.

**`evaluations`** — one `mlflow.genai.evaluate(data=<traces of that pool>, scorers=<the six>,
predict_fn=None)` per axis pool. `data` is the pandas frame from
`mlflow.search_traces(locations=[exp_id], filter_string=..., return_type="pandas")`.

**`registry`** — registered model `dealpoint-agent`, one version per system@model with its metrics,
params and a `config_hash` tag; aliases `champion` → D@gemini, `baseline` → A@haiku,
`cost-floor` → D@qwen; version descriptions carry the verdict sentence.

**`views`** — experiment description (scoring definitions + verdicts, reused from `DASHBOARDS`), and
`data/reports/mlflow_views.json` translating `chart_catalogue()` into
`{question, tag_filter, metric, title}` entries. No chart API exists (§0.4); the tour documents the
manual route.

### 4.4 Manifest

`data/reports/mlflow_manifest.json` on live runs: per-step counts and ids, `tracking_uri`,
`mlflow_version`, `prompt_variant_outputs: "absent"` (§2), and the deltas from the spec's claimed
counts so the record is self-explaining. Committed.

### 4.5 Disk guard

`data/mlflow/` is gitignored. Reuse `dealpoint/eval/disk_guard.py`'s `free_bytes()` /
`record_guard(...)` shape for the new extras; assert the measured `data/mlflow/` footprint stays under
200 MB (§0.5 says to expect ~20 MB).

---

## 5. D3 — `just mlflow-align-judges` (spend-gated, cap $1.50)

Requires the `mlflow-optimize` extra (`dspy`, §0.3). If it is absent, print why and exit non-zero for
`--live`, zero for a dry run.

1. Build the four judges (§4.3 `scorers`).
2. `judge.align(traces)` over the 24 lawyer-scored traces, the HUMAN feedback as target.
3. Re-score the 108 judged traces with each aligned judge; log `judge/<dim>/aligned` assessments.
4. A run `judge-alignment` whose metrics are closeness-to-lawyer before and after, per dimension and per
   family, using **exactly** `metadata_mirror`'s `judge_*_closeness` definition — import it, do not
   reimplement it.
5. Write `data/reports/judge_alignment.json`.
6. One paragraph in `docs/mlflow-tour.md`: did alignment move **trajectory**, the dimension no judge gets
   right? Report the measured answer, whichever way it falls.

**Guards.** Dry run by default; `--live` required. Print call count and dollar estimate before the first
call. Enforce with `dealpoint.eval.spend.assert_within_cap(est_usd)` (`spend.py:562`) — it already raises
`SpendCapError` when the cap is unset or would be exceeded — plus the milestone's own $1.50 ceiling.
Spend lands in `data/results/spend_ledger.jsonl` via `OpenRouterClient(milestone_tag="m9")`
(`dealpoint/llm/client.py:213`), with `purpose="judge_alignment"`.

## 6. D4 — `just mlflow-optimize-prompt` (spend-gated, cap $1.00)

`optimize_prompts(predict_fn=..., train_data=<the 18 playground-armA rows>,
prompt_uris=["prompts:/arm-a-prompt-base/<v>"], optimizer=GepaPromptOptimizer(reflection_model=...,
max_metric_calls=<derived from the cap>), scorers=[<evidence judge>])` — keyword-only (§0.3). Register
the result as a new version of `arm-a-prompt-base` with alias `optimized`, and score it once with the
evidence judge.

Use `max_metric_calls` as the hard budget knob; print the estimate before the first call; same
`assert_within_cap` guard; `purpose="prompt_optimization"`, `milestone_tag="m9"`.

**Skip, do not fail**, when the cap would be exceeded or `gepa`/`dspy` is missing — print the reason and
return 0. If `gepa` is unavailable but `dspy` is, `MetaPromptOptimizer` is a valid substitute; say in the
report which optimizer actually ran.

Per §2, the four hand-written variants' judge scores are not on disk, so the "fifth row" comparison is
reported against the Git reports and the asymmetry is named in the tour.

---

## 7. D5 — `docs/mlflow-tour.md`

Nine stops mirroring `docs/demo-tour.md` (Stop 1 the question and the trap; 2 retrieval; 3 A→D; 4 watch
one agent think; 5 deterministic truth and production scoring; 6 human review, three judges, where they
disagree; 7 iterate on the prompt; 8 diagnose the real failure; 9 what should we deploy), one paragraph
each, pointing at the MLflow surface for the same evidence — Datasets tab, the `retrieval` evaluation run,
the Evaluations comparison of the four GLM runs on `contract_144__q05`, the Traces tab filtered
`tags.case_id = 'contract_144__q05'`, the in-process scorers, the assessment columns and the
judge-alignment run, the prompt registry and the `optimized` version, `search_traces` for
`tags.status = 'CAP_HIT'` and the redacted twin `contract_39__redacted_q05`, and the `Which model?` run
comparison with the `champion` alias.

Header: the tracking URI / proxy URL placeholder, and a link to `docs/demo-tour.md` (and add the reverse
link in `demo-tour.md`'s header — that is a one-line edit, and `demo-tour.md` is not read-only).

**"What MLflow adds"** — in-process deterministic scorers, real span timestamps, judge alignment, prompt
optimization, the model registry. **"What only Braintrust has here"** — monitor dashboards over logs,
online scoring, Topics, Patterns, Loop, Playground, labeling sessions; plus the two M9-specific gaps: no
chart-view API (§0.4) and no prompt-variant outputs on disk (§2). One line each.

**Every number must come from the same source as `demo-tour.md`.** Reuse the `DASHBOARDS` verdict
strings rather than retyping figures.

---

## 8. `justfile` recipes

Match the existing `braintrust-*` block (`justfile:219-256`) — pass args straight through.

```
mlflow-server:
    mlflow server --backend-store-uri sqlite:///data/mlflow/mlflow.db \
      --artifacts-destination data/mlflow/artifacts --host 127.0.0.1 --port 5000

mlflow-sync *ARGS:
    uv run python -m dealpoint.eval.mlflow_mirror "$@"

mlflow-sync-dry-run:
    uv run python -m dealpoint.eval.mlflow_mirror --dry-run

mlflow-align-judges *ARGS:
    uv run python -m dealpoint.eval.mlflow_judges align "$@"

mlflow-optimize-prompt *ARGS:
    uv run python -m dealpoint.eval.mlflow_judges optimize "$@"

mlflow-score-new *ARGS:
    uv run python -m dealpoint.eval.mlflow_judges score-new "$@"
```

`mlflow-score-new` is the documented batch stand-in for the Databricks-only online scoring: run the
judges over traces newer than the last run. It is spend-gated the same way; it may be a thin
`--since`-filtered path over D3's re-scoring code.

---

## 9. Tests and the `gate_m9` gate

Add `"gate_m9: milestone 9 gate"` to `[tool.pytest.ini_options] markers` in `pyproject.toml`.

House style (see `tests/test_braintrust_showroom.py`): module-level `pytestmark`, offline, no network.
`tests/conftest.py` gives `dataset_available` / `require_dataset` and a fast-retry autouse fixture; no
network blocking fixture exists, so keep the tests genuinely offline by never constructing a live client.

Skip cleanly when the extra is missing: `pytest.importorskip("mlflow")` at module scope.

**Fixture:** a session-scoped tmp SQLite tracking URI.

```python
@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory):
    d = tmp_path_factory.mktemp("mlflow")
    return f"sqlite:///{d}/m.db"
```

Real client, no fake, no server — proven to work in §0.1.

**Required tests**

1. **Dry-run counts, exactly** (pure, no `mlflow` needed for the planning functions): 33 runs,
   9 datasets, 8 prompts, 738 traces (265 agent + 406 retrieval + 67 prompt-variant), and assessment
   counts 6 × (agent rows) CODE for `obj/*`, 1296 LLM_JUDGE, 96 HUMAN, 432 DeepEval CODE (108 × 4),
   1044 retrieval CODE (348 × 3) + LlamaIndex `li/*` where present. Derive each expected number in the
   test from the same pure function the step uses, then assert the literal — so a data change fails
   loudly instead of silently agreeing with itself.
2. **Live + idempotency.** `--live` into the tmp store creates the objects; a second `--live` creates
   nothing new (count runs, traces, dataset records, prompt versions before and after). Given §0.5's
   timings, run the trace-heavy steps under `--limit` for this test and the cheap steps
   (`datasets`, `prompts`, `runs`, `registry`, `views`) at full size. Flush before every count.
   The prompt-version check is the one most likely to fail (§4.2) — assert it explicitly.
3. **Spine and twin equality.** For `contract_144__q05` @ `D@glm` and `contract_39__redacted_q05`, every
   assessment value on the MLflow trace equals the `metadata_mirror` value from the same stored row:
   `ga_scored`, `net_accuracy`, `safe`, `judge_mistral_evidence`, `human_professional`, DeepEval task
   completion.
4. **The six scorers.** Each, called as an MLflow scorer on the spine trace, returns the stored row's
   value.
5. **Judge construction.** The four judges are built with `base_url="https://openrouter.ai/api/v1"` and
   the rubric dimension text from `judge_scorer_messages(dim)`, and are never invoked in the offline
   suite.
6. **Guards.** `align` and `optimize` refuse without `--live`, print an estimate, and stop above the cap
   (`SpendCapError`). Test with a monkeypatched cap; no network.
7. **Docs.** `docs/mlflow-tour.md` names every MLflow-unique feature and every named gap, has the nine
   stops, and contradicts no number in `docs/demo-tour.md`.

---

## 10. Verification — run these, judge by exit status

```bash
uv sync --extra mlflow --extra mlflow-optimize
uv run pytest -m "gate_m9 and not needs_network" -q
uv run pytest -m "not needs_network and not needs_model" -q
uv run ruff check .
uv run pyright
```

`mlflow` ships `py.typed`, so pyright resolves it once the extra is installed.

Then the D2 sequence against the real server (operator has it on `127.0.0.1:5000`):
`just mlflow-sync` (dry run) → `just mlflow-sync --live` → `just mlflow-sync --live` again; identical
counts, zero new objects the second time; commit `data/reports/mlflow_manifest.json`.

Then D3 and D4, each within its cap, each recorded in `data/results/spend_ledger.jsonl` with
`milestone_tag: m9`.

---

## 11. Report, do not silently resolve

Surface all of these in the build report:

1. **Counts.** 33 runs not 35; 738 traces not 739; 265 agent traces not 266; 5 Braintrust datasets
   recorded, not 7 (§1).
2. **The prompt-variant gap** (§2) — the spec's §0.1 claim that the Playground judge scores were
   mirrored to disk is not borne out; only 67 ledger keys exist.
3. **`mlflow.start_span` has no timestamp parameters** — the client API is required (§0.2).
4. **`dspy` and `gepa` are needed** for D3 and D4 and are not MLflow dependencies (§0.3); two extras were
   added rather than one.
5. **No chart-view API** in OSS 3.16 (§0.4); the `views` step took the documented fallback.
6. **`register_prompt` is not content-idempotent** (§4.2); idempotency is enforced in our code.
7. **Brief vs spec.** `specs/grilled-product-brief.md` §2.7 makes the observability product a surface and
   never the source of truth — a second cockpit is fully consistent with it, and M9 changes no number
   that the Git reports decide. But the brief's §3 "Not used" row lists *DeepEval,
   OpenTelemetry/OpenInference*; M7a already superseded the DeepEval half, and M9 adds an
   OpenTelemetry-shaped tracing surface. The brief wins on product decisions, and nothing here makes
   MLflow a source of truth — but the "Not used" row is now stale in two places. Report it; do not edit
   the brief.

---

## 12. Addendum — three conventions confirmed after the main plan was written

1. **`gate_m9` is not registered.** `[tool.pytest.ini_options].markers` in `pyproject.toml` stops at
   `gate_m7b`; neither `gate_m8` nor `gate_m9` is there. There is no `--strict-markers`, so an
   unregistered marker only warns — but every prior milestone registered its own. Add
   `"gate_m9: milestone 9 gate",` as part of this build (already required by §9).
2. **No existing test validates tour content.** Nothing under `tests/` references `demo-tour`, so the
   D5 documentation test (§9, test 7) is new ground rather than a pattern to copy. Keep its assertions
   concrete — feature names, gap names, the nine stop headings — rather than fuzzy prose matching.
3. **`specs/mvp/state.json` is factory-managed; do not edit it.** It already carries
   `current_milestone: "m9"` and `next_action: "m9 running in session 056fb019 (attempt 1)"`, written by
   the milestone runner. The builder touches neither it nor `adws/adw_modules/milestones.py`.
4. **`JUDGE_MODELS` is keyed by dimension, not by judge family.**
   `braintrust_showroom.py:63` maps `reasoning`, `evidence` and `professional` to
   `mistralai/mistral-small-3.2-24b-instruct` and `trajectory` to `bytedance-seed/seed-2.0-mini`.
   Build the four `make_judge` judges from that mapping directly (one per dimension in
   `JUDGE_DIMS = ("reasoning", "evidence", "trajectory", "professional")`), prefixing each with
   `openai:/`. Do not assume a family→model mapping; the judge *families* in
   `data/eval/judge_scores.jsonl` (`judge_family`) are a separate, three-valued axis used for the
   `judge/<family>/<dim>` assessment names in §4.3.
5. **`classify_experiment`'s `_ARM` / `_JUDGED` regexes hardcode the index version `e2b4a2b97561`.**
   That is fine for the 29 committed experiment names, but any name that does not match falls through
   to a different branch. Assert in the `runs` test that all 33 planned names classify into a known
   `axis` rather than a fallback, so a silently unclassified run cannot reach the mirror.
6. **Dry run is the default and `--dry-run` always wins.** `braintrust_sync.main()` encodes it as
   `dry_run = "--live" not in argv or "--dry-run" in argv`. Copy that precedence exactly — the
   `just mlflow-sync-dry-run` recipe in §8 passes `--dry-run`, so the module must accept the flag and
   honour it even when `--live` is also present. `dealpoint/cli.py` is not an umbrella CLI (it has only
   the `data` subcommand); every eval module is its own `python -m` entry point with a module-level
   `main(argv: list[str] | None = None) -> int` and `if __name__ == "__main__": sys.exit(main())`.
   `mlflow_mirror.py` follows the hand-scanned-argv family (`braintrust_sync`, `braintrust_cockpit`,
   `braintrust_showroom`), not the argparse family.
7. **265 planned traces come from 262 distinct stored rows — this is correct, not a bug.**
   The sweep plans yield 260 rows and `representative_cases()` yields 6 picks, but **4 of those 6 picks
   are already among the 260**, so there are only **262 distinct stored rows**. The log plan still emits
   **265** entries, because 3 picks are planned a second time under the *full* model variant id
   (`D@anthropic/claude-haiku-4.5`) alongside the short one (`D@haiku`). Both readings are defensible;
   the mirror creates one trace per planned entry, so **265 agent traces and 738 root traces are the
   numbers to assert** (§1), and the `traces` step must iterate the log plan, never a de-duplicated set
   of rows. This makes the `dealpoint.key` choice load-bearing: keying on
   `f"{case_id}:{variant}:{category}"` keeps those 3 pairs distinct, whereas keying on case + category
   alone would collide and silently drop three traces. Assert 265 explicitly so a future de-dup
   refactor fails loudly. For reference, the live bishal-ai ledger holds 269 agent log keys — the 265
   planned plus 4 `--replay` traces that have no offline source and are correctly absent from the mirror.
