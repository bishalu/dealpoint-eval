# The DealPoint tour, MLflow edition

Same rows, second cockpit. This tour sits beside [`docs/demo-tour.md`](demo-tour.md) (the Braintrust
tour, which links back here); neither replaces the other, and no number below differs from the
Braintrust one for the same object -- both are read out of the same stored rows through the same pure
functions (`dealpoint/eval/braintrust_showroom.py`, `braintrust_cockpit.py`, and the M9 mirror,
`dealpoint/eval/mlflow_mirror.py`).

Tracking server: self-hosted, SQLite-backed, on this VM --
`just mlflow-server` (`sqlite:///data/mlflow/mlflow.db`, artifacts under `data/mlflow/artifacts/`,
gitignored like `data/index/`), reached locally at `http://127.0.0.1:5000`, or through the exe.dev HTTPS
proxy once the operator wires it (the factory documents the port, not the URL).

Rebuild everything in a fresh store with one idempotent command:

```
just mlflow-sync            # dry run: prints the plan, writes nothing
just mlflow-sync --live     # writes; a second --live creates nothing new (tag `dealpoint.key`)
```

33 runs in one experiment (`dealpoint-eval`), 738 traces (265 agent + 406 retrieval + 67 prompt-variant),
9 evaluation datasets, 8 registered prompts, four OpenRouter judges, six in-process deterministic
scorers, one registered model (`dealpoint-agent`) with three aliases. Manifest:
`data/reports/mlflow_manifest.json`.

## Act I. Build a measurable agent

### Stop 1. The question, and the trap

**Datasets tab**, `maud-dealpoint-dev` / `maud-dealpoint-test` / `maud-dealpoint-counterfactual`: the
same 58/167/40 cases as the Braintrust datasets, one row per case, expectations = the gold answer. The
counterfactual set is the trap Braintrust's tour opens with: MLflow holds the same 40 redacted/
out-of-scope rows, unchanged.

### Stop 2. Can it retrieve the right evidence?

The **`rag-*` runs** (`rag-m3-dense` through `rag-m3-fusion-rerank`, plus `rag-m7-li-crosscheck`): open
the run comparison (tag `axis = 'retrieval'`), chart `obj/hit_at_5`, `obj/hit_at_10`, `obj/mrr` and
`li/hit_rate`, `li/mrr` side by side (LlamaIndex's independent grader on the same 406 retrieval traces).
Hybrid (dense + BM25, RRF) wins on both scorers, exactly as `docs/demo-tour.md` reports.

### Stop 3. Does agency help? Does better RAG help? Does the skill help?

**Evaluations tab**, filter to the four GLM system runs (`A-z-ai_glm-...` through `D-z-ai_glm-...`),
sort by `dealpoint.key = 'contract_144__q05'`: A abstains, B/C/D answer "Actual knowledge" -- the same
row Braintrust's Grid shows. `mlflow.genai.evaluate` runs the six deterministic scorers in-process here
(Braintrust could not run them server-side); their means become the run metrics `obj/*`.

## Act II. Correctness is not enough

### Stop 4. Watch one agent think

**Traces tab**, filter `tags.case_id = 'contract_144__q05'`. Each trace is a replayed span tree with its
**original timestamps** (`start_time_ns`/`end_time_ns` from the stored row's `t_ms`/`wall_ms`) -- the one
thing the Braintrust replay could not do. Assessment columns show `obj/*`, `judge/<family>/<dim>` and,
for the 24 lawyer-scored packets, `human/<dim>`, right on the trace.

### Stop 5. What deterministic truth captures, and scoring in production

The six `@mlflow.genai.scorer` functions (`grounded_accuracy`, `answer_correct`,
`citation_gold_overlap`, `citation_verbatim`, `abstain_correct`, `skill_adherence`) call
`dealpoint.eval.scorers` directly and run inside this MLflow install -- the in-process scorer Braintrust
never had. `just mlflow-score-new --since <timestamp>` is the batch stand-in for the production
monitoring MLflow OSS does not ship (Databricks-only): it re-scores traces newer than the cutoff instead
of running continuously.

### Stop 6. Human review, three judges, and where they disagree

Assessment columns on the **Traces tab** for the `judge-*` runs' traces: `judge/mistral/<dim>`,
`judge/nvidia/<dim>`, `judge/bytedance/<dim>` next to `human/<dim>` for the 24 packets the lawyer scored
(96 HUMAN assessments). `just mlflow-align-judges --live` ran `judge.align()` for each of the four
dimension judges against those 24 packets (MLflow-unique; Braintrust has no judge alignment) and
re-scored all 432 of the 108 judged traces x 4 dimensions, writing `data/reports/judge_alignment.json`.
Closeness-to-lawyer, before -> after: reasoning 0.9375 -> 0.9375, evidence 0.9444 -> 0.9271, professional
0.9167 -> 0.9583, and **trajectory 0.7847 -> 0.7604** -- alignment moved trajectory, but backwards: the
one dimension no judge gets right on its own got slightly worse, not better, after `.align()` against the
lawyer's 24 packets. (An earlier run of this command lost all 108 trajectory re-scores because
`bytedance-seed/seed-2.0-mini`, a reasoning model, spent its `max_tokens=20` budget on hidden reasoning
and returned an empty response; the fix -- the same `reasoning.enabled=False` flag the base judge runner
already sends -- and a bounded retry with backoff for transient OpenRouter 429s let every re-score land.)

### Stop 7. Iterate on the prompt before touching code

**Prompt registry**: `agent-base-system-prompt` (`baseline`), `arm-a-prompt-cite-first` (`best`),
`judge-panel-rubric` (`judge-panel`), `judge-calibrated-rubric` (`rubric`), `agent-arm-d-skill-injection`
(`skill`), plus the four Playground chat prompts (`arm-a-prompt-{base,terse,cite-first,abstain-first}`).
The 67 `prompt-variant` traces carry the ledger's `(variant, case_id)` pairs and inputs, but **not** the
model's outputs or the judges' scores on them: those were never mirrored to disk outside Braintrust (see
"gaps" below), so the trace exists and is honestly empty rather than backfilled from a Braintrust fetch.
`just mlflow-optimize-prompt --live` ran GEPA/metaprompting on `arm-a-prompt-base` over the 18
`maud-dealpoint-playground-armA` rows with the evidence judge as the objective. **`arm-a-prompt-base`
version 2 is now registered with alias `optimized`** (verified in `data/mlflow/mlflow.db`) -- the fifth
row in the prompt comparison, reported against the Git-recorded scores of the four hand-written variants
rather than a re-run of them.

## Act III. Turn evidence into a deployment decision

### Stop 8. Diagnose the real failure

`mlflow.search_traces(filter_string="tags.status = 'CAP_HIT'")` finds the same trajectories Braintrust's
Debugger walks; the **Traces tab** filtered to `tags.case_id = 'contract_39__redacted_q05'` (the redacted
twin) shows the cap-hit span tree with real timestamps this time. There is no Topics, no Loop and no
Pattern object here -- said plainly in "what only Braintrust has" below -- so the failure-shape clustering
Braintrust's Topics facet does automatically has to be read off the assessment columns and tags by hand
(`tags.category`, `tags.arm`, `tags.model`).

### Stop 9. What should we deploy?

The **`Which model?`** run comparison (tag `axis = 'model'`, arm D held fixed, five models on the 18
judged cases): chart `net_accuracy`, `usd_per_case`, `wall_s_p50`/`wall_s_p90`, `cap_hit_rate`,
`correct_per_dollar`. The registered model **`dealpoint-agent`** carries one version per system@model
with those same metrics as version metadata; alias **`champion`** points at `D@gemini-3.1-flash-lite`,
`baseline` at `A@haiku`, `cost-floor` at `D@qwen3.7-flash` -- the deployment decision as a first-class
object, not a chart someone has to remember to reopen.

## What MLflow adds

- **In-process deterministic scorers.** The six `dealpoint.eval.scorers` functions run as real MLflow
  scorers inside `mlflow.genai.evaluate`; Braintrust could only declare them, never execute them
  server-side.
- **Real span timestamps.** Every replayed trace carries its original `start_time_ns`/`end_time_ns` from
  the stored row's `t_ms`/`wall_ms`, so duration and ordering are exact, not synthetic.
- **Judge alignment.** `judge.align()` against the lawyer's 24 packets is a Braintrust-absent capability
  (Stop 6, `just mlflow-align-judges`).
- **Prompt optimization.** GEPA/metaprompting over the arm-A packets, registered as a new prompt version
  with an alias (Stop 7, `just mlflow-optimize-prompt`).
- **The model registry.** One `dealpoint-agent` object with a version per system@model and named
  aliases, rather than a dashboard someone has to remember to open (Stop 9).

## What only Braintrust has here

- **Monitor dashboards over logs** and the eight ranked-comparison dashboards with a verdict paragraph
  each; MLflow's run comparisons are the closest equivalent, at the run level, not a saved dashboard
  object.
- **Online scoring**, continuous, over live logs; MLflow OSS's nearest equivalent is the batch
  `just mlflow-score-new`.
- **Topics, Patterns, Loop, the Debugger and the Playground UI** are Databricks-only or absent from OSS
  MLflow 3.16; the SQLite store is queryable directly and `search_traces`/`search_runs` filter syntax
  covers the same six saved investigations, by hand rather than as saved objects.
- **Labeling sessions and the review app** (human review with assignment/status workflow) are
  Databricks-only; MLflow's HUMAN assessments hold the same 96 lawyer scores as plain feedback rows, with
  no review-queue UI around them.
- **No chart/view/dashboard API** exists in OSS MLflow 3.16 (verified against the installed 3.16.0):
  the `views` step exports `data/reports/mlflow_views.json` (one entry per Braintrust chart: question,
  tag filter, metric) instead of creating a saved chart object; a run comparison is built by hand from
  that list (filter by tag `axis=<question>`, chart `<metric>`).
- **The 67 prompt-variant outputs and their judge scores** live only in Braintrust (the Playground
  pre-run's model text and LLM-judge scores were never mirrored to `data/reports/`); MLflow mirrors the
  row set (case, variant, ledger key) honestly, not the text.

Project: this VM's tracking store (`sqlite:///data/mlflow/mlflow.db`), reached at
`http://127.0.0.1:5000` or through the operator's exe.dev proxy URL, once wired.
