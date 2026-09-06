# M7b: Braintrust cockpit and the multi-judge demo (reproducible layer)

**Authority:** `specs/grilled-product-brief.md` (§2.5 judges, §2.7 Braintrust as surface), `specs/milestones/m5.md`
(rubric, blinding, judge trio, calibration package), `specs/milestones/m7a.md` (namespaces, experiment names,
score budget, the walkthrough draft and its `[M7b]` markers), and the engineer's instruction of 2026-09-06:
**by the end of M7b there is one good demo that walks the whole multi-judge evaluation process, end to end,
tracked in Braintrust.** M7b is additive. Nothing from M1 to M7a is rerun, retuned or re-judged. M7a's
checkpoint is the starting state; M7b resumes from it.

## How the factory reaches Braintrust: MCP for eyes, REST for the record

The builder runs through `pi-claude-code-provider`, which launches Claude Code with `--strict-mcp-config`
and rejects any MCP server other than its own bridge. Claude Code therefore never loads the Braintrust MCP.
**Pi does**, through `pi-mcp-extension` (streamable-http, bearer `BRAINTRUST_API_KEY`), and the provider
forwards every pi tool to Claude, so the builder sees `mcp_braintrust_<tool>` like any other tool. The
operator sets this up before M7b starts (see "Operator setup"); the factory never edits pi or Claude config.

Two rules keep the M7a reproducibility contract intact:

1. **MCP is for looking, testing and verifying.** `sql_query`, `summarize_experiment`, `resolve_object`,
   `generate_permalink`, `test_facet_on_trace`, `test_evaluator`, `search_docs` and the `list_*`/`get_*`
   tools are how the builder checks that what it synced is what the walkthrough claims.
2. **Anything persistent is created by code, not by a tool call.** Views, the dashboard, review flags,
   human-score ingestion, topics configuration, patterns and evaluators that the demo relies on are created
   by `dealpoint/eval/braintrust_cockpit.py` through the REST API so `just braintrust-cockpit` recreates
   them from Git. The builder may use the matching `create_*`/`update_*` MCP tools to *prototype* a payload,
   but the object that survives is the one the code creates, and the test suite covers the code path with
   a fake client. A demo object that exists only because an agent clicked it once is a defect.

What stays interactive, because no MCP tool exists for it: the human scoring session, Loop investigation
threads, Playground comparisons, and Loop-generated custom trace views. Those are the **cockpit session**
(§7), run by the operator with an MCP-enabled Claude session after this milestone passes. The factory
prepares its checklist; the session records results back into the demo manifest.

The milestone gate covers the factory half. The cockpit session is a recorded human input, like the M5
calibration scores.

## Operator setup (before `adw_mvp.py` launches M7b; not builder work)

1. `pi install npm:pi-mcp-extension` (only after M7a's run has exited; a global extension changes every
   later pi spawn's tool catalog).
2. `~/.pi/agent/mcp.json` (global, mode 600, never in the repo; the extension does no env interpolation):
   ```json
   { "settings": { "toolPrefix": "mcp", "requestTimeoutMs": 60000 },
     "mcpServers": { "braintrust": { "transport": "streamable-http", "url": "https://api.braintrust.dev/mcp",
                                     "lifecycle": "eager", "headers": { "Authorization": "Bearer <BRAINTRUST_API_KEY>" } } } }
   ```
3. `sssf.config.yaml`, builder and reviewer entries: add `mcp_braintrust_sql_query`,
   `mcp_braintrust_summarize_experiment`, `mcp_braintrust_resolve_object`, `mcp_braintrust_generate_permalink`,
   `mcp_braintrust_list_recent_objects`, `mcp_braintrust_list_monitoring_views`,
   `mcp_braintrust_get_monitoring_view`, `mcp_braintrust_search_docs`, `mcp_braintrust_test_facet_on_trace`,
   `mcp_braintrust_test_evaluator` to `tools:` (the provider's tool catalog has a byte limit; name only
   these, not all 42). Builder additionally gets `mcp_braintrust_create_monitoring_view`,
   `mcp_braintrust_update_monitoring_view`, `mcp_braintrust_new_pattern`, `mcp_braintrust_create_facet`
   for prototyping under rule 2. Reviewer gets read-only tools only.
4. Smoke: `just prompt "call mcp_braintrust_list_recent_objects for object_type project and report the names"`
   must name `dealpoint-eval`. Record the pi and extension versions in `data/reports/framework_versions.json`
   at the start of the M7b run (builder does this; it is the one config-adjacent write allowed).

## Budget and disk

- OpenRouter: **$0.00** new metered spend. M7b makes no model calls. Anything the cockpit session spends
  in Playground is the operator's, outside the ledger, and is recorded in `docs/cockpit-session.md`.
- Braintrust score budget: M7b adds at most `human/<dimension>` (4) per human-scored trace on the review
  set (12 to 54 traces) and nothing else. Assert in `tests/test_braintrust_sync.py` as M7a does.
- Disk: no new dependency groups. If any is proposed, the M7a disk guard applies unchanged.

## 1. The hero path: one case, the whole process

The demo is built around **one judged case chosen by rule, not by hand**: from the judged subset
(`subset_hash 5918ef10a7e6`), the case where (a) arms A and D disagree on `grounded_accuracy`, (b) the
three judges disagree most on at least one dimension (max pairwise spread), tie broken by lowest case id.
Record the rule and the pick in `data/reports/demo_manifest.json`. The walkthrough follows this case
through every stage:

1. the dataset row (`judged_calibration`),
2. the agent trace (`case > agent > search_agreement > ... > final_answer`) for two variants (A@Haiku, D@Haiku),
3. the deterministic scores (`obj/`),
4. the **three judge spans** under `scoring` (see §2),
5. the aggregate (`judge/<dimension>`), the human score (`human/<dimension>`), and the agreement statistics,
6. the same numbers in the local report (`data/reports/judges.json`) so the audience sees Braintrust is a
   surface, not the source of truth.

## 2. Judge spans: make the multi-judge process visible in the trace tree

M7a replays `scoring` spans carrying provenance. M7b refines the replay for the judged variants only:

- Under `scoring`, one child span per judge, named `judge/<family>` (`judge/mistral`, `judge/nvidia`,
  `judge/bytedance`), input = the blinded packet (arm and model stripped, exactly what the judge saw),
  output = the judge's JSON (`reasoning, evidence, trajectory, professional, notes`), metadata =
  `judge_model, rubric_version, judge_price_usd, call_cost_usd, subset_hash`.
- One `judge/aggregate` span with the mean-of-judges per dimension, the rounding rule (M5 D5) stated in
  metadata, and per-judge scores in metadata (not as scores; the score budget stays six plus four).
- Replayed from `data/results/` and the calibration packets. **No model calls.** If a stored judge result
  lacks the packet or the raw JSON, the span says so in metadata rather than inventing content.
- `human/<dimension>` scores are logged on the same row when `data/eval/calibration/human_scores.jsonl`
  has an entry for that trace; absent entries log nothing (never zero).

## 3. Human scoring path (Review if the plan allows it, the local form if not)

Human review score configuration in Braintrust is documented as Pro/Enterprise. Probe once, record the
result, and do not guess:

- **If review scores can be configured:** create the four rubric scores as continuous or five-level
  categorical scores named `human/reasoning`, `human/evidence`, `human/trajectory`, `human/professional`
  with the frozen rubric anchors in their descriptions (source: `eval/judges/rubrics.md`, hash
  `cfda9f8cc401`), flag the 12-trace review set with `~__bt_review_lists: PENDING` and `~__bt_assignments`
  = the operator (id via `lookup_users` equivalent REST), and provide `just calibration-pull` that fetches
  the scores back through BTQL into `human_scores.jsonl` (schema unchanged). Local stays canonical.
- **If not:** the operator scores `data/eval/calibration/form.md` as M5 designed; `just calibration` runs
  as today; `just braintrust-cockpit` pushes the rows as `human/<dimension>` scores. The walkthrough shows
  human scores in the experiment table and the trace, not in Review mode, and says why in one sentence.

Either way, `data/reports/judges.json` is the only place that computes Spearman, weighted kappa and
pairwise agreement, and the walkthrough quotes those numbers from it.

## 4. Saved views (REST `POST /v1/view`, idempotent by name)

Create these table views on the project, each with a one-line description that is also its walkthrough
caption. Names are stable; re-running updates rather than duplicates.

| view | view_type | filter / columns |
|---|---|---|
| `Judged traces by variant` | experiments | tag `stage=evaluation`, name prefix `judge-`, columns: variant, judge/* aggregates, obj/grounded_accuracy |
| `Judge disagreement` | experiment (on the hero variant) | rows where max pairwise judge spread ≥ 2 on any dimension (metadata), sorted by spread |
| `Retrieval rescue` | logs | BTQL from `docs/braintrust-queries.md` query 1 |
| `Failure attribution` | logs | query 2 |
| `DeepEval vs judge disagreement` | logs | query 4 |
| `Trajectory inefficiency` | logs | query 6 |
| `Review set (12)` | for_review_experiments or logs | the 12 flagged traces |

Filters are expressed as BTQL strings taken verbatim from `docs/braintrust-queries.md`, so the saved view
and the documented query cannot drift apart (a test compares them).

## 5. Dashboard (REST `POST /v1/view`, `view_type: monitor`)

One dashboard, `DealPoint eval overview`, charts in this order:

1. `obj/grounded_accuracy` by arm, grouped by model (the M4/M6 headline),
2. `judge/<dimension>` mean by variant (four series),
3. judge vs human per dimension on the review set, when human scores exist (otherwise the chart is created
   with a caption "pending human calibration"; it is not omitted, the gap is part of the story),
4. `$/case` by model (M6 Pareto),
5. DeepEval vs `obj/` agreement rate.

Use the documented `custom_charts` layout; keep the payload in `dealpoint/eval/braintrust_cockpit.py` as
data, not string templates, and cover it with a fake-client test. Record the created view ids in the
manifest.

## 6. Demo manifest and the regenerated walkthrough

`data/reports/demo_manifest.json` is written by `just braintrust-cockpit` and holds: project id, every
experiment name and id used by the walkthrough (resolved by name prefix, newest wins, so a re-sync never
strands the doc), dataset ids, view ids, the hero case id and rule, the review set, permalinks generated by
the API for each walkthrough stop, the human-scoring path taken (§3) and its probe result, `git_sha7`,
`rubric_version`, `subset_hash`, `synced_at`.

`docs/demo-walkthrough.md` is **generated** from the manifest by `just demo-walkthrough` (template in
`docs/templates/demo-walkthrough.md.j2` or a plain Python formatter; no new dependency). It keeps the
engineer's order (Datasets, RAG Lab, Agent Systems, Logs trace, Scorers, Review, Eval of Evals, Loop/SQL,
Debugger, Model Economics, Dashboard) and the role diagram, replaces every `[M7b]` marker in the M7a draft
with either the finished content or a `[cockpit session]` marker (§7), and cites real names, ids and observed
numbers from the manifest and reports. Every link in the generated doc is one the manifest can regenerate;
the doc says so in its footer. 5 to 10 minutes read aloud; a hard cap of 1,400 words excluding the
appendix of links.

## 7. Topics, one Pattern, and the cockpit session

**Factory (code, REST, covered by tests):**

- Topics over the synced agent traces (project logs): a preprocessor that renders `case > agent` spans as
  text (question, tool sequence, final answer, `obj/grounded_accuracy`), one facet ("what went wrong, or
  what made it succeed, in one sentence"), clustering enabled. The builder may test the preprocessor and
  facet through `mcp_braintrust_test_preprocessor_on_trace` / `test_facet_on_trace` before committing the
  code that creates them. Record the resulting cluster names in the manifest and whether any maps to a
  `failure_attribution` cause from BTQL query 2.
- One Pattern: the recurring behaviour from query 6 (trajectory inefficiency), with ≥ 3 supporting trace
  ids and a one-paragraph description; created by code, verified with `mcp_braintrust_search_patterns`.

**Cockpit session** (`docs/cockpit-session.md`, written by the factory: a numbered procedure for the
operator and an MCP-enabled Claude session; each step names the object by manifest key, the exact prompt
or question, what to save, and records the result with `just demo-manifest-record --step <n> --note ...`):

1. Score the review set (Review mode or `form.md`), then `just calibration` and `just braintrust-cockpit`.
2. Loop investigation thread over the hero case: BTQL query 2's question in plain words; save the thread;
   record its URL.
3. Playground: the hero case's blinded packet against the calibrated judge prompt, the three judge models
   side by side; record the URL and whether outputs match the stored judge JSON.
4. Custom trace view generated by Loop: "show the three judge spans as a 3 x 4 grid of scores with the
   human row beneath"; save project-wide; record the `tv` parameter.
5. `just demo-walkthrough` once more so the walkthrough embeds the recorded URLs.

## 8. Framework roles reminder

Unchanged from M7a. Braintrust remains the surface; `data/reports/` and Git remain the record. No second
tracing backend, no metric weaker than gold labels, no hardcoded judge stack, no new agent runtime.

## Definition of done (`gate_m7b`; add the marker)

- [ ] `just braintrust-cockpit` is idempotent: run twice, the second run creates nothing new (views and
      dashboard resolved by name, experiments by prefix; asserted with the fake client and once live).
- [ ] Judge spans replayed per §2 for every judged variant; the hero case chosen by the recorded rule; the
      trace tree shows three judge children plus aggregate; per-judge scores in metadata, not scores.
- [ ] Human-scoring path probed and recorded; whichever branch applies is wired end to end
      (`human_scores.jsonl` to `human/<dimension>` in Braintrust; or Review scores to `human_scores.jsonl`).
- [ ] Seven saved views and one dashboard exist with stable names and captions; view filters equal the
      documented BTQL (test).
- [ ] `demo_manifest.json` complete; `docs/demo-walkthrough.md` generated from it, no `[M7b]` markers left,
      only `[cockpit session]` markers where §7 applies; word cap respected; every named object resolves
      (test walks the manifest and the doc).
- [ ] Topics configured and one Pattern created by code (§7); cluster names and pattern id in the manifest.
- [ ] `docs/cockpit-session.md` written per §7; `just demo-manifest-record` works.
- [ ] Score budget assertion extended for `human/`; OpenRouter ledger unchanged by this milestone
      (asserted: no rows tagged `m7b`).
- [ ] Every persistent Braintrust object the walkthrough links to is created by `braintrust_cockpit.py`
      (reviewer cross-checks the manifest against the code; an object with no creating code fails review).
- [ ] Suite, ruff, pyright green. Reviewer verifies: nothing from M1 to M7a rerun; frozen test set untouched
      (hash assertions); no edits to `~/.pi`, `~/.claude.json` or `sssf.config.yaml`; walkthrough claims
      match `judges.json` and `four_arm.json` numbers.

## Out of scope

The cockpit session itself (executed after the gate, by the operator with the MCP-enabled Claude session);
any product UI; any change to `specs/grilled-product-brief.md`, `m5.md` or `m7a.md`; new judges, new cases,
new metered runs.

## Spend guard (engineer's instruction, 2026-09-06 evening; binding)

The org's Braintrust plan reached its 10k monthly score cap during M7a's syncs (every re-sync minted a
suffixed experiment copy and re-logged every score). Pay-as-you-go overage is now on, and the engineer
wants to pay as little as possible before the demo is up. The factory therefore implements, and the
reviewer verifies, all of the following; they are already in the code at the start of attempt 3:

- `braintrust_sync` and `braintrust_cockpit` are **dry-run by default**. A live write needs an explicit
  `--live`; `--dry-run` always wins. `just braintrust-sync` / `just braintrust-cockpit` without `--live`
  never touch the API.
- Before the first live write, `braintrust_cockpit` computes the exact number of scores the run will
  create (`n_live_scores_planned` = human scores + 4 per replayed tree), prints it, records it in
  `demo_manifest.json`, and aborts with `ScoreBudgetError` if it exceeds `LIVE_SCORE_CAP` (600). Extending
  the replay (all judged trees) means extending `planned_replay_trees()`; the cap stays.
- A local ledger, `data/reports/braintrust_score_ledger.jsonl`, records every score written live
  (experiment, key, count, timestamp). Re-runs skip anything in the ledger: **a score is never re-logged**.
  The idempotency proof is "second `--live` run writes zero scores and the ledger is unchanged".
- Per-judge scores live in span metadata, never as scores. Only `judge/<dimension>` aggregates and
  `human/<dimension>` are scores.
- The full M7a `braintrust-sync --live` is **not** run in M7b. The existing experiments are the demo's
  substrate; reads (BTQL, summaries, views, dashboards, permalinks) are free and are how claims are verified.
- Expected live spend for the whole milestone: about 100 to 500 scores, under two dollars. Anything
  larger is a defect.
