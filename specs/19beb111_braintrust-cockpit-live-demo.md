# M7b attempt 3 — Braintrust cockpit and the multi-judge demo: make it real

**Milestone spec (read-only, binding):** `specs/milestones/m7b.md`
**Requirements document (read-only, wins on product decisions):** `specs/grilled-product-brief.md`
**Session:** `19beb111`, attempt 3. Attempts 1 and 2 were interrupted by the operator; their work is
already committed (`b70ca45`).

---

## 1. Starting state — verified on disk, 2026-09-06

Do not re-derive this. It was measured before this plan was written.

**Green already:**

| check | result |
|---|---|
| `uv run pytest -m "gate_m7b and not needs_network" -q` | 23 passed |
| `uv run pytest -m "not needs_network and not needs_model" -q` | 567 passed, 5 deselected |
| `uv run ruff check .` | All checks passed |
| `uv run pyright` | 0 errors, 0 warnings |

**The offline code layer is essentially complete.** `dealpoint/eval/braintrust_cockpit.py` (999 lines)
already has: `hero_case()` (:113) with `HERO_RULE_TEXT` (:46), `judge_spans()` (:201),
`view_definitions()` (:315, seven views), `dashboard_charts()` (:372, five charts) and
`dashboard_definition()` (:408), `topics_config()` (:463) with `topics_preprocessor_code()` (:430),
`pattern_definition()` (:502), `human_score_rows()` (:533) and `assert_human_score_budget()` (:577),
the spend guard (`LIVE_SCORE_CAP = 600` :767, `planned_replay_trees()` :776,
`planned_live_scores()` :781, `assert_live_score_budget()` :785, the ledger :799-818),
`replay_hero_case()` (:730), `push_human_scores()` (:821), `sync_cockpit()` (:857),
`build_demo_manifest()` (:893), `_DryRunRestClient` (:926) and `main()` (:963).
`docs/demo-walkthrough.md` is generated (0 `[M7b]` markers, 6 `[cockpit session]` markers, 1228 words),
`docs/cockpit-session.md` has all five §7 steps, `just demo-manifest-record` works.

**What is NOT done — the whole point of attempt 3.** Every persistent Braintrust object in the
milestone exists only as a *dry-run simulation*. `data/reports/demo_manifest.json` right now says:

```
project_id            "dry-run-project-id"
views                 view-1 .. view-7        (fake ids from _DryRunRestClient)
dashboard             view-8                  (fake id)
topics.id / pattern.id  null / facet-dry-run  (nothing created)
n_human_scores_pushed 0
replay.experiment_name null, replay.variants []
```

`data/reports/braintrust_score_ledger.jsonl` does not exist. The milestone's binding instruction is
*"by the end of M7b there is one good demo that walks the whole multi-judge evaluation process, end to
end, tracked in Braintrust"*, and the DoD requires idempotency asserted **"with the fake client and
once live"**, seven views and a dashboard that **exist**, API-generated permalinks, a pattern id and
Topics cluster names in the manifest. None of that can come from a dry run.

**Already verified sound — do not re-litigate.** The §1/§2 layer was audited independently and
empirically (41 tests in `test_braintrust_cockpit.py` + `test_braintrust_sync.py` pass): `hero_case()`
is deterministic, pinned to `subset_hash` `5918ef10a7e6` by an assertion at :118, tie-broken by
`(-spread, case_id)` at :148, and its rule text and pick are recorded in the manifest
(`contract_32__q04`, spread 4, A wrong / D right on grounded accuracy). `_emit_span_tree`
(`braintrust_sync.py:1697-1717`) has been extended so the judge spans' input (blinded packet), output
(the judge JSON) and metadata all reach the client — an earlier report that content was dropped is
stale. The defect in this area is about *scores*, not spans; see T5b.

Two honest notes for the walkthrough: `hero_case()` records `n_candidates: 2`, a thin pool, and the
"max pairwise spread over both hero variants" reading of §1(b) is one defensible interpretation of a
sentence that does not say which variant supplies the spread. Both are stated verbatim in the recorded
rule, so they are auditable; the walkthrough should not oversell the selection as more forced than it is.

**Attempt 3 is therefore: fix the blockers that would make a live run fail or lie, complete the
manifest to the full §6 shape, run the cockpit live exactly twice (~104 scores, well under the 600
cap), verify every claim through the Braintrust MCP read tools, regenerate the walkthrough from the
live manifest, and add the gate tests that hold the artifacts to that standard.**

**Operator setup is done.** `~/.pi/agent/mcp.json` exists; `mcp_braintrust_*` tools are already listed
for builder and reviewer in `adws/adw_sssf_config/sssf.config.yaml:93-109` and `:134+`. pi is `0.84.4`.
You have `mcp_braintrust_sql_query`, `summarize_experiment`, `resolve_object`, `generate_permalink`,
`list_recent_objects`, `list_monitoring_views`, `get_monitoring_view`, `get_project_settings`,
`search_docs`, `test_facet_on_trace`, `test_preprocessor_on_trace`, `test_evaluator`,
`search_patterns`, plus (prototyping only) `create_monitoring_view`, `update_monitoring_view`,
`new_pattern`, `create_facet`.

---

## 2. Guardrails

- **Read-only files.** `specs/grilled-product-brief.md`, `specs/milestones/m5.md`,
  `specs/milestones/m7a.md`, `specs/milestones/m7b.md`. Do not edit them. Where the brief and the
  milestone spec differ on a product decision, the brief wins and you **report** the difference in your
  envelope; you do not silently resolve it.
- **No config edits.** `~/.pi/**`, `~/.claude.json`, `adws/adw_sssf_config/sssf.config.yaml` are
  untouched. The single config-adjacent write allowed is `data/reports/framework_versions.json`
  (Task 2), which the spec explicitly assigns to you.
- **$0.00 OpenRouter.** M7b makes no model calls. `data/reports/` spend ledger gains no row tagged
  `m7b`. Do not run `just judge`, `just sweep-m4*`, `just pareto-sweep`, `just synth-queries`,
  `just deepeval-crosscheck`, or `dealpoint.eval.smoke --send`.
- **Do not run `just braintrust-sync --live`.** The spec is explicit: the existing M7a experiments are
  the demo's substrate. Reads are free; a full re-sync would mint score volume against a capped plan.
- **Nothing M1–M7a is rerun, retuned or re-judged.** Frozen artifacts (`data/eval/test_subset_v1.json`,
  the judged subset `5918ef10a7e6`, `data/eval/judge_scores.jsonl`, `data/results/**`,
  `data/reports/four_arm.json`, `pareto.json`, `judges.json`) are inputs, never outputs.
- **MCP `create_*` tools are for prototyping only.** Every persistent object the walkthrough links to
  must be created by `dealpoint/eval/braintrust_cockpit.py`. If you prototype a facet or pattern
  payload through MCP, delete the prototype or ensure the code's idempotent upsert claims the same
  name, and note it in your envelope. The reviewer cross-checks the manifest against the code; an
  object with no creating code fails review.
- **Judge-span provenance.** Per-judge scores stay in span metadata. Only `judge/<dimension>` (4) and
  `human/<dimension>` (4) are scores.

---

## 3. Tasks

### T1 — Fix the live API-key resolution (blocker; do this first)

`dealpoint/eval/braintrust_cockpit.py:984` reads the key with
`os.environ.get("BRAINTRUST_API_KEY", "")`. But the key is not in the environment and not in `.env`
(`just` only dotenv-loads `.env`); it lives in `.env.braintrust`, and
`braintrust_adapter.load_braintrust_key()` (`dealpoint/eval/braintrust_adapter.py:81-106`) is the
resolver that finds it (env → `.env.braintrust` → `.braintrust.json`). `braintrust_available()`
(:109) uses the resolver and would return `True`, so today `--live` would pass the availability gate
and then build `RestClient("")` — every REST call would 401 and the run would die partway, possibly
after the SDK path had already written spans.

Fix in `main()`:

- Resolve the key via `load_braintrust_key()`, not `os.environ`.
- Because the `braintrust` SDK reads `BRAINTRUST_API_KEY` from the process environment, also set
  `os.environ["BRAINTRUST_API_KEY"]` from the resolved value before importing/initialising
  `braintrust`, so the REST seam and the SDK seam authenticate as the same identity.
- If the key resolves to empty on a `--live` invocation, exit non-zero with a clear message rather than
  proceeding. (The existing "braintrust unavailable — skipping cleanly, return 0" behaviour is fine for
  *no key at all*, but it must not silently leave a stale dry-run manifest on disk while reporting
  success; print which path was taken.)

Add an offline `gate_m7b` test that `main(["--live"])` with the resolver monkeypatched to return `None`
neither writes the manifest nor claims success, and that with a key it constructs `RestClient` with a
non-empty bearer. Do not make a network call in the test.

### T2 — Record pi and extension versions in `framework_versions.json`

Spec, "Operator setup" item 4: *"Record the pi and extension versions in
`data/reports/framework_versions.json` at the start of the M7b run (builder does this; it is the one
config-adjacent write allowed)."*

`data/reports/framework_versions.json` currently has no pi entry and a stale `git_sha7` (`199ddba`).
Extend `dealpoint/eval/framework_versions.py` so the artifact carries `pi` (from `pi --version`;
currently `0.84.4`) and the `pi-mcp-extension` version (from `pi extension list`, or the extension's
`package.json` under `~/.pi` if the listing does not print a version — record `"unavailable"` with the
reason rather than guessing). Regenerate with `just framework-versions`. Keep the existing keys and
ordering behaviour; this is an addition, not a rewrite. `git_sha7` refreshes naturally.

Note: `pi extension list` writes `[pi-mcp] Could not resolve $ref ...` warnings to stderr. Judge the
command by exit status, not by scanning output.

### T3 — Probe the human-scoring path for real, and record what was observed

`HUMAN_SCORING_PROBE` (`braintrust_cockpit.py:59-83`) is a hand-written constant recording a
docs-derived inference: Starter plan → configured review scores cannot carry four dimensions → branch B
(`local_form`). The spec says *"Probe once, record the result, and do not guess."* The conclusion is
probably right, but the record must show an **observation**, not a deduction.

Do the probe with the MCP tools you have and rewrite the constant to state what you actually saw:

- `mcp_braintrust_get_project_settings` on `dealpoint-eval` — does it expose configured review scores
  / the plan tier? Record the raw finding.
- `mcp_braintrust_search_docs("configure human review scores")` — quote the plan requirement.
- Corroborate with the M7a evidence already on disk: the `num_scores_calendar_months` quota error
  recorded in `data/reports/braintrust_sync.json` (`live_run_status`).

Keep the shape of `HUMAN_SCORING_PROBE` (`probed_via`, `finding`, `decision`, `decision_reason`), add
`probed_at` and, where applicable, the verbatim tool response snippet. If the probe unexpectedly shows
review scores **are** configurable, take branch A of §3 instead: create the four `human/<dimension>`
review scores with the frozen rubric anchors (source `eval/judges/rubrics.md`, hash `cfda9f8cc401`),
flag the 12-trace review set, and add a `just calibration-pull` recipe that fetches scores back through
BTQL into `data/eval/calibration/human_scores.jsonl` (schema unchanged, local stays canonical). Branch
B is the expected outcome; do not build branch A speculatively.

Whichever branch holds, `data/reports/judges.json` remains the only place that computes Spearman,
weighted kappa and pairwise agreement, and the walkthrough quotes those numbers from it.

### T4 — Confirm the Topics / Pattern REST contract, then stop swallowing its errors

`sync_topics_and_pattern()` (`braintrust_cockpit.py:684-705`) posts to `/v1/facet` and `/v1/pattern`
inside bare `try/except Exception` blocks that stash `{"error": ...}` into the manifest, guarded by a
`TOPICS_PATTERN_LIMITATION` note saying the endpoint contract is unconfirmed. On a live run that means
Topics and the Pattern could both silently fail to be created while the manifest still looks
plausible — and the DoD requires *"Topics configured and one Pattern created by code (§7); cluster
names and pattern id in the manifest."*

Confirm the real contract before going live:

1. `mcp_braintrust_search_docs` for the Topics/facet and Pattern REST endpoints and their payload
   shapes (preprocessor, facet, clustering flag; pattern description + supporting trace ids).
2. Prototype the payloads with `mcp_braintrust_test_preprocessor_on_trace` and
   `mcp_braintrust_test_facet_on_trace` against one synced agent trace, so you know the preprocessor
   in `topics_preprocessor_code()` renders `case > agent` spans (question, tool sequence, final answer,
   `obj/grounded_accuracy`) as the spec describes. Prototyping with `create_facet` / `new_pattern` is
   allowed; the surviving object must be the one the code creates.
3. Correct the endpoint paths / payload keys in the code to match what you confirmed.
4. Replace the blanket `except Exception` with behaviour that distinguishes the two clients: the
   dry-run/fake client may keep returning a placeholder, but a **live** run must fail loudly (or record
   a structured, explicitly-flagged failure that the new manifest-completeness test rejects) rather
   than pass a `{"error": ...}` off as a synced object.
5. After the live run, read the resulting cluster names back (MCP `sql_query` / the Topics listing) and
   record them in the manifest, plus whether any cluster maps to a `failure_attribution` cause from
   BTQL query 2 (`docs/braintrust-queries.md`). Verify the pattern with
   `mcp_braintrust_search_patterns`; the pattern already carries ≥ 3 supporting trace ids
   (`_inefficient_trajectory_trace_ids()` :473, covered by a test at
   `tests/test_braintrust_cockpit.py:358`).

If the REST surface for Topics or Patterns genuinely does not exist for this plan, do **not** fake it:
record the probe, the exact error, and treat it as a brief-vs-spec difference to report (§5 below).
That is a legitimate outcome; a manifest that claims a pattern id it does not have is not.

### T4b — Three views/dashboard defects that would ship an empty demo

The seven view names and the BTQL-verbatim test pass, but three bindings are wrong. A saved view that
renders zero rows is a broken walkthrough stop even though every test is green.

1. **`Judge disagreement` filters on a field nothing writes.** Its filter and sort reference
   `metadata.judge_spread_max` (`braintrust_cockpit.py:335-336`), and that key is written by **no code
   anywhere** in the repo (verified: the only two hits are the filter strings themselves). The view will
   be permanently empty. `_max_pairwise_spread()` (:103) already computes the number — emit it into the
   replayed trace metadata (per dimension and as a max) so the filter has something to match, and
   assert in a test that the key the view filters on is a key the replay writes.
2. **`Judge disagreement` is not bound to the hero experiment.** §4 specifies `view_type: experiment`
   *"(on the hero variant)"*, but `_upsert_view` posts every view with `object_type: "project"` and
   `object_id: project_id` (:656-658). Bind this one to the hero experiment's id (available after T5's
   experiment resolution) rather than to the project.
3. **`Review set (12)` points at a dataset name, not the flagged traces.** Its definition is
   `{"dataset": "maud-dealpoint-review-set"}` (:367-369). §4 wants the 12 flagged traces. Drive it from
   the same `review_set` manifest key T5 adds, so the view, the manifest and the walkthrough cannot
   disagree about which traces are in the review set.

Also: §5 says *"Use the documented `custom_charts` layout"*, but the dashboard posts an ad-hoc
`{title, measure, group_by}` shape as `{"charts": [...]}` (:680); `custom_charts` appears nowhere in
the repo. Confirm the real layout with `mcp_braintrust_search_docs` and `get_monitoring_view` on an
existing dashboard, then reshape the payload to match. Keep it as data, not string templates, and keep
the five charts in spec order with chart 3's `"pending human calibration"` caption.

### T5 — Complete `demo_manifest.json` to the full §6 shape

`build_demo_manifest()` (:893) is missing five things §6 requires. Add each as first-class manifest
keys, populated by code, covered by fake-client tests:

| missing key | §6 requirement | how |
|---|---|---|
| `experiments` | *"every experiment name and id used by the walkthrough (resolved by name prefix, newest wins, so a re-sync never strands the doc)"* | Resolve live via `GET /v1/experiment?project_id=...` (or `mcp_braintrust_resolve_object` to confirm the shape first). For each prefix the walkthrough cites — the four-arm `A-`/`B-`/`C-`/`D-` runs, the `judge-` experiments, `m7b-hero-case` — take the newest by created-at. `data/reports/braintrust_runs.json` (11 rows) lists the M7a names and URLs; use it as the offline fallback/cross-check, not as the source of ids. |
| `datasets` | *"dataset ids"* | Resolve the synced dataset names (`DATASET_SETS` plus `judged_calibration`, see `braintrust_sync.py:262-359`) to ids via `GET /v1/dataset?project_id=...`. |
| `review_set` | *"the review set"* | The traces flagged for human review — explicit list of `{case_id, variant_id, packet_id, experiment_name}`. Today it is only implicit in `human_score_rows` (24 rows → 96 scores). Derive it from the same source so the two cannot drift. |
| `permalinks` | *"permalinks generated by the API for each walkthrough stop"* | Add a `permalinks()` builder in `braintrust_cockpit.py` producing one entry per walkthrough stop: the `judged_calibration` dataset, the hero trace for each of `HERO_VARIANTS` (`A@haiku`, `D@haiku`), each of the seven views, the dashboard, and each cited experiment. Build them from ids the live run resolved (org from `GET /v1/project` → `GET /v1/organization`; the existing URL shape is `https://www.braintrust.dev/app/<org>/p/dealpoint-eval/...`, see `data/reports/braintrust_runs.json`). Spot-check at least the hero experiment and the dashboard against `mcp_braintrust_generate_permalink` and fix the builder if they differ. |
| `topics.clusters` | §7 *"Record the resulting cluster names in the manifest and whether any maps to a `failure_attribution` cause"* | From T4 step 5. |

Also print `n_live_scores_planned` **before** the first live write on the live path (today it is only
printed on the dry-run path, `main()` :971-973), so the operator sees the number the spend guard
checked.

Keep `build_demo_manifest()` a pure function of the `sync_cockpit()` result dict so the fake-client
tests can drive it end to end. Extend
`tests/test_braintrust_cockpit.py:329 test_build_demo_manifest_has_every_required_key` with the new
keys.

### T5b — Make the spend guard's number equal what the run actually writes (blocker)

`planned_live_scores()` (`braintrust_cockpit.py:781`) assumes the hero replay emits
`JUDGE_AGGREGATE_DIMENSIONS` (4) `judge/<dimension>` scores per tree. It does not. `_hero_hierarchy()`
calls `log_hierarchy(row, case, doc)` at :721 **without** the `judge_dims=` keyword that
`braintrust_sync.log_hierarchy` (:850-889) accepts and that `sync()` itself passes at :1609. Measured
against the test fakes, the replay emits **~23 `obj/*` scores across the two hero trees and zero
`judge/*` scores**. So:

- The spend guard prints 104 while the run would write ~119. **Under-counting is the wrong direction
  for a cap.** The guard's whole purpose is that the printed number is the number.
- The scores that *do* get written are the wrong ones. §2 and the spend guard both say only
  `judge/<dimension>` aggregates and `human/<dimension>` are scores; a replay writing a pile of `obj/*`
  scores onto a fresh `m7b-hero-case` experiment contradicts that and spends the capped budget on
  duplicates of numbers M7a already logged.

Fix both halves:

1. Pass `judge_dims=` into `log_hierarchy` from `_hero_hierarchy`, using the mean-of-judges dimensions
   `_mean_judge_dims_for_packet()` (`braintrust_sync.py:662`) already computes for that packet — the
   same source `sync()` uses, so the hero trace and the M7a experiments cannot disagree.
2. Ensure the replay logs **only** the four `judge/<dimension>` aggregates as scores per tree, with the
   per-judge scores staying in span metadata (already correct in `judge_spans()`), and no `obj/*`
   re-logging. `braintrust_sync._log_scores_for_row` takes an `allowed_score_names` filter (:1463) —
   use that mechanism rather than inventing a second one.
3. Fix `_ledger_record`'s count too: `replay_hero_case` records the constant
   `JUDGE_AGGREGATE_DIMENSIONS` per tree (:750) regardless of what was written. A ledger that
   misreports counts cannot support the "never re-log" audit. Record what was actually emitted.
4. Make `planned_live_scores()` provably equal to reality with an offline test: run `sync_cockpit`
   against the fake SDK client, count every score the fakes received, and assert it equals
   `planned_live_scores(n_human_scores)`. That test is the guard on the guard; without it the cap is
   decorative.

Do this **before** T6. The pre-flight number is only meaningful once it is true.

### T6 — Pre-flight, then the live run (once)

This writes to a shared external service and spends money. Follow the order exactly.

**Pre-flight (all offline, all must pass before any live call):**

```bash
uv run pytest -m "gate_m7b and not needs_network" -q     # must be green after T1-T5
just braintrust-cockpit                                   # dry run; prints planned live scores
```

Read the dry run's `n_live_scores_planned`. After T5b this must equal the score count the
fake-client test measures. Expected: **104** (96 human scores from 24 review rows ×
4 dimensions, plus 4 `judge/<dimension>` aggregates × 2 hero variants). The cap is 600 and the spec's
expected range for the whole milestone is 100–500 scores, under two dollars.

- If the number is ≤ 150: proceed.
- If it is > 150: **stop and report**. The spec says *"Anything larger is a defect."* Do not raise
  `LIVE_SCORE_CAP`.

**The live run:**

```bash
just braintrust-cockpit --live
```

Then immediately capture the evidence:

- `data/reports/demo_manifest.json` must now show a real `project_id` (not `dry-run-project-id`), real
  view ids (not `view-N`), a real dashboard id, a non-null `pattern.id`, populated `topics.clusters`,
  `n_human_scores_pushed == 96`, `replay.experiment_name == "m7b-hero-case"` and
  `replay.variants == ["A@haiku", "D@haiku"]`.
- `data/reports/braintrust_score_ledger.jsonl` must now exist with one row per `(experiment, key)`
  written. Record its line count and the summed `n_scores` — it should equal
  `n_live_scores_planned`.

If the live run fails partway, do not retry blindly: the ledger is what makes a retry safe. Read it,
confirm what was already written, then re-run — the ledger's skip-if-present logic
(`_ledger_load()` :799, `_ledger_record()` :812) means a retry writes only the remainder. Report any
partial state in your envelope.

### T7 — Second live run: the idempotency proof

```bash
cp data/reports/braintrust_score_ledger.jsonl /tmp/ledger_before.jsonl
just braintrust-cockpit --live
diff /tmp/ledger_before.jsonl data/reports/braintrust_score_ledger.jsonl   # must be empty
```

The DoD's live half is: *"the second run creates nothing new (views and dashboard resolved by name,
experiments by prefix)"* and the spend guard's proof is *"second `--live` run writes zero scores and
the ledger is unchanged."* Confirm:

- every view and the dashboard report `created: false` in the second run's result,
- view ids, dashboard id, pattern id and experiment ids are byte-identical to run 1 in the manifest
  (only `synced_at` and `git_sha7` may move),
- `n_human_scores_pushed == 0` on the second run,
- the ledger diff is empty.

Record this evidence — the two manifests' diff and the ledger diff — in your envelope. The reviewer
will ask for it.

### T8 — Verify every walkthrough claim through MCP (reads are free)

Rule 1 of the spec: MCP is for looking, testing and verifying. Walk the manifest and confirm each
object exists and holds the numbers the walkthrough quotes:

- `mcp_braintrust_list_monitoring_views` / `get_monitoring_view` — the seven views and the dashboard
  exist with the stable names and the captions the code set; the dashboard's five charts are in spec
  order.
- `mcp_braintrust_summarize_experiment` on `m7b-hero-case` and on the `judge-` experiments — the
  scores present are `judge/<dimension>` and `human/<dimension>` and nothing else; no `judge/mistral`
  etc. leaked out of metadata into scores.
- `mcp_braintrust_sql_query` — run BTQL queries 1, 2, 4 and 6 from `docs/braintrust-queries.md` and
  confirm each saved view's filter returns rows (a view whose filter matches nothing is a broken demo
  stop, even if the string comparison test passes).
- `mcp_braintrust_resolve_object` — every experiment and dataset id in the manifest resolves.
- `mcp_braintrust_search_patterns` — the created pattern is findable.
- `mcp_braintrust_generate_permalink` — spot-check the permalinks from T5.
- Confirm the hero trace tree renders as `case > agent > ... > scoring > {judge/mistral, judge/nvidia,
  judge/bytedance, judge/aggregate}` for both variants.

Any mismatch between a live observation and a walkthrough claim is fixed in the **code and the
manifest**, then the walkthrough is regenerated — never by hand-editing `docs/demo-walkthrough.md`.

### T9 — Regenerate the walkthrough from the live manifest

```bash
just demo-walkthrough
```

How the generator works (do not go looking for a `.j2` template — there is none, and the spec allows
either): `dealpoint/eval/demo_walkthrough.py` reads the **frozen M7a draft**
`docs/templates/demo-walkthrough.draft.md` (`DEMO_WALKTHROUGH_DRAFT_PATH`, `dealpoint/config.py:325`),
substitutes its six `[M7b]` markers via `_PARAGRAPH_REPLACEMENTS` (:141) plus the intro rewrite
(:168-173), splices a `## The hero path` section before `## 1. Datasets` (:198), and raises if any
`[M7b]` survives (:201-203). Regenerating from the frozen draft rather than from the previous output
is what makes a re-run idempotent. New content therefore goes into the builder functions in
`demo_walkthrough.py`, never into `docs/demo-walkthrough.md` and never into the draft.

Then check by hand what the tests cannot judge:

- The section order is intact: Datasets, RAG Lab, Agent Systems, Logs trace, Scorers, Review, Eval of
  Evals, Loop/SQL, Debugger, Model Economics, Dashboard, plus the role diagram (`docs/demo-walkthrough.md`
  headings are already in this order).
- Zero `[M7b]` markers; `[cockpit session]` markers only where §7 applies (currently 6 — check each is
  genuinely a step the operator must perform, not a gap you could have closed).
- Under 1,400 words excluding the link appendix (`M7B_MAX_WORDS`, `dealpoint/config.py:329`). Live ids
  and permalinks will push the count up; the current body is 1228 words total. If the cap is
  threatened, move links into the appendix rather than cutting substance.
- The footer states every link is regenerable from the manifest.
- Numbers quoted match `data/reports/judges.json` and `data/reports/four_arm.json` exactly (the
  reviewer checks this). The demo's story is the hero case `contract_32__q04` — arm A wrong, arm D
  right on `obj/grounded_accuracy`, max pairwise judge spread 4 — as `hero_case()` picked it by
  `HERO_RULE_TEXT`.
- `docs/cockpit-session.md`: re-read all five steps now that the manifest is live. Each step must name
  its object by an actual manifest key that now exists, and end with a working
  `just demo-manifest-record --step <n> --note "..."` invocation. Fix any step that references a key
  you renamed or added in T5.

### T10 — Close the test gaps the DoD names

Add these as `gate_m7b`, offline, no network:

1. **The manifest on disk is a live manifest.** Today
   `tests/test_demo_walkthrough.py:101 test_real_manifest_generates_a_valid_walkthrough_when_present`
   is conditional and tolerates the dry-run manifest. Add a test that reads
   `data/reports/demo_manifest.json` and asserts it is the product of a live sync: `project_id` does
   not start with `dry-run`, view ids and dashboard id are not `view-<N>` placeholders, `pattern.id`
   is non-null, `topics.clusters` is non-empty, `permalinks` covers every walkthrough stop,
   `experiments` and `datasets` are populated, `n_human_scores_pushed > 0`,
   `replay.experiment_name == "m7b-hero-case"` with both hero variants. This is the test that stops
   attempt 4 from shipping a simulation again.
2. **Every named object in the walkthrough resolves against the manifest.** Extend the existing
   walkthrough test so each Braintrust name/id/URL appearing in `docs/demo-walkthrough.md` is present
   in `demo_manifest.json` — and, conversely, that every permalink in the doc is one the manifest can
   regenerate.
3. **OpenRouter ledger unchanged by M7b.** No test currently asserts this. The spend ledger rows carry
   `milestone_tag` (`dealpoint/eval/spend.py:231`). Add a test that no ledger row is tagged `m7b`, and
   that the realized total is unchanged from the M7a checkpoint (`3.7133`, per `specs/mvp/state.json`).
4. **Score budget extended for `human/`.** `tests/test_braintrust_sync.py:558
   test_m7b_human_score_budget_extends_m7a_budget_without_touching_it` already covers this but is not
   marked `gate_m7b`. Add the marker so the gate command actually runs it.
5. **Spend guard — currently zero coverage.** No test anywhere references `LIVE_SCORE_CAP`, the
   cockpit's `ScoreBudgetError`, `planned_replay_trees()` or the ledger (verified; the `ScoreBudgetError`
   hits in `tests/test_braintrust_sync.py` are `braintrust_sync`'s separate class). The entire spend
   guard the engineer asked for is untested. Write coverage for: dry-run default, `--live` opt-in, `--dry-run` beats
   `--live`, `ScoreBudgetError` above `LIVE_SCORE_CAP`, ledger skip-if-present. Add whatever is
   missing (in particular a test that `main([])` and `main(["--live", "--dry-run"])` both take the
   `_DryRunRestClient` path and never construct `RestClient`).
6. **T1's key-resolution test** (see T1).
7. **The ledger's never-re-log guarantee, proven offline.** This is the DoD's idempotency proof and it
   currently has *no* offline test, because `_LEDGER_ACTIVE` is armed only inside `main(["--live"])`
   (`braintrust_cockpit.py:797`), so the fake-client tests never exercise the skip path. Worse,
   `tests/test_braintrust_cockpit.py:291` positively asserts `n_spans_2 == 2 * n_spans_1` — a second
   `sync_cockpit` re-emits the whole hero replay. That is correct only while the ledger is inactive;
   live, the ledger must make the second run emit **zero** spans and zero scores. Two statements about
   the same behaviour, neither of them tested where it matters.

   Add a test that points `SCORE_LEDGER_PATH` at `tmp_path` and arms `_LEDGER_ACTIVE`, runs
   `sync_cockpit` twice against the fake clients, and asserts the second run adds no spans, no logged
   rows and no ledger lines. Then narrow the existing `2 *` assertion so its name and docstring say it
   describes the *ledger-inactive* case only, and cross-reference the new test. Do not delete the old
   assertion — make the pair coherent.
7. **Manifest key completeness** for the new §6 keys via the fake client (extend
   `test_build_demo_manifest_has_every_required_key`).

Follow the existing conventions: `pytestmark = pytest.mark.gate_m7b` at module top, fake REST/SDK
doubles rather than mocks of `requests`, no network, no model.

### T11 — Verify and report

```bash
uv run pytest -m "gate_m7b and not needs_network" -q
uv run pytest -m "not needs_network and not needs_model" -q     # ~5.5 min, 567+ tests
uv run ruff check .
uv run pyright
```

All four must be green. Judge each by exit status.

Also confirm before you report done:

- `git status` shows no change to `specs/milestones/*.md`, `specs/grilled-product-brief.md`,
  `adws/adw_sssf_config/sssf.config.yaml`, or anything under `~/`.
- The frozen-artifact hash assertions still pass (they are in the suite; note that they did).
- `data/reports/braintrust_score_ledger.jsonl` is committed — it is the record that makes future runs
  cheap, and without it a later `--live` re-logs everything.

---

## 4. Definition of done, mapped

| DoD item | done when |
|---|---|
| `just braintrust-cockpit` idempotent, fake client **and once live** | T6 + T7; ledger diff empty, ids stable, `created: false` on run 2 |
| Judge spans per §2, hero case by the recorded rule, 3 judges + aggregate, per-judge scores in metadata | already coded and tested offline; T8 confirms the live trace tree |
| Human-scoring path probed, recorded, wired end to end | T3 + T6 (96 `human/<dimension>` scores pushed onto the `judge-<variant>` rows) |
| Seven views + one dashboard, stable names and captions, filters == documented BTQL | already tested (`test_view_btql_filters_match_the_documented_queries_verbatim`); T6 creates them, T8 confirms they exist and return rows |
| Manifest complete; walkthrough generated from it; no `[M7b]`; word cap; every object resolves | T5 + T9 + T10.1/T10.2 |
| Topics + one Pattern created by code; cluster names and pattern id in manifest | T4 |
| `docs/cockpit-session.md` per §7; `just demo-manifest-record` works | already written; T9 revalidates against the live manifest |
| Score budget extended for `human/`; no OpenRouter row tagged `m7b` | T10.3 + T10.4 |
| Every persistent object created by `braintrust_cockpit.py` | guardrails + T4's prototyping discipline; state it explicitly in your envelope |
| Suite, ruff, pyright green; nothing M1–M7a rerun; frozen set untouched; no config edits | T11 |

---

## 5. Report these, do not resolve them silently

Your envelope must carry a short "differences and limitations" section covering, at minimum:

- **Brief vs milestone spec.** `specs/grilled-product-brief.md` is the requirements document and wins
  on any product decision. If nothing conflicts, say so explicitly — the reviewer checks that you
  looked.
- **The human-scoring branch actually taken** (T3) and the observed evidence behind it. §3 of the spec
  wants the walkthrough to say in one sentence why Review mode is not used; confirm the generated doc
  does.
- **Topics / Pattern REST reality** (T4): the confirmed endpoints, or the exact failure if the plan
  does not expose them.
- **Live spend**: `n_live_scores_planned`, scores actually written per the ledger, and the assertion
  that the OpenRouter ledger gained nothing.
- **Anything you prototyped through an MCP `create_*` tool** and what happened to it.
- The cockpit session (§7) is deliberately **out of scope** — it runs after this gate, by the operator.
  Do not attempt steps 1–5 of `docs/cockpit-session.md` yourself.
