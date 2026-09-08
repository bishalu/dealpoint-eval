# M9d: the free Braintrust features we never used, and how the live app plugs in

**Authority:** `specs/grilled-product-brief.md` (§1.4 deal page and "Run live", §1.5 API contract, §2.7 Braintrust
as surface), the Braintrust org as it stands on 2026-09-08 (verified by REST before this spec: project settings hold
only `default_preprocessor`; project scores are the four human sliders and the online rule; datasets are the seven
synced ones plus `maud-dealpoint-judge-packets`, `maud-dealpoint-playground-armA` and a UI-made "Dataset for Prompt
comparer example"; 739 root logs carry the metadata mirror and no first-class tags; 35 experiments), and the
engineer's directive of 2026-09-08: **do the free ones now: baseline, aggregate scores, regressions dataset; study
what we already have in Braintrust so this complements it; theorise, do not build, how the Next.js app would plug
into the live-trace side.** Runs in parallel with M9c; touches Braintrust only, never MLflow.

Everything here is created by code through the REST API (`dealpoint/eval/braintrust_showroom.py` steps, dry run by
default, `--live`, idempotent, ledgered with `n_scores=0`), so `just braintrust-showroom --live` recreates it in a
fresh org. **Zero scores by default**; the one step that would write scores is opt-in and prints its count first.

## 1. What we already have, and what these add

| have | add | why it complements |
|---|---|---|
| 35 experiments on one factorial schema, compared by hand in the Grid | a project **baseline** (D18) | the Grid's improvement/regression colouring and the Summary's deltas only exist relative to a baseline; none is set |
| six `obj/*` scores per row, the trust triple only in log metadata | **aggregate scores** on experiments (D19) | the Experiments page can sort and diff on a composite; today the composites live only on dashboards over logs |
| a hero case, a redacted twin, two lawyer-beats-judges packets, 25 cap-hit traces, all findable by query | a **regressions dataset** (D20) | the plan's "a bad trace becomes a test case in one click" was never done; the next milestone (M8) needs exactly this set to prove the stopping rule |
| logs filtered by metadata expressions | **first-class tags** on logs (D21) | tags are what the filter chips, saved views and online rules key on; the demo filters by typing BTQL today |
| the app instrumented (`@braintrust.traced` tools, `wrap_openai`) but never logging | a written **plug-in design** for the live app (D22) | so the Next.js milestone, when it comes, lands on Braintrust without a second tracing architecture |

## 2. Deliverables

### D18. Baseline and comparison key

- `settings.baseline_experiment_id` = the experiment `A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc` (arm A on GLM,
  the control of the SYSTEM ladder and the cheapest configuration), set with `PATCH /v1/project`. Recorded in the
  showroom manifest with the id resolved by name, never hardcoded.
- `settings.comparison_key` stays `input` after a test confirms every agent, judge and playground experiment
  uses the case id as `input` (they do; the retrieval experiments use the query and are compared only among
  themselves). If any family breaks that, set the key to `[input, metadata.case_id]` and say so.
- The tour's stop 3 gains one sentence: open any arm-D experiment and the Summary now shows deltas against A
  and the Grid colours each case as improvement or regression.

### D19. Aggregate scores (project scores of type weighted / minimum / maximum)

Only aggregates that are **exactly** derivable from the six scores every experiment row carries
(`obj/grounded_accuracy`, `obj/answer_correct`, `obj/citation_gold_overlap`, `obj/citation_verbatim`,
`obj/abstain_correct`, `obj/skill_adherence`):

1. `grounded and verbatim` = minimum(`obj/grounded_accuracy`, `obj/citation_verbatim`): the answer matched the
   expert span and its quote is word for word. The quality floor a lawyer would accept.
2. `citation quality` = weighted mean of `obj/citation_gold_overlap` (0.5) and `obj/citation_verbatim` (0.5).
3. `headline composite` = weighted mean, weights stated in the description: `obj/grounded_accuracy` 0.5,
   `obj/citation_gold_overlap` 0.2, `obj/citation_verbatim` 0.2, `obj/skill_adherence` 0.1. Labelled as a
   composite with chosen weights, never as an accuracy.

Not expressible, and said so in the tour: safe accuracy, precision when answering and net accuracy need a per-row
`misleading` score, and `obj/abstain_correct` is true on an answerable case whenever the system answered, right or
wrong, so no weighted, minimum or maximum of the existing scores equals them. Opt-in
`--score-trust-triple`: writes `trust/safe`, `trust/net`, `trust/correct_outcome` on every canonical experiment
row from the mirror (3 x 986 = 2,958 scores, printed and refused without `--live --score-trust-triple`). Default off.

Created with `POST /v1/project_score` (`score_type` in `weighted`, `minimum`, `maximum`; the exact `config`
shape is probed once live and recorded in the manifest and the tests' fake client). Idempotent by name.

### D20. The regressions dataset

`maud-dealpoint-regressions`, built by rule, not by hand, from rows already in the org:

- every cap-hit on a counterfactual case in the judged pool (the definition-absent loop; 25 traces);
- the two packets where all three judges were wrong and the lawyer right (`32fc075d8413`, `790521a3adab`);
- every misleading answer by the single-shot baseline on the judged pool (A@haiku, 6 cases);
- the hero pair: `contract_144__q05` (A abstained) and `contract_39__redacted_q05` (every D capped).

Each row: `input` = the case id (so it joins every experiment), `expected` = the gold answer (ABSTAIN on
counterfactuals) and gold span, `metadata` = `{reason, source_variant, source_log_id, first_seen_experiment,
rule}`; the `source_log_id` is the Logs root id, found by `metadata.case_id` and `variant_id`, so the row links
back to the trace it came from. One rule function, tested, produces the same rows every run. The tour's stop 4
gains the live action: open the redacted twin in Logs, "Add to dataset" -> `maud-dealpoint-regressions`, and
show the row that the code already created for it (the click is the demo; the code is the record).

### D21. Tags on logs

First-class tags on every root log, merged in one pass (free): `status:<ANSWERED|ABSTAINED|CAP_HIT|EXECUTION_FAILED>`,
`system:<A|B|C|D>`, `model:<short>`, `pool:<judged-18|test-32|retrieval|prompt-variant>`, `category:<...>`,
and `regression` on every log that feeds D20. Two saved Logs views use them: "Cap-hits" and "Regressions".
The online rule's filter can then be `tags` based; it stays on live replays.

### D22. How the live app plugs in (design note, `docs/braintrust-live-app.md`; nothing built)

Written against the brief's deal page and API contract, using only what already exists in the code:

1. **One initialisation.** `braintrust.init_logger(project=PROJECT)` at FastAPI startup behind an env flag;
   `dealpoint/agent/tools.py` is already decorated with `@braintrust.traced` and `dealpoint/llm/client.py`
   already applies `wrap_openai`, so `POST /api/run` produces a full span tree (agent, tool calls, LLM calls
   with real token counts) with no further instrumentation.
2. **Tag every live trace** `category:production`, `system:D`, `model:<the default>`, plus the mirror's
   `case_id` and `question_id` metadata, so the dashboards' `comparable = 0` rule keeps production out of the
   frozen comparisons and a "Production" view keeps it in view on its own.
3. **The online rule widens** from live replays to `category:production`, so every run the deal page triggers
   gets a `judge-professional` score within a minute; the deal page shows it beside the deterministic scores
   that the API computes on the spot.
4. **The trace link on the deal page** is the root span's permalink returned by `POST /api/run`, never a
   guessed URL; the brief already marks it "may be expired; supplementary".
5. **"Add to regressions"** on the deal page is one dataset insert with the trace's root id, the same shape
   D20 writes, so human-flagged failures and rule-found failures land in the same dataset.
6. **The experiments page** reads `data/reports/` (offline by design) and, when Braintrust is reachable,
   the aggregate scores of D19 and the baseline deltas of D18 through the REST API, cached.
7. **Costs**: production traces are processed data only; the judge is cents per run; nothing writes a
   metered score except the online rule, one per run.
8. **What stays out**: the Braintrust AI proxy (optional later, for judge-call caching), Braintrust tools
   (need the local index), any second tracing SDK.

## 3. Gate (`gate_m9d`, offline, fake REST client as in the cockpit tests)

- dry run prints: baseline experiment name and resolved id, three aggregate scores with their inputs and
  weights, the regressions dataset row count (deterministic, currently 33 or the rule's exact count) with
  each row's reason, the tag plan (739 logs, tag counts by key), two views;
- `--live` on the fake creates each object once; a second run creates nothing;
- every regressions row resolves to a log id and to a gold expectation; the hero pair is present;
- `--score-trust-triple` without `--live` prints 2,958 and writes nothing; with the fake and `--live` it
  writes exactly that many score merges and ledgers `n_scores=2958`;
- `docs/braintrust-live-app.md` names the three code seams (`init_logger`, `_traced`, `_maybe_wrap_openai`)
  and every API route of brief §1.5 that touches Braintrust;
- ruff, pyright, full offline suite; then the builder verifies live by REST read-back (settings, project
  scores, dataset row count, a tag filter) and records the ids in the showroom manifest.

## 4. Budget

0 scores, 0 model calls. Processed data: the tag merges and the dataset rows, well under 1 MB.

## 5. Out of scope

Building the Next.js or FastAPI app, live tracing itself, alerts (MCP-only), classifiers, the AI proxy,
anything in MLflow (M9c owns that), any change to scoring definitions or frozen artifacts.

## 6. Registration

`adws/adw_modules/milestones.py`: `m9d`, `spec_path="specs/milestones/m9d.md"`, `gate_marker="gate_m9d"`,
`needs_model=False`. Launch alongside M9c; the two write to different systems and different files
(`braintrust_showroom.py` and docs for M9d; `mlflow_*.py` for M9c).

## 7. Operator go-ahead (2026-09-08, engineer, recorded by the orchestrator)

The live writes this milestone makes to the Braintrust org `bishal.ai`, project `dealpoint-eval` (the key in
`.env.braintrust` resolves to it) are **authorized**: the baseline and comparison-key settings, the three
aggregate project scores, the `maud-dealpoint-regressions` dataset, the tags merged onto the root logs, and the
two saved Logs views. All are additive and idempotent and write 0 scores and make 0 model calls (§4). The
builder may run `just braintrust-showroom --live --only baseline,logtags,aggscores,regressions` and the REST
read-backs of §3 without asking again. The opt-in `--score-trust-triple` remains **not** authorized. Report the
three live-vs-spec numbers (root logs tagged, trust-triple plan count, regressions rows) as measured.
