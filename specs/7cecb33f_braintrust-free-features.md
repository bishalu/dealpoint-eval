# Build plan — M9d: Braintrust baseline, aggregate scores, regressions dataset, log tags, live-app design note

**Spec (read-only):** `specs/milestones/m9d.md`
**Requirements authority (read-only):** `specs/grilled-product-brief.md`
**Milestone id:** `m9d` — already registered in `adws/adw_modules/milestones.py:84-85`. Do not re-register.
**Branch:** `braintrust-org-scoped-ledger` (current).
**Runs in parallel with M9c**, which touches `dealpoint/eval/mlflow_*.py` and `docs/mlflow-*.md`. The only file both milestones need to edit is `pyproject.toml` (the `markers` list). See §9 "Conflict with M9c".

Done means: every item in the spec's §3 gate holds on disk, `uv run pytest -m "gate_m9d and not needs_network" -q` is green, and `uv run pytest -m "not needs_network and not needs_model" -q`, `uv run ruff check .` and `uv run pyright` are green.

---

## 0. What already exists (verified on disk, 2026-09-08)

Read this before writing code; it is all confirmed, not assumed.

### `dealpoint/eval/braintrust_showroom.py` (1155 lines) — the module you extend

- **Step registry:** `STEPS = ("tag", "rows", "logs", "mirror", "raglogs", "review", "params", "prompts", "judges", "views", "playground", "judgeplayground", "promptlogs")` at line 71. `main()` (line 1105) dispatches with `globals()[f"step_{name}"](api, live, manifest)`. Every step has the signature `def step_x(api: Api, live: bool, manifest: dict) -> None`.
- **CLI:** hand-rolled `argv` scanning in `main()`, no argparse. Existing flags: `--live`, `--only a,b`, `--replay <case>:<variant>`, `--run-playground`, `--variants a,b`. Dry run is the default. `--live` without a key prints to stderr and returns 2.
- **REST client:** `class Api` (line 173). `Api(key)`, `self.h = {"Authorization": f"Bearer {key}"}`, `API = "https://api.braintrust.dev/v1"`. Methods: `_call(method, path, **kw)` (backoff on 429 and the fetch endpoint's transient 400), `get(path, params)`, `patch(path, body)`, `post(path, body)`, `resolve_project() -> (project_id, owner_user_id)` (sets module globals `PROJECT_ID`, `OWNER_USER_ID`), `experiments() -> list[dict]` (paged), `fetch_rows(experiment_id) -> list[dict]`. **Paths are relative, without the `/v1` prefix** (`api.get("project", {...})`, `api.post(f"experiment/{id}/insert", {...})`).
- **A second, different client exists:** `braintrust_cockpit.RestClient` (line 800) takes **absolute** `/v1/...` paths. `step_views` (line 1084) imports it plus `_upsert_view` and `_view_data_for` from the cockpit. Keep the two conventions straight; do not mix them in one call site.
- **Root-span detection:** `_is_root(event)` (line 235) — `event["is_root"]` when present, else "no `span_parents`". Never `span_id == root_span_id`.
- **Score guard:** `_assert_scoreless(events)` (line 241) raises if any payload carries a `scores` key. Every new score-free write must pass through it.
- **Ledger:** `_ledger(experiment: str, key: str, n_scores: int = 0)` (line 255) appends one JSON line to `_ledger_path()` = `org_report_path("braintrust_score_ledger.jsonl", override_env="BRAINTRUST_LEDGER_FILE")` = `data/reports/orgs/<org-slug>/braintrust_score_ledger.jsonl`. `_ledger_keys()` (line 263) returns `{(experiment, key)}` — the idempotency check the steps use before writing.
- **Manifest:** `_manifest_path()` = `data/reports/orgs/<org-slug>/showroom_manifest.json`. Written **only on `--live`**, at the end of `main()`, from the `manifest` dict the steps mutate. Existing keys: `mode`, `started_at`, `project`, `project_id`, `org`, `env_file`, `ledger`, plus one key per step.
- **Log fetch:** `_all_root_logs(api)` (line 680) pages `POST project_logs/{PROJECT_ID}/fetch` with `{"limit": 500, "cursor": ...}` and filters by `_is_root`.
- **Log merge:** `step_mirror` (line 692) posts `POST project_logs/{PROJECT_ID}/insert` with `{"events": [...]}` in batches of 100, each event `{"id": <existing log id>, "metadata": {...}, "_is_merge": True}`. This is how you merge onto an existing log. Experiment-row merges (`step_rows`, line 308) additionally carry `"_merge_paths": [["metadata"]]`.
- **The metadata mirror:** `metadata_mirror(row, *, case, judge_dims, human, deepeval, per_judge, category)` (line 354) and `mirror_for(case_id, variant_id, row, ctx, category)` (line 455). Keys it writes that matter here: `system_label`, `model_label`, `variant_label`, `status`, `case_set`, `question_id`, `misleading`, `silent_failure`, `net_accuracy`, `safe`, `answered`, `correct_all`, `correct_answered`, `comparable`, `case_pool`. Root logs also carry, from `_log_one_tree` (line 629): `case_id`, `variant_id`, `arm`, `model`, `category`.
- **Log categories in the org:** `judged`, `agent`, `retrieval`, `prompt-variant`, the six representative-case categories (`successful_direct`, `retrieval_rescue`, `defined_term_cross_ref`, `inefficient_trajectory`, `wrong_answer`, `abstention_counterfactual`) and `live-replay-<stamp>`.
- **`stored_row_index() -> dict[(case_id, variant_id), row]`** (line 475): every stored result row under both id conventions. **This is the offline join you build the tag plan and the regressions rule on.**
- **`agent_run_log_plan(already_planned)`** (line 568), `retrieval_log_rows()` (line 510), `prompt_variant_log_rows(...)` (line 544), `playground_rows()` (line 874), `judge_packet_rows()` (line 921).
- **No helper exists** for: resolving an experiment by name to an id (use `api.experiments()` and match on `name`), listing/creating project scores, listing datasets by name via `Api`, or patching project settings via `Api`. You add the first four.
- **`justfile:255`:** `braintrust-showroom *ARGS: uv run python -m dealpoint.eval.braintrust_showroom "$@"`. No new recipe is needed; the new flags flow through.

### `dealpoint/eval/braintrust_cockpit.py` — the view/settings idioms to copy

- `_upsert_view(rest_client, object_type, object_id, view_type, name, view_data, description=None)` (line 950): `GET /v1/view?object_type&object_id` → match on `(name, view_type)` → `POST /v1/view` (create, no `id` key allowed) or `PATCH /v1/view/{id}` (update). Returns `{"id": ..., "created": bool}`.
- `_view_data_for(definition)` (line 920): `{"btql": ...}` → `{"search": {"filter": [{"btql": ...}]}}` (+ optional `"sort"`). `view_type` for Logs views is `"logs"`.
- Project settings patch idiom (line 1043): `rest_client.patch(f"/v1/project/{project_id}", {"settings": {...}})`. Today the project's settings hold only `default_preprocessor`; `PATCH` merges, so send only the keys you set.
- **No `project_score` code exists anywhere in the repo.** D19 introduces it.
- The online scoring rule was created in the Braintrust UI, not by code. D21's sentence about the rule's filter is **documentation only** — do not try to create or edit an automation via REST.

### Tests

- `tests/test_braintrust_cockpit.py` holds `FakeRestClient` (line 70) — `get`/`post`/`patch` over absolute `/v1/...` paths, keeps `self.views`, `self.functions`, `self.post_calls`, and records `PATCH /v1/project/{id}` into `self.project_settings`. `pytestmark = pytest.mark.gate_m7b` at module level.
- `tests/test_braintrust_showroom.py` has no fake client; it tests pure functions and monkeypatches module globals (`showroom.LEDGER_PATH`, `showroom.MANIFEST_PATH`, `adapter.org_slug`).
- Idempotency style to copy — `test_sync_views_and_dashboard_idempotent_second_run_creates_nothing_new` (line 229): run twice, assert the object count is unchanged, the ids are identical, and every result's `created` is `False`.
- `tests/conftest.py` has no Braintrust fixtures. There is no network guard fixture; offline safety comes from never constructing a real client.
- `pyproject.toml` `[tool.pytest.ini_options] markers` (lines 47-61) lists `gate_m0 … gate_m7, gate_m7b, gate_m9, gate_m9b, needs_network, needs_model`. **`gate_m9d` is missing and you must add it.**
- Lint/type config: `[tool.ruff] line-length = 100`, `extend-exclude = ["adws", ".claude"]`; `[tool.pyright] include = ["dealpoint", "tests"]`, `typeCheckingMode = "basic"`.

### Data facts I verified by running the numbers (use these; do not re-derive from scratch)

Judged pool: `data/eval/judged_subset.json` — `case_ids` (18) and `variants` (6: `A@haiku`, `D@haiku`, `D@glm`, `D@deepseek-v4-flash`, `D@qwen3.7-flash`, `D@gemini-3.1-flash-lite`), each with a `results_path` JSONL. 6 of the 18 cases are `case_set == "counterfactual"`.

Result-row field names (a row of one of those JSONL files):
`case_id`, `case_set` (`test` | `counterfactual`), `arm`, `model`, `question_id`, `finding.answer`, `record.status` (**status lives at `record.status`, not top level**), `scores.grounded_accuracy`, `scores.abstain_correct`, `scores.cap_hit`, `scores.citation_verbatim`, `scores.citation_gold_overlap`, `scores.skill_adherence`.

`misleading` is not stored on the row; it is computed by `metadata_mirror`:
```
correct_all = bool(grounded_accuracy) or (case_set == "counterfactual" and bool(abstain_correct))
misleading  = (status == "ANSWERED") and not correct_all
```

Gold: `dealpoint.eval.cases.find_case(case_id)` returns `gold_answer` and `gold_spans`. Counterfactual cases have `gold_answer == "ABSTAIN"` and `gold_spans == []`.

Packet ids → (case, variant), from `data/eval/calibration/variant_key.json`:
- `32fc075d8413` → `{"arm": "A", "case_id": "contract_99__redacted_q06", "model": "anthropic/claude-haiku-4.5", "variant_id": "A@haiku"}`
- `790521a3adab` → `{"arm": "A", "case_id": "contract_99__q11",           "model": "anthropic/claude-haiku-4.5", "variant_id": "A@haiku"}`

Measured group sizes (I ran this):
- cap-hits on a counterfactual case in the judged pool: **25**, over 6 cases × the arm-D variants.
- misleading answers by `A@haiku` on the judged pool: **6** — `contract_103__q10`, `contract_144__q09`, `contract_32__q04`, `contract_32__q06`, `contract_39__q01`, `contract_99__redacted_q06`.
- the two packets: `(contract_99__redacted_q06, A@haiku)` — **also in the misleading six** — and `(contract_99__q11, A@haiku)`.
- hero pair: `(contract_144__q05, A@haiku)` (ABSTAINED) and `(contract_39__redacted_q05, D@glm)` (CAP_HIT, the tour's Debugger trace) — the latter is **also in the 25**.
- Union keyed on `(case_id, variant_id)`: 25 + 6 + 2 + 2 − 2 overlaps = **33**. This is the spec's 33. It only comes out at 33 if the row key is `(case_id, variant_id)` and the two overlaps are deduplicated with the reasons merged.

Root log ids are **not** on disk. Both `data/reports/orgs/*/showroom_manifest.json` are from partial `--only` runs and have `logs.roots == []`. `source_log_id` must therefore be resolved from the live Logs by matching `metadata.case_id` + `metadata.variant_id`; offline it is `None` in the plan and supplied by the fake client in the gate test.

Offline-computable log plan totals (I ran these): trace logs 265 (`judged` 108, `agent` 151, six representative) + retrieval 406 = **671**, plus 72 prompt-variant logs (18 × 4) = 743 planned. The org reports **739** root logs. Do not hardcode 739 in the rule; see §9 "Numbers that may not match".

### Code seams for the design note (D22) — verified

- `dealpoint/agent/tools.py:27` `def _traced(**span_kwargs) -> Callable[[_F], _F]` — `braintrust.traced` when importable, identity decorator otherwise. Applied at lines 64, 80, 91 to `search_agreement`, `get_section`, `lookup_defined_term`.
- `dealpoint/llm/client.py:134` `def _maybe_wrap_openai(client)` — applies `braintrust.wrap_openai` when `braintrust_adapter.braintrust_available()` (importable **and** a key is loadable), else returns the client unchanged. Called in `OpenRouterClient.__init__` (~line 228).
- `braintrust.init_logger` exists in the installed SDK with signature `(project=None, project_id=None, async_flush=True, app_url=None, api_key=None, org_name=None, force_login=False, set_current=True, state=None, environment=None)`. It is used today only inside showroom steps (`step_logs`, `step_raglogs`, `step_promptlogs`); **no application-level `init_logger` exists.**
- **There is no `dealpoint/api/` and no `web/`.** The design note must say so plainly: the three seams exist, the app does not.
- `PROJECT = os.environ.get("BRAINTRUST_PROJECT", "dealpoint-eval")` (`braintrust_adapter.py:28`). Other env vars: `BRAINTRUST_API_KEY`, `BRAINTRUST_ENV_FILE`, `BRAINTRUST_LEDGER_FILE`.
- `docs/demo-tour.md`: stop 3 is `### Stop 3. Does agency help? Does better RAG help? Does the skill help?` (its "One-liner:" ends the section at ~line 133); stop 4 is `### Stop 4. Watch one agent think` (~line 137-165, one-liner at ~line 165).

---

## 1. Files you will touch

| file | change |
|---|---|
| `dealpoint/eval/braintrust_showroom.py` | four new steps + their pure rule functions + two new CLI flags; extend `STEPS` |
| `tests/test_braintrust_m9d.py` | **new** — the whole `gate_m9d` suite with its own fake client |
| `pyproject.toml` | add `"gate_m9d: milestone 9d gate"` to `markers` |
| `docs/braintrust-live-app.md` | **new** — the D22 design note |
| `docs/demo-tour.md` | one sentence appended to stop 3, one live action added to stop 4 |
| `justfile` | add a `gate-m9d` recipe next to `gate-m9` |

Do **not** touch: `specs/grilled-product-brief.md`, `specs/milestones/m9d.md`, any `mlflow_*` module, any `docs/mlflow-*.md`, any frozen artifact under `data/eval/` or `data/results/`, `adws/adw_modules/milestones.py` (already correct).

---

## 2. D18 — baseline and comparison key (`step_baseline`)

Add `"baseline"` to `STEPS` (put it first, before `"tag"`; it is a project-level setting and the cheapest step).

```python
BASELINE_EXPERIMENT = "A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc"
COMPARISON_KEY = "input"
```

**`def resolve_experiment_id(api: Api, name: str) -> str | None`** — pure-ish helper over `api.experiments()`, matching on `name`. Never hardcode an id.

**`def comparison_key_is_safe(rows_by_experiment: dict[str, list[dict]]) -> tuple[bool, list[str]]`** — a pure function, given `{experiment_name: [root events]}`, returns `(ok, offenders)`. `ok` is true when every experiment whose `classify_experiment(name)["axis"]` is in `("system", "model", "judge", "prompt")` has, for each of its root rows, a string `input` that equals the row's `metadata.case_id` (or, when `metadata.case_id` is absent, is a non-empty string unique within the experiment). Retrieval experiments (`axis == "retrieval"`) are exempt — their `input` is the query and they are only ever compared with each other. This is the test the spec demands before leaving `comparison_key` at `input`.

**`def step_baseline(api, live, manifest)`**

Dry run prints, unconditionally:
```
baseline: A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc (arm A on GLM, the control of the SYSTEM ladder and the cheapest configuration)
baseline: comparison_key stays 'input' (every agent, judge and playground experiment uses the case id as input; retrieval experiments use the query and are compared only among themselves)
```
and when live also prints the resolved id.

Live behaviour:
1. `exp_id = resolve_experiment_id(api, BASELINE_EXPERIMENT)`; if `None`, print a clear failure and leave settings untouched (do not raise — the other steps must still run).
2. Run the comparison-key check by fetching root rows of the non-retrieval experiments (`api.fetch_rows`, filtered by `_is_root`). If `ok` is false, set `settings.comparison_key = ["input", "metadata.case_id"]` instead, and print exactly which experiments forced it.
3. `api.patch("project/" + PROJECT_ID, {"settings": {"baseline_experiment_id": exp_id, "comparison_key": key}})` — `PATCH` merges, so `default_preprocessor` survives; do not read-modify-write the whole settings object.
4. `manifest["baseline"] = {"experiment": BASELINE_EXPERIMENT, "experiment_id": exp_id, "comparison_key": key, "comparison_key_offenders": offenders}`.
5. `_ledger("project", "baseline", 0)` and `_ledger("project", "comparison-key", 0)`.

Idempotency: the `PATCH` is naturally idempotent; also skip the write when `("project", "baseline") in _ledger_keys()` **and** a `GET project` shows the same `baseline_experiment_id`, so a second run reports "already set" and issues no write.

**Tour:** append one sentence to the end of stop 3 in `docs/demo-tour.md`, before its `One-liner:` line, along these lines: *"`A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc` is now the project baseline, so opening any arm-D experiment shows the Summary's deltas against A and the Grid colours each case as an improvement or a regression."*

---

## 3. D19 — aggregate project scores (`step_aggscores`)

Add `"aggscores"` to `STEPS`.

**`def aggregate_score_defs() -> list[dict]`** — pure, tested, the single source of truth:

```python
[
  {"name": "grounded and verbatim", "score_type": "minimum",
   "inputs": ["obj/grounded_accuracy", "obj/citation_verbatim"],
   "description": "MINIMUM of obj/grounded_accuracy and obj/citation_verbatim: the answer matched the expert span AND its quote is word for word. The quality floor a lawyer would accept."},
  {"name": "citation quality", "score_type": "weighted",
   "inputs": ["obj/citation_gold_overlap", "obj/citation_verbatim"], "weights": [0.5, 0.5],
   "description": "Weighted mean, 0.5 obj/citation_gold_overlap + 0.5 obj/citation_verbatim. A composite with chosen weights, not an accuracy."},
  {"name": "headline composite", "score_type": "weighted",
   "inputs": ["obj/grounded_accuracy", "obj/citation_gold_overlap", "obj/citation_verbatim", "obj/skill_adherence"],
   "weights": [0.5, 0.2, 0.2, 0.1],
   "description": "Weighted mean with chosen weights: obj/grounded_accuracy 0.5, obj/citation_gold_overlap 0.2, obj/citation_verbatim 0.2, obj/skill_adherence 0.1. A composite with chosen weights, NOT an accuracy."},
]
```

The `inputs`/`weights` keys above are the plan's own vocabulary. **The exact REST `config` shape for `POST /v1/project_score` is not known offline** — no code in the repo has ever created one. Do this:

1. Write `def _project_score_body(defn: dict) -> dict` that maps a def to the REST body. Start from the documented shape
   `{"project_id": PROJECT_ID, "name": ..., "score_type": ..., "description": ..., "config": {"multi_select": False, "online": None, "destination": None, "weights"/"aggregate": ...}}`
   and treat the `config` sub-shape as **the one thing to probe live**: send one score, read it back with `GET /v1/project_score?project_id=...`, and if the server rejects or normalises the body, adjust `_project_score_body` to what the server accepts.
2. Record the accepted shape verbatim in `manifest["aggregate_scores"]["config_shape"]` **and** in a module-level docstring comment on `_project_score_body` dated `2026-09-08, confirmed live`, exactly as `_upsert_view` and `_view_data_for` document their probed shapes.
3. Mirror that same accepted shape in the gate test's fake client so the offline test asserts the real body, not a guess.

**Idempotency by name:** `GET project_score` with `{"project_id": PROJECT_ID, "limit": 200}`, match on `name`; create with `POST project_score` only when absent, otherwise `PATCH project_score/{id}` with the same body. The project already has four human-slider scores and one online rule score — **never** delete or overwrite those; match strictly on the three new names.

Dry run prints each score with its inputs and weights, e.g.:
```
aggscores: 3 project scores
  grounded and verbatim = minimum(obj/grounded_accuracy, obj/citation_verbatim)
  citation quality = weighted(obj/citation_gold_overlap x 0.5, obj/citation_verbatim x 0.5)
  headline composite = weighted(obj/grounded_accuracy x 0.5, obj/citation_gold_overlap x 0.2, obj/citation_verbatim x 0.2, obj/skill_adherence x 0.1)
  not expressible as an aggregate of the six: safe accuracy, precision when answering, net accuracy (they need a per-row `misleading` score, and obj/abstain_correct is true on an answerable case whenever the system answered, right or wrong)
```

That last line is a required print — the spec demands the limit be said out loud, and the tour must repeat it (see §7).

`manifest["aggregate_scores"] = [{"name", "score_type", "id", "created"} …]`; `_ledger("project", f"score:{name}", 0)` per score.

### `--score-trust-triple` (opt-in, the only step that could write scores)

**`def trust_triple_plan() -> list[dict]`** — pure, offline, over `stored_row_index()` and the canonical experiment plans. Returns one entry per `(experiment_name, case_id, variant_id)` for the **canonical** experiments: those whose `classify_experiment(name)["axis"]` is `system`, `model` or `judge` (the A→D ladder on glm and haiku, the Pareto arm-D runs, the `pareto-*` mirrors, `judged-*` and `judge-*`). Each entry carries the three values read straight from the mirror:
`trust/safe` = `mirror["safe"]`, `trust/net` = `mirror["net_accuracy"]`, `trust/correct_outcome` = `mirror["correct_all"]`.

CLI: `--score-trust-triple`.
- Without `--live`: print `trust triple: would write 3 x <n> = <3n> scores on <k> canonical experiments; refusing without --live --score-trust-triple` and **return without writing anything**.
- With `--live --score-trust-triple`: write via `POST experiment/{id}/insert` with `{"events": [{"id": row_id, "scores": {...}, "_is_merge": True, "_merge_paths": [["scores"]]}]}` in batches of 100, and ledger `_ledger(experiment_name, "trust-triple", n_scores_written)`.
- `_assert_scoreless` must **not** be applied to this path (it is the one legitimate score write). Guard it instead by asserting that this code path is unreachable unless both flags are present, and by printing the count before the first write.
- The step is **never** in `STEPS`; it runs only when the flag is given, like `--run-playground`.

The spec pins `3 × 986 = 2,958`. `986` is a live row count that cannot be reproduced offline from the repo's artifacts (the local canonical result rows total 260 agent + 108 judged + 108 judge, and the `pareto-*` experiments are server-side duplicates). Therefore:
- the gate test asserts `printed_total == 3 * len(trust_triple_plan())` and that nothing was written — **not** the literal 2958;
- the builder records the live number in `manifest["trust_triple"] = {"rows": n, "scores": 3n}` during live verification, and **reports in the final summary whether it is 2,958**. If it is not, that is a spec-vs-org difference to report, not to paper over by bending the canonical-experiment rule.

---

## 4. D20 — the regressions dataset (`step_regressions`)

Add `"regressions"` to `STEPS`.

```python
REGRESSIONS_DATASET = "maud-dealpoint-regressions"
HERO_PAIR = (("contract_144__q05", "A@haiku"), ("contract_39__redacted_q05", "D@glm"))
LAWYER_BEATS_JUDGES_PACKETS = ("32fc075d8413", "790521a3adab")
```

**`def regression_rows() -> list[dict]`** — one pure, deterministic, offline function. Build in this order, keyed on `(case_id, variant_id)`, deduplicating by merging `reason` strings (comma-joined, in rule order) and keeping the **first** rule that matched in `rule`:

1. **`cap_hit_counterfactual`** — for every variant in `judged_subset.json` and every case in its results: `case_set == "counterfactual"` and `record.status == "CAP_HIT"`. Reason: `"cap-hit on a counterfactual case: the definition-absent loop"`. → 25 rows.
2. **`lawyer_beats_judges`** — the two packet ids, resolved through `data/eval/calibration/variant_key.json` to `(case_id, variant_id)`. Reason: `"all three judges were wrong and the lawyer right (packet <pid>)"`. → 2 rows, 1 of which merges into rule 3's set.
3. **`misleading_single_shot_baseline`** — variant `A@haiku` on the judged pool where `status == "ANSWERED"` and not `correct_all` (formula in §0). Reason: `"the single-shot baseline answered and misled"`. → 6 rows.
4. **`hero_pair`** — the two `HERO_PAIR` entries. Reasons: `"hero case: arm A abstained with the passages in hand"` and `"the redacted twin: every arm-D model hit the cap"`. → 2 rows, 1 of which merges into rule 1's set.

**Expected total: 33.** Assert this in the test as `len(regression_rows()) == 33` **and** assert the per-rule counts `(25, 2, 6, 2)` and the merged-overlap count `2`, so a future data change fails loudly with a readable diff rather than silently drifting.

Each row:
```python
{
  "id": f"{case_id}|{variant_id}",          # deterministic dataset row id -> idempotent inserts
  "input": case_id,                          # joins every experiment (comparison_key = input)
  "expected": {"answer": gold_answer, "gold_spans": gold_spans},   # from cases.find_case
  "metadata": {
      "reason": "...",                       # one string, merged in rule order
      "source_variant": variant_id,
      "source_log_id": None,                 # filled live, see below
      "first_seen_experiment": ...,          # see below
      "rule": "cap_hit_counterfactual" | "lawyer_beats_judges" | "misleading_single_shot_baseline" | "hero_pair",
      "case_set": case_set,
      "question_id": question_id,
      "status": record.status,
      "packet_id": pid,                      # only for rule 2
  },
}
```

`first_seen_experiment` is derived, not guessed: for a judged-pool `(case, variant)` it is the `judge-<variant_id>` experiment name; for a case outside the judged pool it is the arm/model experiment name from `braintrust_sync._agent_experiment_plans()`. Write a small `def first_seen_experiment(case_id: str, variant_id: str) -> str` and test it on the hero pair.

`source_log_id`: **`def resolve_source_log_ids(api: Api, rows: list[dict]) -> int`** — fetch `_all_root_logs(api)`, build `{(metadata.case_id, metadata.variant_id): log["id"]}` preferring `category in ("judged", "agent")` over representative/replay categories, and set `row["metadata"]["source_log_id"]` in place. Returns the number resolved. Live only; when a row cannot be resolved, leave `None` and count it — print the unresolved count and list the rows.

`step_regressions(api, live, manifest)`:
- Dry run prints the count and **every row's reason**, one per line: `  contract_39__redacted_q05 | D@glm — cap-hit on a counterfactual case: the definition-absent loop [cap_hit_counterfactual, the redacted twin: every arm-D model hit the cap]`. The gate reads these lines.
- Live: `braintrust.init_dataset(project=PROJECT, name=REGRESSIONS_DATASET, description=...)`, then `ds.insert(id=row["id"], input=..., expected=..., metadata=...)` for every row, `ds.flush()`. Passing an explicit `id` is what makes the second run a no-op update rather than 33 new rows — the same idiom `step_playground` and `step_judgeplayground` already use.
- `manifest["regressions"] = {"dataset": REGRESSIONS_DATASET, "rows": 33, "by_rule": {...}, "log_ids_resolved": n, "unresolved": [...]}`; `_ledger("dataset", "regressions", 0)`.

Dataset description (visible in the UI, so make it carry the argument):
> Failures found by rule, not by hand: every cap-hit on a counterfactual case in the judged pool, the two packets where all three judges were wrong and the lawyer right, every misleading answer by the single-shot baseline, and the hero pair. `input` is the case id so a row joins every experiment; `metadata.source_log_id` links back to the trace it came from.

**Tour:** stop 4 in `docs/demo-tour.md` gains the live action — open the redacted twin in Logs, **Add to dataset** → `maud-dealpoint-regressions`, and point out that the row is already there because the rule created it. Keep it to two or three sentences and place it after the twin paragraph, before the "266 traces live here" paragraph.

---

## 5. D21 — first-class tags on logs (`step_logtags`)

Add `"logtags"` to `STEPS`, after `"mirror"` (tags derive from mirrored metadata).

**`def tags_for_log(metadata: dict) -> list[str]`** — pure, the whole rule, sorted and deduplicated:

- `status:<record status>` when `metadata["status"]` is one of `ANSWERED`, `ABSTAINED`, `CAP_HIT`, `EXECUTION_FAILED`.
- `system:<A|B|C|D>` from `metadata["arm"]`, else parsed from `metadata["variant_id"]` before the `@`.
- `model:<short>` from `metadata["model_label"]`, else `_short(metadata["model"])`.
- `pool:<judged-18|test-32|retrieval|prompt-variant>` — `judged` category → `judged-18`; `agent` category → `test-32`; `retrieval` → `retrieval`; `prompt-variant` → `prompt-variant`; representative and `live-replay-*` categories → no `pool:` tag (they are not a comparison pool; say so in the docstring).
- `category:<category>` from `metadata["category"]`, verbatim, including `live-replay-<stamp>`.
- `regression` when `(metadata["case_id"], metadata["variant_id"])` is in the D20 row set.

Keys absent from a log simply produce no tag — retrieval logs get `pool:retrieval`, `category:retrieval` and nothing else. Test that explicitly.

**`def tag_plan() -> dict`** — offline, from the same local artifacts the log steps use (`judged_subset`, `representative_cases`, `agent_run_log_plan`, `retrieval_log_rows`, the four prompt variants × 18 cases). Returns `{"planned": n, "by_key": {"status": {...}, "system": {...}, "model": {...}, "pool": {...}, "category": {...}, "regression": n}}`. This is what the dry run prints; today it plans 743 logs against the org's 739 (see §9).

`step_logtags(api, live, manifest)`:
- Dry run prints `logtags: <planned> logs planned from local artifacts; tag counts by key: status={...} system={...} model={...} pool={...} category={...} regression=<n>` plus the two view names.
- Live: `_all_root_logs(api)`, compute `tags_for_log(e["metadata"] or {})` per log, merge in **one pass** with `POST project_logs/{PROJECT_ID}/insert`, batches of 100, each event `{"id": e["id"], "tags": tags, "_is_merge": True}`. Braintrust `tags` is a first-class top-level field on the event, not metadata — do **not** nest it under `metadata`. Run `_assert_scoreless` on the batch. Print the actual number of logs tagged alongside the planned number.
- Two saved Logs views via the cockpit's `RestClient` + `_upsert_view` + `_view_data_for`, exactly as `step_views` does:
  - `"Cap-hits"`, `view_type="logs"`, btql `tags includes 'status:CAP_HIT'`
  - `"Regressions"`, `view_type="logs"`, btql `tags includes 'regression'`
  The BTQL operator for a tag membership test is **not confirmed offline**. Probe it live once (the Logs UI filter chip generates it); if `includes` is rejected, fall back to whatever the server accepts and record the accepted expression in `manifest["logtag_views"]` and in a dated comment, the same discipline as `_upsert_view`. The gate test asserts the view names, `view_type == "logs"`, idempotency and that the btql string mentions the tag it filters on — not a specific operator spelling.
- `manifest["logtags"] = {"planned": ..., "tagged": ..., "by_key": {...}, "views": [...]}`; `_ledger("logs", "tags", 0)` and `_ledger("views", "logtag-views", 0)`.

The online scoring rule is **not** touched. The tour and the design note say its filter can now be tag-based; nothing in code creates or edits an automation.

---

## 6. D22 — `docs/braintrust-live-app.md` (design note, nothing built)

A written note, no code. Structure it as the spec's eight numbered points, each grounded in a real file path and line. Requirements the gate checks:

- It names all three seams by their **exact identifiers**: `init_logger`, `_traced`, `_maybe_wrap_openai`, each with its file path.
- It names **every route of brief §1.5** and says, per route, whether it touches Braintrust:
  - `GET /api/agreements`, `GET /api/agreements/{id}/text`, `GET /api/questions`, `GET /api/market/{question_id}` — no Braintrust.
  - `GET /api/cases/{agreement_id}/{question_id}` — reads the cached trace link, returns `braintrust_url|null`; the brief already marks it "may be expired; supplementary".
  - `POST /api/run` — the one route that produces a live trace: one `init_logger` at startup behind an env flag, the existing `@_traced` tools and `wrap_openai` client give the full span tree with real token counts, and the response carries the root span's permalink returned by the SDK, never a guessed URL.
  - `GET /api/reports` — offline by design, reads `data/reports/`; only when Braintrust is reachable does it add D19's aggregate scores and D18's baseline deltas over REST, cached.
- It states plainly that **no FastAPI app and no `web/` exist in this repo today** — this is a design, not a description.
- It covers, one short section each: the tags a live trace carries (`category:production`, `system:D`, `model:<default>`, plus the mirror's `case_id` and `question_id`) and why `comparable = 0` keeps production out of the frozen comparisons while a "Production" view keeps it visible; widening the online rule from live replays to `category:production` so each run gets a `judge-professional` score within a minute, shown beside the deterministic scores the API computes on the spot; "Add to regressions" on the deal page as one dataset insert with the trace's root id in exactly D20's row shape, so human-flagged and rule-found failures land in the same dataset; costs (production traces are processed data only, the judge is cents per run, one metered score per run from the online rule and nothing else); and what stays out (the Braintrust AI proxy, Braintrust tools, any second tracing SDK).
- Cross-link it from `docs/demo-tour.md` and from the module docstring of `braintrust_showroom.py`.

---

## 7. Tour edits (`docs/demo-tour.md`)

Three small, surgical edits. Do not restructure the document.

1. **Stop 3**, before its `One-liner:` — one sentence on the baseline and what the Summary and Grid now show (text in §2).
2. **Stop 3 or the aggregate-scores mention**, wherever the six `obj/*` scores are introduced — one sentence naming the three new aggregates and one sentence saying what is *not* expressible from them (safe accuracy, precision when answering, net accuracy) and why.
3. **Stop 4**, after the redacted-twin paragraph — the "Add to dataset → `maud-dealpoint-regressions`" live action, noting the row is already there because the rule wrote it.

---

## 8. The gate — `tests/test_braintrust_m9d.py`

`pytestmark = pytest.mark.gate_m9d` at module level. Add `"gate_m9d: milestone 9d gate"` to `pyproject.toml` markers and a `gate-m9d` recipe to the justfile mirroring `gate-m9`:

```
# milestone 9d acceptance gates only, offline
gate-m9d:
    uv run pytest -m "gate_m9d and not needs_network and not needs_model" -q
```

**Fake client.** One `FakeApi` in the test module, matching `Api`'s **relative-path** convention (`"project"`, `"experiment"`, `"project_score"`, `f"project_logs/{id}/fetch"`, `f"project_logs/{id}/insert"`, `f"experiment/{id}/insert"`, `f"project/{id}"`). It keeps `self.project_settings`, `self.project_scores`, `self.log_events`, `self.experiment_events`, `self.posts` and seeds a small set of root logs whose metadata covers each family (a `judged` cap-hit, an `agent` answered row, a `retrieval` row, a `prompt-variant` row, and the hero pair) so `resolve_source_log_ids` has something to resolve. For the two saved views reuse the cockpit's `FakeRestClient` shape (absolute `/v1/...` paths) — import it or copy it; do not build a hybrid.

Tests, one per gate bullet:

1. `test_dry_run_prints_the_baseline_name_and_the_comparison_key` — capsys; the experiment name appears, `comparison_key` and `input` appear; with the fake and `--live`, the resolved id appears and `fake.project_settings["baseline_experiment_id"]` equals it.
2. `test_comparison_key_stays_input_because_every_non_retrieval_experiment_keys_on_the_case_id` — `comparison_key_is_safe` returns `(True, [])` on a fixture that mirrors the real families, and `(False, [name])` when one experiment's `input` is not the case id.
3. `test_aggregate_score_defs_are_exactly_derivable_from_the_six_obj_scores` — every input name is in the six; `minimum` has no weights; both `weighted` defs have weights summing to 1.0; the headline weights are `(0.5, 0.2, 0.2, 0.1)` in that order; each description states its weights and the headline one contains the word "composite" and not the word "accuracy" as its label.
4. `test_dry_run_says_which_trust_metrics_are_not_expressible` — capsys; the printed line names safe accuracy, precision when answering and net accuracy, and gives the `abstain_correct` reason.
5. `test_project_scores_are_created_once_and_never_touch_the_existing_four` — seed the fake with the four human sliders and the online-rule score; run twice; assert 5 + 3 scores, ids stable on the second run, and the pre-existing five are byte-identical.
6. `test_regression_rows_are_deterministic_and_33_with_the_rule_counts` — `regression_rows()` twice gives identical output; `len == 33`; per-rule counts `25 / 2 / 6 / 2` before dedup; exactly 2 rows carry a merged reason; ids are unique.
7. `test_every_regression_row_has_a_gold_expectation` — every row's `expected["answer"]` is a non-empty string; every counterfactual row's is `"ABSTAIN"` with `gold_spans == []`; every `test`-set row has at least one gold span.
8. `test_the_hero_pair_is_present` — `("contract_144__q05", "A@haiku")` and `("contract_39__redacted_q05", "D@glm")` are both in the row set, with `hero_pair` reasons on both.
9. `test_every_regression_row_resolves_to_a_log_id` — against the fake's seeded logs, `resolve_source_log_ids` fills every `source_log_id`; the unresolved list is empty.
10. `test_regressions_dataset_is_idempotent` — second live run over the fake inserts the same 33 ids and creates nothing new.
11. `test_tags_for_log_covers_every_family` — one assertion per family: a judged cap-hit yields `status:CAP_HIT`, `system:D`, `model:glm`, `pool:judged-18`, `category:judged`; a retrieval log yields only `pool:retrieval` and `category:retrieval`; a representative log yields no `pool:` tag; a D20 log yields `regression`.
12. `test_tag_plan_counts_by_key_and_is_printed` — `tag_plan()["planned"]` matches the sum of the local families, and the dry-run output contains the counts by key.
13. `test_log_tags_are_merged_in_one_pass_and_are_idempotent` — with the fake and `--live`, every seeded root log gets a `tags` list, the events carry `_is_merge` and no `scores` key, and a second run produces identical tag lists.
14. `test_two_saved_logs_views_exist_and_are_idempotent` — names `Cap-hits` and `Regressions`, `view_type == "logs"`, second run creates nothing, and each btql string mentions the tag it filters on.
15. `test_score_trust_triple_without_live_prints_the_count_and_writes_nothing` — capsys shows `3 x <n> = <3n>`, `fake.posts == []`, and no ledger line was appended.
16. `test_score_trust_triple_with_live_writes_exactly_that_many_and_ledgers_it` — with `--live --score-trust-triple` on the fake, the number of score values written equals `3 * len(trust_triple_plan())` and the ledger row's `n_scores` equals it. Monkeypatch `showroom.LEDGER_PATH` to `tmp_path` (the existing tests' idiom).
17. `test_every_other_live_write_carries_no_scores` — run the four new steps live against the fake without the trust-triple flag; assert no posted event anywhere has a `scores` key and every new ledger row has `n_scores == 0`.
18. `test_design_note_names_the_three_seams_and_every_braintrust_touching_route` — read `docs/braintrust-live-app.md`; assert `init_logger`, `_traced`, `_maybe_wrap_openai` and every route string of brief §1.5 (`/api/agreements`, `/api/agreements/{id}/text`, `/api/questions`, `/api/cases/`, `/api/run`, `/api/market/`, `/api/reports`) appear.
19. `test_tour_stops_3_and_4_carry_the_new_actions` — read `docs/demo-tour.md`; assert the baseline experiment name appears in stop 3 and `maud-dealpoint-regressions` appears in stop 4.

Every test must run with no network and no `BRAINTRUST_API_KEY`. Never construct `Api` or `RestClient` with a real key in a test.

---

## 9. Risks, conflicts and numbers to report

**Conflict with M9c.** Both milestones append to the `markers` list in `pyproject.toml`. Add only the single line `"gate_m9d: milestone 9d gate",` and leave the rest of the block untouched, so the merge is a one-line, non-overlapping addition. Touch no other shared file: M9c owns `mlflow_*.py`, `docs/mlflow-tour.md` and `docs/milestones/m9c.md`; M9d owns `braintrust_showroom.py`, `docs/braintrust-live-app.md` and the two tour edits. If a rebase conflicts in `docs/demo-tour.md`, keep both sides — M9c should not be editing it at all.

**Numbers that may not match the spec.** Three numbers in the spec are live org counts, not repo-derivable facts:
- `739` root logs — the local plan computes 743. Print both; assert neither literal in the gate.
- `986` canonical experiment rows / `2,958` scores — not reproducible offline. Assert `3 × len(trust_triple_plan())`, record the live number in the manifest, and report whether it is 2,958.
- `33` regressions rows — this one **is** repo-derivable and I verified it; assert it as a literal along with the per-rule counts.

**Two unknown REST shapes** must be probed live and then documented in a dated comment and the manifest, exactly the way `_upsert_view` and `_view_data_for` document theirs: the `config` sub-object of `POST /v1/project_score`, and the BTQL operator for a tag-membership filter. Do not invent either and do not let the gate test hardcode a guess beyond what the probe confirms.

**Brief vs. spec.** I found no place where the spec contradicts `specs/grilled-product-brief.md`. The spec's D19 respects the brief's §2.7 score budget by writing zero scores by default; the trust triple is opt-in and outside the six-per-case budget, which is why it is flagged, counted and refused by default. If you find a genuine conflict while building, the brief wins and you **report** the difference in your summary rather than resolving it silently.

**Two clients, two path conventions.** `Api` uses relative paths, `RestClient` uses `/v1/...`. Every new call site must use one or the other consistently; a mixed path is a silent 404.

---

## 10. Order of work

1. `pyproject.toml` marker + `justfile` `gate-m9d` recipe.
2. Pure functions first, with their tests: `aggregate_score_defs`, `regression_rows`, `first_seen_experiment`, `tags_for_log`, `tag_plan`, `trust_triple_plan`, `comparison_key_is_safe`. Get tests 3, 4, 6, 7, 8, 11, 12, 15 green before writing any step.
3. The four steps + the two new CLI flags; wire `STEPS`; `FakeApi` and the live-path tests (1, 2, 5, 9, 10, 13, 14, 16, 17).
4. `docs/braintrust-live-app.md` and the three tour edits; tests 18, 19.
5. `uv run ruff check .`, `uv run pyright`, `uv run pytest -m "gate_m9d and not needs_network" -q`, then the full offline suite.
6. Live verification (needs the key): `just braintrust-showroom` (dry run, read the printed plan), then `just braintrust-showroom --live --only baseline,aggscores,regressions,logtags`. Probe and record the two unknown REST shapes. Read back by REST: project settings (`baseline_experiment_id`, `comparison_key`), the three project scores, `maud-dealpoint-regressions` row count, and one tag filter over Logs. Record every resolved id in the showroom manifest. Then re-run the same command and confirm it creates nothing.
7. Report: what was built, the three live-vs-spec numbers (739/743, 986/2958, 33), the two probed REST shapes, and any brief-vs-spec difference found.

---

## 11. Addendum — the two seams do not activate on the same condition

Confirmed by reading both sites. The design note (D22 §6) must state this explicitly, because it is the one
non-obvious fact about plugging the live app in:

- `dealpoint/agent/tools.py:27` `_traced` catches only `ImportError`. If `braintrust` is **installed**,
  `braintrust.traced(...)` is applied at **module import time**, configured or not; with no key and no current
  logger the span resolves to the SDK's no-op span and the tool returns its normal value.
- `dealpoint/llm/client.py:134` `_maybe_wrap_openai` requires `braintrust_adapter.braintrust_available()` —
  importable **and** a key loadable — before it wraps.

So today the three tools are permanently instrumented and the LLM client is not, and the only thing standing
between the tools and real span trees is whether something in the process called `init_logger`. That is the
entire plug-in point for `POST /api/run`. Because `_traced` is applied at import, the env flag that gates
`init_logger` at FastAPI startup is process-wide by construction and cannot be flipped per request — the note
should say that rather than implying a per-request toggle.
