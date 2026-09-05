# M4.1 — Execution-reliability repair and four-arm re-run

**Spec (read-only):** `specs/milestones/m4_1.md`. **Requirements authority (read-only):**
`specs/grilled-product-brief.md` §1.2, §2.2, §2.4, §6. Neither file may be edited.
**Gate:** `uv run pytest -m "gate_m4 and not needs_network" -q`, plus the full offline suite,
`uv run ruff check .`, `uv run pyright`.

---

## 0. What is actually broken (measured from the committed v1 artefacts — read this first)

I inspected `data/results/test_subset_v1_*.jsonl` and `data/results/spend_ledger.jsonl`. The
numbers below are facts on disk, not guesses; they tell you where to look, but **the probe in
§4 is what confirms each cause** — the record carries no exception text today, which is the
whole reason this milestone exists.

| leg | statuses |
|---|---|
| GLM arm A (32) | 7 ANSWERED, 4 ABSTAINED, **21 `schema_invalid_after_retry`** |
| GLM arm B (32) | 15 ANSWERED, 1 ABSTAINED, 6 schema, 10 CAP_HIT |
| GLM arm C (32) | 19 ANSWERED, 1 ABSTAINED, 4 schema, 8 CAP_HIT |
| GLM arm D (32) | 15 ANSWERED, 2 ABSTAINED, 11 schema, 4 CAP_HIT |
| Haiku arm A (18) | 10 ANSWERED, 7 ABSTAINED, 1 schema |
| Haiku arm D (18) | 2 ANSWERED, 1 schema, **15 `api_error`** |

Two hard signals:

1. **Every one of the 21 GLM arm-A schema failures has `usage.output_tokens == 1200`**, i.e.
   exactly `2 × MAX_TOKENS_FINAL (600)` — both the first attempt and the schema retry were cut
   off at the token ceiling. Successful arm-A cases sit at 130–1166 output tokens. So at least
   part of the GLM schema failure is *truncation*, not bad reasoning: the model writes a preamble
   and/or a long rationale and never reaches the closing brace. The ledger also shows **151 GLM
   calls ending at exactly 300 output tokens** (`MAX_TOKENS_TOOL_TURN`), so tool turns are being
   truncated too. `finish_reason` is available on `ChatResult` and is currently discarded.
2. **Haiku arm D fails on a temporal cliff.** Ledger rows for `test_subset_v1_tranche1_D`:
   the first four cases each made 5–7 successful calls (00:34:56–00:35:57); from 00:36:01
   every remaining case made **at most one** call and then failed. Five cases (`contract_144__q09`,
   `contract_103__q10`, `contract_32__q04`, `contract_99__redacted_q06`, `contract_103__redacted_q08`)
   have `input_tokens == 0`, `wall_ms` 531–687 — a *fast, non-retried* failure, i.e. a 4xx
   (`call_with_retries` re-raises 400s immediately), not a timeout. The rest failed on the
   **second** call, i.e. the first request that carries an assistant message with
   `"content": None` plus `tool_calls` and a `role: "tool"` reply (see `loop.py` lines ~150–190).
   Candidate causes, in the order I would check them: (a) the null-content assistant message
   being rejected on the Anthropic path via OpenRouter; (b) a 429/upstream rate limit that
   `call_with_retries` cannot survive because its backoff is `0.05 · 2^n` capped at 2 s — three
   retries take 0.35 s in total, which is not a backoff; (c) `cache_control` interacting badly
   with the shorter arm-D prefix (arm-D first calls are 1 791–1 848 input tokens; arm-A's are
   2 186–2 482 — but note case `contract_144__q05` succeeded at 1 791 tokens, so this is the
   weakest of the three). **Do not pick one from this list a priori — capture the message and read it.**

CAP_HIT is a legitimate outcome (spec deliverable 3): 10/8/4 CAP_HITs at GLM B/C/D stay, and
`MAX_TOOL_CALLS = 8` is **not** to be touched.

---

## 1. Scope and invariants

**In scope:** `dealpoint/llm/client.py`, `dealpoint/agent/{loop,pipeline,schema,_common}.py`,
`dealpoint/eval/{run,four_arm_sweep,report,spend}.py`, `dealpoint/config.py` (request-shape and
budget constants only), new probe module, new/updated tests, `data/reports/*`, `README.md`.

**Frozen — byte-identical, asserted by a new hash test (§3.F):**
- `data/eval/test_subset_v1.json` (`subset_hash = be2e96433b14`) and `dealpoint/eval/subset.py`'s rule.
- `skills/ma-deal-point-review/**` (`skill_version = f8d255cc169b`).
- `dealpoint.config.ARMS` / `ARM_ORDER` / `ARM_C_RETRIEVER` / `DENSE_RETRIEVER`.
- The text sent to the model: `system_prompt()`, `question_spec_block(q)` for all 12 questions,
  `out_of_scope_question_block(...)`, and the arm-D `skill_block(qid)` composition.
- `dealpoint/eval/scorers.py` and `dealpoint/eval/skill_adherence.py` semantics
  (`SCORE_FIELD_NAMES` unchanged, no scorer edited).

**Explicitly allowed** (spec deliverable 2/3 names these as the expected culprits): request shape
(`response_format`, `max_tokens`, message construction), retry/backoff policy, response parsing,
and new execution-record fields.

**Never:** raise `MAX_TOOL_CALLS`; pad a prompt to make it cacheable; re-run v1; edit the two
spec/brief files; change subset/skill/arm configs/scorers; start any judging or Pareto work.

---

## 2. Order of work (money last)

1. §3 offline: evidence fields, tolerant JSON, option normalisation, request-shape fixes behind
   deterministic tests. Full offline suite green.
2. §4 probes **before** (baseline, on the current code path if cheap — see note) and **after** the
   fix; read `failure_detail`; iterate.
3. §5 v2 sweeps.
4. §6 report + README + gate tests.

---

## 3. Offline changes

### A. Evidence on every failure (spec deliverable 1)

`dealpoint/agent/schema.py` — extend `ExecutionRecord`:

```python
failure_detail: str | None = None   # "<ExcClass>: <first 300 chars>" or the validation error
raw_final_text: str | None = None   # first 1500 chars of the model's final response
finish_reasons: list[str] = Field(default_factory=list)  # one per metered call, in order
```

Add the truncation limits to `dealpoint/config.py`:
`FAILURE_DETAIL_MAX_CHARS = 300`, `RAW_FINAL_TEXT_MAX_CHARS = 1500`.

Wire them:
- `dealpoint/agent/_common.py::build_record` gains `failure_detail`, `raw_final_text`,
  `finish_reasons` keyword arguments (default `None`/`None`/`None → []`), so both arms share one
  construction path.
- Add one helper in `_common.py`:
  `def describe_exception(exc: BaseException) -> str` → `f"{type(exc).__name__}: {str(exc)[:FAILURE_DETAIL_MAX_CHARS]}"`.
  For `openai.APIStatusError` also prefix the HTTP status and, when present, the provider error
  `code`/`type` from `exc.response`/`exc.body` — that is the field that will actually name the
  Haiku arm-D cause. Guard the extraction so a missing attribute can never raise.
- `loop.py` and `pipeline.py`: every `except Exception` around `call_with_retries` becomes
  `except Exception as exc:` and passes `describe_exception(exc)` into `_fail(...)`.
  On `schema_invalid_after_retry`, pass the last validation error as `failure_detail` and the
  last `result.content` (truncated) as `raw_final_text`.
  Record `result.finish_reason` after every successful call.
- `dealpoint/eval/run.py`'s per-case `except Exception as exc` block (currently
  `failure_reason="tool_error"` with the exception discarded, `_ = exc`) sets
  `failure_detail=describe_exception(exc)`.
- `FakeClient` needs nothing new beyond letting a `ScriptedTurn` carry any exception (it already does).

`failure_detail`/`raw_final_text` are **metadata only**: do not add them to `SCORE_FIELD_NAMES`,
do not add a Braintrust score (the six-score budget in `braintrust_adapter.py` is fixed).
They ride along in the result row's `record` dict, which is already `record.model_dump()`.

### B. Tolerant JSON extraction (spec deliverable 2)

New pure function in `dealpoint/agent/schema.py`:

```python
def extract_json_object(raw: str | None) -> str | None
```

Rules, applied in order, first hit wins:
1. `raw.strip()` already parses as a JSON object → return it.
2. A fenced block — ```` ```json … ``` ```` or ```` ``` … ``` ```` — whose body parses.
3. First balanced `{ … }` span found by a brace scanner that respects string literals and
   backslash escapes (so a `}` inside a quote does not close the object), scanning from the first
   `{`; if that span does not parse, try the next `{`.
4. Otherwise `None`.

`validate_finding_json` calls it before `json.loads`; when nothing parses, the error message
becomes `"no JSON object found in response"` and the caller keeps the raw text as evidence.
Truncated output (no closing brace) must still fail — do **not** attempt to repair unbalanced JSON.

### C. Option normalisation (spec deliverable 2)

In `dealpoint/agent/schema.py`, replace the strict `finding.answer not in allowed` check with a
normalising match that maps back to the **exact** option string:

```python
def match_option(answer: str, options: Sequence[str]) -> str | None
```
- Normalise both sides with `dealpoint.data.canonical.canonicalise` (curly quotes/dashes → ASCII,
  whitespace collapsed) then `casefold()` and strip a wrapping pair of `"`/`'`.
- `"ABSTAIN"` matches case-insensitively.
- On a hit, return the canonical option string from `question.options`; `validate_finding_json`
  then rewrites `finding.answer` to that exact string, so `answer_correct`
  (`answer == gold_answer`, exact) is unaffected and the scorers stay untouched.
- No fuzzy/prefix matching. A near-miss is still a failure.
- Out-of-scope questions have `options == ()` — only `ABSTAIN` or a free-form string is allowed
  there, exactly as today (`validate_finding_json`'s existing behaviour for empty options must not
  change; check it and keep it).

### D. Request shape (spec deliverable 3 — the harness, not the benchmark)

`dealpoint/llm/client.py`:
1. **`response_format` capability.** Add a small pinned table + resolver, e.g.
   `SUPPORTS_JSON_SCHEMA: dict[str, bool]` keyed by model id with a documented default, and
   `def response_format_for(model, question) -> dict` returning
   `finding_json_schema(question)` when supported and `{"type": "json_object"}` otherwise.
   The `json_schema` payload itself stays byte-identical. Additionally: if a call raises a 400
   whose message mentions `response_format`/`json_schema`/`structured output`, retry that call
   once with `{"type": "json_object"}`, record the downgrade in `failure_detail` **and** in the
   probe/report notes, so a provider limitation is visible rather than silently absorbed.
   Both arms (`loop.py`, `pipeline.py`) call the resolver instead of `finding_json_schema` directly.
2. **`max_tokens` for the final answer.** Raise `MAX_TOKENS_FINAL` from 600 to a value the probe
   evidence justifies (start at **1200**; the 21 GLM arm-A failures all died at the 600 ceiling and
   the largest successful case used ~1166 across its calls). Keep it a single config constant used
   by both arms and both models — never a per-arm value.
3. **Tool-turn truncation.** Record `finish_reason`; if the probes confirm tool turns are being cut
   at `MAX_TOKENS_TOOL_TURN = 300` in a way that loses tool calls or forces the loop into a
   premature finalisation, raise that constant too (suggest 600) **uniformly for all arms and
   models**, and write the decision + its cost impact into the report. If the probes do not show
   it hurting, leave it at 300 and say so.
4. **Assistant tool-call message.** In `loop.py`, when `result.content` is `None`, emit
   `"content": ""` (or omit the key) rather than `None` on the assistant message that carries
   `tool_calls` — several providers reject a null content field. Cheap, provider-agnostic, and a
   prime suspect for Haiku arm D.
5. **Retry policy.** In `_common.py::call_with_retries`, replace the `0.05 · 2^n` backoff with an
   exponential backoff whose base is a config constant (`API_RETRY_BASE_DELAY_S`, default ~1.0)
   with jitter, honouring `Retry-After` when the exception exposes it. Keep offline tests fast by
   letting the runner/tests inject a smaller base (`API_RETRY_BASE_DELAY_S` monkeypatched, or a
   keyword argument defaulted from config) — the existing `gate_m1` API-error test must stay
   sub-second. Add `openai.APIStatusError` subclasses for 429/5xx only; **do not** make 400
   retryable.
6. Keep `extra_body={"usage": {"include": True}}`, the ledger row shape (the exact `usd` key), and
   `build_system_message`'s single `cache_control` block unchanged. If the probe evidence shows the
   `cache_control` block is what Anthropic is rejecting, the fix is to *omit* the block when the
   static prefix is below the model's minimum cacheable length — never to pad the prefix.

### E. Failure-evidence tests (new `tests/test_failure_evidence_m4_1.py`, `pytest.mark.gate_m4`)

- `FakeClient` scripted to raise `ApiError("upstream 429 rate limited")` → record has
  `status == "EXECUTION_FAILED"`, `failure_reason == "api_error"`,
  `failure_detail` starts with `"ApiError: "` and contains `"429"`, and is ≤ 300 chars
  (test with a 5 000-char message to prove truncation).
- Scripted two invalid finals (`"Here is my answer: {broken"`) → `schema_invalid_after_retry`,
  `raw_final_text` carries the model's text, ≤ 1500 chars (test with a 5 000-char body).
- Same two assertions for arm A via `run_pipeline`.
- `run_eval_set(..., fake=True)` with a case whose document cannot be loaded → the runner's
  `tool_error` row carries a non-empty `failure_detail`.
- `extract_json_object`: bare object; fenced ```` ```json ````; fenced bare ```` ``` ````;
  prose before *and* after (`"Sure! {...} Let me know if…"`); a `}` inside a string value;
  two objects (first wins); truncated/unbalanced → `None`; `None`/`""` → `None`.
- `match_option`: exact; different case; curly quotes (`“All Cash”`); leading/trailing whitespace
  and a wrapping quote pair; `"abstain"`; a non-option (`"All cash or stock"`) → `None`; and an
  end-to-end `validate_finding_json` case proving the returned `finding.answer` is the **exact**
  option string from `QuestionSpec.options` (use q01 `"All Cash"` and q05/q10 whose options contain
  quotes/punctuation).
- `finish_reasons` populated on a normal ANSWERED run.

### F. Frozen-artefact hash assertions (new `tests/test_frozen_artifacts_m4_1.py`, `gate_m4`)

Constants in the test file (they are the record). Assert:
- `json.loads(TEST_SUBSET_V1_PATH)["subset_hash"] == "be2e96433b14"` and sha256 of the file bytes
  equals the pinned digest (compute it once and paste it).
- `skill_version() == "f8d255cc169b"`.
- sha256 of `system_prompt()`, of the concatenation of `question_spec_block(q)` for all 12
  questions in `QUESTION_SPEC` order, and of `skill_block(qid)` for all 12 — pinned digests.
- sha256 of `json.dumps(ARMS, sort_keys=True)` and `ARM_ORDER == ("A","B","C","D")`.
- `SCORE_FIELD_NAMES` equals the current 18-name tuple, verbatim.
- `MAX_TOOL_CALLS == 8`.
Compute each digest from the **current** working tree before you change anything, and pin those
values; if a digest moves later in the build, you have changed the benchmark and must revert.
The pre-existing frozen assertions (`tests/test_subset.py`, `tests/test_skill.py`,
`tests/test_arms.py`, `tests/test_prompts.py`) stay unedited and green.

Also add a golden-vector test for the scorers if you touch anything they read: fixed
`(case, finding, record, doc)` → fixed `score_case` dict. `tests/test_scorers.py` already covers
the semantics; do not duplicate it, just do not let it change.

---

## 4. Probes (spec deliverable 2 + 4) — ≈ $0.05, before any sweep

New module `dealpoint/eval/probe_m4_1.py`, CLI `python -m dealpoint.eval.probe_m4_1`, plus a
`just probe-m4-1` recipe next to `sweep-m4` in the `justfile`.

- **Legs:** `(anthropic/claude-haiku-4.5, D)`, `(z-ai/glm-5.3-flash, A)`, `(z-ai/glm-5.3-flash, B)` — 2 cases each.
- **Case choice (deterministic, recorded):** for each leg, the first two `case_id`s in frozen-subset
  order whose v1 record for that (model, arm) is `EXECUTION_FAILED`. From the v1 files that is
  Haiku D → `contract_7__q07`, `contract_32__q08`; GLM A → `contract_32__q06`, `contract_32__q08`;
  GLM B → `contract_32__q08`, `contract_39__q01`. Resolve them in code from the v1 result JSONLs
  (do not hard-code) and write the resolved ids and the rule into the probe JSON. This is a
  diagnostic selection on a *frozen* subset, it selects nothing that is reported as a result, and
  the v2 sweep runs the whole subset — state that in the report.
- **Rounds:** `before` (current `git stash`-free option: run the probe from the pre-fix code path
  is not required — the v1 result files already give the before-rate, and re-spending on it is
  waste). Take **`before` from the v1 result rows** (rate + `failure_reason` classes, with
  `failure_detail: null` noted as the defect this milestone fixes) and spend only on the
  **`after`** round. If a fix needs more than one iteration, each extra probe round is appended
  with its own timestamp; budget for at most three rounds total (≈$0.09).
- **Ledger tag:** `milestone_tag="m4_1"` for every metered call in this milestone (probes and v2
  sweeps), so M4.1 realised spend is separable from M4's $0.7519.
- **Output `data/reports/m4_1_probes.json`:**
  ```json
  {"rounds": [{"round": "before|after", "ts": "...", "git_sha7": "...",
     "legs": [{"model": "...", "arm": "...", "case_ids": [...], "n": 2,
               "execution_failed": 0, "failure_rate": 0.0,
               "failure_detail_classes": {"BadRequestError: 400 ...": 1},
               "finish_reasons": {"stop": 4}, "realized_usd": 0.0}]}],
   "case_selection_rule": "...", "notes": "..."}
  ```
- **Verify gate:** `after` shows `EXECUTION_FAILED ≤ 10%` per probed (model, arm) — with n=2 that
  means 0/2. If a leg still fails, read the new `failure_detail`, fix, re-probe. If it is provably
  the provider's (e.g. a hard 429 quota), record the exact message and carry it into the report as
  the explanation the DoD allows.
- Before each probe call `assert_within_cap(est)` and `_assert_within_milestone_absolute(est)`
  even though n ≤ 10 (the runner only auto-guards above 10 cases) — do this explicitly in the probe
  module and print the estimate.

---

## 5. v2 sweeps (spec deliverable 4)

**Preserve v1 first, in one commit-safe step, before any v2 write:**
```
data/reports/four_arm.json          -> data/reports/four_arm_v1_execution_defects.json
data/reports/four_arm.md            -> data/reports/four_arm_v1_execution_defects.md
data/reports/four_arm_manifest.json -> data/reports/four_arm_manifest_v1.json
```
Copy (`cp`), do not move-and-regenerate blindly; then **delete/rename the live manifest** so the
v2 run starts a fresh `four_arm_manifest.json`. If v1 legs stay in the live manifest,
`report.load_rows_from_manifest` concatenates v1 and v2 rows into one arm and every number is
wrong. The v1 result JSONLs under `data/results/` keep their `_4d2e361` stems and are untouched;
v2 files get the new `git_sha7` stem automatically. Add a `v1_artifacts` block to the new
`four_arm.json` naming all of these paths.

**Sweeps** (`dealpoint/eval/four_arm_sweep.py`, extended — keep the existing functions working):
- GLM `z-ai/glm-5.3-flash`, arms A–D, full 32-case frozen subset.
- Haiku `anthropic/claude-haiku-4.5`, arms A and D, `tranche=1` (N=18).
- `milestone_tag="m4_1"` on every leg; manifest entries gain `"version": "v2"` and
  `"milestone": "m4_1"`.
- Add a `--v2` (or a dedicated `run_v2()` + `main` flag) entry point so one command runs both legs
  in order; keep `--skip-*` flags. No dev smoke against the network is required — run the wiring
  check with `--fake` instead; if you do spend on a dev smoke, cap it at 2 cases at GLM and record
  it under `dev_loop_spend`.
- `assert_within_cap` + the milestone guard run before each sweep (already wired in `run_eval_set`
  for n > 10); record `est_usd` vs `realized_usd` per leg in the manifest (already done).
- **Budget:** target ≤ $1.00, absolute ≤ $1.50 (`adws/adw_modules/milestones.py` sets
  `DEALPOINT_MILESTONE_ABSOLUTE_USD=1.50`), total OpenRouter ≤ $4.00 with $1.2973 already spent.
  v1 realised: GLM A–D $0.2298 total, Haiku A $0.0604, Haiku D $0.1613 (mostly aborted, so the
  true full-D cost is higher — budget ~$0.45 for it). Expect ≈ $0.85–0.95 realised. Add
  `M4_1_TARGET_USD = 1.00` / `M4_1_MAX_USD = 1.50` to `dealpoint/config.py` and report against them.
  Run `uv run python -m dealpoint.eval.budget four_arm` before the sweeps and record the JSON
  estimate verbatim in the report (`cost.estimate_before_v2`). Note in the report that this
  estimate's `ledger:measured(arm,model)` basis is biased **low** for Haiku arm D because v1's
  arm-D cases aborted early.
- If the projection would breach $1.50 or the $4.00 envelope, drop the Haiku arm-A re-run first
  (arm A at Haiku was already healthy at 5.6% EXECUTION_FAILED) and record the decision — but keep
  arm D, which is the leg the milestone exists to fix, and keep all four GLM arms.

---

## 6. Report, README, and the mechanised DoD

`dealpoint/eval/report.py` (`build_report` stays pure and unit-testable):

1. **Version line.** `report["version"] = "v2 after harness repair"`, plus a
   `report["harness_repair"]` block: the list of fixes applied, the probe file path, and the
   `failure_detail` classes that justified each fix. `render_markdown` prints
   "v2 after harness repair" in the first paragraph; the README block gains the same line
   **without** disturbing the three disclosures the first sentence must keep
   (`budget-scaled`, `not comparable`, `objective` — `tests/test_readme_results.py` asserts them).
2. **Failure-rate table v1 → v2.** New pure function
   `failure_rate_table(v1_rows_by_arm_model, v2_rows_by_arm_model) -> list[dict]` with one row per
   (model, arm): `n_cases`, `execution_failed_v1`, `execution_failed_v2`, `delta`, `cap_hit_v1`,
   `cap_hit_v2`, `failure_detail_classes_v2`. v1 rows are loaded from
   `four_arm_manifest_v1.json`. Rendered in `four_arm.md` and in the README block.
3. **Residual-failure honesty.** For every (model, arm) in {GLM A–D, Haiku D} whose v2
   `execution_failed` mean exceeds 0.10, `report["residual_failures"][f"{model}/{arm}"]` must carry
   a non-empty `explanation` **sourced from `failure_detail`** (quote the class and count) and
   `attribution: "provider" | "harness"`. `render_markdown` and the README print it. Do not paper
   over a residual rate with prose that the JSON does not support (the M4 honesty rule still applies
   to every A→B→C→D sentence, unchanged).
4. **Majority-baseline note (spec deliverable 5).** Add
   `report["majority_baseline_note"]` computed in code, never hard-coded:
   - `subset_overall` = `majority_baseline(subset_cases)["overall"]` → 0.0,
   - `full_test_overall` = `majority_baseline(load_case_set("test"))["overall"]` → 0.461,
   - the sentence: the 0% is **by construction on this subset** — the selection rule in
     `data/eval/test_subset_v1.json` prefers cases where `gold_answer != majority_answer` — and the
     full 167-case MAUD test-set majority baseline is 46.1% for context.
   This sentence must appear in `four_arm.md` **and** in the README block next to the 0.0% line.
5. **Cost.** `cost` gains `m4_1_ledger_total` (`realized_by_tag()["m4_1"]`), keeps
   `m4_ledger_total`, adds `total_ledger_usd` (`realized_usd()`), `m4_1_target_usd`,
   `m4_1_absolute_usd`, and `estimate_before_v2`.
6. `just report` regenerates `four_arm.json`, `four_arm.md` and the README block, as today.
   Braintrust publishing stays optional and must not be required for any number
   (brief §6.7 — no README number may depend on a Braintrust link).

**Gate tests to add (`gate_m4`), in `tests/test_report_m4_1.py`:**
- `failure_rate_table` on fake v1/v2 row sets: correct per-(model, arm) rates and delta.
- `build_report` on fake rows sets `version`, `majority_baseline_note` (both numbers present),
  and `harness_repair`.
- **Report-consistency gate:** if `data/reports/four_arm.json` exists, then for each
  (model, arm) in {GLM A–D, Haiku D} either `execution_failed.mean <= 0.10` **or**
  `residual_failures[key]["explanation"]` is a non-empty string mentioning a failure class.
  `pytest.skip` when the file is absent (same convention as `tests/test_readme_results.py`) —
  but it must not skip once the file exists.
- `data/reports/m4_1_probes.json`, when present, has an `after` round in which every leg's
  `failure_rate <= 0.10` or carries a `provider_attribution` note.
- Extend `tests/test_readme_results.py` (do not weaken any existing assertion) with: the README
  block contains `"v2 after harness repair"`, the v1→v2 failure-rate figures for each rendered
  (model, arm), and the majority-baseline-by-construction sentence.

---

## 7. Definition of done — check each before reporting

- [ ] `failure_detail` / `raw_final_text` populated on synthetic failures (both arms + the runner's
      catch-all), truncation limits tested.
- [ ] Tolerant JSON extraction unit-tested: fenced, prose-wrapped, trailing text, brace-in-string,
      truncated → `None`.
- [ ] Option normalisation tested, and `finding.answer` is rewritten to the exact option string.
- [ ] Frozen-artefact hashes asserted and green; `tests/test_subset.py`, `test_skill.py`,
      `test_arms.py`, `test_prompts.py` unedited and green.
- [ ] `data/reports/m4_1_probes.json` records before (from v1) and after rates per probed
      (model, arm) with `failure_detail` classes; after ≤ 10% per leg or an evidenced provider
      attribution.
- [ ] v2 sweeps complete: GLM A–D × 32, Haiku A/D × 18; `four_arm.json` shows
      `EXECUTION_FAILED ≤ 10%` for the GLM arms and Haiku arm D, or `residual_failures` explains it
      from `failure_detail`.
- [ ] v1 artefacts preserved as `four_arm_v1_execution_defects.{json,md}` +
      `four_arm_manifest_v1.json`, linked from `four_arm.json` and named in `four_arm.md`.
- [ ] Report carries the v2 line, the v1→v2 failure-rate table, the majority-baseline-by-
      construction note with the full-test-set figure, and estimate vs realised.
- [ ] README "Results" regenerated by `just report` and consistent with the JSON.
- [ ] M4.1 realised spend (ledger `milestone_tag == "m4_1"`) ≤ $1.00 target / $1.50 absolute;
      total ledger ≤ $4.00. Report the realised figure.
- [ ] `uv run pytest -m "gate_m4 and not needs_network" -q`, the full offline suite
      (`uv run pytest -m "not needs_network and not needs_model" -q`, 293 passing today),
      `uv run ruff check .` and `uv run pyright` all green.

---

## 8. Notes, risks, and what to report rather than resolve

- **Brief vs spec:** no product-decision conflict found for M4.1. The pre-existing, already-recorded
  divergence (arm C is `hybrid_rrf`, the brief calls it `agent-hybrid-rerank`) stands unchanged —
  keep the existing note in `four_arm.md`. If you hit a new one, the brief wins and you *report* it.
- **`MAX_TOKENS_FINAL` / `MAX_TOKENS_TOOL_TURN` are request shape, not benchmark.** Raising them is
  sanctioned by spec deliverable 2 ("`max_tokens` too small for the final answer at some
  providers"); raising `MAX_TOOL_CALLS` is not. State the new values and their cost effect in the
  report so v1 and v2 efficiency numbers are read as what they are.
- Higher `max_tokens` raises output cost on exactly the cases that used to fail. Watch the ledger
  after the first GLM leg; if realised is tracking above estimate, the Haiku arm-A leg is the
  first thing to drop (§5).
- Do not "fix" CAP_HIT. If v2 CAP_HIT moves because tool turns are no longer truncated, report the
  movement; it is a finding, not a target.
- `pyright` runs over `dealpoint` and `tests` in basic mode; new optional fields and the exception
  introspection helper must be typed defensively (`getattr(..., None)` guards).
- Keep every new metered entry point out of the default test path: nothing in `tests/` may spend
  money except under `needs_model`, and the gate command excludes `needs_network` only — so any
  new metered test must be marked `needs_model` **and** must not be part of `gate_m4`'s offline run.
