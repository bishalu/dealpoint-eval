# M5 — Calibrated multi-judge evaluation (secondary, clearly labelled)

**Spec:** `specs/milestones/m5.md` (read-only). **Requirements:** `specs/grilled-product-brief.md`
§2.5, Appendix C, §6 acceptance 5 (read-only; the brief wins on product decisions and any
difference is *reported*, not silently resolved).

**Out of scope:** M6 Pareto runs, the product UI, the MCP layer, any edit to the brief or the
milestone spec, any edit to `specs/mvp/state.json`, and (see §12) any edit to the README Results
block.

---

## 0. State of the repo as the builder inherits it (verified 2026-09-05)

Everything below was measured, not assumed. Do not re-derive it; do re-verify anything you are
about to depend on.

| Fact | Value |
|---|---|
| `uv run ruff check .` | exit 0 — "All checks passed!" |
| `uv run pyright` | exit 0 — 0 errors |
| `uv run pytest -m "not needs_network and not needs_model" -q` | exit 0 — **333 passed, 4 deselected** (~4 min) |
| `pytest` markers | `gate_m5` and `needs_network` already registered in `pyproject.toml` — no marker work needed |
| `MAX_OPENROUTER_SPEND_USD` | `4` (in `.env`) |
| Ledger realised total | **$2.7612** (`data/results/spend_ledger.jsonl`, 1978 rows) |
| Ledger by tag | m1 $0.4191, m2 $0.1700, m4 $0.7519, m4_1 $1.4202 |
| **Remaining envelope** | **$1.2388 for M5 *and* M6 combined** |
| `eval/judges/` | exists, **empty** — this is repo-root `eval/`, *not* `dealpoint/eval/` |
| `data/eval/calibration/` | exists, **empty** |
| `data/reports/versions.json` | committed; keys incl. `skill_version`, `index_version`, `dataset_version` |
| scipy / pandas | **NOT installed**; numpy **is**. Do not add scipy (CPU-only VM, ~3.5 GB free disk) |
| `specs/mvp/state.json` | currently shows as modified in the worktree — that is the **factory's** edit. Leave it entirely alone |

### 0.1 The envelope is the dominant constraint of this milestone

M5's allocation guide is $0.60 (absolute $1.20), but only **$1.2388** remains in the whole $4.00
envelope and M6 has to fit in what is left. The plan below deliberately sizes M5 at **≈ $0.15
realised** so that ≈ $1.09 survives for M6. This is a recorded decision under the spec's
"Factory latitude" clause, and the report must state it in those terms. Underspending the guide
is always allowed; overrunning the envelope never is.

### 0.2 The traces M5 will judge already exist on disk

M4.1 produced the v2 sweeps at `git_sha7=e3ee9cc`, listed in `data/reports/four_arm_manifest.json`:

| variant | file | n |
|---|---|---|
| arm A @ `anthropic/claude-haiku-4.5` | `data/results/test_subset_v1_tranche1_A_A_anthropic_claude-haiku-4.5_e2b4a2b97561_e3ee9cc.jsonl` | 18 (tranche_1) |
| arm D @ `anthropic/claude-haiku-4.5` | `data/results/test_subset_v1_tranche1_D_D_anthropic_claude-haiku-4.5_e2b4a2b97561_e3ee9cc.jsonl` | 18 (tranche_1) |
| arm D @ `z-ai/glm-5.3-flash` | `data/results/test_subset_v1_D_D_z-ai_glm-5.3-flash_e2b4a2b97561_e3ee9cc.jsonl` | 32 (full subset ⊃ tranche_1) |

**No agent re-run is needed in M5.** The only metered calls M5 makes are judge calls.

Measured status mix (you will need this for the rubric's level-1 anchors and for the report):

- Haiku arm A: 11 `ANSWERED`, 7 `ABSTAINED`.
- Haiku arm D: 8 `ANSWERED`, 9 `CAP_HIT`, 1 `EXECUTION_FAILED`.
- GLM arm D (full 32): 20 `ANSWERED`, 5 `CAP_HIT`, 5 `EXECUTION_FAILED`, 2 `ABSTAINED`.
- **A-vs-D answer disagreement at Haiku on tranche_1: 14 of 18** (treating "no finding" as its own
  value). Verify this number yourself before writing it into the report.

A large fraction of judged traces therefore have **no finding at all**. The rubric and the packet
must handle that as a first-class outcome, not a crash.

### 0.3 Trace record shape (measured)

Each line of a result JSONL is:

```
{arm, case_id, case_set, chunk_version, finding|null, git_sha7, index_version, model,
 question_id, record:{status, failure_reason, failure_detail, raw_final_text, finish_reasons,
                      trajectory:[{tool, args, chunk_ids, char_ranges, result_ref, t_ms}],
                      usage:{input_tokens, output_tokens, cost_usd, cached_tokens,
                             cache_write_tokens, tool_calls, wall_ms}},
 scores:{...18 deterministic scores incl. grounded_accuracy...},
 skill_rules:{rules:{"1".."8"}, n_applicable, n_satisfied, score}, usd}
```

Facts that shape the packet builder:

- Trajectory steps carry **`char_ranges`, never the retrieved text**. Text must be reconstructed
  from the frozen canonical document (`dealpoint.corpus.document.load_document`).
- `search_agreement` steps have 5 `char_ranges` and 5 `chunk_ids`; `get_section` steps have exactly
  one `char_ranges` entry which **can be enormous** (a real one is `[[294865, 352363]]` — 57k
  chars); `lookup_defined_term` steps have **empty** `char_ranges` and `chunk_ids`.
- Max trajectory length observed: 8 steps (the `MAX_TOOL_CALLS` cap).
- Tool call counts across the v2 subset runs: 477 `search_agreement`, 99 `get_section`,
  77 `lookup_defined_term`.

---

## 1. Recorded decisions (factory latitude — write every one of these into the report)

The spec's "Factory latitude" clause makes subset size, judge count, calibration-set size and
slate order defaults. Each decision below is a **recorded decision, not a spec deviation**, and
must appear in `data/reports/judges.json` under a `decisions` array with its reason.

**D1 — Judged subset is 18 cases, not 12.** The spec's default is "12 cases (1 per question)".
The same spec's latitude floor requires that "every question keeps ≥ 1 case **and counterfactuals
stay in**". A 12-case, one-per-question set drops every counterfactual, so it cannot satisfy the
floor. `test_subset_v1.json`'s `tranche_1` is *exactly* 1 test case per question (12) + 4
`redacted` + 2 `out_of_scope` = 18, and is the only tranche for which both M5 Haiku variants have
traces. Judging is priced at ≈ $0.0025/trace, so 18 vs 12 costs ≈ $0.05 more. Reason to record:
"the 12-case default cannot satisfy the latitude floor that counterfactuals stay in; tranche_1 is
the smallest set that does, and the extra 6 cases cost ≈ $0.05."

**D2 — Three variants judged, not two.** Spec deliverable 4 names arm A @ Haiku and arm D @ Haiku.
The spec's own "Judge trio" instruction says "Judge the default **and GLM** in M5". Both are
satisfied by judging **A @ Haiku, D @ Haiku, D @ GLM**. 18 × 3 = **54 traces**; 54 × 3 judges =
**162 calls**. Reason to record: "the judge-trio instruction adds GLM; it costs ≈ $0.045 and gives
M6 a judged data point for a model it reuses rather than re-runs."

**D3 — Statistics are pure-Python, not scipy.** scipy is not installed and must not be added
(CPU-only VM, ~3.5 GB free disk, and the brief's architecture table keeps scorers framework-free).
Spearman ρ, quadratic-weighted Cohen's κ and Pearson are implemented in
`dealpoint/eval/agreement.py` and unit-tested against hand-computed values.

**D4 — M5 targets ≈ $0.15 realised against a $0.60 guide.** See §0.1. Reason: leave ≈ $1.09 of the
$4.00 envelope for M6.

**D5 — Mean-of-judges is rounded to the nearest integer for κ only.** Weighted Cohen's κ is
defined on integer ratings; the mean of three judges is not one. ρ uses the raw mean; κ uses
`round()` of it (ties to even, Python default). Document this in `judges.json` under
`stats_notes` and in `form.md`.

---

## 2. Brief-vs-spec differences to REPORT (do not resolve silently)

Put these in `judges.json` under `brief_differences` and restate them in your final report. None of
them is a blocker; the point is that they are surfaced.

1. **Scale.** Brief §2.5 specifies 40 stratified cases, 6 variants, 240 traces, **720 judge calls**,
   30 hand-scored traces, est. **$15–40**. This milestone runs 18 cases, 3 variants, 54 traces,
   162 calls at ≈ $0.15. Label every judged number **"budget-scaled"**; never present it as the
   brief's benchmark.
2. **Judge families.** Brief §2.5 gives "e.g. GPT + Grok + Qwen" as the example trio while also
   requiring families **not** in the candidate-agent slate. The M6 slate in
   `specs/milestones/openrouter_sweep_2026-09-04.md` now contains OpenAI, xAI *and* Alibaba/Qwen —
   so the brief's own example would violate the brief's own rule. The engineer's trio (Mistral,
   NVIDIA, ByteDance; spare Amazon) satisfies the rule. Record this as "the brief's rule is
   honoured; the brief's illustrative example is not, because the slate changed after the brief
   was written."
3. **Human calibration count.** Brief §2.5 mandates 30 hand-scored traces (~5 per variant); the
   spec sets "whatever is judged, target ≥ 12". 54 packets are produced, and `form.md` names a
   deterministic suggested minimum of 12. The gap between 12 and the brief's 30 is a reported
   difference, and the milestone's pending input carries it.
4. **Output keys.** Brief Appendix C fixes the judge's JSON as
   `{reasoning:int, evidence:int, trajectory:int, professional:int, notes:str}`. **Use exactly
   those key names** — no renaming, no extra model-supplied keys.

---

## 3. Deliverable 1 — Rubrics frozen first

### 3.1 `eval/judges/rubrics.md` (new, repo root `eval/`, **not** `dealpoint/eval/`)

Four dimensions, each with five anchored levels (one sentence per level). Dimension headings and
their JSON keys:

| JSON key | Dimension |
|---|---|
| `reasoning` | Reasoning / analysis quality |
| `evidence` | Evidence sufficiency |
| `trajectory` | Trajectory quality *beyond* deterministic adherence (search strategy, wasted calls, stopping) |
| `professional` | Professional answer quality |

Hard requirements on the file's content:

- Exactly five levels per dimension, labelled `1` … `5`, each a single sentence anchor.
- The **level-1 anchor of every dimension must explicitly cover the no-finding case** — a trace
  that ended `CAP_HIT` or `EXECUTION_FAILED` produced no finding, and that is the majority outcome
  for arm D @ Haiku (10 of 18). Do not leave the judge to improvise.
- The `evidence` dimension is the **only** one that references the expert gold span, and its text
  must say so.
- The `trajectory` dimension text must say it is judged *independently of* the deterministic
  `skill_adherence` score, which the judge never sees.
- A short preamble stating the output contract (the five keys above, integers 1–5, `notes` a
  string) and that the judge must not speculate about which system or model produced the trace.
- No mention of arms, models, or vendor names anywhere in the file.

### 3.2 `dealpoint/eval/rubric.py` (new)

Mirror `dealpoint/agent/skill.py`'s house style exactly:

```python
RUBRICS_PATH = REPO_ROOT / "eval" / "judges" / "rubrics.md"

def rubric_text() -> str: ...
def rubric_version() -> str:          # 12-char sha256 of the file bytes
def stamp_rubric_version() -> None:   # merge-write into data/reports/versions.json
```

`stamp_rubric_version()` must **merge**, never clobber — read the existing JSON, set only
`rubric_version`, write back with `sort_keys=True, indent=2` + trailing newline, exactly as
`stamp_skill_version()` does. Add `RUBRICS_PATH` to `dealpoint/config.py` rather than hard-coding
it in the module (house style: every path lives in config).

### 3.3 Freezing order — non-negotiable

`stamp_rubric_version()` runs, and `data/reports/versions.json` carries `rubric_version`, **before
the first judge call is made**. The judge runner must assert this at start-up and refuse to run if
`versions.json`'s `rubric_version` differs from `rubric_version()` computed from the file. Every
row written to `data/eval/judge_scores.jsonl` carries `rubric_version` **stamped by our code, never
taken from the model's output**.

The spec's "Out of scope" clause is binding: **no rubric revision after judging without re-running
all judges.** If you change one character of `rubrics.md` after judging, you re-run all 162 calls.

---

## 4. Deliverable 2 — Blinding

### 4.1 `dealpoint/eval/blinding.py` (new)

One public entry point:

```python
def build_packet(row: dict, case: dict, doc: Document, *, variant_id: str) -> dict
```

**The packet is built from an allowlist, never by deleting keys from `row`.** The allowlist is the
complete set of top-level packet keys and the blinding test asserts equality against it:

```
packet_id, question_text, options, status, trajectory, finding, gold_span, gold_span_note,
rubric_version
```

Field by field:

- **`packet_id`** — `sha256(f"{case_id}|{variant_id}").hexdigest()[:12]`. Deterministic and stable
  across regenerations. The `case_id → packet_id` direction is recoverable by us; the reverse is
  not recoverable by a reader of the packet.
- **`question_text`** — the question's `gloss` (the one-line plain-English form shown to the model
  in all arms). Resolve via `dealpoint.eval.cases.resolve_question`.
- **`options`** — the exact option strings. For an out-of-scope case `options` is empty; render it
  in the prompt as "(free-form; no fixed option list for this question)".
- **`status`** — `ANSWERED | ABSTAINED | CAP_HIT | EXECUTION_FAILED`. Not an identity; the
  `trajectory` dimension needs it.
- **`trajectory`** — a list of `{step, tool, args, section_refs, retrieved_text}`:
  - `args` is the model-generated tool arguments dict, scrubbed (see §4.2).
  - `section_refs` is derived by mapping each `char_range` onto `doc.sections` and collecting the
    enclosing section `ref`s.
  - `retrieved_text` is reconstructed deterministically and truncated to **≤ 600 chars per
    retrieved item** (the spec's number). Reconstruction rules:
    - `search_agreement` → `doc.text[start:end]` per `char_range`, each truncated to 600 chars.
    - `get_section` → prefer `dealpoint.corpus.document.get_section(doc, ref)`; fall back to the
      `char_range` slice. **Truncate to 600 chars** — a real `get_section` range is 57k chars and
      would blow the packet budget on its own.
    - `lookup_defined_term` → `char_ranges` is empty; re-resolve with
      `dealpoint.corpus.document.defined_term(doc, term)` and truncate. If it does not resolve,
      set `retrieved_text: null` and render "(tool result text not recorded)". Never invent text.
- **`finding`** — `{answer, evidence:[{section_ref, quote}], rationale}`, with `rationale` scrubbed.
  When the row has no finding, `finding` is `null` and the prompt renders
  "The system produced no finding for this case."
- **`gold_span`** — the expert-annotated span **text** (`doc.text[start:end]` per `case["gold_spans"]`,
  joined). Presented in the prompt under a heading that says it is supplied **for the evidence
  sufficiency dimension only**. When `gold_spans` is empty (every counterfactual case),
  `gold_span` is `null` and `gold_span_note` is the neutral string
  **"No expert-annotated span is recorded for this case."** — deliberately *not* "the provision is
  absent", which would hand the judge the gold answer for every redacted and out-of-scope case.
- **`rubric_version`** — stamped by us.

### 4.2 What must NOT be in the packet — and why each one matters

The blinding test asserts each of these individually. They are not cosmetic:

| Excluded | Why |
|---|---|
| `arm`, `model` | The spec's explicit requirement. |
| `scores` (all 18), `skill_rules`, `skill_adherence`, `grounded_accuracy` | **Circularity.** The report correlates judged quality against `grounded_accuracy`; a judge that saw it would make that correlation meaningless. |
| `case_id`, `agreement_id` | A `case_id` like `contract_39__redacted_q05` announces that the document was redacted, i.e. that ABSTAIN is correct. |
| `question_id` | `oos09` announces an out-of-scope probe. Use `question_text` instead. |
| `case["gold_answer"]` | The answer being judged. Only the *span* is supplied, and only for evidence sufficiency. |
| `category`, `reasoning_type`, `required_evidence` | `category == "out-of-scope"` leaks; the others are hints the agent never got. |
| `git_sha7`, `index_version`, `chunk_version` | Run identity. |
| `failure_reason`, `failure_detail`, `raw_final_text` | Harness internals; `status` is the honest, sufficient summary. |

### 4.3 Text scrubbing

Apply a case-insensitive scrub to **model-generated free text only** — `finding.rationale` and
each `trajectory[].args` string value:

- Vendor/family tokens → `[REDACTED-MODEL]`: `anthropic`, `claude`, `haiku`, `sonnet`, `opus`,
  `openai`, `gpt`, `google`, `gemini`, `deepseek`, `qwen`, `alibaba`, `mistral`, `nvidia`,
  `nemotron`, `bytedance`, `llama`, `meta-llama`, `grok`, `x-ai`, `xai`, `z-ai`, `zhipu`, `glm`,
  `minimax`, `moonshot`, `kimi`, `xiaomi`, `mimo`, `nova`, `amazon`, `seed-2`.
- `\barm\s+[A-D]\b` → `[REDACTED-ARM]`.

**Do not scrub `retrieved_text` or `finding.evidence[].quote`.** Those are verbatim merger-agreement
text; a merger agreement may legitimately name a party called Google, and corrupting the evidence
would destroy the very thing the evidence-sufficiency dimension measures. State this scoping choice
in a module docstring so a reviewer sees it was a decision, not an oversight.

### 4.4 The residual limitation — state it, do not hide it

The packet blinds **arm and model identity** (what the spec requires). It cannot blind the *shape*
of the trajectory: arm A is a single-shot pipeline with no tool calls, arm D is an agent loop, so a
knowledgeable reader can often infer which family of system produced a trace. Likewise an empty
option list hints at an out-of-scope probe. Write both limitations into `judges.json` under
`blinding_limitations` and into `form.md`. This is exactly the sort of thing the honesty rule
exists for.

### 4.5 The unblinding key

`data/eval/calibration/variant_key.json` maps `packet_id → {case_id, arm, model, variant_id}`.
It is written **outside** `packets.md` and `form.md`, and `form.md` tells the human scorer not to
open it until after scoring. A test asserts no `packet_id`'s entry appears in `packets.md`.

---

## 5. Deliverable 3 — Judges

### 5.1 The trio and the family constraint

From `specs/milestones/openrouter_sweep_2026-09-04.md` (engineer's instruction):

| model id | family | in $/M | out $/M |
|---|---|---|---|
| `mistralai/mistral-small-3.2-24b-instruct` | Mistral | 0.075 | 0.200 |
| `nvidia/nemotron-3-super-120b-a12b` | NVIDIA | 0.085 | 0.400 |
| `bytedance-seed/seed-2.0-mini` | ByteDance | 0.100 | 0.400 |
| spare: `amazon/nova-lite-v1` | Amazon | 0.060 | 0.240 | (no structured-output flag — JSON via prompt only) |

Candidate (agent) families that a judge may **not** come from: Anthropic, Zhipu, DeepSeek,
Alibaba/Qwen, Google, OpenAI, Meta, Xiaomi, MiniMax, Moonshot, xAI. Encode both lists as module
constants (`JUDGE_FAMILIES`, `CANDIDATE_FAMILIES`) and add an offline test asserting the
intersection is empty. This is the check that keeps the judge panel independent of what it judges.

### 5.2 `dealpoint/eval/judge_slate.py` (new) — verify at run time

The sweep file is dated and its ids may not exist on OpenRouter today. Provide:

```python
def verify_slate(client, *, prices=None) -> dict
```

which, for each candidate judge in order:

1. Confirms the id is present in the live model list (reuse `dealpoint.eval.spend.fetch_prices()`,
   which already handles the network and falls back to the pinned table — record which basis was
   used).
2. Records `prompt` and `completion` price per token.
3. Makes **one tiny JSON smoke call** (a ~20-token synthetic packet, `max_tokens=120`) and confirms
   a parseable four-dimension object comes back.

**Fallback ladder**, applied deterministically and recorded, never silently:

1. Named judge fails → try the spare `amazon/nova-lite-v1` (prompt-only JSON, no `response_format`
   schema — send `{"type": "json_object"}` and lean on tolerant extraction).
2. Spare also fails → pick the cheapest available OpenRouter model whose family is in neither
   `CANDIDATE_FAMILIES` nor the already-selected judge families.
3. Fewer than **2 distinct judge families** survive → **raise and stop**. The latitude clause sets
   ≥ 2 families as a hard floor; do not proceed with one judge.

Every substitution, failure and price goes to `data/reports/judge_slate.json` with `verified_at`,
`price_basis`, and the smoke-call token counts. `verify_slate` takes a client parameter so the
gate test can drive it with `FakeClient` offline.

### 5.3 `dealpoint/eval/judge_run.py` (new) — one call per trace, four dimensions

- Client: `dealpoint.llm.client.OpenRouterClient(milestone_tag="m5")`. It already writes the spend
  ledger row per call, which is what the guard and the report read.
- Set `client.context = {"case_id": <packet_id>, "variant": <variant_id>, "judge": <judge_model>}`
  before each call so cost is attributable. **Do not set a key named `arm`** — `per_case_usd`
  keys off it and you would pollute the agent-cost estimator.
- Request: no tools, `temperature=0`, `max_tokens=400`,
  `response_format={"type": "json_object"}` for every judge (the spare has no structured-output
  flag, and one uniform shape keeps the three judges comparable).
- Prompt: system = `rubric_text()` + the output contract; user = the rendered packet. One call
  returns all four dimensions.
- Parsing: reuse `dealpoint.agent.schema.extract_json_object` (already handles fenced blocks,
  prose-wrapped JSON and a balanced-brace scan). Then validate:
  - all four of `reasoning`, `evidence`, `trajectory`, `professional` present,
  - each an `int` in `1..5` (reject `3.5`, `"4"`, `0`, `6`),
  - `notes` a string, truncated to 500 chars on write.
- On validation failure: **one retry** with an appended "return only the JSON object" instruction.
  Still failing → write a row with `"ok": false`, `"failure_detail"` (exception class + first 300
  chars, matching the M4.1 convention) and **null dimension scores**. Never fabricate or default a
  score; a missing judge is a missing judge and the aggregates must carry its `n`.
- Resumability: before calling, skip any `(packet_id, judge_model)` already present in
  `data/eval/judge_scores.jsonl` with `"ok": true`. A half-finished run must be resumable without
  paying twice.

`data/eval/judge_scores.jsonl` row schema:

```json
{"packet_id":"...","variant_id":"A@haiku","case_id":"contract_39__q01","question_id":"q01",
 "judge_model":"mistralai/mistral-small-3.2-24b-instruct","judge_family":"Mistral",
 "rubric_version":"<12-hex>","ok":true,
 "reasoning":4,"evidence":3,"trajectory":4,"professional":4,"notes":"...",
 "input_tokens":2731,"output_tokens":143,"usd":0.000241,
 "failure_detail":null,"ts":"..."}
```

`case_id`/`variant_id` live in the *scores* file (ours), never in the packet. That is the point of
the two-file split.

### 5.4 Measure the real packet size

The sweep file assumes ≈ 8k input tokens per judge call. The packet as specified is far smaller
(≈ 8 steps × 600 chars + rubric + question + finding + gold span ≈ 2.5–3.5k tokens). **Record the
measured mean input tokens from the first judge leg** and put both the assumed and the measured
figure in `judges.json`. If the measurement is materially below 8k, say so — it is the difference
between an estimate and a measurement, and this project reports measurements.

---

## 6. Deliverable 4 — The judged subset

### 6.1 `data/eval/judged_subset.json`, generated by `dealpoint/eval/subset.py`

Add `generate_judged_subset()` / `write_judged_subset()` to the **existing**
`dealpoint/eval/subset.py`, reusing its `_seeded_key`, `_load_jsonl` and file-writing conventions
(`json.dump(..., sort_keys=True, indent=2, ensure_ascii=False)` + trailing newline) so the existing
byte-for-byte regeneration test style carries over unchanged.

The rule text stored in the file (write it verbatim, it is the record):

```
1. Universe: the 18 case_ids of test_subset_v1.json tranche_1 -- the only tranche for which
   both M5 Haiku variants have completed traces on disk (data/reports/four_arm_manifest.json,
   git_sha7 e3ee9cc). tranche_1 is exactly 1 test case per question (12) + 4 redacted + 2
   out-of-scope.
2. All 18 are judged. Recorded factory-latitude decision (D1): the spec's 12-case
   one-per-question default cannot satisfy the same spec's floor that counterfactuals stay in;
   tranche_1 is the smallest set that satisfies both, and the 6 extra cases cost ~$0.05.
3. Ordering (deterministic, seeded, fixed BEFORE any judge call, computed only from the frozen
   M4.1 v2 agent results -- never from any judge output): ascending by
   (disagreement_class, sha256(f"{SEED}:{case_id}")), where disagreement_class is 0 when arm A
   @ anthropic/claude-haiku-4.5 and arm D @ anthropic/claude-haiku-4.5 produced different
   answer strings -- a missing finding counting as the distinct value "<no finding>" -- and 1
   otherwise. SEED = 42 from dealpoint.config. The ordering exists so that any later
   budget-forced trim takes a deterministic prefix; it does not exclude anything here.
4. Variants judged: A @ anthropic/claude-haiku-4.5 and D @ anthropic/claude-haiku-4.5 (spec
   deliverable 4), plus D @ z-ai/glm-5.3-flash (spec, "Judge trio": judge the default and GLM
   in M5). 18 cases x 3 variants = 54 traces; 54 x 3 judges = 162 calls.
```

File payload keys: `case_ids` (in rank order), `rank` (case_id → int), `disagreement_class`
(case_id → 0/1), `variants` (list of `{variant_id, arm, model, results_path}`), `n_cases`,
`n_traces`, `n_judge_calls`, `seed`, `rule`, `subset_hash` (12-hex sha256 of the ordered
`case_ids` list, same construction as `test_subset_v1.json`), `source_subset`
(`"test_subset_v1"`), `source_tranche` (`1`), `dataset_version`, `version` (`"v1"`).

### 6.2 The freeze rule

The subset is fixed **before** judging and is derived only from label data and the **already
frozen** M4.1 agent outputs — never from any judge output. Deriving `disagreement_class` from the
M4 runs is explicitly sanctioned by the spec ("favouring cases where arms A and D disagree").
Write `judged_subset.json` and commit it before running a single judge call.

---

## 7. Deliverable 5 — Calibration package

### 7.1 `dealpoint/eval/calibration.py` (new)

Writes, into `data/eval/calibration/`:

- **`packets.jsonl`** — canonical machine form, one blinded packet per judged trace (54), sorted by
  `packet_id`.
- **`packets.md`** — the rendered form the human actually scores: a table of contents of
  `packet_id`s, then one `## <packet_id>` section per packet containing exactly what the judge saw
  (question, options, trajectory, finding, gold span under its evidence-sufficiency-only heading),
  followed by a blank score line to fill in. **Identical content to what the judge received** — the
  spec's "the same blinded packet is what humans score" is a testable claim, so make it one:
  render both from one function.
- **`form.md`** — the scoring form. Must contain:
  - the **identical rubric** — embed `rubric_text()` verbatim and record `rubric_version`; a test
    asserts the embedded rubric is byte-identical to `eval/judges/rubrics.md`,
  - the output contract (five keys, integers 1–5, `notes`),
  - the `human_scores.jsonl` row schema and one worked example,
  - the deterministic **suggested minimum set of 12** packet ids (the first 4 by subset rank per
    variant) plus an explicit statement that scoring more is welcome and there is no upper bound,
  - the §4.4 blinding limitations,
  - the D5 note that κ uses the rounded mean-of-judges,
  - the instruction not to open `variant_key.json` until scoring is finished.
- **`human_scores.jsonl`** — created **empty** (git tracks empty files).
- **`human_scores.schema.json`** — the schema the spec asks for, as a real file:

```json
{"packet_id":"<12-hex>","scorer":"<string>","scored_at":"<ISO-8601>",
 "reasoning":1,"evidence":1,"trajectory":1,"professional":1,"notes":"<string>"}
```

- **`variant_key.json`** — §4.5.

Plus `validate_human_scores(rows, known_packet_ids) -> (valid_rows, errors)`: rejects unknown
`packet_id`, non-integer or out-of-range dimensions, missing keys. Errors are collected and
surfaced in `judges.json` under `human_calibration_errors` — never silently dropped.

### 7.2 `just calibration`

New justfile recipe (`uv run python -m dealpoint.eval.calibration`), **offline, no model calls,
idempotent**. It:

1. Rebuilds the calibration package from `judged_subset.json` + the frozen result files.
2. Reads `data/eval/judge_scores.jsonl`.
3. Reads `human_scores.jsonl` if it exists **and is non-empty after validation**.
4. Writes `data/reports/judges.json`.

Running it twice must produce byte-identical output. Add a gate test for that.

---

## 8. Statistics — `dealpoint/eval/agreement.py` (new, pure Python)

Pure functions, no I/O, no numpy dependency in the maths (numpy is available but the functions
should be plain and legible, matching `scorers.py`'s framework-free stance):

```python
def rank_average(xs: Sequence[float]) -> list[float]          # ties get the average rank
def pearson(xs, ys) -> float | None
def spearman(xs, ys) -> float | None                          # pearson over average ranks
def quadratic_weighted_kappa(a, b, min_rating=1, max_rating=5) -> float | None
def exact_agreement(a, b) -> float | None
def mean_abs_diff(a, b) -> float | None
def pairwise_judge_agreement(scores_by_judge: dict[str, dict[str, int|None]]) -> list[dict]
def correlate_with_grounded_accuracy(quality: Sequence[float|None],
                                     grounded: Sequence[bool|None]) -> dict
```

**Degenerate-input guards are mandatory** (this is where these functions break in practice):
`n < 2`, all-`None`, zero variance in either series, and a κ where both raters used a single
rating all return `None` — never `NaN`, never a `ZeroDivisionError`. `judges.json` renders `None`
as JSON `null` with an accompanying `n`, so a reader can tell "not computable" from "zero".

`correlate_with_grounded_accuracy` pairs only indices where **both** values are non-`None`
(`grounded_accuracy` is `None` on `CAP_HIT`/`EXECUTION_FAILED`/abstained rows — a large share of
this data) and reports `pearson` (point-biserial, since `grounded` is boolean), `spearman`, and
`n`. When `n < 30`, attach a `caveat` string saying the correlation is under-powered.

Tests (`tests/test_agreement.py`, `gate_m5`) on synthetic scores:

- perfect monotone agreement → ρ = 1.0; perfect reversal → ρ = −1.0;
- identical integer raters → QWK = 1.0; one hand-worked 2-rater confusion matrix with the expected
  κ computed by hand in the test's comment;
- ties handled by `rank_average` (assert the average-rank vector directly);
- every degenerate guard returns `None`, no exception;
- `pairwise_judge_agreement` returns one entry per unordered judge pair per dimension with the
  right `n` when one judge failed on some packets;
- `correlate_with_grounded_accuracy` on a series containing `None`s pairs the right subset.

---

## 9. Report — `dealpoint/eval/judges_report.py` (new) → `data/reports/judges.json`

`build_judges_report(...)` is **pure over already-loaded rows** (same discipline as
`dealpoint.eval.report.build_report`, so it is unit-testable on fake data with no disk).

Required content (the DoD names the first four explicitly):

```jsonc
{
  "generated_at": "...", "git_sha7": "...", "rubric_version": "<12-hex>",
  "budget_scaled": true,
  "judges": [{"model": "...", "family": "...", "prompt_usd_per_token": ...,
              "completion_usd_per_token": ..., "price_basis": "live|pinned:2026-09-04",
              "n_scored": 54, "n_failed": 0, "substituted_for": null}],
  "judged_subset": {"path": "data/eval/judged_subset.json", "subset_hash": "...",
                    "n_cases": 18, "n_traces": 54, "n_judge_calls": 162,
                    "seed": 42, "rule": "...", "variants": [...]},

  "per_dimension": {                       // <- per-dimension aggregates
    "reasoning": {"mean_of_judges": 3.41, "n": 54,
                  "by_variant": {"A@haiku": {"mean": ..., "n": ...}, ...},
                  "by_judge":   {"mistralai/...": {"mean": ..., "n": ...}, ...}},
    "evidence": {...}, "trajectory": {...}, "professional": {...}
  },
  "pairwise_judge_agreement": {            // <- pairwise judge agreement
    "reasoning": [{"judge_a": "...", "judge_b": "...", "spearman": ..., "qwk": ...,
                   "exact_agreement": ..., "mean_abs_diff": ..., "n": ...}], ...
  },
  "correlation_with_grounded_accuracy": {  // <- correlation with grounded accuracy
    "reasoning": {"pearson": ..., "spearman": ..., "n": 29, "caveat": "n < 30 ..."}, ...
  },

  "human_calibration": "pending",          // <- literal string until human scores exist
  "pending_human_input": "score data/eval/calibration/form.md",
  "human_calibration_errors": [],

  "spend": {"est_usd": ..., "realized_usd": ..., "target_usd": 0.60, "absolute_usd": 1.20,
            "envelope_usd": 4.00, "envelope_realized_usd": ...,
            "envelope_headroom_after_m5": ..., "est_basis": "...",
            "assumed_input_tokens_per_call": 8000, "measured_input_tokens_per_call": ...},

  "decisions": [ /* D1..D5 with reasons */ ],
  "brief_differences": [ /* §2 items 1-4 */ ],
  "blinding_limitations": [ /* §4.4 */ ],
  "stats_notes": [ /* D5 kappa rounding; None-vs-zero convention */ ],
  "caveats": [ "Judge scores are secondary and model-judged; they are never objective truth.",
               "Budget-scaled: 18 cases x 3 variants x 3 judges, not the brief's 40 x 6 x 3.",
               "Human calibration is pending; no judge-human agreement figure exists yet." ]
}
```

When `human_scores.jsonl` validates non-empty, `"human_calibration"` becomes an object carrying,
**per dimension**: Spearman ρ and quadratic-weighted κ of mean-of-judges vs human (κ on the
rounded mean, per D5), the same two per individual judge vs human, `n`, and the scorer ids. The
report builder must support both branches and both must be unit-tested.

Also write a short **`data/reports/judges.md`** rendering the same numbers (house style: every
report JSON in this repo has an `.md` sibling — `four_arm.md`, `tournament.md`). Its first sentence
must state that these are **model-judged secondary scores, budget-scaled, and not objective
truth**, and that human calibration is pending.

---

## 10. Braintrust (deliverable 6)

Extend `dealpoint/eval/braintrust_adapter.py` with a judged-subset path — **do not touch
`SCORE_NAMES`**, which is the M2 six-score budget and is asserted by existing tests.

```python
JUDGE_SCORE_NAMES = ("judge_reasoning", "judge_evidence", "judge_trajectory", "judge_professional")

def run_judge_eval(rows, *, arm, model, case_set, no_send_logs=True, project=PROJECT) -> dict
```

- One experiment **per variant**, named `judged-{experiment_name(arm, model, index_version, git_sha7)}`.
- The four logged scores are the **mean-of-judges** per dimension, normalised to 0–1 as
  `(mean - 1) / 4` (Braintrust scores are conventionally 0–1; record the raw 1–5 mean in metadata
  so nothing is lost).
- **Per-judge scores go in per-case metadata**, exactly as the spec requires, alongside
  `packet_id`, `case_id`, `question_id`, `rubric_version`, `status`, `grounded_accuracy`.
- 4 scores/case × 54 cases = 216 scores — well inside the free tier and consistent with brief §2.7's
  "≤ 6 scores per case (+ 4 judge aggregates on the judged subset)".
- Must stay **offline-safe**: gate on `braintrust_available()`, default `no_send_logs=True`, and
  append to `data/reports/braintrust_runs.json` only on a real send (reuse `_record_braintrust_run`).
- Gate test drives it with `no_send_logs=True`, mirroring `tests/test_braintrust_adapter.py`.

---

## 11. Spend gate and the runner guard

### 11.1 `dealpoint/config.py` additions

```python
JUDGES_MILESTONE_TAG = "m5"
M5_TARGET_USD = 0.60     # allocation guide -- reported, not gated
M5_MAX_USD = 1.20        # absolute per-milestone limit (2x guide) -- enforced
RUBRICS_PATH = REPO_ROOT / "eval" / "judges" / "rubrics.md"
JUDGED_SUBSET_PATH = EVAL_DIR / "judged_subset.json"
JUDGE_SCORES_PATH = EVAL_DIR / "judge_scores.jsonl"
CALIBRATION_DIR = EVAL_DIR / "calibration"
JUDGES_JSON_PATH = REPORTS_DIR / "judges.json"
JUDGES_MD_PATH = REPORTS_DIR / "judges.md"
JUDGE_SLATE_PATH = REPORTS_DIR / "judge_slate.json"
JUDGE_MAX_RETRIEVED_CHARS = 600
JUDGE_MAX_TOKENS = 400
```

### 11.2 Reshape `SWEEP_DEFS["judges"]` in `dealpoint/eval/spend.py`

The current entry prices **Haiku agent cases**, not judge calls:

```python
"judges": {"arms": [], "models": ["anthropic/claude-haiku-4.5"],
           "n_cases": 24, "calls_per_case": 3},
```

That number is meaningless for M5 and would badly overstate the estimate the factory gate reads.
Replace it with a judge-shaped estimate priced directly from token shape × price, since there are
no ledger rows for these models yet:

- Add support for a leg key `"tokens_per_call": {"input": N, "output": M}`. When present,
  `_estimate_leg` prices `n_traces × n_judges` calls directly from `fetch_prices()` for each judge
  id, bypassing the ledger branches (which are agent-case shaped and would return nonsense here).
- `SWEEP_DEFS["judges"]` becomes the M5 shape: 54 traces × the 3 judge ids, `tokens_per_call`
  `{"input": 8000, "output": 300}` (the sweep file's conservative assumption — deliberately an
  upper bound, since the factory's pre-build gate treats the estimate as an upper bound only).
- `estimate("judges")` must still return the mandatory `calls` (int) and `est_usd` (float) keys —
  `adws/adw_modules/spend.py` parses them and `just budget judges` must still print exactly one
  JSON object to stdout with nothing else.

**Before editing, read `tests/test_spend.py` and `tests/test_spend_m4.py`** — both assert on
`SWEEP_DEFS` shape and on `estimate()`'s contract. Update any assertion that names the old judges
shape, and leave the `four_arm` and `pareto` entries untouched.

### 11.3 The gate itself

Before the first judge call, `judge_run.py` must:

1. Compute `est = estimate("judges")`.
2. `assert_within_cap(est["est_usd"])` — the $4.00 envelope.
3. `dealpoint.eval.run._assert_within_milestone_absolute(est["est_usd"])` — honours
   `DEALPOINT_MILESTONE_ABSOLUTE_USD` / `DEALPOINT_MILESTONE_SPEND_START_USD` set by the factory.
   (Import it; do not duplicate the logic.)
4. Assert `rubric_version` is stamped and matches the file (§3.3).
5. Record `realized_usd()` before, and after the run record the delta as M5's realised spend.

Record estimate-vs-realised in `judges.json` and, if the ratio is far off, say so — this project
reports the gap rather than quietly reconciling it.

---

## 12. Files to create and change

### New

| Path | Purpose |
|---|---|
| `eval/judges/rubrics.md` | The frozen rubric (§3.1) |
| `dealpoint/eval/rubric.py` | load / hash / stamp |
| `dealpoint/eval/blinding.py` | packet construction + scrubbing |
| `dealpoint/eval/judge_slate.py` | run-time id/price/JSON verification + fallbacks |
| `dealpoint/eval/judge_run.py` | the metered judging run |
| `dealpoint/eval/agreement.py` | pure-Python statistics |
| `dealpoint/eval/calibration.py` | calibration package + `just calibration` entry point |
| `dealpoint/eval/judges_report.py` | pure report builder + markdown renderer |
| `data/eval/judged_subset.json` | the frozen judged subset |
| `data/eval/judge_scores.jsonl` | raw judge outputs |
| `data/eval/calibration/{packets.jsonl,packets.md,form.md,human_scores.jsonl,human_scores.schema.json,variant_key.json}` | the calibration package |
| `data/reports/{judge_slate.json,judges.json,judges.md}` | reports |
| `tests/test_judges_rubric.py`, `tests/test_judge_blinding.py`, `tests/test_judge_parsing.py`, `tests/test_agreement.py`, `tests/test_judges_report.py`, `tests/test_judged_subset.py`, `tests/test_calibration_package.py`, `tests/test_judge_slate.py` | `gate_m5` |

### Changed

| Path | Change |
|---|---|
| `dealpoint/config.py` | M5 constants and paths (§11.1) |
| `dealpoint/eval/subset.py` | `generate_judged_subset()` / `write_judged_subset()` |
| `dealpoint/eval/spend.py` | reshape `SWEEP_DEFS["judges"]`, add `tokens_per_call` leg support |
| `dealpoint/eval/braintrust_adapter.py` | `JUDGE_SCORE_NAMES` + `run_judge_eval` |
| `tests/test_spend.py`, `tests/test_spend_m4.py` | only if they assert the old judges shape |
| `justfile` | `judge`, `calibration`, `gate-m5` recipes |
| `.gitignore` | add `!data/reports/judges.json`, `!data/reports/judges.md`, `!data/reports/judge_slate.json` under a `# dealpoint (M5 judges)` heading |

### Explicitly NOT changed

- `README.md` — the M5 DoD does not ask for it; `tests/test_readme_results.py` compares the README
  Results block against `four_arm.json`, and M6 regenerates it. Record this scoping choice in the
  report rather than risking M4's verified numbers.
- `specs/mvp/state.json` — the DoD forbids the builder editing it. The pending human input is named
  in the **milestone's evidence** as the literal string
  **`score data/eval/calibration/form.md`**, and also appears in `judges.json` as
  `pending_human_input`.
- `specs/grilled-product-brief.md`, `specs/milestones/m5.md` — read-only.

### New justfile recipes (match the existing comment-above-recipe style)

```make
# M5: verify the judge slate, then judge the frozen judged subset (METERED)
judge *ARGS:
    uv run python -m dealpoint.eval.judge_run "$@"

# M5: rebuild the calibration package and data/reports/judges.json (offline, no model calls)
calibration:
    uv run python -m dealpoint.eval.calibration

# milestone 5 acceptance gates only, offline
gate-m5:
    uv run pytest -m "gate_m5 and not needs_network and not needs_model" -q
```

---

## 13. `gate_m5` tests — the DoD's four named checks and the rest

Every file gets `pytestmark = pytest.mark.gate_m5`. **None may need network or spend money.**
Follow the house style in `tests/test_subset.py` / `tests/test_report_m4.py`: a small `_row(...)`
factory for fake data, `dataset_available` fixture skips for anything needing `data/raw/`.

**DoD check 1 — blinding strips identities** (`test_judge_blinding.py`):
- packet top-level keys == the §4.1 allowlist exactly;
- for a real-shaped synthetic row tagged `arm="D", model="anthropic/claude-haiku-4.5"`, the
  serialised packet contains neither the arm label nor the model id nor `z-ai/glm-5.3-flash`;
- a hostile row whose `rationale` reads "I am Claude Haiku running arm D for Anthropic" scrubs to
  `[REDACTED-MODEL]` / `[REDACTED-ARM]`;
- **no `scores`, no `skill_rules`, no `grounded_accuracy`, no `skill_adherence`** anywhere in the
  packet (the circularity guard);
- no `case_id`, `agreement_id`, `question_id`, `gold_answer`, `git_sha7`, `index_version`,
  `chunk_version`, `failure_detail`, `raw_final_text`;
- every `retrieved_text` ≤ 600 chars, including a synthetic 57k-char `get_section` range;
- a `lookup_defined_term` step with empty `char_ranges` yields `retrieved_text: null`, not a crash
  and not invented text;
- a counterfactual case (empty `gold_spans`) yields `gold_span: null` and the **neutral**
  `gold_span_note` — assert the note does *not* contain "absent", "redacted" or "abstain";
- the human packet rendered into `packets.md` is generated by the same function as the judge's
  packet text (assert equality of the rendered body).

**DoD check 2 — rubric hash** (`test_judges_rubric.py`):
- `eval/judges/rubrics.md` exists; all four dimension headings present; each has levels 1–5;
- `rubric_version()` is 12 lowercase hex chars and stable across two calls;
- `stamp_rubric_version()` into a tmp `versions.json` preserves pre-existing keys;
- the committed `data/reports/versions.json` carries `rubric_version` == `rubric_version()`;
- **every row of `data/eval/judge_scores.jsonl` (when present) has that same `rubric_version`** —
  this is the DoD's "a test asserts the hash in every judge result equals the frozen one";
- the file mentions no vendor, model or arm name.

**DoD check 3 — one-call-four-dims parsing with a fake judge** (`test_judge_parsing.py`), using
`dealpoint.llm.client.FakeClient` + `ScriptedTurn`:
- clean JSON parses to all four ints + notes;
- fenced ```` ```json ```` block parses;
- prose-wrapped JSON parses;
- missing dimension → `ok: false`, **null scores**, no fabricated default;
- out-of-range (`0`, `6`), non-integer (`3.5`, `"4"`) → `ok: false`;
- exactly **one** retry on the first failure, then give up (assert `len(fake.calls) == 2`);
- `rubric_version` on the row comes from our code even if the model emitted a different one;
- `notes` truncated to 500 chars;
- the request carries no `tools`, `temperature == 0`, `response_format == {"type":"json_object"}`.

**DoD check 4 — agreement statistics on synthetic scores** (`test_agreement.py`): see §8.

**Supporting:**
- `test_judged_subset.py` — file exists; `generate_judged_subset()` is deterministic and regenerates
  byte-identically (`filecmp.cmp`, following `tests/test_subset.py`); 18 cases; every one of the 12
  questions has ≥ 1 case; ≥ 1 `redacted` and ≥ 1 `out_of_scope` present; every `case_id` ∈
  `test_subset_v1.json`; `seed` and `rule` recorded; `subset_hash` matches a recomputation;
  ranking is a permutation with no duplicates.
- `test_judges_report.py` — `build_judges_report` pure on fake judge rows; `"human_calibration" ==
  "pending"` and `pending_human_input == "score data/eval/calibration/form.md"` with no human
  scores; the computed branch appears with synthetic human scores; per-dimension, pairwise and
  correlation blocks all present; a failed judge reduces `n` rather than skewing a mean; the spend
  block is present; markdown renders and its first sentence carries the secondary/budget-scaled
  caveat.
- `test_calibration_package.py` — `form.md` embeds the rubric **byte-identically** to
  `eval/judges/rubrics.md`; `human_scores.schema.json` exists and matches the documented keys;
  `validate_human_scores` accepts a good row and rejects unknown `packet_id`, out-of-range and
  non-integer dimensions; `packets.jsonl` line count == number of judged traces; `variant_key.json`
  covers every `packet_id` and none of its `arm`/`model` values appear in `packets.md`;
  `just calibration` run twice is byte-identical.
- `test_judge_slate.py` — `JUDGE_FAMILIES ∩ CANDIDATE_FAMILIES == ∅`; `verify_slate` with a
  `FakeClient` selects the three named judges when all succeed; falls back to the spare when one
  fails; **raises when fewer than 2 families survive**; records prices and `price_basis`.

---

## 14. Execution order (the freeze order is not negotiable)

1. **Offline build, no metered calls.** Rubric → `stamp_rubric_version()` → blinding → subset →
   agreement → report builder → calibration package → all `gate_m5` tests.
2. Get `uv run pytest -m "gate_m5 and not needs_network" -q`, the full offline suite,
   `uv run ruff check .` and `uv run pyright` **all green**. Commit-ready.
3. Write `data/eval/judged_subset.json` and the calibration packets from the frozen result files.
   The subset is now frozen.
4. `just budget judges` → record the estimate and its basis. Confirm
   `realized_usd() + est ≤ 4.00` with room left for M6; record `envelope_headroom_after_m5`.
5. **Verify the judge slate** — 3 tiny metered calls (≈ $0.001) → `data/reports/judge_slate.json`.
   If ids are missing, apply the §5.2 ladder and record every substitution.
6. **Run the judging** — 162 calls, ≈ $0.13 → `data/eval/judge_scores.jsonl`. Resumable.
7. `just calibration` → `data/reports/judges.json` + `judges.md`.
8. Braintrust: `run_judge_eval(..., no_send_logs=False)` per variant **if** `braintrust_available()`;
   otherwise record "not sent (no key)" and move on.
9. Re-run the full offline suite + `ruff` + `pyright`. All green.
10. Confirm `git status` shows **no change to `specs/mvp/state.json`** and no change to the brief or
    `specs/milestones/m5.md`.

**Dev-loop rule:** any exploratory metered call you make while building uses `z-ai/glm-5.3-flash`
(the workhorse), never Haiku. But note you should need **zero** exploratory agent calls — every
trace M5 judges already exists on disk (§0.2).

---

## 15. Verification — the exact commands

```bash
uv run pytest -m "gate_m5 and not needs_network" -q     # the milestone gate
just test                                                # full offline suite (~4 min)
uv run ruff check .
uv run pyright
just calibration && git diff --stat data/reports/judges.json   # idempotent: empty diff
uv run python -m dealpoint.eval.budget judges            # one JSON object, exit 0
```

Judge each by **exit status**, never by grepping its output for words like "error" or "failed" —
those strings appear inside passing output in this repo (the reports themselves discuss
`EXECUTION_FAILED` rates).

Then assert from the ledger:

```bash
uv run python -c "from dealpoint.eval.spend import realized_usd, realized_by_tag; \
print(realized_usd(), realized_by_tag())"
```

- total ≤ **$4.00** (hard),
- `m5` tag ≤ **$1.20** (absolute), target ≤ **$0.60**, expected ≈ **$0.15**,
- headroom left for M6 recorded in `judges.json`.

## 16. Definition of done — mapped to the spec's checkboxes

- [ ] **`gate_m5` offline**: blinding strips identities; rubric hash checked in every judge result;
      one-call-four-dims parsing with a fake judge; agreement statistics unit-tested on synthetic
      scores. (§13)
- [ ] **Spend gate passed**; judges run on the frozen judged subset for the M5 variants;
      `data/reports/judges.json` written with per-dimension aggregates, pairwise judge agreement,
      correlation with grounded accuracy, and `"human_calibration": "pending"`; calibration packets
      and `form.md` written. (§7, §9, §11)
- [ ] **`specs/mvp/state.json` NOT edited by the builder**; the milestone's evidence names the
      pending human input: **`score data/eval/calibration/form.md`**.
- [ ] **Full offline suite, `ruff`, `pyright` green.**
- [ ] Decisions D1–D5, the four brief-vs-spec differences (§2), the blinding limitations (§4.4) and
      the envelope headroom (§0.1) are all written into `judges.json` and into the final report.
- [ ] Nothing anywhere presents a judge score as objective truth.
