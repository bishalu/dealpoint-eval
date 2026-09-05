# M6 — Model cost/quality Pareto experiment: implementation plan

**Spec:** `specs/milestones/m6.md` (read-only). **Authority:** `specs/grilled-product-brief.md`
§2.6, §6 acceptance 6 (read-only). **Slate/judges:** `specs/milestones/openrouter_sweep_2026-09-04.md`.

Arm D is held fixed (frozen index `e2b4a2b97561`, frozen skill `f8d255cc169b`, frozen tools, frozen
32-case subset `test_subset_v1.json`). **Only the model varies.**

---

## §0 — READ THIS FIRST: the spend gate blocks M6 today

The factory runs `uv run python -m dealpoint.eval.budget pareto` before your build phase and compares
`realized + est_usd` against `MAX_OPENROUTER_SPEND_USD` (`.env` = `4`). Measured right now:

```
$ uv run python -m dealpoint.eval.budget pareto
{"est_usd": 3.59166, "calls": 384, "cases": 96, "model": "anthropic/claude-haiku-4.5", ...}
$ ledger realized = 2.871285
→ projected 6.46 > cap 4.00  → SpendCheck(ok=False) → the milestone ESCALATES
```

The cause is the same one M4 hit (`specs/f0eb3bd2_four-arm-experiment-frozen-subset.md` §0.3):
`SWEEP_DEFS["pareto"]` in `dealpoint/eval/spend.py` still holds the **placeholder** shape written at
M2 — 3 models × 32 cases priced at *Haiku* ($0.0374/case), which is not the sweep this milestone
runs. Haiku is **reused, never re-run** (spec deliverable 1), and the actual new models are 15–45×
cheaper per case.

**§1.1 below is a prerequisite, not an optimisation.** Fixing the estimate so it describes the run
you are actually going to make is a correctness fix, not gaming the guard. Do it as the very first
edit of the build, before anything else, and re-run `just budget pareto` to confirm the projection
clears the cap. If the factory escalated on the gate before you were started, apply §1.1 first thing
in the corrective cycle.

`adws/**` is factory code. **Do not edit it.** `m6` has no `absolute_usd` in
`adws/adw_modules/milestones.py`, so the $4.00 envelope is the only external bound; this plan adds
the milestone's own internal stop-floor in product code (§1.1).

---

## §1 — Budget: what runs, what it costs, what does not run

Ledger state at planning time (`dealpoint.eval.spend.realized_by_tag()`):

| tag | realised |
|---|---|
| m1 | $0.4191 |
| m2 | $0.1700 |
| m4 | $0.7519 |
| m4_1 | $1.4202 |
| m5 | $0.1100 |
| **total** | **$2.8713** |
| **headroom to $4.00** | **$1.1287** |

M6's allocation guide is "the remainder" ≈ $1.13. This plan spends **≈ $0.75** of it and holds
≈ $0.38 back.

### 1.1 Resize `SWEEP_DEFS["pareto"]` (prerequisite — do this first)

In `dealpoint/eval/spend.py`, replace the placeholder `"pareto"` entry with a three-leg shape that
models the real run. Add the model lists next to it (import them from
`dealpoint.eval.pareto_slate` once §2 exists, or define them in `spend.py` and have `pareto_slate`
import them — either way there must be **one** list, never two that can drift):

```python
"pareto": {
    "legs": [
        # 1. tool-calling verification probes: 2 dev cases, arm D, per new candidate
        {"arms": ["D"], "models": PARETO_NEW_MODELS, "n_cases": 2},
        # 2. the arm-D sweeps on the frozen 32-case subset, per new candidate
        {"arms": ["D"], "models": PARETO_NEW_MODELS, "n_cases": 32},
        # 3. judging: 18 judged-subset traces per new variant, x 3 judges, at the
        #    M5-measured judge call shape (NOT an agent-case shape).
        {"arms": [], "models": JUDGE_MODELS,
         "n_cases": 18 * len(PARETO_NEW_MODELS),
         "tokens_per_call": {"input": 3400, "output": 120}},
    ],
},
```

`_estimate_leg` already sums a multi-`models` leg per model when `len(models) > 1` (the
`n_models_multiplier` branch only scales single-model legs), and the `tokens_per_call` branch
already prices per-trace judge calls directly from `fetch_prices()`. **No change to `estimate()` or
`_estimate_leg` is needed.** The `3400/120` shape is the M5 *measured* judge call shape (163 judge
ledger rows, mean $0.000394/call), not the old 8000/300 upper bound — record that in the report.

Also add to `dealpoint/config.py`:

```python
# --- Milestone 6: model cost/quality Pareto -------------------------------
PARETO_MILESTONE_TAG = "m6"
M6_TARGET_USD = 1.00     # the M6 allocation guide -- reported, not gated
M6_ENVELOPE_USD = 4.00   # the whole unattended M1-M6 envelope -- never exceeded
# The runner's own stop-floor: no NEW model is started once the projected
# total ledger would cross this. Leaves ~$0.10 of the envelope untouched for
# a re-run of the judging pass or a report regeneration.
M6_STOP_USD = 3.90
PARETO_JSON_PATH = REPORTS_DIR / "pareto.json"
PARETO_MD_PATH = REPORTS_DIR / "pareto.md"
PARETO_SVG_PATH = REPORTS_DIR / "pareto.svg"
PARETO_SLATE_PATH = REPORTS_DIR / "pareto_slate.json"
PARETO_MANIFEST_PATH = REPORTS_DIR / "pareto_manifest.json"
PARETO_PROBE_N_CASES = 2
```

After the edit, `just budget pareto` must print an `est_usd` around **$0.71–0.78** and
`realized + est_usd` must be **< $4.00**. Verify before going further.

### 1.2 The slate (verify every id and price at run time; record what you find)

Reused, **never re-run** (spec deliverable 1, and the "Cheap workhorse model" instruction):

| model | arm-D result already on disk | n_cases | source |
|---|---|---|---|
| `anthropic/claude-haiku-4.5` | `test_subset_v1_tranche1_D_D_anthropic_claude-haiku-4.5_e2b4a2b97561_e3ee9cc.jsonl` | 18 (tranche_1) | M4.1 replication leg |
| `z-ai/glm-5.3-flash` | `test_subset_v1_D_D_z-ai_glm-5.3-flash_e2b4a2b97561_e3ee9cc.jsonl` | 32 (full) | M4.1 headline leg |

Resolve both paths from `data/reports/four_arm_manifest.json` (arm `D`, matching model, prefer the
`version == "v2"` entry) — **never hard-code them**.

New runs, in this ranked order (rows 1–7 of the sweep file, minus the two already run). Prices
below are **live, measured 2026-09-05** and must be re-verified and re-recorded at run time; the
`$/case` column is at the measured arm-D shape (22,285 in / 1,357 out per case, from 67 GLM arm-D
executions in the ledger):

| # | model | family | in $/M | out $/M | $/case | 32-case | role |
|---|---|---|---|---|---|---|---|
| 1 | `qwen/qwen3.7-flash` | Alibaba | 0.030 | 0.130 | 0.00084 | $0.027 | cheapest credible |
| 2 | `deepseek/deepseek-v4-flash` | DeepSeek | 0.084 | 0.168 | 0.00210 | $0.067 | open-weight |
| 3 | `google/gemini-3.1-flash-lite` | Google | 0.250 | 1.500 | 0.00761 | $0.243 | cross-family flash |
| 4 | `xiaomi/mimo-v2.5` | Xiaomi | 0.140 | 0.280 | 0.00350 | $0.112 | breadth of cheap tier |
| 5 | `meta-llama/llama-4-maverick` | Meta | 0.200 | 0.696 | 0.00540 | $0.173 | open-weight, large ctx |

Sweeps $0.622 + probes $0.039 + judging 5 × $0.018 = **$0.751**. Projected total **$3.62**.

**Stretch, in this order, only while the §1.1 stop-floor holds** (the runner decides at run time, not
you): 6 `openai/gpt-5.6-luna-pro` ($0.207 for 34 cases), 7 `minimax/minimax-m2.5` ($0.254),
8 `moonshotai/kimi-k2.5` ($0.445). On the numbers above only #6 could fit and only if the earlier
models come in under estimate; expect none of them to run. That is fine — the DoD floor is ≥ 3
models, and this plan lands 7 (Haiku, GLM + 5 new).

**Not run — record explicitly in `pareto.json` and in the report, with the reason:**

| model | reason |
|---|---|
| `anthropic/claude-opus-*` | not run (budget) — spec deliverable 1 names it as such |
| `anthropic/claude-sonnet-5` | not run (budget) — $1.047 even on the 18-case judged subset, > the $1.13 remaining envelope. Spec deliverable 1 allows the 12/18-case fallback "only if the remaining envelope covers it"; it does not. |
| `x-ai/grok-4.3` | not run (budget) — $0.563 on the 18-case subset; taking it would cost four cheap-tier models, and the sweep file ranks breadth of the cheap tier above depth |

### 1.3 Recorded decisions (factory-latitude clause — decisions, not deviations)

Write every one of these into `pareto.json["decisions"]` with its reason, the same way
`judges_report.DECISIONS` does:

- **D1 — Open-weight slot is `deepseek/deepseek-v4-flash`, not the spec §1 default
  `deepseek/deepseek-v3.2`.** Both are live. The engineer's 2026-09-04 "Slate and judges"
  instruction says to use the ranked slate in `openrouter_sweep_2026-09-04.md`, whose row 3 is
  `deepseek-v4-flash`, and it is 3× cheaper per case ($0.0021 vs $0.0065). Record both prices.
- **D2 — Five new models across five families, no ceiling anchor.** The spec's "stronger
  same-family ceiling" (Sonnet 5) does not fit the envelope even on the reduced subset; the sweep
  file's own guidance is that "breadth of the cheap tier beats depth". Five cheap families expose
  real model-switching opportunities; one ceiling anchor would have bought nothing else.
- **D3 — Judged subset stays at 18 cases (`test_subset_v1` tranche_1), matching M5.** The spec says
  "12-case judged subset"; M5's recorded D1 already settled that a 12-case one-per-question set
  cannot satisfy the same spec's floor that counterfactuals stay in. Reusing M5's exact 18 cases is
  also what makes M5's and M6's judged numbers comparable at all. Cost of the 6 extra cases per
  variant: ~$0.006.
- **D4 — Chart is SVG written by hand, not matplotlib.** matplotlib is not installed, the VM is at
  87% disk (1.3 GB free), and the brief's architecture table keeps reporting framework-free. The
  spec says `pareto.svg`/`png`; SVG satisfies it and is diffable and deterministic.
- **D5 — Probe-then-estimate ordering.** `per_case_usd(arm, model)` returns the
  `ledger:corpus-tokens x live` basis for a model with no ledger history — an approximation from the
  corpus-wide mean tokens/call. After a model's 2-case probe it has real arm-D rows carrying
  `case_id` and `git_sha7`, so branch 1 (`ledger:measured(arm,model)`) applies. Probing first is
  therefore what makes "`assert_within_cap` with the ledger's **measured** cost/case" (spec
  deliverable 2) literally true, not just nominally.

### 1.4 Differences from the brief (report them; do not resolve them silently)

Brief §2.6 and §6 acceptance 6 specify: Sonnet 5 default, Opus ceiling, Haiku, a Gemini Flash-class
model and one open-weight model, over the **full test + counterfactual sets**, ≥ 4 of 5 models, est.
$100–200. This build runs 7 models over a **32-case frozen subset** (18 for Haiku) at a total M1–M6
spend of ≤ $4.00 — the engineer's 2026-09-04 budget-scaling instruction. Record in
`pareto.json["brief_differences"]`, in the same shape as `judges_report.BRIEF_DIFFERENCES`:

1. **scale** — 32 cases vs the full 167-case test set + 40 counterfactuals; budget-scaled, never
   presented as the full benchmark.
2. **slate composition** — the brief's default (Sonnet 5) and ceiling (Opus 5) are both out of
   budget; Haiku is the default here, and the ceiling is unmeasured. The Pareto frontier reported
   is therefore a frontier *of the cheap tier*, and the report must say that a stronger model could
   sit above it.
3. **model count** — the brief asks for ≥ 4 of 5 *named* models; this run completes 7 models, but
   two of the brief's five named ones are absent. Both facts go in the report.
4. **Haiku n_cases** — Haiku's arm-D result is 18 cases (tranche_1), every other model's is 32. Any
   Haiku-vs-other comparison is over the 18 cases they share. The report must carry the common-n
   alongside every such comparison and must never compare an 18-case mean to a 32-case mean without
   saying so.

---

## §2 — Deliverable 1: slate verification (`dealpoint/eval/pareto_slate.py`, new)

Offline-testable; the metered part is one 2-dev-case arm-D probe per candidate.

```python
PARETO_REUSED: tuple[dict, ...]      # haiku, glm: {model, family, role, reuse: True}
PARETO_NEW_MODELS: tuple[str, ...]   # the 5 ranked ids from §1.2 (imported by spend.py)
PARETO_STRETCH_MODELS: tuple[str, ...]
PARETO_NOT_RUN: tuple[dict, ...]     # {model, reason} for opus / sonnet-5 / grok-4.3
PARETO_CANDIDATES: tuple[dict, ...]  # ranked: {model, family, role, rank}

def check_availability(model: str, prices: dict) -> dict
def probe_candidate(client, model, *, case_ids, prices, milestone_tag="m6") -> dict
def verify_slate(client, *, prices=None, case_ids=None, milestone_tag="m6") -> dict
def write_pareto_slate(payload, path=PARETO_SLATE_PATH) -> None
```

Rules, all of them testable:

- **Availability with no spend first.** `fetch_prices()` is the OpenRouter models listing. A model
  absent from it is recorded `{"available": false, "excluded": true, "reason": "not in the
  OpenRouter model listing"}` and **no call is made to it**. Follow `judge_slate.verify_slate`'s
  `prices=None → fetch_prices()`, `prices=<dict> → basis "provided"` convention so the gate test can
  drive the whole thing offline.
- **One probe per candidate**, `PARETO_PROBE_N_CASES = 2` dev cases, arm D, at the candidate model.
  Use the **same two dev case ids for every candidate**, fixed in the module and recorded in the
  payload (`("contract_0__q01", "contract_0__q06")` — one direct, one defined-term, the pair M1
  already uses as its smoke; a defined-term question is the one that actually forces
  `lookup_defined_term`, so it is the one that tests tool calling).
- **Tool-calling verdict.** From the written result rows, per candidate:
  `n_cases`, `n_tool_calls` (sum of `scores.tool_calls`), `n_with_tool_call`,
  `n_execution_failed`, `failure_rate = n_execution_failed / n_cases`,
  `n_valid_finding`, and the `record.failure_detail` strings seen.
  `ok` is true iff `failure_rate == 0` **and** `n_with_tool_call == n_cases`.
- **A failing candidate is recorded and excluded, never swapped** (spec deliverable 1, verbatim).
  There is no fallback ladder here — that is `judge_slate`'s behaviour and it is deliberately not
  copied. If a candidate fails, the next-ranked one is simply the next one already in the list.
- **No model outside the slate is ever contacted.** Assert this in code: `probe_candidate` raises if
  `model not in {c["model"] for c in PARETO_CANDIDATES}`. Router endpoints (`openrouter/auto`,
  `openrouter/fusion`) are not in the list and cannot be reached (operator instruction, 2026-09-05).
- **Ledger discipline.** Every probe row carries `milestone_tag="m6"`, `purpose="probe"` and
  `probe_model=<id>`. `run_eval_set` overwrites `client.context` per case, so add an optional
  `extra_context: dict | None = None` parameter to `dealpoint/eval/run.py::run_eval_set` that is
  merged into the per-case context dict. That is the whole change to `run.py`; default `None` keeps
  every existing call site byte-identical.
- **Spend guard on the probes.** `run_eval_set` only calls `assert_within_cap` when
  `len(selected) > 10`, so a 2-case probe skips it. `probe_candidate` must call
  `assert_within_cap(est)` and `_assert_within_milestone_absolute(est)` itself before running.

`data/reports/pareto_slate.json`:

```json
{
  "verified_at": "<ISO-8601>",
  "price_basis": "live",
  "probe_case_ids": ["contract_0__q01", "contract_0__q06"],
  "probe_n_cases": 2,
  "candidates": [ {"model":..., "family":..., "rank":..., "role":..., "available":...,
                   "prompt_usd_per_token":..., "completion_usd_per_token":...,
                   "supports_tools":..., "supports_structured_outputs":...,
                   "ok":..., "excluded":..., "failure_rate":..., "n_tool_calls":...,
                   "n_execution_failed":..., "failure_details": [...],
                   "est_usd_per_case_after_probe":..., "est_basis":...,
                   "probe_realized_usd":..., "results_path":...} ],
  "reused": [ {"model": "...", "results_path": "...", "n_cases": 18|32,
               "source": "four_arm_manifest.json (v2, m4_1)", "reused": true} ],
  "not_run": [ {"model": "anthropic/claude-opus-...", "reason": "not run (budget)"}, ... ],
  "stretch_considered": [...],
  "probe_spend_usd": <sum>,
  "git_sha7": "..."
}
```

The reused entries carry `"probe": null` with a note that the model's tool calling was already
demonstrated by a completed 32-case (resp. 18-case) M4.1 arm-D leg — re-probing them would spend
money to re-learn a recorded fact, and the spec says reuse, do not re-run.

CLI: `python -m dealpoint.eval.pareto_slate [--dry-run]`. `--dry-run` does availability + pricing
only, writes the payload with `"probe": null` everywhere, spends nothing. Run it that way once
before spending, to confirm every id is live and to catch a renamed model for free.

---

## §3 — Deliverable 2: the sweeps (`dealpoint/eval/pareto_sweep.py`, new)

```python
PARETO_MANIFEST_PATH  # data/reports/pareto_manifest.json -- SEPARATE from four_arm_manifest.json
def load_pareto_manifest(path=PARETO_MANIFEST_PATH) -> list[dict]
def _append_manifest(entry, path=PARETO_MANIFEST_PATH) -> None
def reused_legs() -> list[dict]                     # resolved from four_arm_manifest.json
def run_model_sweep(model, *, milestone_tag="m6") -> dict
def run_sweeps(models=None, *, stop_usd=M6_STOP_USD) -> dict
def main(argv=None) -> int
```

**Never append to `data/reports/four_arm_manifest.json`.** `dealpoint.eval.report.generate_report`
and `dealpoint.eval.subset._resolve_variant_paths` both read it; adding arm-D rows for new models
would silently change the M4 report and could re-point M5's judged variants. A separate manifest
keeps the M4/M4.1 artefacts frozen, which `tests/test_frozen_artifacts_m4_1.py` and
`tests/test_readme_results.py` both depend on.

Per model, in ranked order:

1. Skip if a manifest entry exists whose `results_path` is on disk (**resume**; the sweep must be
   safely re-runnable after an interruption without re-spending).
2. `per_case, basis = per_case_usd("D", model)` — post-probe this is
   `ledger:measured(arm,model)`. Assert it is, and record the basis in the manifest entry; if it is
   still `ledger:corpus-tokens`, the probe did not run and this model must be skipped with a
   recorded reason rather than swept on a guessed price.
3. `est = per_case * 32`. Call, in this order:
   `assert_within_cap(est)` → `_assert_within_milestone_absolute(est)` →
   the stop-floor: `if realized_usd() + est > M6_STOP_USD: stop, record "stopped (stop-floor)"`.
   The stop-floor must also reserve the judging cost for this variant
   (`18 * 3 * measured_judge_call_usd`, ≈ $0.021) — reserve it before starting the sweep, so a model
   is never swept and then left unjudged.
4. `run_eval_set(case_set=f"pareto_D_{slugify_model(model)}", arm="D", model=model,
   case_rows=load_subset_cases("test_subset_v1"), milestone_tag="m6")` — the **full frozen 32-case
   subset**, in file order, unchanged.
5. Append a manifest entry: `leg="pareto"`, `arm="D"`, `model`, `n_cases`, `est_usd`, `est_basis`,
   `realized_usd`, `results_path`, `summary_path`, `git_sha7`, `index_version`, `milestone="m6"`,
   `rank`, `reused=False`.

`reused_legs()` returns manifest-shaped entries for Haiku and GLM built from
`four_arm_manifest.json` (arm D, `version == "v2"`), marked `reused=True` with `realized_usd` copied
from the M4.1 entry and `milestone="m4_1"` — so the report can separate M6's own new spend from
already-paid-for evidence.

`index_version` on every new leg must equal `e2b4a2b97561`. Assert it; a mismatch means the index
moved and the comparison is not held-fixed. Fail loudly rather than reporting it.

CLI: `python -m dealpoint.eval.pareto_sweep [--models a,b] [--dry-run] [--skip-probe]`.
`--dry-run` prints the per-model estimate table and the projected ledger total and spends nothing.
Add a `justfile` recipe:

```
# M6: verify the Pareto slate (probes), then sweep arm D per model (METERED)
pareto-sweep *ARGS:
    uv run python -m dealpoint.eval.pareto_sweep "$@"
```

---

## §4 — Deliverable 3: judging every completed variant

The M5 harness is driven entirely by `data/eval/judged_subset.json`'s `variants` list. Extend the
list; change nothing else about the subset.

### 4.1 `dealpoint/eval/subset.py::generate_judged_subset`

- The 18 `case_ids`, the `rank`, the `disagreement_class` and the `subset_hash` are computed exactly
  as now, from the two frozen Haiku result files. **They must not change.** `subset_hash` is over
  `case_ids` only, so it stays `5918ef10a7e6`. Verify that after regeneration.
- `variants` becomes M5's three (`A@haiku`, `D@haiku`, `D@glm`, from `JUDGED_VARIANT_LEGS` and
  `four_arm_manifest.json`) **plus one per completed M6 model**, read from `pareto_manifest.json` in
  manifest order, with `variant_id = f"D@{short}"` where `short` is a stable short name derived from
  the model id (last path segment, `-` kept: `qwen3.7-flash`, `deepseek-v4-flash`,
  `gemini-3.1-flash-lite`, `mimo-v2.5`, `llama-4-maverick`). Record the `variant_id → model` mapping
  in the payload so nothing has to re-derive it.
- Recompute `n_traces = n_cases * len(variants)` and `n_judge_calls = n_traces * 3`.
- Bump `"version"` to `"v2"` and extend `JUDGED_SUBSET_RULE_TEXT` with a point 5 stating that M6
  appended one variant per completed Pareto model, that the case list and its ranking were **fixed
  before any M6 run and are unchanged**, and that no case was chosen from any model's output.
- Missing rows are tolerated as today (`build_all_packets` skips a case absent from a variant's
  result file). Haiku's 18-case file covers all 18; every 32-case file is a superset.
- Regeneration must stay byte-identical (`tests/test_judged_subset.py::test_regeneration_is_byte_identical`).
  It is deterministic given the two manifests, so it will be — confirm it.

### 4.2 `dealpoint/eval/judge_run.py`

- Add `--milestone-tag` (default `"m5"`) and pass `"m6"` for this run, so every new judge ledger row
  is tagged `m6` (operator instruction, 2026-09-05).
- Before `verify_slate(client)`, set `client.context = {"purpose": "probe", "probe_model": "judge-slate"}`
  and clear it to `{}` afterwards — the slate smoke calls are not case runs and must carry
  `purpose=probe`. The judge trio is in-slate (it is the M5 trio, unchanged), so no new family is
  contacted.
- **Replace the static `estimate("judges")` guard with one sized to the work actually pending.**
  Compute the pending `(packet_id, judge_model)` pairs first (`build_all_packets()` ×
  `slate["judges"]`, minus `_already_scored`), then
  `est = n_pending * measured_judge_call_usd`, where `measured_judge_call_usd` is the mean `usd`
  over ledger rows carrying a `judge` field (currently $0.000394 over 163 rows), falling back to the
  `SWEEP_DEFS["judges"]["tokens_per_call"]` × `fetch_prices()` shape when no such rows exist. Pass
  that to `assert_within_cap` and `_assert_within_milestone_absolute`. The static 54-trace estimate
  is now wrong in both directions and would either block a legitimate run or under-guard a large one.
- Resume is already correct: `_already_scored` skips every `(packet_id, judge_model)` pair with
  `ok: true`, so re-running judges only the new variants and re-attempts only genuine failures.
  Do not change it.
- Judge each completed variant on all 18 cases × 3 judges = 54 calls per variant (the spec's "36
  calls per model" assumes its 12-case default; D3 keeps 18 cases, so it is 54 — record that).

Run it as `just judge --milestone-tag m6` **after** all sweeps land, so one pass covers every new
variant.

### 4.3 `dealpoint/eval/calibration.py`

No structural change needed: `build_calibration_package()` builds a packet for every
`(case, variant)` in `judged_subset.json`, so extending `variants` extends the packets
automatically. Check and adjust only:

- `suggested_minimum` takes the first 4 packets per variant, so it grows from 12 to `4 × n_variants`.
  Update `render_form_md`'s heading text ("Suggested minimum set (12 packets…)") to render the
  actual count rather than a hard-coded 12, and keep the "no upper bound" sentence.
- `data/reports/judges.json` is written by `calibration.main` via `build_judges_report`; the
  `by_variant` blocks key off `variant_id`, so all variants appear with no change. Verify
  `per_dimension.<dim>.by_variant` and `pairwise_judge_agreement` come out with the new keys.
- The `spend` block in `calibration.main` is M5-specific (`M5_TARGET_USD`, `m5` tag,
  `envelope_headroom_after_m5`). Extend it with `m6_realized_usd` (from `realized_by_tag()`),
  `envelope_headroom_after_m6`, and rename nothing that an existing test reads.
- `judges.json["decisions"]` / `["brief_differences"]`: append an M6 entry recording that the judged
  set grew from 3 to `3 + k` variants, the reason, and the new call count.

Run `just calibration` after judging. It is idempotent and offline.

---

## §5 — Deliverable 4: the report (`dealpoint/eval/pareto_report.py`, new)

Same discipline as `dealpoint/eval/report.py`: a **pure** builder over already-loaded rows plus thin
disk-driven loaders, so the whole thing is unit-testable offline on fake results.

### 5.1 Pure functions

```python
def pareto_frontier(points: list[dict], *, x_key="usd_per_case", y_key="grounded_accuracy",
                    id_key="model") -> list[str]
def per_model_metrics(rows: list[dict]) -> dict
def judged_quality(judge_rows: list[dict], variant_id: str) -> dict
def percentile(values: list[float], q: float) -> float | None
def build_pareto_report(...) -> dict
def render_markdown(report: dict) -> str
def render_svg(report: dict) -> str
def regenerate_readme_pareto(report, readme_path=README_PATH) -> None
```

**`pareto_frontier` must be deterministic and total** — this is the spec's named unit-test target:

- Minimise `x` (realised $/case), maximise `y` (grounded accuracy). A point is dominated iff another
  point has `x <= ` and `y >= ` with at least one strict inequality.
- Points with a `None` on either axis are excluded from the frontier and recorded as
  `frontier: false, frontier_note: "not comparable (missing <axis>)"`.
- Exact ties on **both** axes: keep the lexicographically-smallest `id_key` on the frontier and mark
  the other(s) `frontier: false, tied_with: <id>` — never both, never a coin flip.
- Implementation: sort by `(x asc, y desc, id asc)`, sweep keeping a running best `y`, a point joins
  the frontier iff its `y` is strictly greater than every `y` seen so far. Return ids in sorted-x
  order.
- Unit-test: empty input, one point, all-dominated chain, all-on-frontier, exact ties on x only, on
  y only, on both, and a `None` axis.

`per_model_metrics(rows)` reuses `dealpoint.eval.report.arm_metrics` verbatim (it already produces
`grounded_accuracy`, `execution_failed`, `cap_hit`, `abstain_recall`, `false_abstain`,
`skill_adherence`, `usd`, `wall_ms` with means and n). Do not reimplement any scorer. Add on top:
`latency_median_ms`, `latency_p90_ms` (from each row's `scores.wall_ms`, nearest-rank percentile,
pure Python), `usd_per_case` (**realised**: `sum(row.usd) / n_cases`, from the result rows, not from
the estimate), `total_usd`, `n_cases`.

`judged_quality` computes, per variant, per dimension, the **mean of judges** over `ok: true` rows
plus an overall mean, each with its `n`, and stamps `"secondary": true` and
`"label": "model-judged, secondary, never objective truth"` on the block. Variants with no judge
rows get `null` and a reason.

### 5.2 `data/reports/pareto.json`

```json
{
  "generated_at": "...", "git_sha7": "...", "version": "v1",
  "budget_scaled": true,
  "budget_scaled_note": "<32-case frozen subset, 18 at Haiku; not the brief's full design>",
  "held_fixed": {"arm": "D", "index_version": "e2b4a2b97561",
                 "skill_version": "f8d255cc169b", "subset": "test_subset_v1",
                 "subset_hash": "be2e96433b14", "n_cases": 32,
                 "sentence": "Only the model varies."},
  "models": {
    "<model id>": {
      "family": "...", "rank": 1, "reused": false, "n_cases": 32,
      "prompt_usd_per_token": ..., "completion_usd_per_token": ..., "price_basis": "live",
      "grounded_accuracy": {"mean":..., "n":...},
      "abstain_recall": {...}, "execution_failed": {...}, "cap_hit": {...},
      "answer_correct": {...}, "citation_gold_overlap": {...}, "skill_adherence": {...},
      "latency_median_ms":..., "latency_p90_ms":...,
      "usd_per_case":..., "total_usd":...,
      "judged_quality": {"secondary": true, "label": "...", "variant_id": "D@...",
                         "reasoning": {"mean":..., "n":...}, "evidence": {...},
                         "trajectory": {...}, "professional": {...},
                         "overall": {"mean":..., "n":...}},
      "frontier": true, "results_path": "...", "est_usd":..., "est_basis":..."
    }
  },
  "frontier": {"rule": "<the deterministic rule, in words>",
               "x": "usd_per_case", "y": "grounded_accuracy",
               "models": ["...", "..."]},
  "not_run": [{"model": "...", "reason": "not run (budget)"}],
  "slate": {"path": "data/reports/pareto_slate.json", "probe_spend_usd": ...,
            "excluded": [{"model": "...", "failure_rate": ..., "reason": "..."}]},
  "spend": {"target_usd": 1.00, "envelope_usd": 4.00,
            "m6_realized_usd": ..., "m6_probe_usd": ..., "m6_sweep_usd": ...,
            "m6_judge_usd": ..., "by_milestone": {...},
            "total_realized_usd": ..., "envelope_headroom": ...,
            "estimate_vs_realised": [{"model":..., "est_usd":..., "realized_usd":..., "basis":...}]},
  "comparability": {"haiku_n_cases": 18, "others_n_cases": 32,
                    "common_case_ids": [...], "note": "<see §1.4 item 4>"},
  "majority_baseline": {...},
  "decisions": [ ...§1.3... ],
  "brief_differences": [ ...§1.4... ],
  "caveats": [ "Judged quality is secondary and model-judged; it is never objective truth.",
               "The frontier is a frontier of the cheap tier: no ceiling model was affordable.",
               "..." ]
}
```

Probe spend is listed **separately** from sweep spend (operator instruction, 2026-09-05): sum ledger
rows with `milestone_tag == "m6"` and `purpose == "probe"` for `m6_probe_usd`; rows with a `judge`
field for `m6_judge_usd`; the rest for `m6_sweep_usd`. The three must sum to `m6_realized_usd` —
assert it in the report generator and in a test.

### 5.3 Chart — `data/reports/pareto.svg`

Hand-written SVG string, no dependency (D4). Requirements:

- x-axis = realised $/case (log scale is fine and is easier to read across a 45× price range — say
  so in the axis label), y-axis = grounded accuracy 0–1.
- One marker per completed model, labelled with the model's short name.
- **Latency encoded** (brief §2.6: "latency encoded") — marker radius scaled by `latency_median_ms`,
  with a legend giving the radius→ms mapping.
- Frontier models joined by a polyline and drawn filled; dominated models drawn hollow.
- Deterministic: no timestamp, no random ids, coordinates rounded to 2 decimals, so re-running
  produces a byte-identical file. Test that.

### 5.4 Markdown + README

- `data/reports/pareto.md` via `render_markdown`. First sentence must carry, like `judges.md` does:
  **budget-scaled**, **objective vs judged** ("grounded accuracy is objective; judged quality is
  secondary and model-judged"), and **not comparable to MAUD leaderboard numbers**.
- README: add a **second** generated block with its own markers,
  `<!-- BEGIN PARETO -->` / `<!-- END PARETO -->`, under a `## Model cost/quality Pareto (M6)`
  heading. Use the same regex-substitution pattern as `report.regenerate_readme`; append the block
  if the markers are absent.
  **Do not touch the `<!-- BEGIN RESULTS -->` block** — `tests/test_readme_results.py` (gate_m4)
  pins every number in it to `four_arm.json`.
- The PARETO block renders: the held-fixed sentence, the per-model table (model, n_cases,
  grounded_accuracy with its scored-n denominator in `(n_scored/n_cases)` form — the M4 convention,
  abstain_recall, execution_failed, cap_hit, median/p90 latency, $/case, total $), a frontier line,
  the judged-quality column clearly labelled secondary, the "not run (budget)" list, the
  comparability note, and the total M1–M6 spend.

### 5.5 CLI + recipes

`python -m dealpoint.eval.pareto_report` reads `pareto_manifest.json` + `four_arm_manifest.json`
(reused legs) + `judge_scores.jsonl` + `judged_subset.json` + `pareto_slate.json` + the ledger, and
writes `pareto.json`, `pareto.md`, `pareto.svg` and the README block.

```
# M6: verify the Pareto slate at run time (2 dev cases per candidate, METERED)
pareto-slate *ARGS:
    uv run python -m dealpoint.eval.pareto_slate "$@"

# M6: regenerate data/reports/pareto.json + pareto.md + pareto.svg + README Pareto block
pareto-report:
    uv run python -m dealpoint.eval.pareto_report

# milestone 6 acceptance gates only, offline
gate-m6:
    uv run pytest -m "gate_m6 and not needs_network and not needs_model" -q
```

---

## §6 — Tests (all `pytestmark = pytest.mark.gate_m6` unless noted)

The DoD names three offline gates; these five files cover them plus the traceability and spend
assertions.

**`tests/test_pareto_frontier.py`** — the frontier computation, on synthetic points only, no disk:
empty; single point; strictly-improving chain (all on frontier); strictly-dominated chain (one on
frontier); tie on x only (higher y wins, lower y off); tie on y only (cheaper wins); exact tie on
both (lexicographically-smallest id on frontier, other marked `tied_with`); a `None` on either axis
excluded with a reason; and **order-independence** — shuffling the input list gives the same result.

**`tests/test_pareto_report.py`** — `build_pareto_report` on fake per-model rows built the way
`tests/test_report_m4.py::_row` builds them (copy that helper's shape):
- every §2.6 metric present per model, with its `n`;
- `usd_per_case` is realised (`sum(usd)/n`), not the estimate;
- `latency_median_ms` / `latency_p90_ms` correct on a known list;
- judged quality is the mean of judges, carries `"secondary": true`, and is `null` with a reason for
  a variant with no judge rows;
- `not_run` entries survive into the payload with their reasons;
- a model whose rows are all `execution_failed` still appears, with `grounded_accuracy.n == 0` and
  `mean == None`, and is excluded from the frontier rather than crashing it;
- `spend.m6_probe_usd + m6_sweep_usd + m6_judge_usd == m6_realized_usd`;
- `render_markdown`'s first sentence carries budget-scaled + objective/judged + not-comparable;
- `render_svg` emits one marker per completed model and is byte-identical across two calls.

**`tests/test_pareto_slate.py`** — `verify_slate` with `FakeClient` + a provided price table, the
way `tests/test_judge_slate.py` does it:
- every candidate available and tool-calling → all `ok`, none excluded;
- a candidate absent from the price table → `available: false`, `excluded: true`, and
  **`FakeClient.calls` shows no call was made to it**;
- a candidate that returns a finding with **no tool call** → recorded with its `failure_rate` and
  `excluded: true`, and the surviving list is **not** back-filled with a substitute (the spec's
  "excluded, not silently swapped" — assert the slate is one shorter, not the same length);
- `probe_candidate` raises for a model outside `PARETO_CANDIDATES` (router endpoints included);
- prices and the price basis are recorded on every candidate.

**`tests/test_readme_pareto.py`** — every number in the README PARETO block is traceable to
`pareto.json` (the DoD's "a test compares"), modelled on `tests/test_readme_results.py`: skip
cleanly if `pareto.json` does not exist yet, but **fail** if it exists and the block does not, or if
any rendered percentage, `$/case`, latency or `(n_scored/n_cases)` denominator is absent from the
block. Assert the first sentence carries budget-scaled, objective-vs-judged and not-comparable.

**`tests/test_spend_m6.py`** —
- `SWEEP_DEFS["pareto"]` has the three legs of §1.1 and its models are exactly
  `PARETO_NEW_MODELS` / the judge trio (one list, no drift);
- `estimate("pareto")["est_usd"] > 0` and `realized_usd() + est_usd <= 4.00`;
- **`realized_usd() <= 4.00`** — the DoD's envelope assertion, read straight from the ledger;
- every ledger row with `milestone_tag == "m6"` names a model in
  `PARETO_CANDIDATES ∪ JUDGE_TRIO ∪ {SPARE_JUDGE}` — no router endpoint, no off-slate model
  (operator instruction, 2026-09-05);
- every `m6` row that is not a case run (no `case_id`, or `case_set` starting `dev_` from a probe)
  carries `purpose == "probe"` with a `probe_model`;
- the three m6 spend buckets sum to the m6 tag total.
- These last four skip cleanly when no `m6` rows exist yet, so the gate is green before the metered
  phase and meaningful after it — the same convention `tests/test_readme_results.py` uses.

**Existing tests to update** (say why in the commit message; both are legitimate M6 extensions, not
loosened assertions):
- `tests/test_judged_subset.py::test_three_variants_recorded` — becomes: the three M5 variant ids
  are still present **and** there is exactly one variant per completed model in
  `pareto_manifest.json`, and `n_traces == n_cases * len(variants)`. Keep every other assertion in
  that file unchanged (18 cases, every question covered, redacted + oos present, seed, rule,
  `subset_hash` unchanged, rank a permutation, byte-identical regeneration).
- `tests/test_calibration_package.py` — already derives its expected counts from
  `judged_subset["n_traces"]`, so it should pass unchanged. Confirm; if `render_form_md`'s
  hard-coded "12 packets" heading is asserted anywhere, update the assertion with the heading.

Do not touch `tests/test_frozen_artifacts_m4_1.py`. Its pinned digests (`SUBSET_HASH`,
`SKILL_VERSION`, `ARMS_SHA256`, the prompt hashes) are the "held fixed" guarantee this milestone
rests on; if any of them moves, arm D was not held fixed and the whole experiment is void.

---

## §7 — Build order

**Phase A — offline, no spend. Everything below must be green before a single metered call.**

1. §1.1 `SWEEP_DEFS["pareto"]` + the `dealpoint/config.py` constants. Verify
   `uv run python -m dealpoint.eval.budget pareto` prints `est_usd ≈ 0.71–0.78` and that
   `realized + est < 4.00`. **The spend gate cannot pass until this is done.**
2. `dealpoint/eval/pareto_slate.py` (§2) + `run_eval_set(extra_context=...)` (§2).
3. `dealpoint/eval/pareto_sweep.py` (§3).
4. `dealpoint/eval/subset.py` judged-subset extension (§4.1) — code only; the file on disk is
   regenerated in Phase B once the manifests exist.
5. `dealpoint/eval/judge_run.py` changes (§4.2) and `dealpoint/eval/calibration.py` changes (§4.3).
6. `dealpoint/eval/pareto_report.py` (§5).
7. All of §6. `uv run pytest -m "gate_m6 and not needs_network" -q`, the full offline suite,
   `uv run ruff check .`, `uv run pyright` — all green.

**Phase B — metered, in this exact order. Stop and record if any step's guard fires.**

8. `just pareto-slate --dry-run` — availability + pricing only, **$0.00**. Confirm all five ids are
   live and tool-calling is advertised. If one is gone, record it as unavailable and move to the
   next-ranked candidate in the sweep-file list; do not invent a substitute.
9. `just pareto-slate` — 5 candidates × 2 dev cases ≈ **$0.04**. Read
   `data/reports/pareto_slate.json`. Any candidate with a non-zero `failure_rate` is excluded and
   reported with that rate; it is not swapped.
10. `just pareto-sweep --dry-run` — the per-model estimate table on the now-**measured** per-case
    costs. Sanity-check the projected total against $3.90 before proceeding.
11. `just pareto-sweep` — arm D × 32 frozen cases per surviving model ≈ **$0.62**, ranked order,
    guard before each model, stop-floor honoured. Resumable.
12. `uv run python -m dealpoint.eval.subset` — regenerate `judged_subset.json` with the new
    variants. **$0.00.** Confirm `subset_hash == "5918ef10a7e6"` and the 18 case ids are unchanged.
13. `just judge --milestone-tag m6` — 54 calls per new variant ≈ **$0.09**. Resumable; re-run if it
    stops part-way.
14. `just calibration` — packets, form, `judges.json`, `judges.md`. **$0.00.**
15. `just pareto-report` — `pareto.json`, `pareto.md`, `pareto.svg`, README PARETO block. **$0.00.**
16. Re-run the whole check set (offline suite, gate_m6, ruff, pyright) and confirm
    `realized_usd() <= 4.00`.

If the stop-floor fires at step 11 with fewer than 5 new models, that is a **success**, not a
failure: the DoD floor is ≥ 3 models (Haiku + two others) and Haiku + GLM are already in hand.
Record which models were stopped and why in `pareto.json["decisions"]` and carry on to step 12.

---

## §8 — Definition of done, mapped

| DoD item | Where it is satisfied | How it is verified |
|---|---|---|
| `gate_m6` offline: Pareto-frontier computation unit-tested | §5.1, §6 | `tests/test_pareto_frontier.py` |
| `gate_m6` offline: report generator tested on fake per-model results | §5, §6 | `tests/test_pareto_report.py` |
| `gate_m6` offline: slate verification tested with a fake client | §2, §6 | `tests/test_pareto_slate.py` |
| Spend gate passed | §1.1 | `just budget pareto`; `tests/test_spend_m6.py` |
| ≥ 3 models (Haiku + two others) complete on the frozen subset | §1.2, §3 | `pareto_manifest.json` + reused legs; asserted in `tests/test_pareto_report.py` against `pareto.json` |
| `pareto.json`, chart, `pareto_slate.json` exist | §2, §5 | files on disk; `tests/test_readme_pareto.py` |
| Judged subset covers every completed variant | §4.1, §4.2 | `tests/test_judged_subset.py` (updated) |
| Calibration packets cover every judged trace | §4.3 | `tests/test_calibration_package.py` (`n_traces`) |
| Total realised M1–M6 spend ≤ $4.00, asserted from the ledger | §5.2, §6 | `tests/test_spend_m6.py` |
| README regenerated; every number traceable to a report file | §5.4 | `tests/test_readme_pareto.py` |
| Full offline suite, ruff, pyright green | §7 | the check phase |

---

## §9 — Guardrails

- **Never re-run Haiku or GLM on arm D.** Their results are on disk and paid for; the spec says
  reuse. Re-running them would cost $0.53 + $0.06 and change nothing.
- **Never edit `data/eval/*.jsonl` or `test_subset_v1.json` by hand.** The subset is frozen; its
  hash is pinned by `tests/test_frozen_artifacts_m4_1.py`.
- **Never append to `four_arm_manifest.json`.** §3.
- **Never edit `adws/**`, `specs/grilled-product-brief.md` or `specs/milestones/m6.md`.**
- **Never edit `specs/mvp/state.json`** (the M5 DoD's rule; it applies here too — the factory owns it).
- **No rubric change.** `rubric_version` stays `cfda9f8cc401`; `assert_rubric_frozen()` enforces it
  and any change would invalidate every M5 judge score.
- **`MAX_TOOL_CALLS` stays 8** and `MAX_TOKENS_FINAL` stays 1200. A model that hits the cap or
  fails is a first-class result (brief Appendix D), not noise to tune away. Report
  `execution_failed` and `cap_hit` per model honestly; a cheap model that cannot tool-call reliably
  is exactly the finding this milestone exists to produce.
- **No new dependency.** No matplotlib, no scipy, no pandas. 1.3 GB of disk is left.
- **Every metered call carries `milestone_tag="m6"`**, and every non-case-run call carries
  `purpose="probe"` with the model id being verified.
- **No model outside the slate or the judge trio is ever called** — router endpoints
  (`openrouter/auto`, `openrouter/fusion`) included.
- If a measured number contradicts this plan (a price moved, a model is gone, a sweep costs more
  than estimated), **the measurement wins**: record it in `pareto.json`, adjust, and say so. Do not
  bend a result to match a table in this document.
