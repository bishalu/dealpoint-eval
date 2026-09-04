# M4 — Four-arm experiment on the frozen test set

**Spec:** `specs/milestones/m4.md` (read-only). **Requirements:** `specs/grilled-product-brief.md`
§1.6, §2.2, §2.4, §2.7, §6, Appendix B (read-only).
**Repo:** `/home/exedev/repos/dealpoint-eval`, HEAD `055a6fc`. M0.1–M3 passed.

---

## 0. Read this before you touch anything

Four things in this milestone are irreversible or expensive if done in the wrong order. They are
the whole risk surface.

### 0.1 The freeze order is the milestone

> "**No test run may be used to change any config, prompt, or skill; the skill, arm configs and the
> subset are frozen (hashes recorded) before the first test case runs.**" — spec §5

So the build is strictly two phases, and **you must commit between them**:

- **Phase 1 (offline, free).** Everything in §2–§8 below: skill files, adherence scorer, arm
  configs, arms C/D wiring, frozen subset, report generator, tests. `gate_m4` offline green,
  full suite green, ruff green, pyright green. **Commit.** This commit is the freeze.
- **Phase 2 (metered).** §9–§11: caching probe, spend gate, the two sweeps, report generation,
  README. After the first frozen-subset case runs, you may **not** edit `skills/**`,
  `ARMS`, `dealpoint/agent/prompts.py`, or `data/eval/test_subset_v1.json`. If a sweep reveals a
  defect in those, stop and report it — do not silently fix and re-run.

Bug fixes to the *runner*, the *report generator* or the *scorers* after a sweep are fine (they do
not change what the model saw); note any such fix in the report.

### 0.2 You are allowed a cheap dev smoke — use it

The spec says "No dev sweep." That prohibits a full scored dev sweep reported as a result. It does
**not** prohibit the workhorse-model smoke the spec's own "Cheap workhorse model" clause authorises
("the default model for every dev-loop, smoke, and exploratory metered call … the builder's own
testing included").

**Before the frozen sweeps, run ≤ 6 `dev` cases at `z-ai/glm-5.3-flash`, arms C and D, to prove the
new wiring works end to end** (~$0.01). Arms C and D have never made a metered call in this repo.
Discovering that arm C's retriever construction throws, or that arm D's skill block breaks
structured output, is worth $0.01 here and costs ~$1.00 if you find it during the Haiku sweep.
Record the smoke in the report under "dev-loop spend".

### 0.3 The spend gate currently FAILS — you must fix it first

The factory runs the spend gate before your metered phase:
`uv run python -m dealpoint.eval.budget four_arm`, compared against M4's `absolute_usd=3.00`
(`adws/adw_modules/milestones.py:42-46`, factory code — **do not edit it**).

Measured now:

```
$ uv run python -m dealpoint.eval.budget four_arm
{"est_usd": 3.088341, "calls": 512, "cases": 128, ...}     # 3.088 > 3.00  -> gate BLOCKS
```

Because `SWEEP_DEFS["four_arm"]` still describes the pre-re-sizing design (4 arms × 32 cases, all
at Haiku). It no longer describes the sweep this spec asks for. **Updating it (§9.1) is a
prerequisite for the gate passing**, and it is a correctness fix, not gaming: the estimate must
describe the run you are actually going to make.

### 0.4 The estimator silently returns $0.00 for GLM — fix it

```
per_case_usd("D", "z-ai/glm-5.3-flash") -> (0.0, 'ledger:tokens x live')
```

`dealpoint/eval/spend.py::per_case_usd` branch 4 computes `_mean_tokens_per_call(model_rows)` where
`model_rows` are ledger rows **for that model**. GLM has zero ledger rows, so mean tokens = 0, so
the estimate is $0.00. An estimate of zero passes every cap and measures nothing.

Fix in §9.2: when a model has no ledger rows, fall back to the corpus-wide mean tokens/call across
all models (measured: 4952.1 input, 145.8 output over 96 rows) and say so in the returned `basis`.

---

## 1. What already exists (do not rebuild it)

| Thing | Where | Note |
|---|---|---|
| Arm A (pipeline) | `dealpoint/agent/pipeline.py::run_pipeline` | module-level `ARM = "A"` |
| Arm B (agent loop) | `dealpoint/agent/loop.py::run_agent` | module-level `ARM = "B"`, cap 8 |
| Prompts | `dealpoint/agent/prompts.py` | `system_prompt()`, `question_spec_block(q)`; **no skill seam yet** |
| Tools | `dealpoint/agent/tools.py` | 3 tools + `tool_schemas()` |
| Execution record | `dealpoint/agent/schema.py` | `ExecutionRecord{status, failure_reason, trajectory[], usage, ...}`, `TrajectoryStep{tool, args, result_ref, chunk_ids, char_ranges, t_ms}` |
| Scorers | `dealpoint/eval/scorers.py` | all §2.4 metrics; `skill_adherence()` is a **stub returning `None`** |
| Runner | `dealpoint/eval/run.py` | `--arm` choices `["A","B"]` only; seam comment at :243-245 |
| Spend | `dealpoint/eval/spend.py` | `per_case_usd`, `estimate`, `assert_within_cap`, `SWEEP_DEFS` |
| Braintrust | `dealpoint/eval/braintrust_adapter.py` | 6 scores incl. `skill_adherence`; `braintrust_runs.json` |
| Arm C retriever | `dealpoint/config.py::ARM_C_RETRIEVER` | frozen `hybrid_rrf`, `arm_c_index_version()` = `8ae2fc1e24e9` |
| Retriever builder | `dealpoint/corpus/retrievers.py::build_retriever(config, dense=, sparse=)` | takes a `RetrieverConfig` |

**Absent, and yours to create:** `skills/`, `README.md`, `just report`, `data/reports/four_arm.*`,
`data/eval/test_subset_v1.json`, any `skill_version`.

---

## 2. Skill files (deliverable 1)

Create `skills/ma-deal-point-review/SKILL.md` — generic procedure, the **8 rules of Appendix B
verbatim in substance**, ≤ ~800 tokens (≤ ~3,200 chars; assert it in a test).

Each rule must be phrased so it maps 1:1 onto the deterministic check in §3. Keep the numbering
1–8 identical to Appendix B so the scorer's rule ids line up. Do not invent a 9th rule.

Create `skills/ma-deal-point-review/playbooks/q01.md … q12.md` — **3–5 lines each** (assert
`3 <= non-blank line count <= 5` in a test). Per brief §1.6/Appendix B each playbook says: where the
term lives, what distinguishes the options, the trap case. Write them from
`dealpoint/data/questions.py::QUESTION_SPEC` (each `QuestionSpec` has `.id`, `.maud_question`,
`.gloss`, `.options`, `.canonical_query`, `.reasoning_type`, `.required_evidence`) — these are
domain hints, not answers. **Never encode a gold answer or a majority answer in a playbook.** Add a
test asserting no playbook contains any question's majority answer string.

New module `dealpoint/agent/skill.py`:

```python
SKILL_DIR = REPO_ROOT / "skills" / "ma-deal-point-review"   # add to config.py

def skill_version() -> str:
    """12-char sha256 over SKILL.md + every playbooks/*.md, sorted by path,
    hashing (relative_path, bytes) pairs. Pure, no I/O beyond reading them."""

def load_skill() -> str: ...
def load_playbook(question_id: str) -> str | None: ...   # None for oos* questions

def skill_block(question_id: str) -> str:
    """SKILL.md + the ONE applicable playbook, as a system-prompt block.
    Loads no other playbook (spec deliverable 1)."""

def stamp_skill_version() -> None:
    """Merge-write `skill_version` into data/reports/versions.json without
    rebuilding the index."""
```

Also add `"skill_version": skill_version()` to the `versions_payload` in
`dealpoint/corpus/build_index.py:192-203` so a future rebuild keeps it. **Do not rebuild the index**
— use `stamp_skill_version()` now. `tests/test_tournament.py:360` reads `versions.json`; an
additive key is fine, but re-run that test.

`dealpoint/eval/braintrust_adapter.py::_skill_version()` already reads this key — it will start
returning a real value automatically.

---

## 3. Skill-adherence scorer (deliverable 2) — computed for ALL arms

New module `dealpoint/eval/skill_adherence.py`. Each Appendix-B rule is a pair of pure predicates
over `(case, finding, record, doc)`:

```python
@dataclass(frozen=True)
class Rule:
    id: int                 # 1..8, matching Appendix B
    name: str
    applicable: Callable[..., bool]
    satisfied: Callable[..., bool]

RULES: tuple[Rule, ...]     # exactly 8

def evaluate(case, finding, record, doc) -> dict:
    """-> {"score": float|None, "n_applicable": int, "n_satisfied": int,
           "rules": {"1": "satisfied"|"violated"|"n/a", ...}}
       score = n_satisfied / n_applicable, or None when n_applicable == 0."""
```

Rule semantics (all deterministic, all from the trajectory/finding/labels — never from a model
judgement):

| # | Applicable when | Satisfied when |
|---|---|---|
| 1 | always | ≥ 1 `search_agreement` step in the trajectory |
| 2 | `case["required_evidence"]` is not null (q05–q08, q11) | a `lookup_defined_term` step whose `args["term"]` matches a term from `scorers.parse_required_evidence` (reuse `scorers._term_matches`) |
| 3 | any retrieved text contains a cross-reference (regex `Section\s+\d+\.\d+` / `Article\s+[IVXL]+`) | a later `get_section` step whose `args["section_ref"]` matches one of those references |
| 4 | a retrieved chunk's text ends mid-list (stripped text ends with `;`, `,`, ` and`, ` or`) | a later step whose `char_ranges` start within ~200 chars of that chunk's end (the next chunk/section was read) |
| 5 | `finding` is not None and has evidence | ≤ 3 citations **and** every quote verbatim (`scorers.locate_quote`) **and** every quote's located range overlaps some trajectory `char_range` |
| 6 | `record.status == "ABSTAINED"` | ≥ 2 `search_agreement` steps with **distinct** normalised (casefold/whitespace-collapsed) queries |
| 7 | `record.status == "ABSTAINED"` | the provision is genuinely absent — i.e. `case["case_set"] == "counterfactual"` (no gold spans). An abstention on a case that has gold spans violates rule 7 |
| 8 | `finding` is not None | rationale ≤ 80 words **and** it names the chosen option (answer substring, casefolded) or a section ref |

Then replace the stub in `dealpoint/eval/scorers.py`:

```python
def skill_adherence(case, finding, record, canonical_text) -> float | None:
    ...  # delegates to skill_adherence.evaluate(...)["score"]
```

Keep `score_case`'s signature; put the per-rule detail dict into the result row as a **separate**
`skill_rules` key (metadata, not a Braintrust score — the 6-score budget is fixed).

**Two existing tests assert the stub is `None` and must be updated deliberately:**
`tests/test_scorers.py:342` (`assert scores["skill_adherence"] is None`) and
`tests/test_braintrust_adapter.py:144`. Change them to the new real values; do not delete them.

Unit-test each rule's *applicable* and *satisfied* legs on synthetic trajectories (spec DoD).

**Honesty note for the report:** rules 2/3/4/6 need tools arm A does not have, so arm A's
adherence is computed over a smaller applicable set. The report must print
`n_applicable` and `n_satisfied` per rule per arm, not just the ratio, so an A-vs-D adherence
comparison cannot be read as like-for-like. State this in `four_arm.md`.

---

## 4. Arm configs + the arm-diff test (deliverable 3)

In `dealpoint/config.py`, next to `ARM_C_RETRIEVER`:

```python
DENSE_RETRIEVER: dict = {"kind": "dense", "name": "dense", "fetch_k": RETRIEVER_DEFAULT_K,
                         "rerank_model": None, "multi_query": False, "rrf_k": RRF_K}

ARMS: dict[str, dict] = {
    "A": {"loop": "pipeline", "retriever": DENSE_RETRIEVER,  "skill": False},
    "B": {"loop": "agent",    "retriever": DENSE_RETRIEVER,  "skill": False},
    "C": {"loop": "agent",    "retriever": ARM_C_RETRIEVER,  "skill": False},
    "D": {"loop": "agent",    "retriever": ARM_C_RETRIEVER,  "skill": True},
}
ARM_ORDER: tuple[str, ...] = ("A", "B", "C", "D")
```

Exactly three keys per arm, so "differs by exactly one config key" is checkable. A→B differs in
`loop`; B→C in `retriever`; C→D in `skill`.

Test (`gate_m4`): for each adjacent pair in `ARM_ORDER`, `len([k for k in a if a[k] != b[k]]) == 1`,
and assert *which* key it is (`loop`, `retriever`, `skill` respectively) — a test that only counts
would pass if two arms swapped the wrong variable.

Brief §2.2 names arm C `agent-hybrid-rerank`; the M3 tournament winner is `hybrid_rrf` **without**
rerank (`dealpoint/config.py:176-188` records the rationale). Brief §2.2 also defines arm C's backend
as "tournament winner", so the measurement governs. **Carry this forward: label arm C
`agent-hybrid-rrf` in the report/README and restate the reason in `four_arm.md`.** Report it, do not
re-open it.

---

## 5. Wire arms C and D into the agent and the runner

`dealpoint/agent/loop.py::run_agent` — add two keyword params, defaults preserving today's
behaviour so the eight existing `run_agent(...)` call sites in `tests/test_agent_loop.py` and
`tests/test_smoke_m1.py` stay green:

```python
def run_agent(case, doc, retriever, client, question, model,
              index_version=None, *, arm: str = "B", skill_block: str | None = None):
```

Build the static prefix as:

```python
static_prefix = system_prompt() + "\n\n" + question_spec_block(question)
if skill_block:                      # arm D only
    static_prefix += "\n\n" + skill_block
```

and pass `arm=arm` into `build_record(...)` in place of the module-level `ARM`. Keep `ARM = "B"` as
the default. Do **not** put skill text in `prompts.py` — that module's docstring promises a reviewer
it contains no skill rules; keep that true by composing in `loop.py`.

`dealpoint/agent/run.py`: `--arm` choices → `["A","B","C","D"]`; build the arm's retriever and,
for D, the skill block.

`dealpoint/eval/run.py`:
- `--arm` choices → `["A","B","C","D"]`; replace the seam comment at :243-245.
- `_build_retriever(fake, arm)`: for A/B keep `LazyRetriever()`; for C/D build the frozen arm-C
  retriever via `build_retriever(RetrieverConfig(**ARM_C_RETRIEVER), dense=DenseRetriever(),
  sparse=BM25Retriever())`. **Construct the retriever once per sweep, not per case** — Qdrant local
  mode allows one client per path, and `BM25Retriever` caches per document.
- Dispatch: `run_pipeline` for A, `run_agent(..., arm=arm, skill_block=skill_block(qid) if
  ARMS[arm]["skill"] else None)` for B/C/D.
- `index_version` stamped into rows: `arm_c_index_version()` for C/D, `index_version()` for A/B.
- Keep the `--fake` path working for all four arms (`OfflineChunkRetriever` for C/D too).

Extend `tests/test_eval_run.py`'s parametrise to `["A","B","C","D"]` under `--fake`.

---

## 6. Frozen discriminative subset (deliverable 4)

Write `dealpoint/eval/subset.py` producing `data/eval/test_subset_v1.json`. **The rule and the seed
live in the file**; a `gate_m4` test regenerates it and asserts byte-for-byte equality (same
convention as the M0 determinism gate: `sort_keys=True`, `indent=2`, trailing newline).

Chosen **before any test case runs, from labels only** — never from model output. The only
case-level signal used is `gold_answer != majority_answer`, which is a MAUD label fact.

### 6.1 The rule (write it into the JSON as `rule`)

1. **Question priority order** — reasoning types where the arms should differ come first
   (`QuestionSpec.reasoning_type`): `defined-term`, `cross-ref`, `carve-out` →
   `q05, q06, q07, q08, q09, q10, q11, q12`; then the rest → `q01, q02, q03, q04`.
   Within a group, ascending question id.
2. **2 test cases per question (24)** — from `data/eval/test.jsonl`, prefer cases where
   `gold_answer != majority_answer` (the majority baseline already gets these wrong, so arm
   differences are visible rather than masked by a degenerate question). Every question has ≥ 5
   such cases (verified: min 5 at q08), so the preference never runs dry. Order within the pool by
   `sha256(f"{SEED}:{case_id}")` ascending; take the first 2. `SEED = 42` from `dealpoint.config`.
3. **8 counterfactual cases** — 6 `kind == "redacted"`, at most one per question, taking questions
   in priority order (→ q05, q06, q07, q08, q09, q10: six distinct questions, ≥ 4 ✓); within a
   question, the same seeded hash order. Plus 2 `kind == "out_of_scope"`, first two by seeded hash
   order of their case ids.
4. **Total 32.**

### 6.2 Tranches — so the Haiku N is a whole-question prefix

The spec sizes the Haiku replication as "the largest whole-question prefix of the frozen subset"
that projects ≤ $1.00. Make that a first-class, pre-registered property of the file rather than an
after-the-fact slice:

- **`tranche_1` (18 cases)** = the **rank-1** case of each of the 12 questions (priority order)
  + 4 redacted (q05, q06, q07, q08) + 2 out-of-scope.
  Satisfies the spec's floor: ≥ 1 case per question ✓, counterfactuals included ✓.
- **`tranche_2` (14 cases)** = the rank-2 case of each of the 12 questions + the remaining 2
  redacted (q09, q10).

`case_ids` in the file is `tranche_1 + tranche_2` in that order, so "prefix" is literal.

### 6.3 File shape

```json
{
  "version": "v1",
  "seed": 42,
  "n_cases": 32,
  "rule": "<the prose of §6.1 and §6.2, verbatim>",
  "question_priority": ["q05", "...", "q04"],
  "tranche_1": ["<18 case ids>"],
  "tranche_2": ["<14 case ids>"],
  "case_ids": ["<all 32, tranche_1 then tranche_2>"],
  "subset_hash": "<12-char sha256 over the ordered case_ids>",
  "dataset_version": "81ed82cd7552",
  "generated_from": {"test": "data/eval/test.jsonl", "counterfactual": "data/eval/counterfactual.jsonl"}
}
```

Add `just subset` (regenerate) and a loader `load_subset()` returning the case rows, resolving each
id across `test`/`counterfactual` via `dealpoint.eval.cases.find_case`. Add
`--subset test_subset_v1` to `dealpoint/eval/run.py` selecting exactly these cases in file order
(and `--tranche 1` to run only tranche 1).

Tests (`gate_m4`): regeneration is byte-identical; 32 cases; 24 test + 6 redacted + 2 oos; every
question appears ≥ 1 (in fact exactly 2); redacted spans ≥ 4 questions; `tranche_1` has all 12
questions; no case id appears twice.

---

## 7. Report generator (deliverable 6)

New module `dealpoint/eval/report.py` → `data/reports/four_arm.json` + `four_arm.md`.
Add `just report` → `uv run python -m dealpoint.eval.report`, and a `report` subcommand is optional.

Input: the result JSONLs written by the sweeps under `data/results/` (select by
`(case_set, arm, model, git_sha7)` recorded in a small manifest the sweeps write, or by globbing and
filtering on the subset's case ids — prefer an explicit manifest, it is less fragile).

`four_arm.json` must contain:

- `sweeps`: one entry per (model, arm) leg with `n_cases`, `est_usd`, `realized_usd`, `basis`,
  `tranche`, `experiment_name`.
- `arms`: per arm, per model — **every §2.4 metric**, overall and **per question**:
  `gold_seen, gold_first_rank, answer_correct, citation_verbatim, citation_gold_overlap,
  fabrication, grounded_accuracy, abstain_recall, false_abstain, redacted_fabrication,
  required_evidence_met, tool_calls, cap_hit, execution_failed, skill_adherence,
  input_tokens, output_tokens, usd, wall_ms`.
- `skill_adherence_detail`: per arm, per rule id → `{n_applicable, n_satisfied, rate}`.
- `majority_baseline`: from `scorers.majority_baseline` over the subset.
- `config_diff`: for each adjacent arm pair, the single differing key and both values.
- `paired`: **per-case flips** for each adjacent pair (A→B, B→C, C→D) and for A→D, at each model:
  `{"gained": [case_ids], "lost": [case_ids], "unchanged_correct": n, "unchanged_wrong": n}` on
  `grounded_accuracy`. With N=32 this is the evidence — print the case ids, not just counts.
- `cost`: `estimate_vs_realised` per leg, plus `m4_ledger_total` from
  `spend.realized_by_tag()["m4"]`, plus the envelope and the target/absolute.
- `caching`: the §9.3 finding (measured static-prefix tokens, provider minimum, cached_tokens
  before/after, cost/case before/after, and the decision not to pad).
- `budget_scaled`: `true`, with the sentence explaining what was scaled and why.
- `arm_d_improves_over_c`: **computed boolean** (per model), plus the delta. See §7.1.
- `versions`: dataset/chunk/index/arm_c_index/skill versions + git sha.

`four_arm.md`: the same as human tables — per-arm overall table, per-question tables, the paired
flip lists, the majority baseline row alongside every accuracy, the adherence per-rule table with
applicable counts, estimate-vs-realised, and the caveats.

### 7.1 The honesty rule, mechanised

> "If arm D does not improve `grounded_accuracy` over C, the report says so and shows adherence,
> fabrication, abstention, trajectory and efficiency deltas instead; no wording may claim an
> improvement the numbers do not support (the reviewer checks the README sentences against the
> JSON)."

Do **not** hand-write the verdict sentence. The generator computes
`arm_d_improves_over_c` from the JSON and emits one of two pre-written sentence templates, filled
with the actual numbers. Same for A→B and B→C. A test feeds fake results in which D **regresses**
and asserts the rendered markdown contains no improvement claim and does contain the adherence /
fabrication / abstention / trajectory / efficiency delta table.

### 7.2 README

Create `README.md` with a `<!-- BEGIN RESULTS -->` / `<!-- END RESULTS -->` block that `just report`
rewrites from `four_arm.json`. **The first sentence of the Results section must state** (brief §6.8,
spec deliverable 6):

- the MAUD non-comparability caveat (this task is document→(answer, citation); MAUD supplies the
  span; scores are not comparable to MAUD leaderboard numbers),
- the objective-vs-judged distinction (everything here is deterministic Python over expert labels;
  no model judging in M4),
- the **budget-scaled subset** (32 frozen cases, N=18 at Haiku — not the brief's full-scale design).

Also include the dataset licence/attribution (MAUD v1, CC BY 4.0, Zenodo 7500064).

A `gate_m4` test parses the README's Results block and asserts its numbers equal `four_arm.json`
(spec DoD: "a test compares the numbers").

Add to `.gitignore` (the `data/reports/*` allowlist pattern):

```
# dealpoint (M4 four-arm experiment)
!data/reports/four_arm.json
!data/reports/four_arm.md
```

---

## 8. Offline gate (`gate_m4`)

`pytestmark = pytest.mark.gate_m4` (marker already registered in `pyproject.toml`). Suggested files:
`tests/test_skill.py`, `tests/test_skill_adherence.py`, `tests/test_arms.py`,
`tests/test_subset.py`, `tests/test_report_m4.py`.

Must cover, per the spec's DoD:
- each adherence rule's applicable/satisfied legs on synthetic trajectories;
- the arm-diff test (§4);
- the report generator on **fake** results (including the D-regresses case of §7.1);
- skill files exist, SKILL.md ≤ ~800 tokens, each playbook 3–5 lines, no gold/majority answers leak;
- `skill_version` is stamped in `versions.json` and matches a fresh `skill_version()`;
- subset regenerates byte-for-byte and satisfies its structural invariants;
- README numbers match `four_arm.json` (skip cleanly if the report has not been generated yet, in
  the style of `tests/conftest.py::require_dataset` — but it must **not** skip once `four_arm.json`
  exists).

`uv run pytest -m "gate_m4 and not needs_network" -q` must be green, and so must the full offline
suite, `uv run ruff check .`, `uv run pyright`.

---

## 9. Metered phase — preparation

### 9.1 Re-define the `four_arm` sweep to describe the actual run

`dealpoint/eval/spend.py::SWEEP_DEFS["four_arm"]` must model two legs. Extend the sweep schema with
an optional `legs` key and teach `estimate()` to sum them (keep the old single-leg shape working for
`judges`/`pareto`):

```python
"four_arm": {
    "legs": [
        {"arms": ["A", "B", "C", "D"], "model": "z-ai/glm-5.3-flash",      "n_cases": 32},
        {"arms": ["A", "D"],           "model": "anthropic/claude-haiku-4.5", "n_cases": 18},
    ],
},
```

Projected from the measured ledger shape (4952.1 in / 145.8 out per call, 4 calls/case):

| leg | case-runs | $/case | projected |
|---|---|---|---|
| GLM, arms A–D, 32 cases | 128 | $0.00163 | **$0.209** |
| Haiku, arms A+D, 18 cases | 36 | $0.02272 | **$0.818** |
| | | **total** | **≈ $1.03** |

Against realized $0.5454 → projected $1.57 of the $4.00 envelope; M4 realised target ≤ $1.50,
absolute ≤ $3.00 ✓; leaves M5 ($0.60) + M6 headroom ✓. And $1.03 < M4's `absolute_usd` $3.00, so
the factory spend gate passes.

**Why N=18 at Haiku:** the next whole-question prefix up is the full 32 (2 arms × 32 = $1.454),
which breaks the spec's $1.00 ceiling for that leg. 2 arms × 24 = $1.09, also over, and 24 is not a
whole-question tranche. 18 = tranche_1 = 1 case/question + 6 counterfactual is the largest that
fits. Record N and the projection in the report, as the spec requires.

Update the `judges` sweep's `n_cases` only if M5's plan needs it — out of scope here.

### 9.2 Fix the zero-cost estimate for unseen models

In `per_case_usd` branch 4: if `model_rows` is empty, fall back to the mean tokens/call over **all**
ledger rows and return basis `f"ledger:corpus-tokens x {price_basis}"`. Add a unit test with a
ledger fixture that has no rows for the queried model, asserting the estimate is > 0.

### 9.3 Step 0 — diagnose prompt caching (do this before any sweep)

**Measure first, and expect to record a negative result.** The static prefix
(system prompt + question spec + tool schemas) measures **≈ 613–730 tokens** across the 12
questions (system+question block 1,224–1,693 chars; tool schemas 1,230 chars). Anthropic's minimum
cacheable prefix for the Haiku class is **2,048 tokens**. Our prefix is roughly a third of that, so
it is very likely **below the provider's minimum cacheable length** — which is exactly the case the
spec anticipates:

> "If the static prefix … is below the provider's minimum cacheable length for that model, record
> that fact in the report and move on — **never pad a prompt to make it cacheable**."

Note also that the request shape is already correct: `dealpoint/llm/client.py::build_system_message`
sends the system message as a content-block list with `cache_control: {"type": "ephemeral"}`, which
is what OpenRouter's Anthropic caching path requires. There is no block-placement bug to fix.

Procedure:
1. Run the **2-call probe** on `anthropic/claude-haiku-4.5` (identical static prefix twice, e.g. one
   dev case run twice, or two calls in a tiny script). ~$0.05.
2. Read `cached_tokens` from the new ledger rows (`data/results/spend_ledger.jsonl` already carries
   `cached_tokens` and `cache_write_tokens`; all 96 existing rows are 0).
3. Record in `four_arm.json.caching`: measured static-prefix tokens, the provider minimum, observed
   `cached_tokens`, and measured **cost/case before and after**.
4. If it turns out the prefix *is* above the minimum and still uncached, fix block
   placement/order per OpenRouter's Anthropic caching docs and re-probe once. **Do not pad.**
5. If caching starts working, the spec permits raising the Haiku N to 32 within the same $1.00 —
   re-run the estimator and record the decision. (Do not count on this.)

### 9.4 Spend gate

`assert_within_cap(est)` must pass **before each sweep** — the runner already calls it for runs
> 10 cases (`dealpoint/eval/run.py:118-123`). Record every projection in the report. `.env` has
`MAX_OPENROUTER_SPEND_USD` set; do not edit it.

---

## 10. The sweeps

Order matters — cheapest and most informative first.

1. **Dev smoke (§0.2):** ≤ 6 `dev` cases, arms C and D, `z-ai/glm-5.3-flash`. Prove the wiring.
   ~$0.01. Not a reported result.
2. **Caching probe (§9.3):** 2 calls at Haiku. ~$0.05.
3. **Headline — GLM, arms A, B, C, D, full 32-case frozen subset.** ≈ $0.21.
   This is the constant-model four-arm comparison the DoD names.
4. **Replication — Haiku, arms A and D, `tranche_1` (N=18).** ≈ $0.82.
   Record N and the projection in the report.
5. **Arms B and C at Haiku only if headroom remains** under the M4 target after the above
   (it will not: 2 more arms × 18 ≈ $0.82 would take M4 to ~$1.85, over the $1.50 target though
   under the $3.00 absolute). **Default: do not run them**; record "not run (budget)" with the
   arithmetic. If you do run them, the spec requires the reason be recorded in the report.

After each sweep, publish the rows to Braintrust (one experiment per arm×model,
`{arm}-{model}-{index_version}-{git_sha7}`) and confirm names land in
`data/reports/braintrust_runs.json`. Score volume ≈ (128 + 36) × 6 ≈ 984, well inside the free tier.

Then `just report`, then regenerate the README block, then re-run the full gate.

---

## 11. Definition of done (mirror of the spec)

- [ ] `gate_m4` offline green: adherence rules unit-tested (applicable/satisfied on synthetic
      trajectories); arm-diff test; report generator tested on fake results (including the
      D-regresses honesty case); skill files exist and `skill_version` is stamped.
- [ ] `data/eval/test_subset_v1.json` frozen and regenerable byte-for-byte.
- [ ] `data/reports/four_arm.json` covers 4 arms at GLM on the full 32-case subset and arms A/D at
      Haiku on the recorded N (=18); per-question tables, paired flips, majority baseline and
      estimate-vs-realised cost present.
- [ ] README Results section regenerated by `just report`, consistent with the JSON (test compares
      numbers), carrying the MAUD non-comparability caveat, the objective-vs-judged distinction and
      the budget-scaled subset **in the first sentence**.
- [ ] Spend gate passed before sweeps; M4 ledger totals in the report. Realised: target ≤ $1.50,
      absolute ≤ $3.00, envelope $4.00 never exceeded.
- [ ] Full offline suite, `uv run ruff check .`, `uv run pyright` green.

---

## 12. Notes to carry into the report (not to resolve silently)

1. **Spec-internal inconsistency about which model is the headline.** Deliverable 5 (re-sized
   2026-09-04) and the Definition of done both say the **headline comparison is at
   `z-ai/glm-5.3-flash`** on the full subset, with **Haiku as the A/D replication**. The older
   "Cheap workhorse model" paragraph at the foot of the same spec says the reverse ("This does not
   change the constant model of the headline comparison (Haiku)"). The re-sized deliverable and the
   DoD are the later, more specific instruction and they agree with each other, so **follow them**:
   GLM headline, Haiku replication. Record the discrepancy in `four_arm.md` as a noted
   spec-internal inconsistency resolved by precedence. This is a spec-vs-spec conflict, not
   spec-vs-brief; the brief §2.2 only requires the arms share *a* constant model, which holds
   within each leg.
2. **Arm C's name.** Brief §2.2 calls it `agent-hybrid-rerank`; the frozen M3 winner is
   `hybrid_rrf` without rerank. Brief §2.2 defines the backend as "tournament winner", so the
   measurement governs and the label follows it (`agent-hybrid-rrf`). Restate the M3 rationale.
3. **Adherence is not like-for-like between arm A and arms B/C/D** (§3) — print applicable counts.
4. **Prompt caching** — record the measured finding, and explicitly record that the prompt was
   **not** padded to reach the cacheable minimum.
5. **Budget-scaled** — every headline number is from a 32-case (18 at Haiku) frozen subset, not the
   brief's full 167-case test set. Never present it as the full benchmark.
6. Brief §7 M4's relative gate ("no arm regresses vs its previously recorded dev result") has no
   prior four-arm dev record to compare against — M2's only recorded run is 3 dev cases at arm B.
   State that the relative gate is vacuous this milestone and why, rather than inventing a baseline.
