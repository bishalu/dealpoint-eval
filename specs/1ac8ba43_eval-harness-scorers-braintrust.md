# M2 — Eval harness: deterministic scorers, spend accounting, Braintrust adapter, smoke eval

**Spec:** `specs/milestones/m2.md` (read-only). **Requirements:** `specs/grilled-product-brief.md`
§2.1, §2.4, §2.7, §3 (read-only, wins on product decisions).
**Baseline at planning time:** `git d2b052d`, offline suite 130 passed / 1 deselected, `ruff` 0,
`pyright` 0. Do not regress any of these.

---

## 0. Read this first — the two facts that constrain every decision

### 0.1 The M2 money envelope is $0.12, not $0.50

`MAX_OPENROUTER_SPEND_USD=4` is set in `.env`. The milestone allocation guide is **M1+M2 ≤ $0.50**.
The ledger already holds M1's spend:

```
data/results/spend_ledger.jsonl : 68 rows, all milestone_tag="m1",
all model=anthropic/claude-haiku-4.5, total usd = $0.375386
=> remaining under the $0.50 M1+M2 allocation: $0.1246
```

Measured arm-B/Haiku cost per case, from M1's 2-case smoke runs (grouped by ledger minute):
**$0.021–$0.026 per case** (7–9 calls per 2-case run, $0.0369–$0.0518 per run).

Therefore **M2 may run the agent against OpenRouter exactly once, on 3 dev cases**
(≈ 3 × $0.026 = **$0.078**), bringing M1+M2 to ≈ **$0.453 ≤ $0.50**. That leaves ~$0.047 of slack.

Consequences you must design to, not discover later:

- **`just smoke --send` must NOT re-run the agent.** The spec's DoD says "the **same 3 results** are
  sent once". Re-running would cost a second $0.078 and break the $0.50 allocation. `--send` replays
  the **persisted result rows** through the Braintrust adapter with zero model calls.
- **Do not run `python -m dealpoint.eval.budget <sweep> --sample` during this milestone.** `--sample`
  runs 3 dev cases and would cost another ~$0.078. Build the flag, unit-test it with a fake client,
  never invoke it metered. The DoD only requires `just budget four_arm` (no `--sample`) to print JSON.
- **There must be exactly one metered code path**, so the pytest gate and the `just` recipe cannot
  both spend. See §6.
- Anything that fails and needs a retry costs a further $0.078. Get the offline half completely
  green — including a full `--fake` end-to-end run — **before** the first metered call.

### 0.2 Spec-vs-brief differences — report them, do not silently resolve

Record all three in your report under a "brief vs milestone spec" heading. Implement as stated here.

| # | Brief | `m2.md` | Resolution |
|---|---|---|---|
| 1 | §7 milestone table: smoke = **5 dev cases** | **3 dev cases** | Follow `m2.md` (3). Its "Budget scaling (engineer decision)" section explicitly authorises smaller sizes, and §0.1 above shows 5 cases would not fit. Record as **budget-scaled**. |
| 2 | §2.4 names `abstain_recall` (counterfactual) and `false_abstain` (dev/test) | names a single `abstain_correct` | Implement `abstain_correct` per the spec as the per-case scorer, **and** derive `abstain_recall` and `false_abstain` in the summary aggregate so the brief's metric names still exist. The brief wins on the metric vocabulary; the spec wins on the per-case field name. |
| 3 | §2.7 "Smoke gate does not log to Braintrust" | DoD requires the same 3 results be sent once so one experiment is visible | Not a contradiction — both hold. The smoke itself runs with `no_send_logs=True`; a **separate, explicit** `--send` invocation publishes the already-computed rows. Brief §6 acceptance and §7 gate both call for "one real dev run visible in Braintrust". |

---

## 1. What exists that you must build on (verified, do not re-derive)

**`dealpoint/eval/` does not exist yet.** Create it with `__init__.py`.

### Agent entry points — exact signatures

```python
# dealpoint/agent/loop.py      (arm B)
def run_agent(case: dict, doc: Document, retriever: Retriever, client,
              question, model: str, index_version: str | None = None
              ) -> tuple[Finding | None, ExecutionRecord]

# dealpoint/agent/pipeline.py  (arm A) — identical signature
def run_pipeline(case: dict, doc: Document, retriever: Retriever, client,
                 question, model: str, index_version: str | None = None
                 ) -> tuple[Finding | None, ExecutionRecord]
```

Both read only `case["case_id"]`. `run_pipeline` **always** calls `retriever.search(...)`;
`run_agent` only does so if the model asks for a tool.

### Schemas (`dealpoint/agent/schema.py`)

```python
Evidence(section_ref: str, quote: str)
Finding(answer: str, evidence: list[Evidence], rationale: str)   # evidence == [] iff answer == "ABSTAIN"
TrajectoryStep(tool: str, args: dict, result_ref: str|None,
               chunk_ids: list[str], char_ranges: list[tuple[int,int]], t_ms: int)
Usage(input_tokens, output_tokens, cost_usd: float, wall_ms, tool_calls,
      cached_tokens, cache_write_tokens)
ExecutionRecord(status: "ANSWERED"|"ABSTAINED"|"CAP_HIT"|"EXECUTION_FAILED",
                failure_reason, trajectory: list[TrajectoryStep], usage: Usage,
                wall_ms, model, arm, case_id, index_version, chunk_version)
```

### Corpus / data

- `dealpoint.corpus.document.load_document(document_id) -> Document`, with
  `Document(document_id, agreement_id, text, sections, body_start, body_end)`.
  `text` is the canonical string; **all offsets index into it**; it contains no newlines.
- `dealpoint.corpus.document.defined_term(doc, term) -> DefinedTerm | None`, with
  `DefinedTerm(term_as_written, section_ref, start, end, text)`.
- `dealpoint.corpus.chunks.chunk_document(doc) -> list[Chunk]` and `chunk_version() -> str`.
  `Chunk(agreement_id, chunk_id, section_ref, start, end, text)`.
- `dealpoint.corpus.retrievers.index_version() -> str` (pure, no I/O; currently `e2b4a2b97561`),
  `LazyRetriever()`, `DenseRetriever()`, `.search(agreement_id, query, k=5) -> list[Chunk]`.
- `dealpoint.data.canonical.normalise_quote(s) -> str` (alias of `canonicalise`) — **the only**
  normalisation permitted before a verbatim check.
- `dealpoint.data.questions.QuestionSpec(id, maud_question, text_type, category, gloss,
  options: tuple[str,...], canonical_query, reasoning_type, required_evidence: str | None)`,
  `QUESTION_BY_ID`, `OUT_OF_SCOPE_QUESTION_BY_ID`.

The five non-null `required_evidence` values, verbatim:

```
q05: 'defined term "Knowledge" (or "Knowledge of the Company")'
q06: 'defined term Material Adverse Effect (or Company MAE)'
q07: 'defined term Superior Proposal'
q08: 'defined term Intervening Event'
q11: 'defined term Material Adverse Effect'
```

### Case row schemas (`data/eval/*.jsonl`)

`dev.jsonl` (58 rows, 5 agreements) and `test.jsonl` (167 rows) share:
`agreement_id, case_id, case_set, gold_answer, gold_spans:[{start,end}], majority_answer,
question_id, required_evidence, span_type`.

`counterfactual.jsonl` (40 rows = 30 `redacted` + 10 `out_of_scope`) adds `kind`, plus
`source_case_id` + `redacted_ranges` for redacted, or `question_text` for out-of-scope
(which has `majority_answer: null`, `span_type: null`, `gold_answer: "ABSTAIN"`).

### Document-id and question resolution — promote, do not duplicate

`dealpoint/agent/run.py` already has `_find_case`, `_resolve_document_id` and `_resolve_question`.
**Move all three into a new `dealpoint/eval/cases.py`** as public `find_case(case_id)`,
`resolve_document_id(case)`, `resolve_question(case)`, plus a new `load_case_set(name)`, and have
`dealpoint/agent/run.py` import them. Do not fork the logic — a redacted case must be run against
its own redacted document (`case_id` is the document id) in both the CLI and the runner.

### LLM client (`dealpoint/llm/client.py`)

`OpenRouterClient(api_key=None, base_url=..., milestone_tag=MILESTONE_TAG, ledger_path=...)`.
Every call appends one ledger row:
`{ts, milestone_tag, model, calls:1, usd, input_tokens, output_tokens, cached_tokens,
cache_write_tokens}` (+ `usd_missing: true` when OpenRouter omitted cost).
`FakeClient(script=[ScriptedTurn(...)])` never touches network or ledger.

### Environment

- `openai==3.8.0` and `braintrust` are installed. `braintrust.Eval` accepts
  `name, data, task, scores, experiment_name, metadata, no_send_logs, project_id, ...`.
  `braintrust.traced`, `braintrust.wrap_openai`, `braintrust.init_dataset` all exist.
- `BRAINTRUST_API_KEY` lives in `.braintrust.json` and `.env.braintrust` — **both gitignored, and
  neither is loaded by `just`'s `set dotenv-load` (which only reads `.env`)**. The adapter must load
  it explicitly (see §5). Never print, log or echo the value.
- `api.braintrust.dev` and `openrouter.ai/api/v1/models` are both reachable from this VM.
- OpenRouter pricing for `anthropic/claude-haiku-4.5` today: prompt `$1.00/M`, completion `$5.00/M`.
  **Verified: `input_tokens*1e-6 + output_tokens*5e-6` reproduces `usd` exactly on all 68 ledger
  rows.** Use this to sanity-check your pinned table.

---

## 2. `dealpoint/eval/scorers.py` — pure functions, no I/O

Every scorer is a pure function. No file reads, no network, no globals. Signature convention:

```python
def scorer(case: dict, finding: Finding | None, record: ExecutionRecord,
           canonical_text: str) -> bool | int | float | None
```

`None` means **not applicable to this case** and must be excluded from aggregates — it is never
silently coerced to `0`/`False`. This is the single most important convention in the module; an
`answer_correct` of `None` on a `CAP_HIT` case must not drag the mean down as if it were a wrong
answer.

### 2.1 Geometry helpers (private, but unit-tested through the scorers)

- `MIN_GOLD_OVERLAP_CHARS = 50` → put in `dealpoint/config.py`.
- `overlap_chars(a: tuple[int,int], b: tuple[int,int]) -> int` — `max(0, min(a1,b1) - max(a0,b0))`.
- `gold_ranges(case) -> list[tuple[int,int]]` from `case["gold_spans"]`.
- `locate_quote(quote: str, canonical_text: str) -> list[tuple[int,int]]` — normalise the quote with
  `normalise_quote`, then return **every** occurrence span in `canonical_text` (`str.find` loop).
  Empty list ⇒ the quote is not verbatim. Returning all occurrences matters: a short quote can appear
  more than once, and gold overlap must be judged on the best-overlapping occurrence, deterministically.

### 2.2 The scorers

| Name | Returns | Definition |
|---|---|---|
| `gold_seen` | `bool \| None` | `True` if **any** `char_range` on **any** trajectory step overlaps **any** gold span by ≥ 50 chars. `None` when the case has no gold spans (all counterfactual cases). |
| `gold_first_rank` | `int \| None` | 1-based index of the first gold-overlapping (≥ 50 chars) entry in the `char_ranges` of the **first `search_agreement` step** in the trajectory. `None` if there is no search step, no gold spans, or no hit. |
| `answer_correct` | `bool \| None` | `finding.answer == case["gold_answer"]`, exact string. `None` unless `record.status == "ANSWERED"`. |
| `citation_verbatim` | `bool \| None` | `True` iff **every** quote in `finding.evidence` locates in the canonical text. `None` when there is no finding or `evidence == []`. |
| `citation_gold_overlap` | `bool \| None` | `True` iff ≥ 1 quote has **some** occurrence overlapping **some** gold span by ≥ 50 chars. `None` when the case has no gold spans or there is no finding. |
| `fabrication` | `bool \| None` | `True` iff any quote fails to locate. `None` when there is no finding or `evidence == []`. (Mirror of `citation_verbatim`.) |
| `grounded_accuracy` | `bool \| None` | `answer_correct AND citation_gold_overlap`. `None` if **either** input is `None`. **This is the headline metric.** |
| `abstain_correct` | `bool` | If `case["case_set"] == "counterfactual"`: `record.status == "ABSTAINED"`. Otherwise: `record.status != "ABSTAINED"`. Always a `bool` — applicable to every case. |
| `redacted_fabrication` | `bool \| None` | `fabrication`, but `None` unless `case.get("kind") == "redacted"`. |
| `required_evidence_met` | `bool \| None` | See §2.3. `None` when `case["required_evidence"]` is `None`. |
| `tool_calls` | `int` | `record.usage.tool_calls`. |
| `cap_hit` | `bool` | `record.status == "CAP_HIT"`. |
| `execution_failed` | `bool` | `record.status == "EXECUTION_FAILED"`. |
| `input_tokens` / `output_tokens` / `usd` / `wall_ms` | `int`/`float` | From `record.usage` (`usd` ← `usage.cost_usd`) and `record.wall_ms`. |

`skill_adherence` is **M4**. Provide `def skill_adherence(...) -> None: return None` with a docstring
saying so, because the Braintrust score list must carry all six names from M2 onward (§5).

### 2.3 `required_evidence_met` — deterministic, no prose matching at score time

Add `parse_required_evidence(spec: str | None) -> tuple[str, ...]` that extracts the candidate term
strings from the five prose values in §1. Rule: strip a leading `defined term `, strip surrounding
quotes, split on `(or ` / `)`, strip each part, drop empties. Assert in a unit test that the five
real values parse to exactly:

```
q05 -> ("Knowledge", "Knowledge of the Company")
q06 -> ("Material Adverse Effect", "Company MAE")
q07 -> ("Superior Proposal",)
q08 -> ("Intervening Event",)
q11 -> ("Material Adverse Effect",)
```

`required_evidence_met` is `True` if **either**:

1. **direct** — some trajectory step has `tool == "lookup_defined_term"` whose `args["term"]`,
   casefolded and stripped, equals a candidate (or a candidate is a casefolded substring of it); or
2. **by coverage** — for some candidate, `defined_term(doc, candidate)` resolves to a block, and some
   trajectory step's `char_ranges` overlaps that block by ≥ 50 chars.

Rule 2 is what makes the check honest: the brief (§2.4) says "retrieved the required block **by any
tool**", so a `search_agreement` that happens to return the definition counts. Because rule 2 needs
the `Document`, `required_evidence_met` takes an extra `doc: Document | None = None` parameter and
falls back to rule 1 alone when `doc is None`.

### 2.4 The aggregate

```python
def score_case(case, finding, record, doc) -> dict   # every field above, by name
def majority_baseline(cases: list[dict]) -> dict     # per question + overall
def summarise(rows: list[dict]) -> dict              # means over non-None, plus derived rates
```

- `score_case` returns a flat dict with **every** score name as a key, `None` included. The DoD
  requires "result rows with every score field" — a missing key is a failure, a `None` value is not.
- `majority_baseline`: for each `question_id`, predicting `case["majority_answer"]`, the fraction of
  cases where it equals `gold_answer`; plus an `overall`. Skip cases with `majority_answer is None`
  (the 10 out-of-scope cases). **DoD: this must match `data/eval` labels** — the test computes it
  straight from `dev.jsonl`+`test.jsonl` and asserts it equals an independent recount in the test.
- `summarise` computes, for every boolean/numeric field, `{mean, n}` over **non-`None`** values only,
  and additionally derives the brief's §2.4 names:
  - `abstain_recall` = mean of `status == "ABSTAINED"` over `case_set == "counterfactual"` rows,
  - `false_abstain` = mean of `status == "ABSTAINED"` over dev/test rows.

### 2.5 The three synthetic traps (DoD requires all three)

In `tests/test_scorers.py`, build cases and records by hand (no disk, no model):

1. **Fabricated quote** — a dev case, `ANSWERED`, correct answer, one quote that is *not* a substring
   of the canonical text. Assert `citation_verbatim is False`, `fabrication is True`,
   `citation_gold_overlap is False`, `grounded_accuracy is False`, `answer_correct is True`.
   *A correct answer with an invented citation is not grounded.*
2. **Redacted case answered with an invented quote** — `kind == "redacted"`, `case_set ==
   "counterfactual"`, `ANSWERED` with a quote absent from the redacted canonical text. Assert
   `redacted_fabrication is True`, `abstain_correct is False`. Also assert that the *same* record on
   a non-redacted case gives `redacted_fabrication is None`.
3. **Correct "No" is not an abstention** — a dev case with `gold_answer == "No"`, status `ANSWERED`,
   `answer == "No"`, one real verbatim quote overlapping gold. Assert `answer_correct is True`,
   `abstain_correct is True`, `grounded_accuracy is True`, and that `summarise` counts it in
   `false_abstain`'s denominator with value `False`. Brief §2.4: *"Correct **No** ≠ abstention."*

Also unit-test: `gold_first_rank` ordering; a 49-char overlap scoring `False` and 50 scoring `True`
(the boundary, both directions); `answer_correct is None` on `CAP_HIT` and on `EXECUTION_FAILED`;
`abstain_correct is True` for an `ABSTAINED` counterfactual and `False` for an `ABSTAINED` dev case;
a quote occurring twice where only the second occurrence overlaps gold ⇒ `citation_gold_overlap` is
`True`; `required_evidence_met` satisfied by rule 1 and, separately, by rule 2 on a synthetic doc.

---

## 3. `dealpoint/eval/spend.py` — ledger reading, estimation, the cap

### 3.1 Ledger

```python
LEDGER_PATH = SPEND_LEDGER_PATH                      # dealpoint.config
def read_ledger(path=LEDGER_PATH) -> list[dict]      # tolerant: skip blank/corrupt lines
def realized_usd(path=LEDGER_PATH) -> float          # sum of "usd", rounded to 6
def realized_by_tag(path=LEDGER_PATH) -> dict[str, float]
```

`realized_usd` must agree with `adws/adw_modules/spend.py::realized_usd` to the cent — the factory's
guard and the product's estimator must never disagree about what has been spent.

### 3.2 Attributing cost to (arm, model) — a gap you must close

M1's ledger rows carry **no arm and no case id**, so "measured mean cost/case per (arm, model)"
cannot be computed from them. Close it going forward, without rewriting history:

Give `OpenRouterClient` a mutable `context: dict[str, str]` attribute (default `{}`), merged into
every ledger row it writes. The runner sets `client.context = {"arm": ..., "case_id": ...,
"case_set": ...}` before each case. Old rows simply lack the keys; `read_ledger` must tolerate that.
Also let the runner pass `milestone_tag="m2"` through the existing constructor argument, so M2's
spend is separable from M1's in the ledger.

### 3.3 Pricing

```python
PINNED_PRICES_DATE = "2026-09-04"
PINNED_PRICES = {                       # USD per token
    "anthropic/claude-haiku-4.5": {"prompt": 1e-6, "completion": 5e-6},
    # add the M6 slate ids as they are known; unknown model -> KeyError surfaced by estimate()
}
def fetch_prices(timeout=20) -> tuple[dict, str]   # ("live"|"pinned") from
                                                   # https://openrouter.ai/api/v1/models
```

`fetch_prices` returns the pinned table with basis `"pinned:<date>"` on **any** failure (no network,
timeout, missing model, unparseable payload). It never raises. Unit-test the fallback by monkey-
patching the fetch to raise; mark any test that really hits OpenRouter `needs_network`.

### 3.4 `estimate`

```python
def estimate(sweep_name: str, ledger_path=LEDGER_PATH) -> dict
# -> {"sweep": str, "calls": int, "est_usd": float, "per_call_usd": float,
#     "basis": str, "cases": int, "per_case_usd": float, "model": str, "arms": [...]}
```

`adws/adw_modules/spend.py::estimate` requires `est_usd` and `calls` to be present — **both keys are
mandatory and `est_usd` must be a plain float**, or the M4/M5/M6 spend gate breaks.

Sweep definitions (a module-level table, so M4–M6 change one dict, not the code):

| sweep | shape | cases | calls |
|---|---|---|---|
| `four_arm` | arms A–D × the M4 frozen subset (32 cases) @ `anthropic/claude-haiku-4.5` | 128 | 128 × measured calls/case |
| `judges` | 24 traces × 3 judges, one call per trace | 24 | 72 |
| `pareto` | arm D × 32 cases × 3 models | 96 | 96 × measured calls/case |

Cost basis, in strict priority order — the first that yields data wins, and the chosen one is
reported verbatim in `basis`:

1. `"ledger:measured(arm,model)"` — ledger rows carrying both `arm` and `case_id`: group by
   `(arm, model)`, mean `usd` per distinct `case_id`.
2. `"ledger:measured(model)"` — ledger rows for the model with a `case_id` but no `arm`.
3. `"results:measured"` — mean `usd` over rows in `data/results/*.jsonl` for that (arm, model).
4. `"ledger:tokens x live"` / `"ledger:tokens x pinned:<date>"` — measured mean input/output tokens
   per call from the ledger × the price table. **This is the branch that will fire for `just budget
   four_arm` in M2** (M1 rows have no `case_id`), and measured calls/case comes from M1's smoke:
   68 calls over the runs recorded ⇒ use `EST_CALLS_PER_CASE = 4` as a config constant, documented
   as measured from M1, not guessed.

Mean tokens/call in the current ledger: **4792 in / 146 out** ⇒ ≈ $0.00552/call, ≈ $0.0221/case at 4
calls/case. `estimate("four_arm")` should therefore land near **$2.83** for 512 calls — comfortably
over the $1.50 M4 allocation, which is a **true and useful signal for M4's planner**, not a bug to
tune away. Do not fudge the number; M4 will shrink its subset in response, exactly as its spec says.

`--sample` (`budget` CLI only): run 3 dev cases first, then add `sample_est_usd` and
`sample_realized_usd` (ledger delta across the sample). **Do not execute this metered in M2** (§0.1);
test it with `FakeClient`.

### 3.5 `assert_within_cap`

```python
class SpendCapError(RuntimeError): ...
def cap_usd() -> float | None     # os.environ, then dotenv_values(".env") — mirror adws/spend.py
def assert_within_cap(est_usd: float, ledger_path=LEDGER_PATH) -> None
```

Raises `SpendCapError` when the cap is **unset or unparseable** (message must name
`MAX_OPENROUTER_SPEND_USD`), and when `realized + est > cap` (message must carry realized, est and
cap). Returns `None` silently otherwise. The `.env` fallback matters: `just` loads `.env`, a bare
`uv run pytest` does not.

Gate tests must not depend on the operator's real `.env`. Use `monkeypatch.setenv` /
`monkeypatch.delenv` **and** point `cap_usd`'s dotenv read at a `tmp_path` file, or inject the cap
via an argument the test can control. A test that passes only because `.env` happens to say `4` is
not a test.

### 3.6 `dealpoint/eval/budget.py`

`python -m dealpoint.eval.budget <four_arm|judges|pareto> [--sample]` → prints
`json.dumps(estimate(...), indent=2, sort_keys=True)` to stdout, exit 0. Nothing else on stdout —
`adws/adw_modules/spend.py::estimate` slices from the first `{` to the last `}`, so any stray
diagnostic must go to **stderr**. Unknown sweep ⇒ exit 2 with the valid names on stderr.

---

## 4. `dealpoint/eval/run.py` — the runner

```
python -m dealpoint.eval.run --set dev|test|counterfactual --arm A|B --model <id>
                             [--limit N] [--cases id,id] [--fake] [--out-dir DIR]
```

### 4.1 Behaviour

1. Load the case set via `dealpoint.eval.cases.load_case_set`. Apply `--cases` (explicit ids, order
   preserved as given) then `--limit N` (first N after a deterministic sort by `case_id`).
2. **If the selected case count > 10 and not `--fake`: call `assert_within_cap(estimate_for_run)`
   before the first model call.** Compute the run's own estimate from measured cost/case × n_cases
   via `spend.estimate`-style logic (factor the basis resolution into a reusable
   `spend.per_case_usd(arm, model) -> tuple[float, str]`). ≤ 10 cases skips the check, so the 3-case
   smoke is not blocked.
3. Build the client: `--fake` ⇒ `FakeClient` with a scripted valid finding; otherwise
   `OpenRouterClient(milestone_tag="m2")`.
4. Build the retriever: `--fake` ⇒ `OfflineChunkRetriever` (below); otherwise `LazyRetriever()`.
5. Per case: `resolve_document_id` → `load_document` → `resolve_question` → set `client.context` →
   `run_agent` (arm B) or `run_pipeline` (arm A) → `score_case(case, finding, record, doc)`.
   **Wrap each case in `try/except Exception`**: a crash becomes a synthesised `EXECUTION_FAILED`
   record with `failure_reason="tool_error"` and the run continues. One bad case must not throw away
   a metered sweep.
6. Write results and summary; print the summary to stdout.

**`OfflineChunkRetriever`** — needed because arm A always retrieves, so `--fake` would otherwise
require the Qdrant index. Implement in `dealpoint/eval/run.py` (or `cases.py`): `search(agreement_id,
query, k)` loads the document, calls `chunk_document(doc)`, and returns the first `k` chunks.
Deterministic, offline, no index, no embeddings.

### 4.2 Output

Result path — note the model id contains `/`, which **must** be slugified (`/` → `_`):

```
data/results/{set}_{arm}_{model_slug}_{index_version}_{git_sha7}.jsonl
e.g. data/results/dev_B_anthropic_claude-haiku-4.5_e2b4a2b97561_d2b052d.jsonl
summary: data/results/{same stem}_summary.json
```

`git_sha7` from `git rev-parse --short=7 HEAD`; on failure (no git) use `"nogit"`. Put a
`git_sha7()` helper in `dealpoint/eval/cases.py` or a small `dealpoint/eval/_version.py` — do **not**
import from `adws/`, which is factory code and excluded from lint.

One JSONL row per case, written with `json.dumps(row, sort_keys=True, ensure_ascii=False)`:

```json
{"case_id": ..., "case_set": ..., "arm": ..., "model": ..., "question_id": ...,
 "index_version": ..., "chunk_version": ..., "git_sha7": ...,
 "finding": {...} | null, "record": {...}, "scores": {...every score field...},
 "usd": <realised, from record.usage.cost_usd>}
```

`usd` is duplicated at the top level on purpose — spec deliverable 6: "every result row carries
realised `usd`", and `estimate()`'s `results:measured` basis reads it there.

Summary JSON: `{set, arm, model, index_version, chunk_version, git_sha7, n_cases, started_at,
ended_at, scores: summarise(rows), majority_baseline: {...}, realized_usd, est_usd, est_basis,
status_counts: {ANSWERED: n, ...}}`. `est_usd` vs `realized_usd` side by side is spec deliverable 6.

`data/results/` is **not** gitignored (only `data/reports/*` is), so result files are committed —
which matches brief §3.5.

---

## 5. `dealpoint/eval/braintrust_adapter.py`

**Import `braintrust` lazily inside functions**, never at module top level, so `pyright` and the
offline suite stay clean when it is absent. Same rule the existing code applies to `openai`/`qdrant`.

```python
def experiment_name(arm, model, index_version, git_sha7) -> str
    # f"{arm}-{model_slug}-{index_version}-{git_sha7}"
def experiment_metadata(arm, model, index_version, skill_version, git_sha, case_set) -> dict
def braintrust_available() -> bool           # key loadable AND import succeeds
def load_braintrust_key() -> str | None      # os.environ, then .env.braintrust, then .braintrust.json
def run_eval(rows, *, arm, model, case_set, no_send_logs: bool = True, project="dealpoint-eval")
def push_datasets(sets=("dev","test","counterfactual"), version="v1")
```

- **Project** `dealpoint-eval`. **Experiment name** `{arm}-{model}-{index_version}-{git_sha7}`
  (slugify the model's `/`). **Metadata** exactly `{arm, model, index_version, skill_version,
  git_sha, case_set}` — `skill_version` is `None` until M4; read it from
  `data/reports/versions.json` when present.
- **Exactly six scores per case**, no more — the free tier meters scores at 10k/month
  (brief §2.7): `grounded_accuracy`, `answer_correct`, `citation_gold_overlap`,
  `citation_verbatim`, `abstain_correct`, `skill_adherence` (returns `None` until M4).
  **Everything else — `gold_seen`, `gold_first_rank`, `fabrication`, `redacted_fabrication`,
  `required_evidence_met`, `tool_calls`, `cap_hit`, `execution_failed`, tokens, usd, wall_ms — goes
  in per-case metadata, never in `scores`.** Add a unit test asserting `len(SCORE_NAMES) == 6` and
  that the list equals the six names above; a seventh score is a budget defect.
- `run_eval` takes **already-computed result rows** and replays them: the `task` returns the stored
  finding, the scorers read the stored `scores` dict. This is what makes `just smoke --send` free
  (§0.1). Do not make the Braintrust path re-invoke the agent.
- `no_send_logs=True` by default. Only `--send` passes `False`.
- **`wrap_openai` and `@traced`** (brief §2.7, spec deliverable 3): apply `wrap_openai` to the
  `openai.OpenAI` instance inside `OpenRouterClient` when braintrust is importable and a key is
  present, else leave the client untouched; and decorate the three tool functions in
  `dealpoint/agent/tools.py` with `@traced` behind a no-op fallback decorator when braintrust is
  absent. Neither may change behaviour offline — assert that with a test that the tools still return
  identical `ToolResult`s with braintrust unimportable.
- **After a successful send**, append to `data/reports/braintrust_runs.json`:
  `{experiment_name, project, case_set, arm, model, index_version, git_sha, n_cases, ts, url}`
  (create the file as a list if absent). **`data/reports/*` is gitignored — add
  `!data/reports/braintrust_runs.json` to `.gitignore`**, or the DoD's "name recorded in
  `data/reports/braintrust_runs.json`" leaves nothing in the repo.
- `push_datasets` uses `braintrust.init_dataset(project="dealpoint-eval",
  name=f"maud-dealpoint-{set}-v1")` and inserts one record per case. Expose it as
  `python -m dealpoint.eval.braintrust_adapter --push-datasets`. Network, not metered — its test is
  `needs_network`, and it must be idempotent enough to re-run.

---

## 6. Recipes (`justfile`) — one metered path only

`justfile` already has `set positional-arguments`, so `*ARGS` forwards with `"$@"`.

```make
# score one arm x model over a case set: just eval B anthropic/claude-haiku-4.5 dev --limit 5
eval ARM MODEL SET *ARGS:
    uv run python -m dealpoint.eval.run --arm {{ARM}} --model {{MODEL}} --set {{SET}} "$@"

# 3 dev cases, Haiku, arm B, offline scoring; --send publishes those same rows to Braintrust
smoke *ARGS:
    uv run python -m dealpoint.eval.smoke "$@"

# print the JSON cost estimate for a sweep: just budget four_arm
budget SWEEP:
    uv run python -m dealpoint.eval.budget {{SWEEP}}

# milestone 2 acceptance gates only, offline (excludes the metered smoke)
gate-m2:
    uv run pytest -m "gate_m2 and not needs_network and not needs_model" -q
```

### `dealpoint/eval/smoke.py` — the single metered entry point

This module is the **only** thing in M2 that may call OpenRouter, and it must be impossible to spend
twice by accident:

- `just smoke` (no flag): if the canonical smoke result file already exists **and** was produced at
  the current `git_sha7` + `index_version`, print it and exit 0 **without any model call**. Otherwise
  run 3 dev cases (arm B, `anthropic/claude-haiku-4.5`) through `dealpoint.eval.run`, score offline,
  write results, and additionally invoke the Braintrust adapter with `no_send_logs=True` (so the
  adapter path is genuinely exercised, per the DoD, at zero score cost).
- `just smoke --send`: **loads the persisted rows and sends only.** It must refuse with a non-zero
  exit if the result file is missing — that refusal is what prevents a second metered run.
- Assert, in `smoke.py` itself: ledger delta > 0 for a fresh run, ledger delta ≤ `M2_SMOKE_MAX_USD`,
  and total ledger ≤ `M1_M2_MAX_USD`.
- Add to `dealpoint/config.py`:
  ```python
  M2_SMOKE_CASE_IDS = ("contract_0__q01", "contract_0__q06", "contract_0__q12")
  M2_SMOKE_MAX_USD = 0.10          # 3 cases at the measured ~$0.026/case, with slack
  M1_M2_MAX_USD = 0.50             # the milestone allocation, asserted from the ledger
  MIN_GOLD_OVERLAP_CHARS = 50
  EST_CALLS_PER_CASE = 4           # measured from M1's ledger
  EVAL_MILESTONE_TAG = "m2"
  ```
  All three case ids are present in `dev.jsonl` (verified). They give a direct question (`q01`,
  gold `All Cash`), a defined-term question with non-null `required_evidence` (`q06`, gold
  `"Would" (reasonably) be expected to`), and a **gold-`No`** question (`q12`) — which exercises the
  "correct No is not an abstention" path on a real metered run, not only in a unit test.

---

## 7. Tests

New files: `tests/test_scorers.py`, `tests/test_eval_run.py`, `tests/test_spend.py`,
`tests/test_braintrust_adapter.py`, `tests/test_smoke_m2.py`.

Marker discipline, following `tests/test_smoke_m1.py`: the metered test carries
`@pytest.mark.gate_m2` and `@pytest.mark.needs_model` and **never** `needs_network` (the factory gate
runs `-m "gate_m2 and not needs_network"`, so adding it would silently skip the gate forever).
Every marker is already registered in `pyproject.toml`.

### Offline (`gate_m2`, no network, no model) — must all pass before any metered call

- **`test_scorers.py`** — every metric, the three synthetic traps of §2.5, the 50-char boundary,
  `None`-vs-`False` semantics, `parse_required_evidence` on the five real spec strings, and
  `majority_baseline` matching a recount straight from `data/eval/dev.jsonl` + `test.jsonl`.
- **`test_eval_run.py`** — `--fake --set dev --limit 3` for **both arm A and arm B**; assert the
  JSONL exists at the exact templated path, has 3 rows, every row carries **every** score key
  (compare against the full expected key set, so a renamed scorer fails loudly), `usd` is present at
  top level, and the summary JSON has `est_usd`, `realized_usd`, `majority_baseline`. Assert no
  ledger row was written (FakeClient must not touch it). Use `--out-dir tmp_path` so the test never
  writes into `data/results/`.
- **`test_spend.py`** — `assert_within_cap` raises `SpendCapError` on unset cap (both env and dotenv
  absent) and on overrun; passes under the cap; `estimate("four_arm"|"judges"|"pareto")` returns a
  dict with `calls`/`est_usd`/`per_call_usd`/`basis` and `est_usd > 0`; the JSON printed by
  `python -m dealpoint.eval.budget four_arm` parses and satisfies
  `adws/adw_modules/spend.py`'s contract (`est_usd` and `calls` present); `fetch_prices` falls back
  to the pinned table when the fetch raises, and the pinned Haiku prices reproduce a real ledger
  row's `usd` to within a cent.
- **`test_braintrust_adapter.py`** — `SCORE_NAMES` has exactly 6 entries and equals the spec's list;
  `experiment_name` renders `B-anthropic_claude-haiku-4.5-e2b4a2b97561-d2b052d`-shaped strings;
  `experiment_metadata` has exactly the six required keys; the module imports and these functions
  work **with `braintrust` unimportable** (monkeypatch `sys.modules`); the `@traced` fallback leaves
  tool output unchanged.

### Metered (`gate_m2 and needs_model`) — `tests/test_smoke_m2.py`

Skip cleanly (`pytest.skip`, not error) when `OPENROUTER_API_KEY` is unset, the dataset is absent,
or the dense index is missing. Then: run the 3 smoke cases via `dealpoint.eval.smoke`, and assert
**0 `EXECUTION_FAILED`**, every row has a full `scores` dict, ledger delta > 0 and ≤
`M2_SMOKE_MAX_USD`, and **total ledger ≤ `M1_M2_MAX_USD` ($0.50)** — the DoD's "M1+M2 realised total
≤ $0.50 (asserted from the ledger)". Reuse the persisted result file if it is already current, so
re-running the gate does not re-spend.

The Braintrust send is **not** a pytest test — it is the one-off `just smoke --send`, run by hand
once, leaving its record in `data/reports/braintrust_runs.json`.

---

## 8. Order of work

1. `dealpoint/eval/__init__.py`, `cases.py` (promote the three helpers out of `agent/run.py`; keep
   `agent/run.py` working and its tests green).
2. `scorers.py` + `tests/test_scorers.py`. Get this fully green first — it is pure and free.
3. `spend.py`, `budget.py` + `tests/test_spend.py`. Add the config constants and the
   `OpenRouterClient.context` attribute.
4. `run.py` + `OfflineChunkRetriever` + `tests/test_eval_run.py`. Prove `--fake` end to end for
   both arms.
5. `braintrust_adapter.py` + `tests/test_braintrust_adapter.py`; `wrap_openai` and `@traced` wiring
   behind availability checks; `.gitignore` negation for `braintrust_runs.json`.
6. `smoke.py`, the four `justfile` recipes, `tests/test_smoke_m2.py`.
7. **Full offline gate green + `ruff` + `pyright` green.** Only then:
8. `just smoke` (≈ $0.078, the milestone's only metered run), then `just smoke --send` (free), then
   `just budget four_arm` to confirm valid JSON. Optionally `push_datasets` (network, unmetered).

---

## 9. Definition of done — verify each, do not assume

- [ ] `uv run pytest -m "gate_m2 and not needs_network" -q` green: scorer units incl. the three
      traps; `--fake` runner rows carry every score field; majority baseline matches `data/eval`;
      `assert_within_cap` raises on unset cap and on overrun.
- [ ] `uv run pytest -m "gate_m2 and needs_model" -q` green: 3 dev cases, Haiku, arm B, 0
      `EXECUTION_FAILED`, scores computed, `no_send_logs=True`.
- [ ] `just smoke --send` run **once**; one experiment visible in Braintrust project
      `dealpoint-eval` with the six scores and the six metadata keys; its name in
      `data/reports/braintrust_runs.json` (and that file not gitignored).
- [ ] No other metered run in M2. Ledger rows exist for every metered call, tagged `m2`.
      **Total ledger ≤ $0.50.**
- [ ] `just budget four_arm` prints valid JSON containing `calls` and `est_usd`.
- [ ] `uv run pytest -m "not needs_network and not needs_model" -q` green (was 130 passed),
      `uv run ruff check .` exit 0, `uv run pyright` exit 0.
- [ ] Report records: the milestone as **budget-scaled**; the realised M2 spend against the $0.12
      remaining envelope; the `estimate("four_arm")` figure and its `basis` as an input to M4; and
      the three brief-vs-spec differences from §0.2.

## 10. Out of scope

Tournament (M3), skill and `skill_adherence` logic (M4), arms C/D, judges (M5), Pareto (M6), the
product UI, the MCP layer. Do not edit `specs/grilled-product-brief.md` or `specs/milestones/m2.md`.
