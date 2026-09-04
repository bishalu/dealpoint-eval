# Deal-Point Eval — Implementation Plan

**Source brief:** `specs/grilled-product-brief.md` (committed at 9029ecf). This plan does not
replace it; it corrects it where recon proved it wrong and turns Milestone 0 into something a
builder can execute without deciding anything.

**Scope of THIS build request: Milestone 0 (data foundation) + repo scaffolding.**
Milestones 1–8 are specified at interface level only, so that M0's data contracts are right.
Each later milestone gets its own `just sdlc` request against the brief. Do not build M1+ now.

---

## 0. Recon results — corrections to the brief

Every number below was measured on this VM on 2026-09-03 against the real dataset and the live
OpenRouter API. Where the brief disagrees, **this section wins**.

### 0.1 Confirmed as written
| Brief claim | Verdict |
|---|---|
| Zenodo record 7500064, `maud_v1.zip`, 29,153,514 B, md5 `aad11e3f68cd4f3e0067a15b923a19ce` | ✅ exact |
| `data/contracts/contract_{0..151}.txt` | ✅ 152 files, contiguous 0–151 |
| All 12 Appendix-A `question` strings | ✅ all 12 exist **verbatim**, including the *two* spaces in `Fiduciary exception:  Board determination standard-Answer (no-shop)` |
| All 12 `text_type` / `category` values | ✅ exact as tabled |
| 92 distinct questions, 144 (question, subquestion) pairs, 22 text_types, 7 categories | ✅ exact |
| Majority ≤ 0.72 for all 12 | ✅ measured: q01 .586, q02 .580, q03 .507, q04 .526, q05 .553, q06 .625, q07 .476, q08 .516, q09 .592, q10 .423, q11 .592, q12 .559 |
| Qdrant local mode, native hybrid | ✅ dense + sparse prefetch + `models.Fusion.RRF` verified working on-disk, no server |
| OpenRouter native tool calling | ✅ live call to `anthropic/claude-haiku-4.5` returned a well-formed `tool_calls` |

### 0.2 WRONG — must be handled differently

**C1. The three CSVs partition *items*, not contracts. You must read all three.**
`MAUD_train.csv` (25,827 rows), `MAUD_dev.csv` (6,753), `MAUD_test.csv` (6,651). All 152
contracts appear in **all three** files. Union of `data_type == "main"` = 20,623 rows, and
`(contract_name, question, subquestion)` is unique across the union (20,623 keys, every count
exactly 1). Reading only `MAUD_train.csv` silently loses ~35% of gold labels.
→ `labels.py` loads and concatenates all three. MAUD's own split is irrelevant to us; our split
is by agreement (§3.6).

**C2. Party names are NOT on line 1.** Line 1 is `Exhibit 2.1` (or `EXHIBIT 2.1`, `EX-2.1`,
`Exhibit (d)(10)`) in the large majority of contracts. Files are CRLF with a `\ufeff` BOM.
A tolerant regex over the first 20k canonical chars reaches only **129/152 = 84.9%**; the 23
failures use forms like `by and among:` (colon), `entered into by and among`, `among:`,
`TRANSACTION AGREEMENT`, `AMENDED AND RESTATED AGREEMENT AND PLAN OF MERGER`.
→ Parse with the tolerant regex, and commit a hand-checked override file
`data/agreement_names.json` for any of the **20 selected** agreements the regex misses. Only 20
matter, so this is bounded work. Never fail the build on a name.

**C3. 77.3% of gold spans contain `(Page N)` markers.** The brief never mentions them. They are
inside the span text and are not present in the contract. They must be stripped before
alignment or they poison every fragment that carries one.

**C4. 36.6% of gold spans are multi-excerpt, separated by a blank line** (`\n\n`) in the raw CSV
field, *in addition to* the 41.8% that use `<omitted>`. Splitting only on `<omitted>` leaves
concatenated excerpts that never align. Split on both.

**C5. Line-anchored heading regexes return zero matches on canonical text.** Canonicalisation
collapses whitespace, so `^\s*Section\s+\d+\.\d+` matches nothing. Heading detection must be
**inline** (verified: 212–668 headings/contract found inline). See §3.3 for the cross-reference
and table-of-contents problem this creates.

**C6. `just` was not installed.** I installed it during recon: `uv tool install rust-just` →
`/home/exedev/.local/bin/just` (1.58.0). It is present now, but do not assume; the `justfile`
must not be the only way to run anything.

**C7. `BAAI/bge-reranker-base` is 1.04 GB — too big for 2.4 GB free disk.** fastembed's
cross-encoder list offers `Xenova/ms-marco-MiniLM-L-6-v2` at **0.08 GB**, which loads in 1.6 s
and reranks 20 docs in 0.08 s. Use that (it is the same ms-marco MiniLM the brief lists as its
alternative).

**C8. Alignment is 97.5%, not ~99%, and one question is far worse.** Measured on a seeded
300-case sample over the 12 questions:
- fragment level: 76.2% exact, 21.3% fuzzy (`partial_ratio_alignment`, cutoff 90), **2.5% unaligned**
- case level: **98.7%** have ≥1 fragment aligned; **96.0%** have *all* fragments aligned
- worst question by all-fragments rate: **q03 `Initial matching rights period (COR)` = 0.667 (12/18)**.
The brief's "alignment ≥ 95%" gate is ambiguous and, on the strict reading, sits 1 point above
the measured value. §6 sets two separate thresholds with headroom instead.

**C9. q10 has a 9th answer option and a literal `'None'` null sentinel.** 47/151 rows carry the
answer string `'None'` (this is the null, not an option). The brief lists 7 options; the data
has 8 non-null options — it omits `'"Violation" of fiduciary duties'` (n=1). Null detection must
test the literal string `None`, not just empty.

**C10. Not every contract answers all 12 questions.** Non-null counts: q01 152, q02 138, q03 148,
q04 152, q05 152, q06 152, q07 145, q08 128, q09 152, q10 104, q11 152, q12 152. Per contract,
the number of answerable questions is `{9:1, 10:22, 11:50, 12:79}`, mean **11.36**.
→ 20 agreements yields ≈227 cases; dev(5) ≈57 ≤ 60 ✅, test(15) ≈170 ≤ 180 ✅. The brief's
sizing holds, but only because it wrote "≤".

**C11. `adws/adw_modules/quality.py` is protected — the builder cannot wire the test gate.**
`defaults.protected_files` includes `adws/adw_modules/`, and `permissions.py` checks protection
before the builder's unrestricted `writes`. quality.py currently ships placeholder `echo`
commands that **exit 0**, so the SDLC test phase passes vacuously. See §8 (operator action).

### 0.3 Verified environment facts
- Python 3.12.3, uv 0.12.9, node v22.23.2, git present. **bun absent.** 2 vCPU, 7 GB RAM.
- Disk: **2.4 GB free** after recon cleanup. Dataset extracts to **170 MB**.
- `openai` 3.8.0: `client.chat.completions.parse` and `client.responses` both present.
- `braintrust` exposes `Eval`, `wrap_openai`, `traced`, `init_dataset`. **`Eval(..., no_send_logs=True)`
  exists** — that is exactly the brief's "smoke gate does not log" requirement, no workaround needed.
- OpenRouter realised cost: `extra_body={"usage": {"include": True}}` → `response.usage.cost`
  (float USD) plus `usage.cost_details`. Verified live: a haiku call returned `cost=0.000852`.
- `response_format={"type":"json_schema", "json_schema":{...,"strict":true}}` verified working
  through OpenRouter on Anthropic.
- Model IDs confirmed live (`tools` + `structured_outputs` on all): `anthropic/claude-sonnet-5`
  ($2/$10 per M), `anthropic/claude-opus-5` ($5/$25), `anthropic/claude-haiku-4.5` ($1/$5),
  `google/gemini-3.5-flash` ($1.5/$9), `google/gemini-3.1-flash-lite` ($0.25/$1.5),
  `deepseek/deepseek-v3.2` ($0.269/$0.40), `qwen/qwen3.8-27b`, `moonshotai/kimi-k3`,
  `x-ai/grok-4.6`, `openai/gpt-5.6-luna`. Note `gemini-3.5-flash` costs *more* than
  `claude-haiku-4.5`; use `gemini-3.1-flash-lite` if the slate wants a genuinely cheap tier.
- fastembed `BAAI/bge-small-en-v1.5`: 65 MB, dim 384, 20 texts in 0.16 s. Both models cached = 152 MB.
- Dependency weight: light set + `qdrant-client[fastembed]` = **250 MB** site-packages.
  `llama-index-core` + `llama-index-retrievers-bm25` resolves to ~26 extra packages (nltk,
  networkx, sqlalchemy, tiktoken, pystemmer) — no torch, but heavy for what it provides. See §7.3.

---

## 1. Repo state — do not disturb

`git status` is dirty on paths this work does not own:
```
 M adws/adw_*.py, adws/adw_modules/{agents,runner}.py, .gitignore,
   .claude/skills/sssf/apps/visualizer/vite.config.ts
?? adws/adw_modules/braintrust_tracing.py
?? .claude/skills/sssf/apps/visualizer/node_modules/
```
**Touch none of it.** `adws/` is protected anyway. Append to `.gitignore` only; do not rewrite it.

---

## 2. Deliverables (M0)

```
pyproject.toml                     # new: project deps, pytest markers
justfile                           # APPEND recipes only, keep existing SSSF ones
.gitignore                         # APPEND ignores only
dealpoint/
  __init__.py
  config.py                        # paths, SEED=42, thresholds, constants
  data/
    __init__.py
    download.py                    # zenodo fetch, md5 verify, extract
    canonical.py                   # canonicalise() + normalise_quote()
    sections.py                    # section map + parser_coverage
    labels.py                      # MAUD CSV union, gold answers
    align.py                       # gold span alignment
    questions.py                   # QUESTION_SPEC — the 12, frozen
    select.py                      # 20-agreement selection
    cases.py                       # dev/test/counterfactual JSONL
  cli.py                           # `python -m dealpoint.cli data ...`
data/raw/                          # gitignored (zip + extracted)
data/derived/                      # gitignored (canonical texts, section maps, alignment)
data/eval/*.jsonl                  # COMMITTED
data/agreement_names.json          # COMMITTED (name overrides, C2)
data/reports/                      # gitignored except .gitkeep
tests/
  test_canonical.py test_sections.py test_labels.py test_align.py
  test_questions.py test_select.py test_cases.py test_determinism.py
  conftest.py
```

Everything in M0 runs **offline** except `download.py`.

---

## 3. Module specifications

### 3.1 `config.py`
```python
SEED = 42
ZENODO_URL = "https://zenodo.org/records/7500064/files/maud_v1.zip?download=1"
ZIP_MD5    = "aad11e3f68cd4f3e0067a15b923a19ce"
ZIP_BYTES  = 29_153_514
N_CONTRACTS = 152
N_MAIN_ROWS = 20_623          # integrity assertion, union of all 3 CSVs

MIN_FRAGMENT_CHARS   = 20
FUZZY_CUTOFF         = 90
MIN_FRAGMENT_COVER   = 0.50   # case kept if aligned fragments cover >= 50% of fragment chars
MIN_PARSER_COVERAGE  = 0.80   # see 3.3
MAX_SECTION_CHARS    = 20_000
N_AGREEMENTS, N_DEV, N_TEST = 20, 5, 15
N_REDACTED, N_OUT_OF_SCOPE  = 30, 10
```
All paths derived from a `REPO_ROOT` resolved from `__file__`, never from cwd.

### 3.2 `canonical.py`
```python
def canonicalise(raw: str) -> str
def normalise_quote(s: str) -> str    # alias of canonicalise; used on model output later
```
Exact, ordered steps (this pipeline is what produced the 97.5% alignment figure — do not vary it):
1. strip `\ufeff`
2. `unicodedata.normalize("NFKC", s)`
3. `str.translate` map: `' ' ' "` → ASCII quotes; en/em dash and `\u2212` → `-`; `\xa0` → space;
   `\u200b` → removed
4. `\r\n` and `\r` → `\n`
5. `re.sub(r"\s+", " ", s)` — collapses newlines too
6. `.strip()`

**All offsets everywhere in this project index into this string.** Persist each canonical text to
`data/derived/canonical/{agreement_id}.txt` so offsets are stable and inspectable.

Tests: idempotence (`canonicalise(canonicalise(x)) == canonicalise(x)`), each transform in
isolation, and a golden fixture.

### 3.3 `sections.py`
Because canonicalisation removes newlines (C5), headings are found **inline**, which drags in two
false-positive classes. Both are handled deterministically:

- **Table of contents.** Every contract opens with a dense heading run (median gap 116–840 chars
  in the body vs. far tighter in the TOC). Rule: walk matches from index 0; while the next 10
  matches all sit within 200 chars of each other, they are TOC. Body starts at the first match
  after that run. Require ≥10 matches to call it a TOC at all; some contracts have none.
- **Cross-references.** `Section 6.3(b)` appears constantly in prose (c0: 203 inline `Section N.N`
  hits vs 170 real headings). Rule: accept a candidate as a heading only if its `(article, section)`
  tuple **advances** — same article with `section == last + 1`, or a new, higher article. Everything
  else is a cross-reference and is skipped.

```python
@dataclass(frozen=True)
class Section:
    ref: str        # "6.3" or "ARTICLE VI"
    title: str      # text up to the first sentence end, <= 120 chars, may be ""
    start: int      # inclusive, canonical offsets
    end: int        # exclusive, = next section's start
def parse_sections(canonical: str) -> list[Section]
def parser_coverage(canonical: str, sections: list[Section]) -> float
```
`parser_coverage` = fraction of **body** characters lying in a section of length
≤ `MAX_SECTION_CHARS`. (Naive "chars under a heading" is ~1.0 for every contract and measures
nothing; this measures whether the map is fine-grained enough to be useful.) Measured
median gap 116–840, max gap 8,963–37,839 — so a 20k ceiling separates good maps from bad.

Emit per-agreement `{n_sections, median_len, max_len, parser_coverage}` to
`data/derived/sections/{agreement_id}.json`; this table is a reported data-quality artifact.

### 3.4 `labels.py`
```python
@dataclass(frozen=True)
class Label:
    contract_name: str; question: str; subquestion: str
    text: str; answer: str | None; label: str; text_type: str; category: str
def load_labels() -> dict[tuple[str, str, str], Label]   # union of all 3 CSVs, data_type=="main"
def is_null_answer(a: str | None) -> bool                # "" | "None" | "<NONE>" | "nan" -> True (C9)
```
`csv.field_size_limit(10**9)` is required — span fields reach 16,768 chars.
Integrity test: exactly `N_MAIN_ROWS` rows, `N_CONTRACTS` contracts, all keys unique.

### 3.5 `align.py`
```python
def fragments(raw_span: str) -> list[str]
def align_span(canonical: str, raw_span: str) -> AlignResult
@dataclass(frozen=True)
class AlignResult:
    ranges: list[tuple[int,int]]     # merged, sorted
    n_fragments: int; n_aligned: int
    modes: list[str]                 # "exact" | "fuzzy" per aligned fragment
    covered_fraction: float          # aligned fragment chars / total fragment chars
```
`fragments()`, in order:
1. split raw span on the literal `<omitted>` **and** on `\n\n` (C4)
2. strip `(Page N)` markers anywhere via `re.sub(r"\(Page\s+\d+\)", " ", part)` (C3)
3. `canonicalise` each part
4. drop parts shorter than `MIN_FRAGMENT_CHARS`

`align_span()` per fragment: `canonical.find(frag)` → `"exact"`; else
`rapidfuzz.fuzz.partial_ratio_alignment(frag, canonical, score_cutoff=90)` → `"fuzzy"` using
`(dest_start, dest_end)`; else unaligned. Merge overlapping ranges.

Gold span for a case = the merged ranges. Note `(contract, text_type) → text` is **unique**
(0/2870 keys have >1 distinct text), and q06/q11 share byte-identical MAE-definition text for all
152 contracts — so caching alignment on `(contract, text_type)` is correct and halves the work.

Cache alignment output to `data/derived/alignment.jsonl`.

Tests: a synthetic doc with a planted exact fragment; one with a single-char corruption (must go
fuzzy); one with `<omitted>`; one with `(Page 12)`; one unalignable (must return empty, not raise).

### 3.6 `questions.py`
Frozen `QUESTION_SPEC: tuple[QuestionSpec, ...]` of exactly 12, verbatim from Appendix A —
all 12 strings are confirmed present in the CSV, so **copy them character-for-character**,
especially `Fiduciary exception:  Board determination standard-Answer (no-shop)` (two spaces).
```python
@dataclass(frozen=True)
class QuestionSpec:
    id: str                  # "q01".."q12"
    maud_question: str
    text_type: str
    category: str
    gloss: str               # one-line plain English, shown to the model in ALL arms
    options: tuple[str, ...] # exact answer strings, majority first
    canonical_query: str     # for the M3 tournament
    reasoning_type: str
    required_evidence: str | None
```
Add q10's missing 8th option `'"Violation" of fiduciary duties'` (C9).
**Test:** every `maud_question` and every `option` string appears in the real CSV, and the
`text_type` matches. This test is the guard against a typo silently zeroing a question's accuracy.

### 3.7 `select.py`
Eligibility for an agreement:
- `parser_coverage >= MIN_PARSER_COVERAGE`
- ≥ 10 of the 12 questions have a non-null gold answer **and** a usable aligned span

Then seeded greedy (`random.Random(SEED)`): repeatedly take the agreement that adds the most
unseen (question, answer-option) pairs, ties broken by lowest `contract_name` for determinism,
stratified so the dev/test split preserves the q01 Type-of-Consideration mix. 20 total → 5 dev,
15 test, split by agreement. Write `data/eval/selection.json` with the chosen ids, the rejected
ids **and their rejection reason**.

### 3.8 `cases.py`
Emits three committed JSONL files. A case:
```json
{"case_id":"contract_41__q06","agreement_id":"contract_41","question_id":"q06",
 "gold_answer":"\"Would\" (reasonably) be expected to",
 "gold_spans":[{"start":123456,"end":124001}],
 "span_type":"MAE Definition","majority_answer":"\"Would\" (reasonably) be expected to",
 "required_evidence":"Material Adverse Effect","case_set":"test"}
```
Inclusion: non-null gold answer, ≥1 aligned fragment, `covered_fraction >= MIN_FRAGMENT_COVER`.
Excluded cases are listed with reasons in `data/eval/excluded.json`.

**Counterfactual set (40).**
- **30 redacted:** seeded stratified pick across the 12 questions from *test* cases. Delete the
  aligned gold ranges from the canonical text (no marker, no placeholder), producing a derived
  document `contract_41__redacted_q06` with its **own** canonical text in `data/derived/canonical/`
  and its own section map. Offsets shift — that is why it is a separate document, not a patch.
  Fields: `source_case_id`, `redacted_ranges` (offsets in the *original*), `gold_answer:"ABSTAIN"`.
- **10 out-of-scope:** hand-written questions no merger agreement answers, asked against test
  agreements, `gold_answer:"ABSTAIN"`. Put the 10 question strings in `questions.py` as
  `OUT_OF_SCOPE_QUESTIONS` so they are versioned, not invented at runtime.

**Determinism:** sort by `case_id`; write with `json.dumps(obj, sort_keys=True, ensure_ascii=False)`
and `\n` endings. `test_determinism.py` rebuilds into a temp dir and asserts **byte equality**
with the committed files.

### 3.9 `cli.py`
`python -m dealpoint.cli data [--force-download] [--skip-download]` runs the whole chain:
download → canonicalise → sections → labels → align → select → cases, printing a summary table
(alignment rates, parser coverage, case counts). Idempotent; re-running must not change any
committed file.

---

## 4. `pyproject.toml`

M0 needs only: `rapidfuzz`, `pydantic`, `python-dotenv`, `pytest`. Declare the rest in an optional
group so M0 stays installable in seconds and the disk stays clear:
```toml
[project.optional-dependencies]
agent = ["openai", "braintrust"]
retrieval = ["qdrant-client[fastembed]", "bm25s"]
api = ["fastapi", "uvicorn"]
```
Pytest markers, registered to avoid `PytestUnknownMarkWarning`:
```toml
[tool.pytest.ini_options]
markers = [
  "gate_m0: milestone 0 gate", "gate_m1: ...", "gate_m2: ...", "gate_m3: ...",
  "gate_m4: ...", "gate_m5: ...", "gate_m6: ...", "gate_m7: ...",
  "needs_network: requires internet", "needs_model: spends money on OPENROUTER_API_KEY",
]
```

---

## 5. `justfile` (append; keep every existing SSSF recipe)
```
data:            uv run python -m dealpoint.cli data
test:            uv run pytest -m "not needs_network and not needs_model" -q
gate-m0:         uv run pytest -m "gate_m0 and not needs_network and not needs_model" -q
```

---

## 6. M0 acceptance gates (pytest, marked `gate_m0`)

Thresholds carry headroom over measured values so a gate failure means something broke, not that
reality was 0.5 points under an aspirational number.

| # | Gate | Threshold | Measured |
|---|---|---|---|
| 1 | fragment-level alignment rate over all 12 questions × 152 contracts | ≥ 0.95 | 0.975 |
| 2 | case-level ≥1-fragment-aligned rate | ≥ 0.95 | 0.987 |
| 3 | case-level all-fragments-aligned rate | ≥ 0.90 | 0.960 |
| 4 | agreements selected | == 20 (5 dev / 15 test) | ~150 eligible |
| 5 | dev cases ≤ 60, test cases ≤ 180, counterfactual == 40 | — | ~57 / ~170 |
| 6 | every selected agreement `parser_coverage` | ≥ 0.80 | — |
| 7 | rebuild == committed JSONL, byte-for-byte | exact | — |
| 8 | all 12 `maud_question` + option strings exist in the CSV | exact | verified |
| 9 | label integrity: 20,623 main rows, 152 contracts, unique keys | exact | verified |

Gate 3 is deliberately 0.90, not the brief's 0.95: the strict measure is 96.0% and **q03 alone
sits at 0.667**. Report per-question alignment in the summary so q03's weakness stays visible
rather than being averaged away.

---

## 7. Design decisions for later milestones (fix now, build later)

### 7.1 Canonical offsets are the contract between every component
Retriever chunks, gold spans, model quotes, UI highlights and redaction all index the canonical
string. Nothing else may define an offset. The verbatim-citation scorer is
`normalise_quote(model_quote) in canonical_text` — the same function that built the corpus, which
is the only reason that check can be trusted.

### 7.2 Braintrust
`Eval(..., no_send_logs=True)` for the smoke gate (verified present — no env-var trickery).
Scores stay ≤ 6/case per the free-tier budget; everything else rides in `metadata`, and local
JSONL under `data/results/` remains the source of truth.

### 7.3 Retrieval: go framework-free, keep the interface
The brief permits LlamaIndex behind a `Retriever` interface. Recon says take the interface and
skip the framework: `llama-index-core` + `retrievers-bm25` adds ~26 packages (nltk, networkx,
sqlalchemy, tiktoken, pystemmer) to a 2.4 GB disk, while dense/BM25/RRF/rerank are each ~30 lines
against `qdrant-client[fastembed]` + `bm25s` (1 package), and Qdrant's local mode already does
hybrid RRF natively — verified working on-disk. Define
```python
class Retriever(Protocol):
    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]: ...
```
and implement `DenseRetriever`, `BM25Retriever`, `HybridRRFRetriever`, `RerankRetriever`. If a
later milestone wants LlamaIndex for comparison, it slots in behind the same Protocol.
Embeddings `BAAI/bge-small-en-v1.5` (65 MB); reranker **`Xenova/ms-marco-MiniLM-L-6-v2`** (0.08 GB),
not `bge-reranker-base` (1.04 GB, C7).

### 7.4 Agent loop
`openai` SDK → `https://openrouter.ai/api/v1`, native `tools`, and
`extra_body={"usage":{"include":True}}` on **every** call so `usage.cost` gives realised USD per
case (verified live). Structured final output via
`response_format={"type":"json_schema","json_schema":{...,"strict":true}}` (verified). Hard cap 8
tool calls → `CAP_HIT`.

### 7.5 Model slate (IDs verified available today; re-verify at M6)
Default `anthropic/claude-sonnet-5`; ceiling `anthropic/claude-opus-5` (run last); cheap
`anthropic/claude-haiku-4.5`; Google `google/gemini-3.1-flash-lite` (cheaper than
`gemini-3.5-flash`, which costs more than Haiku); open-weight `deepseek/deepseek-v3.2`.
Judges from non-candidate families: `openai/gpt-5.6-luna`, `x-ai/grok-4.6`, `qwen/qwen3.8-27b`.

### 7.6 Disk budget
2.4 GB free. Dataset 170 MB + venv 250 MB + models 152 MB ≈ 0.6 GB, leaving ~1.8 GB for
`web/node_modules` at M7. Keep `data/raw/` gitignored and delete the zip after extraction.

---

## 8. Operator action item — the test gate is not wired (C11)

`adws/adw_modules/quality.py` still holds placeholder `echo` commands that exit 0, so the SDLC
test phase passes without running anything. The builder **cannot** fix this: `adws/adw_modules/`
is in `defaults.protected_files`, and `permissions.py` checks protection before the builder's
otherwise-unrestricted write permission.

The builder therefore provides `just test` and `just gate-m0`, and **the operator** makes this
one-line edit:
```python
def test(run) -> QualityCheckResult:
    return _run(QualityCheckSpec(
        name="test", area="backend", operation="build",
        argv=["uv", "run", "pytest", "-m", "not needs_network and not needs_model", "-q"],
        timeout_seconds=600), run)
```
and deletes the `lint`/`typecheck`/`build` placeholders from `run_quality()`'s list, or replaces
them. Until that happens, treat a green test phase as unproven.

---

## 9. Out of scope for this request
M1–M8: agent loop, tools, Qdrant index, scorers, Braintrust runner, retrieval tournament,
four-arm sweep, judge panel, model Pareto, FastAPI, Next.js, MCP. Also excluded, per brief §4:
CUAD, ACORD, DeepEval, OpenTelemetry, multi-agent, full-context arm, all 92 questions.

---

## 10. Definition of done

1. `just data` (or `uv run python -m dealpoint.cli data`) runs end to end on a clean checkout and
   prints alignment rates, per-agreement parser coverage, and case counts.
2. `just gate-m0` is green offline; no test needs the network or a model key.
3. `data/eval/{dev,test,counterfactual}.jsonl`, `selection.json`, `excluded.json` and
   `agreement_names.json` are committed; `data/raw/` and `data/derived/` are ignored.
4. A second `just data` changes no committed byte (`git status` clean).
5. Every §0.2 correction is implemented — verify each of C1–C11 explicitly before reporting done.
