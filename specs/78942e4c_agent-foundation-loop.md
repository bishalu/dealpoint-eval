# M1 — Agent foundation: chunks, dense index, three tools, bounded loop

**Spec (read-only):** `specs/milestones/m1.md`
**Requirements authority (read-only):** `specs/grilled-product-brief.md` §1.2, §1.3, §2.2, §3, §3.4, §3.5
**Do not edit either file.** Out of scope: M2+ (scorers, Braintrust, tournament, skill, arms C/D), the UI, MCP.

---

## 0. What the planner measured (trust these; they are facts, not guesses)

I ran the recon myself and probed the live API. Everything below is measured on this VM today,
not inferred. Several items change how you must build.

### 0.1 Environment

| Fact | Value |
|---|---|
| Offline suite now | `uv run pytest -m "not needs_network and not needs_model" -q` → **exit 0, 70 passed, 235 s** |
| `uv run ruff check .` | exit 0 |
| `uv run pyright` | exit 0 |
| `OPENROUTER_API_KEY` | set in `.env` |
| `MAX_OPENROUTER_SPEND_USD` | `4` |
| Network | no proxy needed; `openrouter.ai` → 200, `huggingface.co` → 200 |
| Free disk | **2.0 GB** (`/dev/root`, 79% used) |
| `openai`, `qdrant-client`, `fastembed` | **not installed**, but already resolved in `uv.lock` |
| `torch` in lockfile | **absent** — the retrieval extra is ONNX-only. Good. |

**`pyproject.toml` already declares both optional groups** (`agent = ["openai","braintrust"]`,
`retrieval = ["qdrant-client[fastembed]","bm25s"]`). Spec deliverable 5 is therefore *install*, not
*add*. Do not re-add them; just:

```
uv sync --extra agent --extra retrieval
```

Resolved wheel payload ≈ 64 MB (onnxruntime 23 MB, numpy 17 MB, pillow 7 MB, grpcio 7.6 MB,
tokenizers 4 MB, openai 1.7 MB, qdrant-client 0.4 MB, fastembed 0.1 MB); installed ≈ 200–300 MB,
plus ≈ 130 MB for the `bge-small-en-v1.5` ONNX weights on first embed. Against 2.0 GB free this
fits, but **check `df -h .` after install and after the index build and record both numbers in
your report.** If disk drops below ~400 MB, stop and report rather than improvising.

### 0.2 The three OpenRouter behaviours M1 depends on — probed live

I spent ≈ $0.031 of the envelope on read-only probes against `anthropic/claude-haiku-4.5`
(planner recon, made with raw `urllib`, so **not** in the ledger; declare it in your report as
planner recon spend so the M1+M2 ≤ $0.50 arithmetic stays honest).

1. **Native tool calling + `usage: {"include": true}` works.** The response carries
   `usage.cost` (realised USD), `usage.prompt_tokens`, `usage.completion_tokens`, and
   `usage.prompt_tokens_details.{cached_tokens,cache_write_tokens}`. Use `usage.cost` for the
   ledger's `usd`. `finish_reason` was `tool_calls`.
2. **`response_format` with a strict `json_schema` works *while `tools` is still present*.**
   The final turn returned `finish_reason: "stop"` and a valid JSON object. This is exactly the
   shape the loop's final answer needs — you do not have to drop the tools array to get
   structured output.
3. **Prompt caching works, but silently no-ops on small prefixes.** Measured:

   | static prefix | 1st call | 2nd identical call |
   |---|---|---|
   | 3,445 prompt tokens | `cached=0, cache_write=0` | `cached=0` — **cache never engaged** |
   | 5,506 prompt tokens | `cached=0, cache_write=5,178` | `cached=5,178` |
   | 18,774 prompt tokens | `cache_write=18,442` | `cached=18,442` (cost $0.0237 → $0.0024) |

   **This is the one finding that bites the spec.** Spec deliverable 2 requires "prompt caching on
   for the static prefix (system prompt + question spec + tool schemas) … so the tool loop does not
   re-bill the prefix every turn". M1's static prefix is ~1–2k tokens — **below the provider's
   minimum cacheable block**, so `cache_control` on it will be accepted and do nothing.

   **Do this:** implement `cache_control` correctly anyway (it is free and becomes effective in
   M4–M6 when the skill and playbooks enlarge the prefix), record `cached_tokens` /
   `cache_write_tokens` in every ledger row, and **write no test that asserts caching engaged.**
   Report the measured threshold as an environment finding. Do not pad the prompt to reach the
   threshold — paying for filler tokens to win a caching discount is a net loss and a benchmark lie.

   Placement that verified correct: `cache_control: {"type":"ephemeral"}` on the **last content
   block of the static prefix**. Send the system message as a content-block list and mark the final
   static block; everything before the mark is cached.

### 0.3 On-disk reality the corpus loader must handle

- `data/derived/canonical/{doc_id}.txt` — **182 files**: 152 base contracts + 30 redacted variants
  (`contract_103__redacted_q09.txt`). Canonical text, already `canonicalise()`d.
- `data/derived/sections/{doc_id}.json` — 182 files.
- **Gotcha (verified):** base-contract section files have keys
  `{body_start, body_end, giant_single_section, max_section_chars, n_sections, parser_version,
  sections, structural_coverage}`, but **redacted-variant files omit `body_start` and `body_end`**
  (`cases.write_redacted_documents` does not write them). `document.py` must treat those two as
  optional — do not `KeyError` on a redacted document.
- Section records are `{ref, title, start, end}`. `ref` is `"ARTICLE I"` / `"ARTICLE 1"` /
  `"1.1"` / (rare bare-toplevel fallback) `"1"`. Zero-padded source headings are **already
  normalised to unpadded** by `sections.py` (`f"{article}.{section}"` over `int()`), so on-disk
  `1.01` is stored as `"1.1"`. Roman vs arabic article style **varies per contract** (contract_0
  → `ARTICLE I`; contract_46 → `ARTICLE 1`) — `get_section` must normalise both.
- **`data/derived/` and `data/reports/*` are gitignored.** `data/reports/` has an allowlist of
  `!` negations. `data/results/` and `data/index/` are currently un-ignored.
- The 20 selected agreements: dev = `contract_0, 1, 3, 5, 46`; test = the other 15
  (`data/eval/selection.json`).

### 0.4 Defined terms — measured on all 5 dev contracts

`lookup_defined_term("Material Adverse Effect")` **must** tolerate the prefixed form. Measured:

| contract | first MAE-style definition found |
|---|---|
| contract_0 | `"Material Adverse Effect" means` |
| contract_1 | `"Company Material Adverse Effect" means` |
| contract_3 | `"Material Adverse Effect" means` |
| contract_5 | `"Company Material Adverse Effect" means` |
| contract_46 | `"Company Material Adverse Effect" means` |

In 3 of 5 dev contracts the bare term is **never** defined — only the `Company …` form exists. A
naive exact-term regex fails the spec's own acceptance item ("`Material Adverse Effect` resolves").
Every contract also defines a `Parent Material Adverse Effect`; prefer the `Company`/bare form over
the `Parent` form when both match. A generic `"X" means|shall mean|has the meaning` sweep found 79
definitions in contract_0, so the pattern generalises.

### 0.5 Sizing

- 20 selected + 30 redacted = **50 documents to index**. At ~1,600 chars/chunk that is
  **≈ 12,000 chunks**; 384-dim float32 ≈ **18 MB** of vectors. Comfortable.
- Embedding 12k chunks on 2 CPU cores: expect **several minutes**. This must be a one-off CLI
  build, **never** work done inside the test suite (see §6.3).

---

## 1. Divergences to report, not silently resolve

The brief wins on product decisions. Each of these is a *report* item for your final summary; none
is a blocker.

1. **M1 gate size (budget-scaled).** Brief §7 M1 gate says "5 dev cases end-to-end on Haiku:
   0 `EXECUTION_FAILED`, ≥ 1 `grounded_accuracy` hit". The milestone spec says **2 dev cases**,
   ≤ $0.15. The spec's own "Budget scaling" section declares this the engineer's instruction under
   a $4.00 envelope. **Follow the milestone spec (2 cases)** and record it in the report as
   "budget-scaled", never as the full benchmark.
2. **`grounded_accuracy` is not available in M1.** It is an M2 scorer (brief §2.4) and M2 is
   explicitly out of scope. The milestone spec correctly substitutes a weaker, self-contained
   property: every ANSWERED finding has ≥ 1 quote that is a verbatim substring of canonical text.
   Implement that; note in the report that the brief's `grounded_accuracy` hit is deferred to M2.
3. **Tool signature vs Retriever signature.** Brief §1.3 gives the *model-facing* tool as
   `search_agreement(query, k=5)` over "the *current* agreement" — no document argument. The
   milestone spec gives the *internal* `Retriever` Protocol as `search(agreement_id, query, k=5)`.
   These are not in conflict: the document is bound by the harness, not chosen by the model.
   Build both exactly as written (§3.3, §4.2). Never expose a document selector to the model —
   that would let it read another agreement.
4. **Execution-record `usage` shape.** Brief §1.2 nests
   `usage: {input_tokens, output_tokens, cost_usd, wall_ms, tool_calls}`; the milestone spec lists
   `usage` and `wall_ms` as siblings. Implement the brief's superset (§4.1) and expose `wall_ms` at
   the top level too, so both readings are satisfied.
5. **`dealpoint.eval.spend.estimate` does not exist yet.** The budget-scaling boilerplate names it,
   but `dealpoint/eval/` is M2. For M1, assert the ≤ $0.15 smoke cost **directly from the ledger
   delta**. Do not build `eval/spend.py` — that is M2 scope.
6. **Prompt caching cannot engage at M1's prefix size.** See §0.2.3. Report the measured numbers.

---

## 2. Configuration — `dealpoint/config.py` (extend, do not restructure)

Append a clearly commented M1 block. Keep the existing style: constants derived from `REPO_ROOT`,
no CWD dependence.

```python
# --- Milestone 1: chunking, index, agent ----------------------------------
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = DATA_DIR / "results"
SPEND_LEDGER_PATH = RESULTS_DIR / "spend_ledger.jsonl"
INDEX_VERSION_TXT_PATH = REPORTS_DIR / "index_version.txt"
VERSIONS_JSON_PATH = REPORTS_DIR / "versions.json"

# Chunking. The brief (§3.4) specifies "~400 tokens / 50 overlap". Tokens are
# not a framework-free unit, so the parameters are stored in CHARS at the
# corpus's measured chars/token ratio (see data/reports/chunk_report.json);
# the ratio is measured once with the bge tokenizer and recorded, never guessed.
CHUNK_TARGET_CHARS = 1600      # ~400 tokens at ~4.0 chars/token
CHUNK_OVERLAP_CHARS = 200      # ~50 tokens
CHUNK_ALGO_VERSION = "1"       # bump to invalidate chunk_version deliberately

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QDRANT_COLLECTION = "dealpoint_chunks"
RETRIEVER_DEFAULT_K = 5

MAX_TOOL_CALLS = 8                  # brief §1.3 hard cap -> CAP_HIT
MAX_TOOL_RESULT_CHARS = 1500        # per tool result, section_ref preserved
MAX_TOKENS_TOOL_TURN = 300
MAX_TOKENS_FINAL = 600
MAX_DEFINITION_CHARS = 6000         # ceiling on a returned defined-term block
LLM_TEMPERATURE = 0
API_MAX_RETRIES = 3                 # API errors -> 3 retries w/ backoff
SCHEMA_MAX_RETRIES = 1              # schema-invalid -> exactly one retry
RATIONALE_MAX_WORDS = 80
EVIDENCE_MAX_ITEMS = 3

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
M1_SMOKE_CASE_IDS = ("contract_0__q01", "contract_0__q06")  # direct + defined-term
M1_SMOKE_MAX_USD = 0.15
```

Both smoke case ids are verified present in `data/eval/dev.jsonl`.

---

## 3. `dealpoint/corpus/` — new package

### 3.1 `document.py`

```python
@dataclass(frozen=True)
class Document:
    document_id: str          # "contract_0" or "contract_103__redacted_q09"
    agreement_id: str         # base contract: document_id.split("__")[0]
    text: str                 # canonical text, offsets index into THIS
    sections: list[Section]   # reuse dealpoint.data.sections.Section
    body_start: int | None    # absent for redacted variants (see §0.3)
    body_end: int | None

def load_document(document_id: str) -> Document
def get_section(doc: Document, ref: str) -> Section | None
def defined_term(doc: Document, term: str) -> DefinedTerm | None
```

- `load_document` reads `CANONICAL_DIR/{id}.txt` + `SECTIONS_DIR/{id}.json`. Call
  `check_parser_version(path, payload.get("parser_version"))` from `dealpoint.data.sections` so a
  stale section map raises `StaleDerivedDataError` with the existing "Run `just data`" message —
  reuse it, do not invent a second guard. Use `.get()` for `body_start`/`body_end`.
- Raise a clear `FileNotFoundError` naming `just data` when the derived files are absent.
- **`get_section(ref)` normalisation** — accept everything the model will actually emit:
  `"6.3"`, `"Section 6.3"`, `"§ 6.3"`, `"Section 6.03"`, `"6.3(b)"`, `"Section 6.3(b)(ii)"`,
  `"Article VI"`, `"ARTICLE 6"`, mixed case, stray whitespace. Algorithm: strip a leading
  `Section`/`Sec.`/`§`/`Article`; drop trailing parenthetical subsection groups; for `N.M` forms
  re-render as `f"{int(n)}.{int(m)}"` (kills zero-padding); for article forms resolve roman↔arabic
  and compare on the integer. Return `None` on no match — never raise, never guess a neighbour.
- **`defined_term(term)`** — build a case-insensitive regex:
  optional `"`/`'`, an optional prefix from `(Company|Parent|Target|Purchaser|Buyer|Seller|Acquiror)`
  (one or two words), the term, optional closing quote, whitespace, then
  `means|shall mean|shall have the meaning|has the meaning|have the meaning`.
  Collect **all** matches; rank them: exact/bare term first, then `Company …`, then any other
  prefix, `Parent …` last (measured rationale in §0.4); tie-break on lowest offset for determinism.
  Return `DefinedTerm(term_as_written, section_ref, start, end, text)` where `end` =
  `min(next definition's start, start + MAX_DEFINITION_CHARS, enclosing section end)` — bounded and
  deterministic. Offsets are canonical.

### 3.2 `chunks.py`

```python
@dataclass(frozen=True)
class Chunk:
    agreement_id: str   # the DOCUMENT id (base or redacted variant) — the retrieval scope key
    chunk_id: str
    section_ref: str
    start: int
    end: int
    text: str

def chunk_document(doc: Document) -> list[Chunk]
def chunk_version() -> str
```

- **Field naming:** the spec's field list is `{agreement_id, chunk_id, section_ref, start, end, text}`
  — keep it verbatim, and document in the docstring that for a redacted variant this field holds the
  *document* id, because that is the scope key that keeps a redacted variant's chunks from
  colliding with its original's. Getting this wrong silently corrupts the whole abstention suite.
- **Section-bounded:** never let a chunk cross a section boundary. Sections shorter than
  `CHUNK_TARGET_CHARS` become exactly one chunk. Longer sections sub-split on a stride of
  `CHUNK_TARGET_CHARS - CHUNK_OVERLAP_CHARS`, preferring a whitespace boundary within a small
  lookback so chunks do not split mid-word. Text before the first section (front matter) is
  chunked under `section_ref = ""`, so a gold span in the recitals is still retrievable.
- `chunk_id` deterministic and self-describing: `f"{agreement_id}:{start}-{end}"`.
- `chunk_version()` = first 12 hex of a sha256 over a canonical JSON of
  `{CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS, CHUNK_ALGO_VERSION, PARSER_VERSION}`. Pure, no I/O.
- **Measure the ratio once.** Write a small one-off (or a `--report` flag on the index builder)
  that tokenises a sample of chunks with the bge tokenizer and records the observed chars/token in
  `data/reports/chunk_report.json` alongside mean/median/p90 chunk chars and token counts. This is
  what makes the "~400 tokens" claim evidence rather than assertion. If the measured mean lands far
  from 400 tokens (say outside 300–500), adjust `CHUNK_TARGET_CHARS` once, rebuild, and record it.

### 3.3 `retrievers.py`

```python
class Retriever(Protocol):
    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]: ...

class DenseRetriever:
    def __init__(self, client=None, collection: str = QDRANT_COLLECTION) -> None: ...
    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]: ...
```

- Qdrant **local/on-disk** mode: `QdrantClient(path=str(INDEX_DIR))`. Not the in-memory mode
  (the index must survive the process) and not a server.
- Prefer `qdrant-client`'s built-in fastembed integration (`add`/`query`) or an explicit
  `TextEmbedding("BAAI/bge-small-en-v1.5")` + `upsert`/`query_points` — either is fine; keep the
  embedding model id read from `EMBEDDING_MODEL` in one place.
- **Scope every query to one document** with a Qdrant payload filter on `agreement_id`. A search
  that can return another document's chunks is a correctness bug, not a ranking flaw.
- Payload per point: all six `Chunk` fields, so a hit rehydrates without touching disk.
- `index_version()` = first 12 hex of sha256 over canonical JSON of
  `{chunk_version, EMBEDDING_MODEL, retriever config: {collection, distance, k default}}`.
- **Import discipline:** `qdrant_client`/`fastembed` must be imported *inside* `DenseRetriever`
  (module-level import guarded or deferred), so that `chunks.py`, `document.py`, `tools.py` and the
  whole offline gate keep working if the retrieval extra is missing. The offline tests must not
  require ONNX.

### 3.4 `build_index.py` — the one-off builder

`python -m dealpoint.corpus.build_index [--force]`:

1. Resolve the 50 document ids: 20 from `selection.json` + the 30 `kind == "redacted"` `case_id`s
   from `counterfactual.jsonl`.
2. Load, chunk, embed, upsert into `INDEX_DIR`.
3. Write `data/reports/index_version.txt` (bare hash + newline, matching `parser_version.txt`
   style) and `data/reports/versions.json` with
   `{parser_version, dataset_version, chunk_version, index_version, embedding_model,
   n_documents, n_chunks}` — deterministic (`sort_keys=True`, trailing newline), no timestamps,
   so it never fights a determinism test.
4. Print a summary: documents, chunks, index_version, elapsed, and `du -sh data/index`.

Add a `just index` recipe next to `just data`.

---

## 4. `dealpoint/agent/` — new package

### 4.1 `schema.py`

Pydantic v2 (already a dependency).

- `Evidence`: `section_ref: str`, `quote: str`.
- `Finding`: `answer: str`, `evidence: list[Evidence]`, `rationale: str`.
  Validators: `1 <= len(evidence) <= 3` **unless** `answer == "ABSTAIN"`, in which case `evidence`
  must be `[]`; `len(rationale.split()) <= 80`. A question-aware check
  (`answer in question.options or answer == "ABSTAIN"`) lives in the loop, where the question spec
  is in hand — keep the model class question-agnostic.
- `Status = Literal["ANSWERED","ABSTAINED","CAP_HIT","EXECUTION_FAILED"]`.
- `TrajectoryStep`: `tool`, `args: dict`, `result_ref: str | None`, `chunk_ids: list[str]`,
  `char_ranges: list[tuple[int,int]]`, `t_ms: int`.
- `Usage`: `input_tokens`, `output_tokens`, `cost_usd`, `wall_ms`, `tool_calls`,
  plus `cached_tokens`, `cache_write_tokens` (§0.2.3).
- `ExecutionRecord`: `status`, `failure_reason: str | None`, `trajectory: list[TrajectoryStep]`,
  `usage: Usage`, `wall_ms: int`, `model`, `arm`, `case_id`, `index_version`, `chunk_version`.
  `failure_reason` values: `None | "schema_invalid_after_retry" | "api_error" | "tool_error" |
  "cap_hit"`.
- `finding_json_schema(question) -> dict` — the strict JSON schema handed to `response_format`,
  with `answer` as an **enum of that question's exact option strings plus `"ABSTAIN"`**. This is
  what makes the option strings binding rather than advisory. Set
  `"additionalProperties": false` and `"strict": true` (probe-verified, §0.2.2).

### 4.2 `tools.py`

Pure functions over a `Document` + a `Retriever`; no LLM, no global state.

```python
def search_agreement(doc, retriever, query, k=5) -> ToolResult
def get_section(doc, section_ref) -> ToolResult
def lookup_defined_term(doc, term) -> ToolResult
```

- `ToolResult` carries the rendered `text` (what the model sees), plus `chunk_ids` and
  `char_ranges` for the trajectory.
- **Rendering:** every returned block is prefixed with its `section_ref` and canonical offsets,
  e.g. `[section 6.3 | chars 12345-13890] <text>`, and truncated to
  `MAX_TOOL_RESULT_CHARS` **per result** with the section ref preserved. Truncate the *text*, never
  the header, and mark the cut (`…[truncated]`) so the model knows it is looking at a fragment.
- A miss returns a plain, honest message (`no section matching "6.3" in this agreement`) — never an
  exception into the loop, never a silent empty string. The agent needs to be able to tell
  "absent" from "broken"; that distinction is the whole abstention suite.
- `tool_schemas()` returns the OpenAI-format tool definitions. `search_agreement` exposes
  **only** `query` and `k` (brief §1.3 — the document is bound by the harness, §1.3 of this plan).

### 4.3 `prompts.py`

- `question_spec_block(question: QuestionSpec, doc: Document) -> str` — the block shown in **every**
  arm: MAUD question text, the one-line gloss, the exact option strings (verbatim, one per line),
  the `ABSTAIN` option and when it applies, the output contract (1–3 evidence items, verbatim
  quotes, rationale ≤ 80 words).
- `system_prompt() -> str` — role, grounding rule (answer only from this agreement), verbatim-quote
  rule, abstain-only-when-genuinely-absent rule.
- **No skill content.** That is M4. A reviewer will check this.
- For out-of-scope counterfactual cases the question has no options; support a free-form variant
  where `answer` is `ABSTAIN` or a short string. Keep it in this module so M2 can reuse it.

### 4.4 `loop.py` — arm B

`run_agent(case, doc, retriever, client, question, model) -> tuple[Finding | None, ExecutionRecord]`

- Native tool calling. Append each assistant tool call + tool result to the message list.
- **Hard cap `MAX_TOOL_CALLS = 8`.** On the 9th requested call: stop, `status = CAP_HIT`,
  `failure_reason = "cap_hit"`, **no finding**. Count *tool calls*, not turns — a turn returning
  two calls consumes two.
- Final answer requested with `response_format = finding_json_schema(question)`,
  `max_tokens = MAX_TOKENS_FINAL`; tool turns use `MAX_TOKENS_TOOL_TURN`. `temperature = 0`.
- **Schema-invalid → exactly one retry → `EXECUTION_FAILED`** with
  `failure_reason = "schema_invalid_after_retry"`. "Invalid" covers: unparseable JSON, Pydantic
  validation failure, `answer` not in options ∪ {ABSTAIN}, evidence-count violation, rationale
  > 80 words. Feed the validation error back on the retry — a blind retry is a wasted call.
- **API errors → 3 retries with exponential backoff → `EXECUTION_FAILED`**
  (`failure_reason = "api_error"`). Retry on connection/timeout/429/5xx; do **not** retry a 400
  (a malformed request will never succeed). These are independent of the schema retry.
- **Quote normalisation:** run every model quote through
  `dealpoint.data.canonical.normalise_quote` (already exists, an alias of `canonicalise`) before
  any verbatim substring check. The document text is already canonical, so this is the only
  normalisation needed.
- `answer == "ABSTAIN"` → `ABSTAINED`; otherwise `ANSWERED`. A correct "No" is `ANSWERED`
  (brief §2.4) — do not conflate it with abstention.
- Record every tool call in `trajectory` with returned chunk ids and char ranges, and `t_ms`.

### 4.5 `pipeline.py` — arm A

No loop, no tools: dense top-5 chunks → **one** call → finding. Same
`question_spec_block`, same JSON schema, same validation and retry rules, same execution record
(`trajectory` holds the single implicit retrieval). This is the RAG baseline; keep it genuinely
one call so the A→B contrast isolates agency (brief §2.2).

### 4.6 `run.py` — CLI

`python -m dealpoint.agent.run --case <case_id> --arm A|B --model <id> [--fake]`

Prints `{"finding": ..., "record": ...}` as JSON to stdout. Resolves the case from
`dev/test/counterfactual` JSONL by `case_id`; for a redacted case the document is the **variant**
(`case_id` itself), not the base contract. `--fake` uses `FakeClient` and must work with no key and
no network. Exit non-zero on `EXECUTION_FAILED` so the CLI is usable in a shell chain.

---

## 5. `dealpoint/llm/client.py`

Per the spec, both the real client and `FakeClient` live in this module.

- `OpenRouterClient` wrapping the `openai` SDK with
  `base_url="https://openrouter.ai/api/v1"`, key from `OPENROUTER_API_KEY`.
- Every call passes `extra_body={"usage": {"include": True}}` (probe-verified) and
  `temperature=0`, with `max_tokens` bounded per call type by the caller.
- `cache_control` on the last static prefix block (§0.2.3).
- **Ledger:** after every metered call append one line to `data/results/spend_ledger.jsonl`:
  ```json
  {"ts": "...", "milestone_tag": "m1", "model": "...", "calls": 1, "usd": 0.0037,
   "input_tokens": 3445, "output_tokens": 55, "cached_tokens": 0, "cache_write_tokens": 0}
  ```
  The key **must** be `usd` — `adws/adw_modules/spend.py::realized_usd()` sums exactly that field,
  and the whole unattended spend guard for M4–M6 depends on it. Create `data/results/` if missing.
  Append-only, one JSON object per line, flushed per call (a crash mid-sweep must not lose the
  record of money already spent).
- Take `usd` from `usage.cost`. If the field is absent, write `usd: 0.0` **and** a
  `"usd_missing": true` marker rather than silently under-reporting spend.
- `FakeClient`: constructed with a scripted list of turns (tool calls and/or a final content
  string), replays them in order, records the messages it was given so tests can assert on the
  prompt, and **never** touches the network or the ledger. It must be able to script: a normal
  2-tool-then-answer run, a run that requests 9+ tool calls (cap), a run returning invalid JSON
  then valid (schema retry), a run returning invalid twice (EXECUTION_FAILED), an ABSTAIN answer,
  and an API-error sequence (raise N times, then succeed / never succeed).

---

## 6. Tests — `tests/`, all marked `pytest.mark.gate_m1`

### 6.1 Offline (FakeClient) — the acceptance list

Follow the existing convention: `pytest.skip(...)` cleanly when derived data is absent
(see `tests/conftest.py::require_dataset`), never a hard error.

- `tests/test_corpus_document.py`
  - `get_section` on a **synthetic** document: `"6.3"`, `"Section 6.3"`, `"§ 6.3"`, `"6.03"`,
    `"6.3(b)"`, `"Article VI"` vs `"ARTICLE 6"`, unknown ref → `None`.
  - `defined_term` on a synthetic document: quote styles, `shall mean`, `has the meaning`,
    prefixed `Company X`, and correct offsets (`text[start:end]` really is the returned block).
  - **On one real selected agreement: `Material Adverse Effect` resolves** (spec requirement).
    Assert on ≥ 2 dev contracts including one where only the `Company …` form exists
    (contract_1, contract_5 or contract_46 — see §0.4), so the prefix tolerance is actually gated.
  - A redacted variant loads without `body_start`/`body_end` (§0.3).
- `tests/test_chunks.py` — chunks never cross a section boundary; offsets round-trip
  (`doc.text[c.start:c.end] == c.text`); overlap is as configured; **deterministic** (chunk twice,
  identical `chunk_id`s); `chunk_version` is stable and changes when a param changes.
- `tests/test_agent_loop.py` — with `FakeClient`: terminates normally; **cap → `CAP_HIT`** with no
  finding and exactly 8 recorded tool calls; **schema-invalid → one retry → `EXECUTION_FAILED`**
  with `failure_reason="schema_invalid_after_retry"`; a valid retry succeeds; **ABSTAIN path**
  yields `ABSTAINED` with empty evidence; API error retried 3× then `EXECUTION_FAILED`;
  a correct "No" answer is `ANSWERED`, not `ABSTAINED`.
- `tests/test_verbatim.py` — **verbatim normalisation**: a quote differing only by curly quotes,
  en-dash, NBSP and collapsed whitespace still matches canonical text after `normalise_quote`;
  a genuinely fabricated quote does not.
- `tests/test_agent_tools.py` — each tool returns text carrying `section_ref` and offsets;
  truncation caps at `MAX_TOOL_RESULT_CHARS` **and** preserves the ref; misses return a message,
  not an exception. Use a stub `Retriever` (a plain list) — no Qdrant, no ONNX.
- `tests/test_llm_client.py` — ledger row shape and the `usd` key; `FakeClient` writes no ledger
  row; the request body carries `usage.include`, `temperature=0` and `cache_control` (assert on the
  captured payload, not on a live call).
- `tests/test_agent_schema.py` — evidence 1–3, `[]` only for ABSTAIN, rationale ≤ 80 words,
  `answer` enum built from the question's real options.

### 6.2 Retrieval sanity (offline once built)

`tests/test_index_sanity.py` — skip cleanly if `INDEX_DIR` or `index_version.txt` is absent.
Sample **5 dev cases** deterministically (sorted `case_id`, seed 42) and assert that a
`search_agreement` using the question's `canonical_query` returns, **for at least one of the 5**, a
chunk overlapping that case's gold span. The spec is explicit that this is a sanity check, not a
quality gate — do not tighten it into a threshold, and do not tune chunking against it.

### 6.3 Runtime discipline — read this before you write a test

The factory's `test` quality block runs the **whole** offline suite with a **600 s timeout**
(`adws/adw_modules/quality.py`), and it already takes **235 s**. You have roughly 300 s of headroom
for every M1 test combined.

- **No test may build the index or download model weights.** The index is built once by
  `python -m dealpoint.corpus.build_index`; tests only read it.
- **No test may load all 152 contracts.** Use synthetic documents plus 1–3 named dev contracts.
- Cache expensive loads in session-scoped fixtures.
- If the suite creeps past ~450 s, stop and report it rather than shipping a gate that times out
  intermittently in the factory.

### 6.4 Metered smoke — `tests/test_smoke_m1.py`

Mark **`gate_m1` and `needs_model` only — NOT `needs_network`.** The factory gate runs
`pytest -m "gate_m1 and not needs_network"` (plus `needs_model` when a key is present); marking it
`needs_network` would silently exclude it from the gate forever.

- Skip cleanly if `OPENROUTER_API_KEY` is unset or the index is missing.
- 2 dev cases (`M1_SMOKE_CASE_IDS`: one defined-term q06, one direct q01), **arm B**,
  `anthropic/claude-haiku-4.5`.
- Assert: **0 `EXECUTION_FAILED`**; every `ANSWERED` finding has ≥ 1 quote that is a verbatim
  substring of the canonical text after `normalise_quote`; the ledger gained rows with `usd > 0`;
  and the **ledger delta across the smoke is ≤ `M1_SMOKE_MAX_USD` ($0.15)**. Measure the delta by
  reading the ledger total before and after — an expensive smoke is a defect, per the spec.
- Budget arithmetic: at the measured ~$0.0037 per ~3.4k-token call, a capped 8-call case with
  growing context lands in the low cents; 2 cases should come in well under $0.15. If it does not,
  that is a real signal (runaway context or an unbounded tool result) — fix the cause, do not raise
  the ceiling.

---

## 7. Repo plumbing

- **`.gitignore` — needed, verified.** `data/reports/*` is ignored with an allowlist, so
  `index_version.txt` and `versions.json` are currently **ignored** (I checked with
  `git check-ignore`). Add:
  ```
  !data/reports/index_version.txt
  !data/reports/versions.json
  !data/reports/chunk_report.json
  data/index/
  ```
  `data/index/` must be ignored (tens of MB of binary). `data/results/` is already un-ignored;
  the brief (§3.5) says results JSONL is committed, so leave the ledger tracked.
- **`justfile`** — add next to `gate-m0`:
  ```
  index:      uv run python -m dealpoint.corpus.build_index
  gate-m1:    uv run pytest -m "gate_m1 and not needs_network and not needs_model" -q
  agent *ARGS: uv run python -m dealpoint.agent.run "$@"
  ```
- **`pyproject.toml`** — extras already exist (§0.1). If `pyright` complains about the new
  third-party imports, prefer installing the extras over adding ignores.
- New packages need `__init__.py` with a one-line docstring, matching the existing house style.

---

## 8. Order of work

1. `uv sync --extra agent --extra retrieval`; record `df -h .` before/after.
2. `config.py` additions; `.gitignore` and `justfile` plumbing.
3. `corpus/document.py` + tests (synthetic first, then the real-contract MAE assertions).
4. `corpus/chunks.py` + determinism tests.
5. `agent/schema.py`, `agent/prompts.py`, `agent/tools.py` + tests with a stub retriever.
   **Everything to this point is offline and needs neither Qdrant nor a key** — keep it green.
6. `llm/client.py` with `FakeClient`; `agent/loop.py`, `agent/pipeline.py` + the full FakeClient
   acceptance matrix (§6.1). Still no network.
7. `corpus/retrievers.py` + `corpus/build_index.py`; build the index for real (50 documents);
   write `index_version.txt`, `versions.json`, `chunk_report.json`; then `test_index_sanity.py`.
8. `agent/run.py`; verify `--fake` end to end.
9. Metered smoke last, once everything else is green — it is the only step that spends money.

---

## 9. Definition of done — verification commands

Judge every command by its **exit status**, never by grepping its output for the word "error".

```bash
uv run pytest -m "gate_m1 and not needs_network" -q      # the milestone gate (incl. metered smoke)
uv run pytest -m "not needs_network and not needs_model" -q   # full offline suite
uv run ruff check .
uv run pyright
uv run python -m dealpoint.agent.run --case contract_0__q01 --arm B --model anthropic/claude-haiku-4.5 --fake
test -s data/reports/index_version.txt && test -s data/reports/versions.json
```

Checklist, mapped to the spec:

- [ ] Offline `gate_m1` (FakeClient): loop terminates; cap → `CAP_HIT`; schema-invalid → retry →
      `EXECUTION_FAILED`; ABSTAIN path; verbatim normalisation; `get_section` and
      `lookup_defined_term` on a synthetic doc **and** on a real selected agreement with
      `Material Adverse Effect` resolving.
- [ ] Metered smoke: 2 dev cases, arm B, Haiku → 0 `EXECUTION_FAILED`; every ANSWERED finding has
      ≥ 1 verbatim quote; ledger rows with `usd > 0`; smoke total ≤ $0.15 asserted from the ledger.
- [ ] Dense index built for 20 + 30 documents; `index_version.txt` and `versions.json` written;
      retrieval-sanity test passes on ≥ 1 of 5 sampled dev cases.
- [ ] Full offline suite, `ruff`, `pyright` green; suite runtime still comfortably under 600 s.

## 10. Report these in your final summary

1. Measured **prompt-caching threshold** (§0.2.3) and the fact that M1's prefix sits below it, so
   `cache_control` is implemented but inert until M4 — with the measured numbers.
2. The **budget-scaled** M1 gate (2 dev cases, not the brief's 5) and that `grounded_accuracy` is
   deferred to M2 — stated as scaling, never as the full benchmark.
3. **Realised spend**: ledger total for M1, plus the ≈ $0.031 of planner recon probes made outside
   the ledger, against the M1+M2 ≤ $0.50 allocation.
4. Measured **chars/token** ratio and the resulting chunk-size distribution (`chunk_report.json`).
5. **Disk** before/after install and index build.
6. Anything where the brief and the milestone spec pulled in different directions (§1) and which
   you followed.
