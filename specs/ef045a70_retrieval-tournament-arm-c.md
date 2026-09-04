# M3 — Retrieval tournament on dev; freeze arm C

**Spec:** `specs/milestones/m3.md` (read-only). **Requirements:** `specs/grilled-product-brief.md`
§2.3, §3, §3.4, Appendix A (read-only; wins on product decisions).
**Baseline at planning time:** `git 2397b90`; `uv run pytest -m "not needs_network and not needs_model" -q`
= **186 passed, 3 deselected** (4 min); `uv run ruff check .` = 0; `uv run pyright` = 0 errors.
Do not regress any of these.

**This milestone is LLM-free.** The tournament makes zero model calls. The only money you may spend
is your own dev-loop probing, and it must use `z-ai/glm-5.3-flash` (§0.4).

---

## 0. Read this first — five facts measured on this VM today

Everything in §0 was executed and verified during planning. Do not re-derive it; do not design
around a guess that contradicts it.

### 0.1 No new dependencies are needed, and LlamaIndex is not worth adopting

Already installed and working in `.venv`: `bm25s 0.3.11`, `fastembed 0.8.0` (which ships
`fastembed.rerank.cross_encoder.TextCrossEncoder`), `qdrant-client 1.19.0`, `numpy 2.5.2`.
`PyStemmer` is **not** installed and is **not** needed (§0.3).

LlamaIndex was measured, not guessed. `uv venv /tmp/li-probe && uv pip install llama-index-core
llama-index-retrievers-bm25` **succeeded**: **178 MB** standalone, **no torch, no transformers, no
nvidia wheels**. Largest packages: numpy 30 MB + numpy.libs 28 MB, `llama_index` 25 MB,
sqlalchemy 14 MB, pillow.libs 14 MB, networkx 8.1 MB, nltk 7.4 MB, aiohttp 6.4 MB, PIL 5.5 MB,
pydantic_core 4.7 MB, tiktoken 3.7 MB. Marginal cost into this repo's venv (numpy/pillow/pydantic
already present) ≈ **100 MB**. Free disk on `/` is **1.6 GB**.

**Decision: do not adopt LlamaIndex.** It fits the budget, but it buys nothing here — `bm25s` gives
BM25 in three calls, RRF is eight lines of pure Python, and `fastembed` already gives the
cross-encoder the spec names. The spec permits LlamaIndex ("*may* be used"), it does not require it.
**Record the measured 178 MB / ~100 MB marginal figure and this reasoning in your report** — the spec
asks for the installed size to be stated, and "measured, then declined" is the answer. Do not add
`llama-index-*` to `pyproject.toml`.

### 0.2 The cross-encoder works, costs ~1.9 s per call, and lives in `/tmp`

```python
from fastembed.rerank.cross_encoder import TextCrossEncoder
ce = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
scores = list(ce.rerank(query, [chunk.text for chunk in candidates]))  # list[float], higher = better
```

Measured: construction + first-time download **1.5 s**; **1.87–2.08 s** per `rerank` call over 20
documents averaging 1,781 chars, on this 2-vCPU box; peak RSS **1.31 GB** for the reranker alone and
**1.77 GB** with dense + bm25 + reranker all live in one process (7.9 GB total, 4.3 GB free — no OOM
risk, but do not hold two rerankers).

The weights land in **`/tmp/fastembed_cache/models--Xenova--ms-marco-MiniLM-L-6-v2` (88 MB)**;
`FASTEMBED_CACHE_PATH` is unset, so fastembed defaults there. The existing bge embedder is already
cached alongside it (65 MB). **`/tmp` is not durable.** Therefore: **no `gate_m3` test may construct
a real `TextCrossEncoder`.** Every rerank unit test injects a fake scorer (§2.4). A test that needs
the real model is `needs_network` and skips cleanly when the download fails.

### 0.3 `bm25s` API, verified on a real document

```python
import bm25s
tok = bm25s.tokenize(texts, stopwords="en", stemmer=None, show_progress=False)  # -> Tokenized
r = bm25s.BM25(); r.index(tok, show_progress=False)
q = bm25s.tokenize([query], stopwords="en", stemmer=None, show_progress=False)
idx, scores = r.retrieve(q, k=k, show_progress=False)   # numpy arrays, shape (1, k)
```

On `contract_0` (227 chunks): tokenize **0.02 s**, index **0.01 s**, retrieve **< 0.001 s**. Building
a per-document BM25 index is essentially free; cache it per document anyway.

Two gotchas you must handle, both observed:
- `retrieve(q, k=n+50)` **raises `ValueError`** ("k of 277 is larger than the number of available
  scores, which is 227"). **Always clamp `k = min(k, len(chunks))`.** Document chunk counts in dev
  range 227–310, so `k=20` is safe today, but clamp anyway — the counterfactual redacted documents
  are shorter.
- An empty/all-stopword query returns zero scores rather than raising. Return `[]` for a
  blank query rather than a list of arbitrary rank-0 chunks.

### 0.4 Qdrant local mode allows exactly ONE client per path

```
RuntimeError: Storage folder .../data/index is already accessed by another instance of Qdrant
client. If you require concurrent access, use Qdrant server instead.
```

Consequences you must design to:
- The tournament constructs **one** `DenseRetriever` and shares it across every candidate config.
  Never build a fresh `DenseRetriever()` per config or per case.
- The tournament CLI cannot run while `pytest` (which builds a `DenseRetriever` in
  `tests/test_index_sanity.py`) is running. Run them sequentially, never concurrently.
- `QdrantClient(location=":memory:")` works fine and is the right thing for the in-test tiny index
  (verified: create_collection + upsert + `query_points` returns the payload intact).

### 0.5 A 10-case dry run of the tournament — the shape of the result

Executed during planning: 10 dev cases sampled deterministically (`sorted by case_id`, every
`len//10`-th → `contract_0__q01/q06/q11, contract_1__q04/q09, contract_3__q02/q07,
contract_46__q01/q06/q11`), k=10, hit = chunk overlaps a gold span by ≥ 50 chars, dense/bm25
fetch_k = 20, RRF k=60, rerank over the hybrid top-20:

| config | hit@5 | hit@10 | MRR | wall (10 cases) |
|---|---|---|---|---|
| dense / canonical | 0.70 | 0.70 | 0.583 | 1.4 s |
| bm25 / canonical | 0.90 | 1.00 | 0.685 | 0.1 s |
| hybrid-rrf / canonical | 0.90 | 1.00 | 0.705 | ~0 s |
| hybrid-rrf+rerank / canonical | 0.90 | 0.90 | **0.733** | 17.1 s |
| dense / maud | 0.10 | 0.20 | 0.110 | 1.4 s |
| bm25 / maud | 0.40 | 0.50 | 0.283 | ~0 s |
| hybrid-rrf / maud | 0.30 | 0.40 | 0.164 | ~0 s |
| hybrid-rrf+rerank / maud | **0.50** | 0.50 | 0.333 | 16.5 s |

Three things this tells you, all of which belong in your report:
1. **BM25 beats dense badly on the canonical queries** (0.90 vs 0.70 hit@5). The canonical queries
   are keyword-dense legal boilerplate; that is exactly BM25's home ground. The DoD's "winner's
   hit@5 ≥ dense hit@5 on canonical queries" is therefore easy to satisfy — but do not treat it as
   satisfied until the *full* 58-case run says so.
2. **The raw MAUD question text is the hard run, as the spec intends** — dense collapses to 0.10.
   Reranking is the only thing that recovers it (0.50). This is the finding that justifies arm C.
3. **Reranking can demote a hit out of the top 10** (hit@10 0.90 vs hybrid's 1.00 on canonical). Your
   per-stage instrumentation must record demotions honestly, not just promotions.

These are 10-case numbers on a sample and are **not** the answer. The real run is all 58 dev cases.

### 0.6 Full-run wall clock — budget ~10 minutes, not hours

58 dev cases × 2 query types = 116 evaluations per config. From §0.2/§0.3/§0.5:
dense ≈ 0.14 s/query → 16 s; bm25 ≈ free; hybrid fusion ≈ free; rerank ≈ 1.9 s/query → **220 s**;
multi-query fusion (3 extra variants: 3 extra dense + 3 extra bm25 queries, then rerank) ≈ 49 s +
220 s. **Total ≈ 8–10 minutes** for all six configs. Print progress; do not add a cache.

---

## 1. Brief-vs-spec differences — report them, do not silently resolve

Record both under a "brief vs milestone spec" heading in your report. Implement as resolved here.

| # | Brief | `m3.md` | Resolution |
|---|---|---|---|
| 1 | §4 MVP exclusions: "**LLM-generated tournament queries**" are excluded outright | deliverable 1 permits "frozen, pre-generated alternate queries stored in `data/eval/tournament_queries.json` (never generated at run time)" | **The brief wins.** The alternate queries are **hand-written by you**, committed, and frozen. They are pre-generated and never generated at run time, so `m3.md` is satisfied too; and they are not LLM-generated, so the brief's exclusion is honoured. Do not call any model to produce them. Record this as the resolution. |
| 2 | §2.3: candidates listed as `dense, bm25, hybrid-rrf, hybrid-rrf+rerank, hybrid+multiquery-fusion(+rerank)` | deliverable 1 marks `MultiQueryFusionRetriever` "optional" | Build it. It is cheap (§0.6) and the brief lists it as a tournament arm. If it loses, that is a result, not a failure. |

There is **no** difference on `index_version`: brief §2.3 already defines it as "hash of (chunking
params, embedding model, retriever config)", which is what §4 below implements.

---

## 2. Deliverable 1 — the candidate retrievers

All of this lands in **`dealpoint/corpus/retrievers.py`** (extending the existing module; do not
create a parallel one). Every candidate satisfies the existing Protocol **unchanged**:

```python
class Retriever(Protocol):
    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]: ...
```

`agreement_id` carries the **document** id (a redacted variant passes its own `case_id`) — see the
`Chunk.agreement_id` docstring in `chunks.py`. Do not change that convention.

Two design rules that make the whole gate offline and cheap; follow them for every class:

- **Every expensive dependency is constructor-injectable with a `None` default.** `DenseRetriever`
  already takes `client=None`; give it `embedder=None` too. `BM25Retriever` takes
  `chunks_provider=None`. `RerankRetriever` takes `scorer=None`. When the argument is `None` the
  class lazily builds the real thing on first use; when it is provided, nothing is imported and
  nothing is downloaded. This is what lets `gate_m3` run on a synthetic corpus with no network,
  no `/tmp` cache and no Qdrant file lock.
- **Lazy imports stay lazy.** `fastembed`, `qdrant_client` and `bm25s` are imported *inside*
  methods/constructors, never at module top level — the module is imported by the offline suite and
  by `pyright`, and the `retrieval` extra is optional.

### 2.1 `reciprocal_rank_fusion` — a pure function, unit-tested on its own

```python
RRF_K = 60  # dealpoint/config.py

def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Chunk]], rrf_k: int = RRF_K
) -> list[Chunk]:
    """Fuse ranked lists by sum of 1/(rrf_k + rank), rank 1-based. Deterministic."""
```

Score a chunk by `sum(1/(rrf_k + rank))` over every list it appears in, then sort by
`(-score, chunk_id)`. **The `chunk_id` tiebreak is mandatory** — without it the output depends on
dict insertion order and the report stops being reproducible. No I/O, no imports beyond `Chunk`.

### 2.2 `BM25Retriever`

```python
class BM25Retriever:
    def __init__(self, chunks_provider: Callable[[str], list[Chunk]] | None = None) -> None
    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]
```

- Default `chunks_provider` = `lambda doc_id: chunk_document(load_document(doc_id))` — the **same
  M1 chunks** the dense index holds. The spec is explicit: chunks are fixed at M1's `chunk_version`.
  Never re-chunk.
- Cache the built `(BM25, chunks)` per `agreement_id` in an instance dict. Building is 0.03 s
  (§0.3) but the tournament calls each document ~12–24 times per config.
- Tokenize with `stopwords="en", stemmer=None, show_progress=False` on **both** corpus and query.
- **Clamp `k = min(k, len(chunks))`** before `retrieve` (§0.3) and return `[]` for a query whose
  tokenization is empty.

### 2.3 `HybridRRFRetriever`

```python
class HybridRRFRetriever:
    def __init__(self, dense: Retriever, sparse: Retriever,
                 fetch_k: int = HYBRID_FETCH_K, rrf_k: int = RRF_K) -> None
```

`search` fetches `fetch_k` (=20) from each leg, fuses with `reciprocal_rank_fusion`, returns the
top `k`. Fusion happens **in Python**, not in Qdrant — the spec allows either ("Qdrant sparse+dense
RRF, **or fusion in Python**"), and Python fusion needs no re-index, no sparse vectors in the
collection, and no change to `data/index/`. Say so in the report.

### 2.4 `RerankRetriever`

```python
Scorer = Callable[[str, list[str]], list[float]]   # (query, doc_texts) -> scores, higher = better

class RerankRetriever:
    def __init__(self, base: Retriever, model: str = RERANK_MODEL,
                 fetch_k: int = RERANK_FETCH_K, scorer: Scorer | None = None) -> None
```

`search` takes `base.search(agreement_id, query, k=fetch_k)` (=20), scores those texts, sorts by
`(-score, chunk_id)` (tiebreak again mandatory), returns the top `k`. The default scorer lazily
constructs one shared `TextCrossEncoder(RERANK_MODEL)` on first use and wraps
`list(ce.rerank(query, texts))`. **Build the encoder at most once per process** (§0.2: 1.31 GB RSS).

### 2.5 `MultiQueryFusionRetriever`

```python
class MultiQueryFusionRetriever:
    def __init__(self, base: Retriever, variants_by_query: dict[str, list[str]],
                 fetch_k: int = HYBRID_FETCH_K, rrf_k: int = RRF_K) -> None
```

The Protocol only passes a query *string*, not a question id, so the variant table is keyed by
**query text**: build `variants_by_query` from `tournament_queries.json` mapping **both** the
question's `canonical_query` **and** its `maud_question` to the same variant list. On `search`,
run `base.search` for the incoming query plus every variant, RRF-fuse the lists, return top `k`.
A query with no entry in the table falls back to a single `base.search` — never an error.

**`data/eval/tournament_queries.json`** (committed; `data/eval/` is *not* gitignored — verified):

```json
{
  "note": "Hand-written frozen alternate queries for the M3 multi-query fusion candidate. Never generated at run time; never LLM-generated (brief §4).",
  "queries": {
    "q01": {"canonical": "<exact copy of QUESTION_BY_ID['q01'].canonical_query>",
            "variants": ["...", "...", "..."]},
    "...": {}
  }
}
```

Exactly **3 variants per question, 12 questions**. Write them yourself as a competent M&A reader
would rephrase the target: one leaning on the defined-term/heading vocabulary, one on the operative
verb phrasing, one on the option-discriminating words (e.g. for q03, "business days" vs "calendar
days"). Keep them short, keyword-dense, and free of the question's answer. A `gate_m3` test asserts:
the file parses, has all 12 question ids, each `canonical` field is **byte-identical** to
`QUESTION_BY_ID[qid].canonical_query`, and each has ≥ 1 variant with no duplicates.

### 2.6 The frozen-config plumbing

```python
@dataclass(frozen=True)
class RetrieverConfig:
    name: str                    # "dense" | "bm25" | "hybrid_rrf" | "hybrid_rrf_rerank" | ...
    kind: str                    # which class to build
    fetch_k: int = HYBRID_FETCH_K
    rrf_k: int = RRF_K
    rerank_model: str | None = None
    multi_query: bool = False

def build_retriever(config: RetrieverConfig, *, dense: Retriever, sparse: Retriever,
                    variants_by_query: dict[str, list[str]] | None = None,
                    scorer: Scorer | None = None) -> Retriever
```

`build_retriever` takes the shared `dense`/`sparse` instances as arguments — it must **never**
construct a `DenseRetriever` itself (§0.4, one Qdrant client per path). A `config_to_dict` /
`config_from_dict` pair (or `dataclasses.asdict`) is what gets hashed into `index_version` and
written into `config.py` at freeze time.

---

## 3. Deliverable 2 — `dealpoint/eval/tournament.py`

### 3.1 Shape: a pure core plus a thin CLI

```python
def evaluate_config(
    cases: list[dict],
    retriever: Retriever,
    query_for: Callable[[dict], str],
    k: int = TOURNAMENT_K,
) -> dict: ...

def run_tournament(
    cases: list[dict],
    configs: list[RetrieverConfig],
    *,
    dense: Retriever,
    sparse: Retriever,
    variants_by_query: dict[str, list[str]] | None = None,
    scorer: Scorer | None = None,
    k: int = TOURNAMENT_K,
) -> dict: ...

def render_markdown(report: dict) -> str: ...
def main(argv: list[str] | None = None) -> int: ...
```

`run_tournament` takes **already-loaded case rows and already-constructed dense/sparse retrievers**.
That is what makes the DoD's "the tournament script runs on ≤ 10 dev cases with a tiny index inside
the test" a three-line test (§5.2) instead of a subprocess.

### 3.2 Metrics and qrels

For one case and one ranked list of `k` chunks:
- a chunk **hits** iff `overlap_chars((chunk.start, chunk.end), gold) >= MIN_GOLD_OVERLAP_CHARS`
  for some gold span of the case. **Reuse `dealpoint.eval.scorers.overlap_chars` and
  `gold_ranges`** — do not re-implement the geometry; the whole project has exactly one definition
  of "overlaps gold" and it is already unit-tested.
- `hit@5` = 1 if any hit at rank ≤ 5; `hit@10` likewise at ≤ 10; `MRR` = `1/first_hit_rank`, else
  `0.0`. Aggregate = mean over cases. Skip a case with no gold spans (dev has none, but be safe).
- Query types: `"canonical"` → `QUESTION_BY_ID[case["question_id"]].canonical_query`;
  `"maud"` → `.maud_question`. Both run for every config.

### 3.3 Per-stage instrumentation (the spec's real ask)

The report must say *which component surfaced or promoted the gold chunk*, per case. For each
(case, query_type) record a `stages` object:

```json
{"dense_rank": 3, "bm25_rank": 1, "fused_rank": 1, "reranked_rank": 2,
 "surfaced_by": "both",          // "dense" | "bm25" | "both" | "none"
 "rerank_effect": "demoted"}     // "promoted" | "demoted" | "unchanged" | "n/a"
```

Ranks are the 1-based rank of the **first gold-overlapping chunk** in that stage's list, or `null`.
`surfaced_by` compares whether the gold chunk appears in the dense top-`fetch_k`, the bm25
top-`fetch_k`, both, or neither. `rerank_effect` compares `reranked_rank` to `fused_rank`
(`null` → better counts as promoted; better → `null` counts as demoted). §0.5 shows demotions
happen — record them.

### 3.4 Output

- **`data/reports/tournament.json`** — deterministic (`json.dump(..., indent=2, sort_keys=True,
  ensure_ascii=False)` + trailing newline), containing:
  `{schema_version, generated_from: {chunk_version, index_version, dataset_version, git_sha7,
  n_cases, case_set: "dev"}, k, rrf_k, fetch_k, rerank_model, configs: [asdict(cfg), ...],
  results: {<config>: {<query_type>: {hit_at_5, hit_at_10, mrr, n_cases, wall_seconds}}},
  per_case: [{case_id, question_id, agreement_id, config, query_type, first_hit_rank, stages}],
  winner: {config, query_type_basis: "canonical", hit_at_5, dense_hit_at_5, margin},
  llamaindex: {adopted: false, measured_install_mb: 178, marginal_mb: ~100, reason: "..."}}`.
  Wall-clock per config **per query type** is required by the spec — record both and the total.
- **`data/reports/tournament.md`** — human tables: one table per query type (rows = configs,
  columns = hit@5 / hit@10 / MRR / wall), a short "which component found it" table rolled up from
  `stages` (counts of `surfaced_by` and `rerank_effect`), the winner sentence, and the
  LlamaIndex-declined note. Regenerable from the JSON alone.

**Both files are gitignored today** (`.gitignore` line 19: `data/reports/*`). **Add**
`!data/reports/tournament.json` and `!data/reports/tournament.md` under a
`# dealpoint (M3 retrieval tournament)` heading, exactly as M2 did for `braintrust_runs.json`.
Without this the DoD's artifacts never reach the repo.

### 3.5 The `test.jsonl` prohibition — two independent enforcements

The DoD demands both, so implement both:

1. **The runner refuses `--set test`.** `--set` accepts a free string (do **not** use
   `choices=["dev"]`, which produces an argparse error rather than the refusal the DoD describes)
   and the code raises `SystemExit(2)` with a message naming the milestone rule for anything other
   than `dev`. Unit-test that `main(["--set", "test"])` exits non-zero and that
   `main(["--set", "counterfactual"])` does too.
2. **The module's source text contains no path to the test set.** `tests/test_tournament.py` reads
   `Path(dealpoint.eval.tournament.__file__).read_text()` and asserts `"test.jsonl" not in src` and
   `"TEST_JSONL_PATH" not in src`. This constrains your imports: you may call
   `dealpoint.eval.cases.load_case_set("dev")` (that module names the constant, yours must not),
   but you may not import `TEST_JSONL_PATH` into `tournament.py`.

Careful: a docstring sentence like "never reads data/eval/test.jsonl" would fail your own grep.
Phrase the docstring as "never reads the frozen test set" instead.

### 3.6 CLI

```
python -m dealpoint.eval.tournament [--set dev] [--limit N] [--configs a,b,c]
                                    [--k 10] [--out-dir DIR] [--no-rerank]
```

`--limit N` takes the first N after a deterministic `sorted(key=case_id)`. `--no-rerank` skips the
two rerank configs, for a fast smoke while you are iterating. Print a per-config progress line with
elapsed seconds — the full run is ~10 minutes (§0.6) and a silent process looks hung.

Add to `justfile`, next to the existing `gate-m2` block:

```make
# run the LLM-free retrieval tournament over the dev set: just tournament [--limit 10]
tournament *ARGS:
    uv run python -m dealpoint.eval.tournament "$@"

# milestone 3 acceptance gates only, offline
gate-m3:
    uv run pytest -m "gate_m3 and not needs_network and not needs_model" -q
```

---

## 4. Deliverable 3 — freezing arm C

### 4.1 `index_version` — generalise without changing the existing value

`dealpoint/corpus/retrievers.py::index_version()` currently returns **`e2b4a2b97561`** and hashes
`{chunk_version, embedding_model, retriever_config: {collection, distance, default_k}}`. That value
is embedded in committed result filenames under `data/results/`, in
`data/reports/{index_version.txt, versions.json, braintrust_runs.json}`, and asserted literally in
`tests/test_braintrust_adapter.py`. **A no-argument `index_version()` call must keep returning
`e2b4a2b97561` byte-for-byte.** Add a `gate_m3` test that pins it.

Extend it instead:

```python
def index_version(collection=QDRANT_COLLECTION, distance="Cosine",
                  default_k=RETRIEVER_DEFAULT_K,
                  retriever_config: dict | None = None) -> str:
    """... When `retriever_config` is given it replaces the default dense config in the
    hashed payload, which is how arm C's frozen retrieval identity is derived
    (brief §2.3: index_version = hash of chunking params, embedding model, retriever config)."""
```

and a convenience `def arm_c_index_version() -> str: return index_version(retriever_config=ARM_C_RETRIEVER)`.

### 4.2 What goes where

- **`dealpoint/config.py`** gains `ARM_C_RETRIEVER: dict` — the winning `RetrieverConfig` as a
  plain, sorted-key dict literal, with a comment naming the tournament run it came from
  (`tournament.json`'s `git_sha7` + the winner's hit@5 vs dense hit@5). This is the freeze.
- **`data/reports/versions.json`** gains `arm_c_retriever` (the same dict) and
  `arm_c_index_version`, and **keeps `index_version` unchanged** (`e2b4a2b97561`) — that key is the
  identity of the *physical* dense index under `data/index/`, which arms A and B use and which M3
  does not rebuild.
- **`data/reports/index_version.txt`** is rewritten to hold **arm C's** `index_version`, per the
  DoD's literal wording and brief acceptance criterion 4 ("arm C's `index_version` is pinned in
  config and stamped on every experiment"). Nothing reads this file's *content* today (verified:
  `smoke.py` and `run.py` both call the function, `test_index_sanity.py` only checks existence), so
  this is safe — but `dealpoint/corpus/build_index.py` writes it, so **update `build_index.py` to
  write `arm_c_index_version()` there and to emit both keys into `versions.json`**, or the next
  `just index` silently reverts the freeze.
- **Arms A/B are untouched.** `dealpoint/eval/run.py` still calls `index_version()` for them. Wiring
  arms C/D into the runner is **M4's** job, not yours; leave a one-line comment in `run.py` naming
  `ARM_C_RETRIEVER` as the seam, and do not add `"C"`/`"D"` to its `--arm` choices.

### 4.3 How the winner is chosen — decide it in code, before you look at the numbers

Write the rule into `run_tournament` and state it in the report:

> The winner is the config with the highest **hit@5 on canonical queries**; ties broken by MRR on
> canonical queries, then by hit@5 on MAUD queries, then by lower wall-clock, then by config name.

Then assert the DoD's condition — `winner.hit@5 >= dense.hit@5` on canonical queries — as a `gate_m3`
test reading `tournament.json`. Because the winner is *defined* as the argmax over a set that
contains `dense`, this holds by construction; the test exists so a future edit that breaks it fails
loudly.

**If the tournament winner is plain `bm25` rather than a hybrid** (§0.5 makes that plausible), freeze
`bm25` and say so plainly. Arm C's name in the brief is `agent-hybrid-rerank`, but the brief also
says arm C's backend is "tournament winner" — the measurement wins over the label. Record the
discrepancy between the arm's *name* and its *frozen config* in the report so M4's planner sees it,
and do **not** hand-pick a hybrid to make the label true.

---

## 5. Tests

New files: `tests/test_retrievers_m3.py`, `tests/test_tournament.py`. Both `pytestmark =
pytest.mark.gate_m3`. **No test may hit the network, download a model, or open `data/index/`**
(§0.2, §0.4). The `gate_m3` marker is already registered in `pyproject.toml`.

### 5.1 `tests/test_retrievers_m3.py`

Build a synthetic corpus in the test: ~12 hand-made `Chunk`s over 2 fake document ids, with
distinctive vocabulary and known offsets. Then:

- **Protocol conformance** — for every candidate (`DenseRetriever` with an injected in-memory
  Qdrant client + fake embedder, `BM25Retriever` with an injected `chunks_provider`,
  `HybridRRFRetriever`, `RerankRetriever` with an injected scorer, `MultiQueryFusionRetriever`):
  `isinstance(obj, Retriever)` is not usable on a plain Protocol, so assert structurally —
  `search(doc_id, query, k=3)` returns a `list[Chunk]` of length ≤ 3, every chunk belongs to the
  requested document, and results are identical across two calls (determinism).
  For the dense case build the tiny index with `QdrantClient(location=":memory:")` and a fake
  embedder that maps text → a small deterministic vector (e.g. normalised token-hash counts into 8
  dims) — verified working in §0.4.
- **RRF ordering** — a hand-computed case: list A `[c1, c2, c3]`, list B `[c3, c1]`, `rrf_k=60`.
  Assert the exact score order and that the returned order matches `sorted by (-score, chunk_id)`.
  Add a case where two chunks tie on score and assert the `chunk_id` tiebreak decides.
- **Rerank ordering** — inject a scorer returning a fixed permutation and assert the output order
  is exactly the descending-score order of the base's top-`fetch_k`, that the base is called with
  `k=fetch_k` (not the caller's `k`), and that a tie falls back to `chunk_id`.
- **BM25 clamping** — `search(doc_id, query, k=1000)` on a 12-chunk corpus returns 12 chunks and
  does not raise (§0.3), and a whitespace-only query returns `[]`.
- **Multi-query fusion** — a variants table maps one query to two variants; assert the base
  retriever is called three times with the three distinct strings, and that an unknown query
  triggers exactly one call.
- **`index_version` stability** — `index_version() == "e2b4a2b97561"`;
  `index_version(retriever_config=ARM_C_RETRIEVER) != index_version()`; the arm-C value is stable
  across calls and equals `arm_c_index_version()`.
- **`tournament_queries.json`** — the four assertions of §2.5.

### 5.2 `tests/test_tournament.py`

- **The DoD's tiny end-to-end run** — 10 synthetic case rows (with `gold_spans` chosen so some
  configs hit and some miss) + the synthetic corpus + in-memory dense + injected scorer, through
  `run_tournament(...)`. Assert: every config × every query type appears in `results`; each carries
  `hit_at_5`, `hit_at_10`, `mrr`, `n_cases`, `wall_seconds`; `0.0 <= mrr <= 1.0`;
  `hit_at_5 <= hit_at_10`; `per_case` has `len(cases) * n_configs * 2` rows; every `stages` object
  has the five keys of §3.3; `winner` is present and its `config` is one of the config names.
- **Determinism** — two `run_tournament` calls produce identical `results` and `per_case` (drop
  `wall_seconds` before comparing).
- **No test set, two ways** — the source grep of §3.5(2), and `main(["--set", "test"])` /
  `main(["--set", "counterfactual"])` both exiting non-zero with a message on stderr.
- **The committed report** — read `data/reports/tournament.json` and assert: it exists;
  `generated_from.case_set == "dev"` and `n_cases == 58`; every one of the six configs is present
  for both query types; `winner.hit_at_5 >= winner.dense_hit_at_5`;
  `generated_from.chunk_version == chunk_version()`. Assert `data/reports/tournament.md` exists and
  is non-empty. **Do not skip this test when the file is missing** — the DoD requires the artifact,
  so its absence must fail.
- **The freeze** — `data/reports/index_version.txt` (stripped) equals `arm_c_index_version()`;
  `versions.json` carries `arm_c_index_version` equal to the same value, `arm_c_retriever` equal to
  `ARM_C_RETRIEVER`, and an unchanged `index_version == "e2b4a2b97561"`;
  `ARM_C_RETRIEVER["name"] == tournament.json["winner"]["config"]`.

### 5.3 One optional `needs_network` test

A single test marked `@pytest.mark.gate_m3 @pytest.mark.needs_network` that constructs the real
`TextCrossEncoder` and asserts it scores a relevant passage above an irrelevant one. It must
`pytest.skip` cleanly on any exception during construction. **Never mark it `needs_model`** (it
spends nothing) and **never** let the offline gate depend on it — the factory runs
`-m "gate_m3 and not needs_network"`.

---

## 6. Config constants to add (`dealpoint/config.py`)

```python
# --- Milestone 3: retrieval tournament -------------------------------------
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # fastembed cross-encoder, 88 MB, CPU
RRF_K = 60                  # standard reciprocal-rank-fusion constant
HYBRID_FETCH_K = 20         # per-leg depth before fusion (brief §2.3: "hybrid top-20")
RERANK_FETCH_K = 20         # candidates re-scored by the cross-encoder
TOURNAMENT_K = 10           # ranked-list depth the metrics are computed over
TOURNAMENT_QUERIES_PATH = EVAL_DIR / "tournament_queries.json"
TOURNAMENT_JSON_PATH = REPORTS_DIR / "tournament.json"
TOURNAMENT_MD_PATH = REPORTS_DIR / "tournament.md"
TOURNAMENT_MILESTONE_TAG = "m3"

# The cheap workhorse for every dev-loop/smoke/exploratory metered call from M3 on
# (engineer's instruction, 2026-09-04). Verified on OpenRouter 2026-09-04:
# prompt $0.075/M, completion $0.25/M; supports tools, tool_choice, response_format,
# structured_outputs. M3 itself is LLM-free.
WORKHORSE_MODEL = "z-ai/glm-5.3-flash"

ARM_C_RETRIEVER: dict = { ... }   # written at freeze time, §4.2
```

Also add the workhorse to `dealpoint/eval/spend.py::PINNED_PRICES`:
`"z-ai/glm-5.3-flash": {"prompt": 7.5e-08, "completion": 2.5e-07}` — **these exact values were
fetched live from `https://openrouter.ai/api/v1/models` during planning on 2026-09-04**, and
`fetch_prices()` returns `basis == "live"` on this VM (426 models). Re-verify at build time and
record what you saw; the milestone spec asks for the price to be recorded.

---

## 7. Order of work

1. `reciprocal_rank_fusion` + `BM25Retriever` + `HybridRRFRetriever` + `RerankRetriever` +
   `MultiQueryFusionRetriever` + `RetrieverConfig`/`build_retriever` in `retrievers.py`; the
   `embedder=` injection on `DenseRetriever`; the `retriever_config=` parameter on `index_version`.
2. `tests/test_retrievers_m3.py` fully green. This is pure, offline and free — get it green before
   anything touches the real index.
3. Hand-write `data/eval/tournament_queries.json` (12 × 3 variants) and its test.
4. `dealpoint/eval/tournament.py` (core + CLI + markdown) and `tests/test_tournament.py`'s synthetic
   half. Green.
5. `.gitignore` negations, the two `justfile` recipes.
6. **Smoke the real thing small first:** `just tournament --limit 10 --no-rerank` (~2 s), then
   `just tournament --limit 10` (~40 s). Sanity-check the numbers against §0.5 — they should be
   close, not identical (§0.5 sampled every 5th case, `--limit 10` takes the first 10).
7. **The full run:** `just tournament` (~10 min, 58 cases × 2 query types × 6 configs). Confirm
   `pytest` is not running concurrently (§0.4).
8. Freeze: `ARM_C_RETRIEVER` in `config.py`, `versions.json` + `index_version.txt` updated,
   `build_index.py` updated to match, then the freeze tests in `tests/test_tournament.py`.
9. `uv run pytest -m "gate_m3 and not needs_network" -q` green, then the **full** offline suite
   (`uv run pytest -m "not needs_network and not needs_model" -q`, was 186 passed), then
   `uv run ruff check .` and `uv run pyright`, both exit 0.

---

## 8. Definition of done — verify each, do not assume

- [ ] `uv run pytest -m "gate_m3 and not needs_network" -q` green: every candidate conforms to the
      Protocol on a synthetic corpus; RRF and rerank ordering unit-tested including tiebreaks;
      `run_tournament` runs on 10 cases against an in-memory tiny index inside the test.
- [ ] `data/reports/tournament.json` exists and is committed (`.gitignore` negation added), covering
      **all 58 dev cases**, **all six configs**, **both query types**, with `hit@5`, `hit@10`, `MRR`,
      per-stage instrumentation and wall-clock per config per query type.
      `data/reports/tournament.md` renders the tables.
- [ ] The winner's canonical-query `hit@5` ≥ dense's, asserted by a test that reads the JSON.
- [ ] `ARM_C_RETRIEVER` pinned in `dealpoint/config.py`; `arm_c_index_version` in `versions.json`;
      `index_version.txt` holds arm C's value; `build_index.py` writes the same thing; a
      no-argument `index_version()` still returns `e2b4a2b97561`.
- [ ] Nothing in `dealpoint/eval/tournament.py` names the test set (source grep) and the runner
      exits non-zero on `--set test` and `--set counterfactual`.
- [ ] Full offline suite green (≥ 186 passed), `uv run ruff check .` exit 0, `uv run pyright`
      0 errors.
- [ ] **Zero metered calls in M3.** `data/results/spend_ledger.jsonl` gains no rows. If you spent
      anything on your own dev loop it was on `z-ai/glm-5.3-flash`; report the amount and the
      verified price.
- [ ] Report records: the two brief-vs-spec differences of §1; the LlamaIndex measurement (178 MB
      standalone / ~100 MB marginal, no torch) and the decision to decline; the verified
      `z-ai/glm-5.3-flash` price; the full-dev-set table; the winner and the rule that chose it;
      and — if the winner is not a hybrid+rerank — the mismatch between arm C's brief name
      (`agent-hybrid-rerank`) and its frozen config.

## 9. Out of scope

Arms C/D in the eval runner (M4), the skill and `skill_adherence` (M4), judges (M5), Pareto (M6),
the product UI, the MCP layer, any test-set run, any change to `data/eval/{dev,test,counterfactual}.jsonl`
or to the built index under `data/index/`. Do not edit `specs/grilled-product-brief.md` or
`specs/milestones/m3.md`.
