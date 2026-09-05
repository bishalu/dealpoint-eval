# M7a — Framework roles made explicit: LlamaIndex RAG lab, DeepEval cross-check, Braintrust sync

**Milestone spec (read-only, authoritative for scope):** `specs/milestones/m7a.md`
**Requirements document (read-only, wins on product decisions):** `specs/grilled-product-brief.md`
**Gate:** `uv run pytest -m "gate_m7 and not needs_network" -q` green, plus the full offline suite,
`uv run ruff check .` and `uv run pyright`.

M7 is **additive**. M1–M6 results, the frozen subsets, the skill, the arm configs, the canonical
scorers and the M3 winner are not rewritten, retuned or rerun. The only pre-existing behaviour this
milestone deliberately changes is the README results block (an explicit engineer instruction in the
spec) and one M6 spend-test predicate that would otherwise become false for a reason that has
nothing to do with M6 (see §0.3 — read that before doing anything else).

---

## 0. Read this first — the five sharp edges

These are measured facts about the repo as it stands at `HEAD` (`60eadff`), not guesses. Each one
will silently break the build if you meet it by accident instead of on purpose.

### 0.1 `data/reports/` is gitignored with an explicit allowlist

`.gitignore` contains `data/reports/*` followed by a list of `!data/reports/<file>` un-ignore lines,
one per artifact each milestone added. **Every new report file you write must get its own `!` line**,
or it will not be committed and the reviewer will find the walkthrough referencing files that are not
in the repo. Add, under a new `# dealpoint (M7a framework roles)` heading:

```
!data/reports/li_rag_eval.json
!data/reports/li_rag_eval.md
!data/reports/deepeval_crosscheck.json
!data/reports/deepeval_crosscheck.md
!data/reports/btql_investigations.json
!data/reports/representative_cases.json
!data/reports/braintrust_sync.json
!data/reports/framework_versions.json
!data/reports/m7a_disk_guard.json
```

`data/eval/synthetic_dev_queries.jsonl` lives under `data/eval/`, which is **not** ignored — nothing
to do there. `docs/` is not ignored either.

### 0.2 The factory's spend gate calls `budget m7a`, which does not exist yet

`adws/adw_modules/milestones.py` line 62 registers M7a with
`budget_argv=[..., "-m", "dealpoint.eval.budget", "m7a"]` and `absolute_usd=2.50`. But
`dealpoint/eval/budget.py` takes `choices=sorted(SWEEP_DEFS)` and `SWEEP_DEFS` currently holds only
`four_arm`, `judges`, `pareto`. Verified:

```
$ uv run python -m dealpoint.eval.budget m7a
error: argument sweep: invalid choice: 'm7a' (choose from 'four_arm', 'judges', 'pareto')
```

**You must add an `m7a` entry to `SWEEP_DEFS`** (§8) or the milestone's pre-build spend gate fails
before your code ever runs. Do this early — it is a two-minute change that unblocks the runner.

### 0.3 An existing M6 test will go red the moment M7a spends a cent — and it must be *scoped*, not raised

`tests/test_spend_m6.py::test_realized_usd_within_envelope` asserts:

```python
def test_realized_usd_within_envelope():
    """DoD (spec, verbatim): total realised OpenRouter spend across M1-M6 is
    <= $4.00, asserted from the ledger."""
    assert realized_usd() <= 4.00
```

Measured now: `realized_usd() == 3.61394`, by tag
`{m1: 0.41912, m2: 0.170014, m4: 0.751879, m4_1: 1.420234, m5: 0.110038, m6: 0.742655}`.
Headroom before that assertion breaks: **$0.3861**. M7a's own soft target is $1.00.

The assertion's own docstring says *"across M1–M6"*, and `realized_usd()` sums the whole ledger
including tags that did not exist when M6 was written. The honest repair is to make the test measure
what it claims to measure, using the `realized_by_tag()` helper that already exists in
`dealpoint/eval/spend.py`:

```python
M1_M6_TAGS = ("m1", "m2", "m3", "m4", "m4_1", "m5", "m6")

def test_realized_usd_within_envelope():
    """DoD (M6 spec, verbatim): total realised OpenRouter spend across M1-M6 is
    <= $4.00, asserted from the ledger. Scoped to the M1-M6 milestone tags so a
    later milestone's spend cannot retroactively falsify M6's own DoD claim;
    M7a's ledger discipline is asserted separately in tests/test_spend_m7.py
    against the $6.00 global cap.
    """
    by_tag = realized_by_tag()
    m1_m6 = round(sum(by_tag.get(t, 0.0) for t in M1_M6_TAGS), 6)
    assert m1_m6 <= 4.00
```

**Do not raise the number to 6.00.** That would quietly restate M6's finding as something weaker
than what M6 actually achieved. Scoping keeps M6's claim exactly as strong as it was and makes it
permanently true. Record this edit in `deepeval_crosscheck.md`'s decisions section (or a short
`decisions` block in `li_rag_eval.json` — pick one and be consistent) as a recorded decision with
this reasoning, so the reviewer sees it was deliberate.

The **global** `MAX_OPENROUTER_SPEND_USD=6` cap in `.env` is untouched and is what
`assert_within_cap` enforces. Headroom to it: **$2.386**. M7a's absolute is $2.50, so the global cap
binds first — that is fine and is exactly the belt-and-braces the spec wants.

### 0.4 Disk — measured, not assumed (the spec requires a re-measurement; here it is)

The spec asks for a fresh measurement before touching the live environment. Done, with a scratch venv
(`uv venv /tmp/m7probe`, since removed):

| step | scratch venv size |
|---|---|
| empty venv | 1 MB |
| + `llama-index-core`, `llama-index-retrievers-bm25` | **178 MB** |
| + `deepeval` | **244 MB** (so deepeval's marginal cost ≈ 66 MB) |

Into the *live* `.venv` (340 MB, 67 packages) the dry-run says `rag-lab` adds 41 new distributions and
`deepeval` adds 37, of which several overlap (`aiohttp`, `jinja2`, `yarl`, `tenacity`, `setuptools`,
`nest-asyncio`, `frozenlist`, `multidict`, `propcache`, `markupsafe`, `aiosignal`,
`aiohappyeyeballs`). Neither pulls torch, transformers or nvidia wheels.

Free space now: **1.2 GB**. Expected after both groups: **~0.95–1.0 GB**, comfortably above the
spec's 500 MB floor. Keep `.claude/skills/sssf/apps/visualizer/node_modules` (146 MB) — the spec says
so explicitly.

**Implement the guard anyway** and record its output, because the spec's DoD asks for it: before
installing, `shutil.disk_usage("/")`, run `uv pip install --dry-run` for the group, install, measure
again, and write `data/reports/m7a_disk_guard.json` with `free_before_bytes`, `free_after_bytes`,
`group`, `packages_added`, `floor_bytes: 500_000_000`, `within_guard: true`. If a group would cross
the floor, install `rag-lab` only, skip `deepeval`, and say so in both reports rather than shrinking
something else.

### 0.5 Braintrust is live and reachable — verified, so no capability is speculative

All of the following were run and succeeded:

- `dealpoint.eval.braintrust_adapter.load_braintrust_key()` returns a 51-char key (from
  `.braintrust.json`; `.env.braintrust` also carries `BRAINTRUST_API_KEY`). `braintrust_available()`
  is `True`. `braintrust.login()` succeeds; `init_dataset(project="dealpoint-eval", ...)` returns a
  handle.
- `bt` CLI is at `/home/exedev/.local/bin/bt` and works **only when `BRAINTRUST_API_KEY` is exported**
  (`bt status` shows everything unset; `bt projects list --json` errors without the env var). There
  is no `-p` short flag and no global `--env-file` — use `BRAINTRUST_DEFAULT_PROJECT=dealpoint-eval`
  and `--project` on subcommands that take it.
- Project `dealpoint-eval` exists: `3eb5eb88-e6f5-41c7-b38d-20bbe49b9577`. Eleven experiments are
  already logged (`data/reports/braintrust_runs.json`), e.g.
  `judged-A-anthropic_claude-haiku-4.5-e2b4a2b97561-e3ee9cc` (id
  `cdc7bead-7d7b-48d6-97d1-ea2a55526c50`), `A-z-ai_glm-5.3-flash-e2b4a2b97561-4d2e361`.
- **BTQL over REST works.** `POST https://api.braintrust.dev/btql`, bearer token, body
  `{"query": "..."}`. It rejects a query with no `from:` and one with no
  `select:`/`dimensions:`/`measures:`. Both of these return real data:
  ```
  from: experiment('<uuid>') | select: id, metadata | limit: 2
  from: experiment('<uuid>') | dimensions: metadata.arm as arm | measures: count(1) as n
  ```
  `bt sql --non-interactive --json "<btql>"` hits the same endpoint and surfaces the same errors, so
  either path is fine; prefer the REST call from Python so results are capturable into JSON.
- `claude mcp list` exits 0 and shows `braintrust: https://api.braintrust.dev/mcp (HTTP) - ! Needs
  authentication`. **Record that verbatim and stop.** The spec forbids re-adding it and forbids
  MCP-dependent work; authentication happens after the restart, in M7b.

Note the metadata gotcha visible above: the judged experiments carry rich metadata at the
*experiment* level but the per-row `metadata` came back `{}` for that experiment. Your new sync must
put the common metadata on **every row**, not only on the experiment, or the BTQL slice queries in §6
will return `null` group keys (the `dimensions: metadata.arm` query above returned
`[{"arm": null, "n": 108}]` — that is exactly the failure mode to avoid).

---

## 1. What already exists — the exact names you will build against

Do not re-derive these; they are read from the code at `HEAD`.

### Retrieval (`dealpoint/corpus/retrievers.py`)

```python
class Retriever(Protocol):
    def search(self, agreement_id: str, query: str, k: int = 5) -> list[Chunk]: ...
```

`Chunk` (`dealpoint/corpus/chunks.py`, frozen dataclass):
`agreement_id, chunk_id, section_ref, start, end, text`. Chunk ids are
`f"{doc.document_id}:{start}-{end}"`.

Concrete retrievers: `DenseRetriever` (Qdrant local + fastembed), `BM25Retriever` (bm25s),
`HybridRRFRetriever`, `RerankRetriever`, `MultiQueryFusionRetriever`, plus the pure function
`reciprocal_rank_fusion(ranked_lists, rrf_k)`. Assembled by:

```python
build_retriever(config: RetrieverConfig, *, dense, sparse, variants_by_query=None, scorer=None) -> Retriever
RetrieverConfig(name, kind, fetch_k=20, rrf_k=60, rerank_model=None, multi_query=False)
```

The six frozen M3 candidates are `dealpoint.eval.tournament.DEFAULT_CONFIGS`:
`dense`, `bm25`, `hybrid_rrf`, `hybrid_rrf_rerank`, `multi_query_fusion`,
`multi_query_fusion_rerank`.

**Qdrant local mode allows exactly one client per path.** Build `DenseRetriever()` and
`BM25Retriever()` **once** and pass them into every `build_retriever` call, exactly as
`tournament.main` and `eval/run.py::_build_retriever` already do. A second construction raises
`RuntimeError: already accessed by another instance`.

Versions (`data/reports/versions.json`): `chunk_version=8e5e8ba56765`, `index_version=e2b4a2b97561`,
`arm_c_index_version=8ae2fc1e24e9`, `dataset_version=81ed82cd7552`, `parser_version=29cc01eda19b`,
`skill_version=f8d255cc169b`, `rubric_version=cfda9f8cc401`.

M3 winner (`data/reports/tournament.json` + `dealpoint.config.ARM_C_RETRIEVER`): **`hybrid_rrf`**,
canonical hit@5 `0.9138` vs dense `0.8103`, MRR `0.7319`. Metric key names in the report are
`hit_at_5`, `hit_at_10`, `mrr`, `n_cases`, `wall_seconds`, under
`results[<config>][<"canonical"|"maud">]`.

### Gold-span truth (`dealpoint/eval/scorers.py`)

`overlap_chars(a, b)`, `gold_ranges(case)`, and `MIN_GOLD_OVERLAP_CHARS = 50`. The tournament's
`_first_hit_rank(chunks, golds)` is the canonical "is this chunk a hit" rule:

```python
any(overlap_chars((chunk.start, chunk.end), g) >= MIN_GOLD_OVERLAP_CHARS for g in golds)
```

**This is the definition of `expected_ids` for LlamaIndex.** There is no pre-existing
"gold-bearing chunk ids" helper — you will write one (§3) and it must use exactly this rule.

Canonical scorers, all `(case, finding, record, canonical_text) -> value | None`:
`grounded_accuracy`, `answer_correct`, `citation_verbatim`, `citation_gold_overlap`, `gold_seen`,
`gold_first_rank`, `fabrication`, `abstain_correct`, `redacted_fabrication`, `tool_calls`,
`cap_hit`, `execution_failed`, `input_tokens`, `output_tokens`, `usd`, `wall_ms`.
`required_evidence_met(..., doc=None)` and `skill_adherence(..., doc=None)` take the extra `doc`.
`score_case(case, finding, record, doc) -> dict` returns all 18 by name. **`None` means
not-applicable and must never be coerced to `False`.**

### Case sets and result rows

`dealpoint.eval.cases`: `load_case_set("dev"|"test"|"counterfactual")`, `find_case(case_id)`,
`resolve_document_id(case)` (redacted cases resolve to their own document), `resolve_question(case)`,
`git_sha7()`, `slugify_model(model)`.

Dev: **58 cases, all with gold spans, across 5 documents.** Measured: **148 gold-bearing
(chunk, case) pairs**, 1–7 per case.

Result rows live at `data/results/*.jsonl` (59 files; `spend_ledger.jsonl` is in the same directory
and must be skipped). Row shape:

```
case_id, case_set, arm, model, question_id, index_version, chunk_version, git_sha7,
finding {answer, evidence[{section_ref, quote}], rationale} | null,
record {status, failure_reason, trajectory[], usage{}, wall_ms, model, arm, case_id,
        index_version, chunk_version, failure_detail, raw_final_text, finish_reasons[]},
scores {…18 names…}, skill_rules {score, n_applicable, n_satisfied, rules{}}, usd
```

Trajectory step: `{tool, args, result_ref, chunk_ids, char_ranges, t_ms}`. This is **enough to
replay a trace offline with zero model calls**: `dealpoint/eval/blinding.py::_build_trajectory_step`
already reconstructs the retrieved text per step from `char_ranges` + the document, and
`_enclosing_section_ref` resolves section refs. Reuse that module's approach for the Braintrust log
replay (§5) rather than inventing a second reconstruction.

Manifests that map (arm, model) → results file: `data/reports/four_arm_manifest.json`,
`data/reports/pareto_manifest.json`. `dealpoint.eval.report.load_rows_from_manifest(path)` reads them.

### Judging (M5/M6)

`data/eval/judged_subset.json`: 18 case ids (tranche_1 of `test_subset_v1`), `subset_hash
5918ef10a7e6`, and **6 variants** — `A@haiku`, `D@haiku`, `D@glm`, `D@deepseek-v4-flash`,
`D@qwen3.7-flash`, `D@gemini-3.1-flash-lite` — each with a concrete `results_path`. 108 traces.

Dimensions are exactly `("reasoning", "evidence", "trajectory", "professional")`
(`judge_slate.JUDGE_DIMENSIONS`, `rubric.py`, `eval/judges/rubrics.md`, `rubric_version cfda9f8cc401`).

**The non-candidate-family policy lives in `dealpoint/eval/judge_slate.py`** and nowhere else:
`CANDIDATE_FAMILIES` (11 families a judge may never come from), `JUDGE_TRIO`, `SPARE_JUDGE`,
`PROVIDER_PREFIX_FAMILY`, `_family_for_model_id`. The **verified** judges are in
`data/reports/judge_slate.json`: `mistralai/mistral-small-3.2-24b-instruct` (Mistral),
`nvidia/nemotron-3-super-120b-a12b` (NVIDIA), `bytedance-seed/seed-2.0-mini` (ByteDance).
`DEEPEVAL_MODEL`'s default must be **resolved from these**, never hardcoded (§4).

**Human scores: `data/eval/calibration/human_scores.jsonl` is 0 bytes.** n = 0. Every human
comparison in the DeepEval report is `"pending"`, and `judges.json` already records
`human_calibration: "pending"`. Do not fabricate a number and do not re-label anything.

### Spend and metering

`dealpoint/llm/client.py::OpenRouterClient` writes one ledger line per call to
`data/results/spend_ledger.jsonl`, merging `self.context` into the row. Set `client.context` before
each call to attach `milestone_tag` (from the constructor) plus `purpose`, `case_id`, etc. — this is
how M6 tagged probes, and `tests/test_spend_m6.py` asserts on exactly that shape. `FakeClient` never
touches the network or the ledger.

`dealpoint/eval/spend.py`: `read_ledger`, `realized_usd`, `realized_by_tag`, `fetch_prices` (live
OpenRouter with a pinned fallback), `per_case_usd(arm, model)`, `estimate(sweep)`,
`assert_within_cap(est_usd)`, `SpendCapError`, `SWEEP_DEFS`.

Workhorse: `dealpoint.config.WORKHORSE_MODEL = "z-ai/glm-5.3-flash"` (prompt `7.5e-8`, completion
`2.5e-7` per token). This is the `RAG_SYNTH_MODEL` default.

### Braintrust adapter as it stands (`dealpoint/eval/braintrust_adapter.py`)

`PROJECT = "dealpoint-eval"`. `SCORE_NAMES` (exactly 6), `METADATA_KEYS` (exactly 6:
`arm, model, index_version, skill_version, git_sha, case_set`), `JUDGE_SCORE_NAMES` (4, kept
separate), `experiment_name(arm, model, index_version, git_sha7)` →
`"{arm}-{model_slug}-{index_version}-{git_sha7}"`, `experiment_metadata(...)`,
`load_braintrust_key()`, `braintrust_available()`, `run_eval(rows, ...)`, `run_judge_eval(rows, ...)`,
`push_datasets(sets, version)`, `_record_braintrust_run(entry)` →
`data/reports/braintrust_runs.json`.

`braintrust` is imported **lazily inside every function**, never at module top level. Keep that rule
— `tests/test_braintrust_adapter.py::test_module_works_with_braintrust_unimportable` reloads the
module with `sys.modules["braintrust"] = None` and asserts it degrades cleanly.

### Test conventions

`tests/conftest.py` has one autouse fixture (`_fast_retry_backoff`) and the `dataset_available` /
`require_dataset` pair that **skips cleanly** when `data/raw/` is absent. Every test file sets one
`pytestmark = pytest.mark.gate_mN`. Markers `gate_m0`…`gate_m7`, `needs_network`, `needs_model` are
all already declared in `pyproject.toml` — **`gate_m7` exists; you do not need to add it.**

---

## 2. Dependency groups, versions, disk guard

**`pyproject.toml`** — add to `[project.optional-dependencies]`:

```toml
rag-lab = ["llama-index-core", "llama-index-retrievers-bm25"]
deepeval = ["deepeval"]
```

Install with `uv pip install -e ".[rag-lab]"` then `-e ".[deepeval]"`, running the §0.4 guard around
each. Do **not** add them to the default `dependencies` — the offline suite must stay green without
them, and every import of either framework must be lazy (inside a function), matching the existing
`openai`/`qdrant`/`braintrust` convention.

**Justfile** — add under a new `# ── dealpoint (M7a framework roles) ─` heading:

```
# M7a: LlamaIndex RAG lab -- native eval on canonical dev retrieval + synthetic-query study
li-rag-eval *ARGS:
    uv run python -m dealpoint.rag_lab.report "$@"

# M7a: generate the frozen synthetic dev query set (METERED, cheap workhorse)
synth-queries *ARGS:
    uv run python -m dealpoint.rag_lab.synthetic "$@"

# M7a: DeepEval independent cross-check on the judged subset (METERED)
deepeval-crosscheck *ARGS:
    uv run python -m dealpoint.eval.deepeval_adapter "$@"

# M7a: recreate every Braintrust artifact from Git/local sources; idempotent
braintrust-sync *ARGS:
    uv run python -m dealpoint.eval.braintrust_sync "$@"

# M7a: run the six BTQL investigations and save results
btql *ARGS:
    uv run python -m dealpoint.eval.btql "$@"

# milestone 7 acceptance gates only, offline
gate-m7:
    uv run pytest -m "gate_m7 and not needs_network and not needs_model" -q
```

**`data/reports/framework_versions.json`** — write from `importlib.metadata.version` for
`llama-index-core`, `llama-index-retrievers-bm25`, `deepeval`, `braintrust`, `qdrant-client`,
`fastembed`, `bm25s`, `openai`, `pydantic`, plus `python`, `git_sha7`. Every M7 report embeds this
dict under a `framework_versions` key, and every Braintrust experiment carries it in metadata (the
spec names `framework_versions` in the common metadata list).

---

## 3. LlamaIndex RAG lab — `dealpoint/rag_lab/`

New package: `__init__.py`, `adapters.py`, `evaluate.py`, `synthetic.py`, `report.py`.

**Framework boundary, stated in the package docstring and enforced by review:** LlamaIndex owns
retriever composition, framework-native RAG evaluation, and the synthetic-query study. It does not
own the parser, canonical sections, MAUD offsets, the agent loop, the tools, the answer schema or
the canonical scorers. **MAUD gold-span overlap remains benchmark truth.** LlamaIndex node ids and
qrels never replace it and never drive selection.

### 3.1 `adapters.py` — wrap the frozen candidates, do not rebuild them

```python
def chunk_to_node(chunk: Chunk) -> "TextNode"      # id_=chunk.chunk_id, text=chunk.text,
                                                    # metadata={agreement_id, section_ref, start, end}
class ProjectRetrieverAdapter(BaseRetriever):       # llama_index.core.retrievers.BaseRetriever
    def __init__(self, retriever: Retriever, agreement_id: str, k: int) -> None: ...
    def _retrieve(self, query_bundle) -> list[NodeWithScore]: ...
def build_li_retriever(config: RetrieverConfig, *, dense, sparse, variants_by_query, scorer,
                       agreement_id: str, k: int) -> "ProjectRetrieverAdapter"
```

`_retrieve` calls `self._retriever.search(self._agreement_id, query_bundle.query_str, k=self._k)`
and wraps each `Chunk` in `NodeWithScore(node=chunk_to_node(c), score=1.0 / rank)` — the project
retrievers return a *ranked list*, not scores, so a reciprocal-rank score preserves order honestly
without inventing a similarity number. Say that in the docstring.

Import `llama_index` **inside** the functions/`__init__`, never at module top level. `BaseRetriever`
is a class you must subclass, so import it inside a small factory rather than at import time:

```python
def _base_retriever_cls():
    from llama_index.core.retrievers import BaseRetriever
    return BaseRetriever
```
and build the adapter class lazily, or guard the module with a `TYPE_CHECKING` import plus a runtime
subclass created in the factory. Either is acceptable; the requirement is that
`import dealpoint.rag_lab.adapters` succeeds with LlamaIndex absent.

**Assert identity before evaluating.** `evaluate.py` must call
`dealpoint.corpus.retrievers.index_version()` and `dealpoint.corpus.chunks.chunk_version()` and
compare them against `data/reports/versions.json` (`e2b4a2b97561` / `8e5e8ba56765`), raising if they
differ. Record both in the report. This is the spec's "same chunks and indexes
(`chunk_version`/`index_version` asserted)" requirement.

### 3.2 `evaluate.py` — LI metrics beside project metrics, on all 58 dev cases

```python
def gold_bearing_chunk_ids(case: dict, chunks: list[Chunk]) -> list[str]
```
Uses the canonical rule verbatim: `overlap_chars((c.start, c.end), g) >= MIN_GOLD_OVERLAP_CHARS` for
any `g in gold_ranges(case)`. Import `overlap_chars`/`gold_ranges`/`MIN_GOLD_OVERLAP_CHARS` from
`dealpoint.eval.scorers` — **do not reimplement the geometry.** These are the `expected_ids`.

```python
def evaluate_retriever_li(cases, config, *, dense, sparse, ..., k) -> dict   # hit_rate, mrr, n, per_case
def evaluate_retriever_obj(cases, config, *, dense, sparse, ..., k) -> dict  # obj/gold_span_hit@k, obj/gold_span_mrr
```

The project-side numbers must come from the **existing** `dealpoint.eval.tournament.evaluate_config`
(reuse it; it already computes `hit_at_5`/`hit_at_10`/`mrr` with `_first_hit_rank`), not a fresh
implementation. Name them `obj/gold_span_hit@5`, `obj/gold_span_hit@10`, `obj/gold_span_mrr` in the
report to match the spec's namespace.

LlamaIndex side: `RetrieverEvaluator.from_metric_names(["hit_rate", "mrr"], retriever=adapter)`,
then per case `await evaluator.aevaluate(query=..., expected_ids=[...])` (or the sync
`evaluate(...)`), reading `result.metric_vals_dict`. **Verify this API against the installed
`llama-index-core` version before writing the loop** — the package is not installed yet and the
0.14.x surface is what this plan assumes; if `RetrieverEvaluator` or `from_metric_names` has moved,
adapt and record the actual import path in the report's `framework_versions` note rather than
silently substituting a hand-rolled metric. If LlamaIndex's `RetrieverEvaluator` genuinely cannot be
used, that is a finding to report, not to paper over.

Queries: the canonical query per question,
`dealpoint.data.questions.QUESTION_BY_ID[case["question_id"]].canonical_query` — the same one the
tournament uses. Retrieval is per document: `resolve_document_id(case)`.

**Comparison block**, per the spec:
- per retriever, both metric sets side by side;
- **ranking-order agreement**: Spearman over the six retrievers' ranks under LI-`hit_rate` vs
  `obj/gold_span_hit@5` (and mrr vs mrr). Use the existing
  `dealpoint.eval.agreement.spearman(xs, ys)` — do not write a new one;
- **disagreement cases in both directions** — LI hit / MAUD miss and MAUD hit / LI miss — each with a
  `cause` label drawn from a fixed vocabulary: `chunk_identity`, `partial_overlap`,
  `duplicate_relevant_chunks`, `section_boundary`, `reranking`. Assign the label by a documented
  deterministic rule (e.g. *`partial_overlap`* when the top chunk overlaps a gold span by
  `0 < overlap < 50` chars; *`duplicate_relevant_chunks`* when ≥2 chunks in the case's chunk list
  share ≥50-char gold overlap; *`section_boundary`* when the gold span straddles two chunks'
  `section_ref` values; *`reranking`* when the config has a `rerank_model` and the pre-rerank fused
  rank was a hit; *`chunk_identity`* otherwise). Write the rule text into the JSON so it is auditable;
- **1–2 worked examples with real case ids**, quoting the query, the top chunk ids, the gold ranges
  and why the two metrics disagree.

Because `expected_ids` are the *same* chunk ids the project rule marks as gold-bearing, LI hit_rate
and `obj/gold_span_hit@k` should agree almost everywhere. **A near-total agreement is a valid and
interesting result** — report it plainly (it demonstrates the adapters are faithful) rather than
manufacturing disagreements. If disagreement count is 0 in a direction, say so explicitly with the
reason (the two definitions coincide by construction at k, differing only where `k` truncation or
rerank ordering bites).

### 3.3 `synthetic.py` — the frozen synthetic query set (METERED, small)

Scope: **≤ 2 questions per gold-bearing *dev* chunk only.** Distinct gold-bearing chunks across the
58 dev cases number ~130–148 (148 (chunk, case) pairs measured; dedupe by `chunk_id`). Generate with
LlamaIndex's question generator (`llama_index.core.evaluation.generate_question_context_pairs`, or
`RagDatasetGenerator` — verify what the installed version exposes), model from `RAG_SYNTH_MODEL`
env var defaulting to `dealpoint.config.WORKHORSE_MODEL` (`z-ai/glm-5.3-flash`).

Route the LLM through the project's own `OpenRouterClient` so every call lands in the ledger with
`milestone_tag="m7a"` and `purpose="synthetic_query"`. LlamaIndex wants an `LLM` object; write a thin
`llama_index.core.llms.CustomLLM` subclass in `synthetic.py` that delegates `complete()` to
`OpenRouterClient.chat(...)` and sets `client.context = {"purpose": "synthetic_query", "chunk_id":
…}` before each call. **A LlamaIndex-native OpenAI LLM pointed at OpenRouter would bypass the ledger
— do not do that.**

Order of operations, mandatory (spec, "Budget and disk"):
1. Count target chunks; estimate calls and USD from `fetch_prices()` and a measured token shape.
2. Run a **≤ 3-chunk calibration sample**; record projected vs realised.
3. `assert_within_cap(est_usd)`; proceed only if both the global cap and M7a's own absolute hold.
4. Generate; write `data/eval/synthetic_dev_queries.jsonl`, one row per question:
   `{query_id, question, chunk_id, agreement_id, case_id, question_id, generator, generator_version,
     prompt_hash, model, usd, ts}`.
5. Write a sidecar provenance block into `li_rag_eval.json` under `synthetic`: generator id/version,
   prompt hash, model, n_chunks, n_queries, calibration (projected vs realised), total usd, and the
   file's sha256.

Then evaluate **every M3 candidate** on this set with the same LI + obj metric pair, using the
generating chunk id as the single `expected_id`.

Guardrails to state in code and report: it never touches the test set (only
`load_case_set("dev")`), never changes the frozen M3 winner, never becomes benchmark truth, never
drives selection. It answers exactly one question — *does the frozen winner stay strong under a
different query distribution?* — answered with a yes/no sentence plus the ranking under both
distributions.

### 3.4 `report.py` → `data/reports/li_rag_eval.json` + `.md`

JSON keys: `schema_version`, `generated_from` (chunk_version, index_version, dataset_version,
git_sha7, n_cases, case_set), `framework_versions`, `retrievers` (per config: `li` and `obj` metric
blocks), `ranking_agreement` (spearman values + the two rank orders), `disagreements` (list with
`case_id, config, direction, cause, detail`), `cause_rule`, `worked_examples`, `synthetic`
(provenance + per-config metrics + the winner-stability verdict), `frozen_assertions`
(the version hashes checked, and the sha256 of `data/eval/test_subset_v1.json` and
`data/reports/tournament.json` proving neither moved), `decisions`, `brief_differences`.

Markdown regenerable from the JSON alone — follow `tournament.render_markdown`'s shape.

---

## 4. DeepEval — `dealpoint/eval/deepeval_adapter.py`

A **thin adapter**, not a second harness. No new trace store; Braintrust remains primary
observability. Import `deepeval` lazily.

**Environment hygiene:** set `DEEPEVAL_TELEMETRY_OPT_OUT=1` (and `ERROR_REPORTING=0`) in the module
before importing deepeval, and never call `deepeval login` / require `CONFIDENT_API_KEY`. DeepEval
ships posthog and will otherwise phone home; the offline suite must not depend on network.

### 4.1 Evaluator model resolution — from config, never hardcoded

```python
def resolve_evaluator_model(env_value: str | None = None) -> dict:
    """DEEPEVAL_MODEL if set; else the first model in data/reports/judge_slate.json's
    verified judges whose family is not in judge_slate.CANDIDATE_FAMILIES.
    Returns {model, family, source, price_basis}. Raises if nothing qualifies.
    """
```
Read `JUDGE_SLATE_PATH` (`data/reports/judge_slate.json`) for the *verified* trio, and
`judge_slate.CANDIDATE_FAMILIES` / `_family_for_model_id` for the policy. If `DEEPEVAL_MODEL` is set
explicitly, still check it against `CANDIDATE_FAMILIES` and record the check. Record the resolved
provider/model/family/source in **both** the results rows and the report. Today this resolves to
`mistralai/mistral-small-3.2-24b-instruct` (Mistral) — but resolve it, do not write it down.

Wrap the model in a `DeepEvalBaseLLM` subclass delegating to `OpenRouterClient` with
`milestone_tag="m7a"`, `purpose="deepeval"`. Same reasoning as §3.3: deepeval's built-in model
clients would bypass the ledger.

### 4.2 Subset and mapping

Subset = **the M5 judged subset × already-judged variants**: the 18 case ids and the 6 variants in
`data/eval/judged_subset.json` (`A@haiku`, `D@haiku`, `D@glm`, `D@deepseek-v4-flash`,
`D@qwen3.7-flash`, `D@gemini-3.1-flash-lite`) — 108 traces, trace-for-trace aligned with objective,
judge and (pending) human. Load rows from each variant's `results_path`.

```python
def row_to_test_case(row: dict, case: dict, doc: Document) -> "LLMTestCase"
```
- `input` = the question gloss + options (use `resolve_question(case)`, same as `blinding.py`);
- `actual_output` = the finding's answer + rationale (or an explicit "no finding produced" string
  when `finding is None`, mirroring how the rubric treats it);
- `retrieval_context` = the reconstructed retrieved text per trajectory step — **reuse
  `dealpoint.eval.blinding._build_trajectory_step`'s reconstruction** rather than writing a second one;
- `tools_called` = the trajectory's tool names in order, with args;
- `expected_tools` = derived from the case's `required_evidence` via
  `dealpoint.eval.scorers.parse_required_evidence` → `lookup_defined_term` when a defined term is
  named, plus `search_agreement` (the always-applicable first step, per skill rule 1). Document the
  derivation in the report; it must be a pure function of the case, never of the output.

**This mapping function must be pure and framework-free** (returning a plain dict that a tiny shim
converts into `LLMTestCase`), so the offline gate test can assert on the mapping without deepeval
installed. That is the single most important design decision in this section.

### 4.3 Metrics

`task_completion`, `tool_correctness`, `argument_correctness` (where applicable — only cases whose
`required_evidence` names a term give a checkable argument), `step_efficiency` (DeepEval-native if
the installed version has one; otherwise a **documented `GEval`** definition whose criteria string is
written verbatim into the report and hashed). No generic RAG metrics weaker than MAUD truth — say so
in the report and name the ones deliberately omitted (faithfulness, answer relevancy, contextual
precision/recall) with the reason: gold-span overlap already answers the same question with expert
labels.

### 4.4 Comparisons and conclusion

- DeepEval ↔ deterministic: `grounded_accuracy`, `required_evidence_met`, `tool_calls` vs
  `MAX_TOOL_CALLS=8`;
- ↔ calibrated judge dimensions: per-trace mean-of-judges per dimension, from
  `data/eval/judge_scores.jsonl` keyed `(case_id, variant_id)`; correlate with
  `dealpoint.eval.agreement.spearman` / `quadratic_weighted_kappa` / `correlate_with_grounded_accuracy`
  — all already exist;
- ↔ human: **n = 0**, so every human cell is `"pending"` with `n: 0` reported explicitly;
- step efficiency ↔ human trajectory quality: `"pending"` for the same reason;
- **surface disagreement cases** with case ids and both scores.

Conclude with exactly one of `KEEP_CORE_DIAGNOSTIC` / `KEEP_OPTIONAL_ANALYSIS` /
`REMOVE_NO_ADDED_SIGNAL`, in a `classification` field, with a `classification_rationale` that cites
the measured correlations. **Derive it from the evidence you actually get** — if DeepEval's metrics
track `grounded_accuracy` and the judge dimensions closely enough to add no independent signal,
`REMOVE_NO_ADDED_SIGNAL` is the honest and acceptable answer, and the spec's own no-bloat rule
expects tested-and-rejected features to be documented rather than hidden.

Output: `data/reports/deepeval_crosscheck.json` + `.md`, including `resolved_evaluator`,
`framework_versions`, `subset` (case ids, variants, hashes), `metrics`, `comparisons`,
`disagreements`, `classification`, `spend`, `decisions`, `brief_differences`.

**Metered discipline:** estimate → ≤3-trace calibration sample → compare → `assert_within_cap` →
run. 108 traces × 1 call at Mistral-small's price is roughly $0.02–0.05; a calibration sample will
tell you exactly. If the projection exceeds the remaining envelope, cut to a documented deterministic
prefix of the judged subset's `rank` ordering (the `rank` field exists precisely for this) and say so.

---

## 5. `just braintrust-sync` — `dealpoint/eval/braintrust_sync.py`

One command, **idempotent**, recreating everything from Git/local sources. The walkthrough must never
depend on a permanent trace id: record stable case ids and experiment names alongside any trace id.
Starter tier retains logs 14 days — that is precisely why re-creatability is the requirement.

Structure the module as **pure mapping functions + a thin client-calling driver**, so the offline
tests drive the whole mapping through a fake client:

```python
def score_namespace(name: str) -> str                     # -> "obj/grounded_accuracy" etc.
def common_metadata(row, *, stage, extra=None) -> dict
def dataset_rows(set_name: str) -> list[dict]
def experiment_plan() -> list[dict]                       # name, stage, tags, metadata, rows
def log_hierarchy(row, case, doc) -> dict                 # nested span tree, no model calls
def representative_cases() -> dict                        # rule + selections
def review_set() -> list[dict]                            # the 12 blinded traces
def assert_score_budget(plan) -> None
def sync(client, *, dry_run: bool = False) -> dict        # the driver
```

### 5.1 Namespaces and score budget

Score names are namespaced `obj/`, `li/`, `judge/`, `deepeval/`, `procedure/`. Common metadata on
**every row**: `stage, arm, reasoning_type, retriever, case_type, model, index_version,
skill_version, git_sha, framework_versions` (see §0.5 — row-level, not just experiment-level).

`assert_score_budget` enforces: **M7 experiments ≤ 12 scores/case on subsets ≤ 60 cases; M4/M6 sweeps
keep six.** Raise a clear error naming the experiment and count. Secondary diagnostics go to
metadata. This function is directly unit-tested (§7).

Leave `braintrust_adapter.SCORE_NAMES` at six and untouched — `tests/test_braintrust_adapter.py`
asserts `len(SCORE_NAMES) == 6` and the exact tuple. The new budget lives in the sync module.

### 5.2 Experiments

Named/tagged by stage:
- RAG: `rag-m3-dense`, `rag-m3-bm25`, `rag-m3-hybrid`, `rag-m3-hybrid-rerank`, fusion variants where
  retained, `rag-m7-li-crosscheck`, `rag-m7-synthetic` — tagged `stage=rag`;
- agent systems: the existing A–D experiment names from `braintrust_runs.json`, tagged `stage=agent`;
- evaluation systems: `judge-…` (existing `judged-…` names), `deepeval-…`, tagged `stage=eval`;
- economics: the M6 Pareto experiments, tagged `stage=economics`.

**Existing experiments are re-logged from canonical rows only if needed for tags — never re-run.**
Every row is a replay of a stored result row; no model is ever called by this command.

### 5.3 Logs with a readable hierarchy — replayed, zero model calls

`case` → `agent` → `search_agreement` (retrieval stages as child spans) → `lookup_defined_term` /
`get_section` → `final_answer` → `scoring` spans carrying `obj/`, `judge/`, `deepeval/` provenance.
Build with `braintrust.start_span` / `Logger` nesting; the payload comes entirely from the stored
`record.trajectory` and the reconstruction described in §4.2.

**Representative case selection — by rule, documented, not by convenience.** Six categories:
successful direct; retrieval-rescue; defined-term/cross-reference; inefficient trajectory; wrong
answer; abstention/counterfactual. Write the selection rule as a deterministic predicate over the
stored rows, e.g.:

| category | predicate (over a canonical result row) |
|---|---|
| successful direct | `grounded_accuracy is True` and `reasoning_type == "direct"` and `tool_calls <= 2` |
| retrieval-rescue | `grounded_accuracy is True` and `gold_first_rank` is None or `> 1`, and ≥2 `search_agreement` steps |
| defined-term / cross-ref | `required_evidence` is not None and a `lookup_defined_term` or `get_section` step exists |
| inefficient trajectory | `cap_hit is True`, or `tool_calls >= 6` with `grounded_accuracy is not True` |
| wrong answer | `answer_correct is False` |
| abstention / counterfactual | `case_set == "counterfactual"` and `record.status == "ABSTAINED"` |

Ties broken by ascending `sha256(f"{SEED}:{case_id}")`, the same seeded-hash convention
`dealpoint/eval/subset.py::_seeded_key` already uses. Write rule + selections to
`data/reports/representative_cases.json` with `case_id`, `variant_id`, `category`, `experiment_name`,
and (if a send happened) the trace id — **case id and experiment name first, trace id last and
optional**.

### 5.4 Datasets, scorers, prompts, parameters, tools

- **Datasets** (mirrors, never hand-edited): dev, test, counterfactual, judged/calibration subset,
  synthetic-query set. Metadata `question_id, agreement_id, reasoning_type, case_type,
  source_hashes`. Extend the existing `push_datasets` rather than forking it.
- **Scorers** as Braintrust functions that **import the canonical implementations** from
  `dealpoint.eval.scorers`: `grounded_accuracy`, `answer_correct`, `citation_verbatim`,
  `citation_gold_overlap`, `required_evidence_met`, `skill_adherence`. Judge and DeepEval provenance
  stay in distinct namespaces. Pass/fail thresholds only where meaningful (booleans have a natural
  one; `skill_adherence` is a fraction and should not get an invented threshold).
- **Prompts**: base agent system instructions (`dealpoint/agent/prompts.py`), arm-D skill injection
  (`skills/ma-deal-point-review/SKILL.md` + the applicable playbook, via
  `dealpoint.agent.skill.skill_block`), and the calibrated judge rubric (`eval/judges/rubrics.md`) —
  each with its source hash (`skill_version f8d255cc169b`, `rubric_version cfda9f8cc401`).
  **Canonical ownership stays in Git**; Braintrust holds a mirror.
- **Parameters**: one compact versioned schema of non-secret runtime parameters — `arm`,
  `retriever config`, `top_k`, `max_tool_calls`, `skill version`, `model alias`, `dataset/split`,
  `mode` — built by *exposing existing config* (`ARMS`, `ARM_C_RETRIEVER`, `RETRIEVER_DEFAULT_K`,
  `MAX_TOOL_CALLS`), not by re-declaring it.
- **Tools**: expose `search_agreement` / `get_section` / `lookup_defined_term` as Braintrust tools
  **only if** thin adapters can run in Braintrust's function environment without duplicating the
  index/data layer. They almost certainly cannot — the tools need the 68 MB Qdrant index, the
  64 MB derived canonical corpus and fastembed weights. **Verify, then document the decision with
  the measured reason. Do not ship placeholder stubs.** A documented "no" is a passing outcome here.

### 5.5 Human review preparation — the 12-trace blinded calibration set

Rule: span reasoning types, mix successes and failures, ≥ 1 abstention, ≥ 2 judged variants.
Methodology must extend to ~30 later **unchanged** — so implement it as
`review_set(n: int = 12)` with a deterministic ordering, not as a hand-picked list of twelve.

Reuse `dealpoint/eval/blinding.py::build_packet` / `render_packet_text` — the blinding is already
frozen and the same packets the judges saw. Sync them so the engineer can score against the frozen
M5 rubric fields in Braintrust Review. **Existing local human scores remain canonical and are
displayed, never re-labelled** — today there are none (`human_scores.jsonl` is empty), so the synced
set carries `human_score: null` and the report says `n = 0, pending`.

### 5.6 Idempotency and the offline path

`sync()` takes an injected client. With `--dry-run` (or a fake client) it performs the entire mapping
and returns the plan without touching the network — that is what the gate tests drive. Re-running
against the live project must not duplicate: key datasets by name, experiments by name, and
prefer update/upsert over blind insert. Record the run in
`data/reports/braintrust_sync.json`: what was created vs found-existing, counts, score-budget check
results, timestamps, `framework_versions`.

Gate the live path on `braintrust_available()` and skip cleanly (never fail) when the key is absent,
matching `tests/test_braintrust_judge_eval.py`'s convention.

---

## 6. BTQL investigations — `dealpoint/eval/btql.py`

Implement **and actually run** six queries via the API, using the verified REST shape from §0.5:

```python
def run_btql(query: str, *, api_key: str, timeout: float = 60) -> dict
    # POST https://api.braintrust.dev/btql, {"query": query}, Bearer auth
```

Remember: every query needs a `from:` **and** one of `select:` / `dimensions:` / `measures:`.

The six investigations:
1. **Retrieval rescue** — cases where a later `search_agreement` surfaced gold that the first missed.
2. **Failure attribution** — `EXECUTION_FAILED` / `CAP_HIT` grouped by model and arm.
3. **Reasoning slices** — `grounded_accuracy` by `reasoning_type` (direct, numeric, structured,
   defined-term, cross-ref, carve-out).
4. **DeepEval disagreement** — traces where `deepeval/` and `obj/` disagree.
5. **Model economics** — cost per correct answer by model.
6. **Trajectory inefficiency** — tool calls vs outcome.

Save results where the product allows, and write **exact reproducible queries + results** to
`docs/braintrust-queries.md` and `data/reports/btql_investigations.json`. Each entry:
`{id, title, question, btql, executed_at, row_count, rows, notes}`. The markdown must be
copy-pasteable — include the `curl` form with the endpoint and a `$BRAINTRUST_API_KEY` placeholder.

If a query returns nothing because 14-day retention has purged the underlying logs, **record that
outcome honestly** (`row_count: 0`, `note: "logs purged by 14-day starter-tier retention; rerun
`just braintrust-sync` first"`) rather than reshaping the question until it returns rows. Run
`just braintrust-sync` before the investigations so there is fresh data to query.

Mark the network-dependent test path `needs_network`; the offline gate test covers query
*construction* and result *parsing* against a recorded fixture.

---

## 7. README results block + tests

**Engineer's instruction (spec, 2026-09-05):** `regenerate_readme` writes the block between the
README markers. Change what it writes to:

- per model, a table of arms with `grounded_accuracy` shown as **"x% (k of n scored)"**;
- plus the **one-line majority-baseline note**;
- **nothing else** — no execution-failure table, no cap-hit line, no v1-vs-v2 table, no residual
  failure bullets, no per-arm verdict sentences.

Those diagnostics **stay, complete and honest**, in `data/reports/four_arm.md` and `four_arm.json`,
which the README links to. Do not delete anything from the reports.

Edit `dealpoint/eval/report.py::_readme_results_block` (lines ~1108–1185). Keep:
- the leading disclosure sentence — it carries `budget-scaled`, `not comparable`, `objective`, which
  `tests/test_readme_results.py` asserts **and** which brief §6 acceptance 8 requires
  (MAUD non-comparability caveat, objective-vs-judged distinction). This is a brief requirement, so
  it survives the trim;
- `report["majority_baseline_note"]["sentence"]` as the single majority-baseline line (it already
  contains both the 0.0% subset figure and the 46.1% full-test context);
- the per-model arm tables, with the cell rendered `f"{pct} ({n_scored} of {n_cases} scored)"`.

Delete from the block: the `failure_rate_table` section, the `Overall EXECUTION_FAILED / CAP_HIT`
line, the `residual_failures` bullets, and the `verdicts` `C_to_D` sentence.

Ensure the prose immediately after `<!-- END RESULTS -->` links `data/reports/four_arm.md` **and**
`data/reports/four_arm.json` explicitly — the README is an introduction for readers who do not know
the project; the reports are the record.

Then **rewrite `tests/test_readme_results.py`** to the new shape:
- still asserts the three disclosures in the first sentence;
- asserts the majority-baseline note sentence is present;
- asserts, per (model, arm) in `four_arm.json`, that the percentage and the literal
  `f"({n_scored} of {n_cases} scored)"` appear;
- **asserts absence**: `"EXECUTION_FAILED rate by arm"` not in block, `"v1 -> v2"` not in block,
  `"Residual failure"` not in block, and no `verdicts[*]["C_to_D"]["sentence"]` in block;
- asserts the README links to `data/reports/four_arm.md`;
- keeps the existing clean-skip when `four_arm.json` is absent.

Leave `pytestmark = pytest.mark.gate_m4` (it is the M4 artifact's contract) **and add**
`pytest.mark.gate_m7` alongside it, so this milestone's gate also covers the change:
`pytestmark = [pytest.mark.gate_m4, pytest.mark.gate_m7]`.

Regenerate with `just report`. **Do not hand-edit the README block** — it must come from the
generator, and the test compares the two.

`<!-- BEGIN PARETO -->` / `<!-- END PARETO -->` and `tests/test_readme_pareto.py` are untouched.

---

## 8. Spend plumbing

**`dealpoint/config.py`** — add an M7a section following the existing pattern:

```python
# --- Milestone 7a: framework roles (LlamaIndex, DeepEval, Braintrust sync) ---
M7A_MILESTONE_TAG = "m7a"
M7A_TARGET_USD = 1.00     # soft target -- reported, not gated
M7A_MAX_USD = 2.50        # absolute per-milestone limit -- enforced
GLOBAL_ENVELOPE_USD = 6.00
RAG_SYNTH_MODEL_ENV = "RAG_SYNTH_MODEL"
DEEPEVAL_MODEL_ENV = "DEEPEVAL_MODEL"
LI_RAG_EVAL_JSON_PATH = REPORTS_DIR / "li_rag_eval.json"
LI_RAG_EVAL_MD_PATH = REPORTS_DIR / "li_rag_eval.md"
SYNTHETIC_DEV_QUERIES_PATH = EVAL_DIR / "synthetic_dev_queries.jsonl"
DEEPEVAL_CROSSCHECK_JSON_PATH = REPORTS_DIR / "deepeval_crosscheck.json"
DEEPEVAL_CROSSCHECK_MD_PATH = REPORTS_DIR / "deepeval_crosscheck.md"
BTQL_INVESTIGATIONS_PATH = REPORTS_DIR / "btql_investigations.json"
REPRESENTATIVE_CASES_PATH = REPORTS_DIR / "representative_cases.json"
BRAINTRUST_SYNC_PATH = REPORTS_DIR / "braintrust_sync.json"
FRAMEWORK_VERSIONS_PATH = REPORTS_DIR / "framework_versions.json"
M7A_DISK_GUARD_PATH = REPORTS_DIR / "m7a_disk_guard.json"
M7A_REVIEW_SET_N = 12
M7A_MAX_SCORES_PER_CASE = 12
M7A_MAX_SUBSET_FOR_WIDE_SCORES = 60
```

**`dealpoint/eval/spend.py`** — add the `m7a` sweep so `budget m7a` resolves (§0.2). Two legs, both
priced by the `tokens_per_call` shape (neither is agent-case shaped, so the ledger branches would
return nonsense — same reasoning the `judges` leg already documents):

```python
"m7a": {
    "legs": [
        # synthetic-query generation: <=2 questions per gold-bearing dev chunk
        # (~148 gold-bearing (chunk, case) pairs measured; deduped by chunk_id),
        # one generation call per chunk at the cheap workhorse.
        {"arms": [], "models": [WORKHORSE_MODEL], "n_cases": 148,
         "tokens_per_call": {"input": 2200, "output": 220}},
        # DeepEval: one evaluator call per (trace, metric) over the 18-case
        # judged subset x 6 already-judged variants = 108 traces, 4 metrics.
        {"arms": [], "models": [<resolved evaluator model id>], "n_cases": 108 * 4,
         "tokens_per_call": {"input": 3400, "output": 160}},
    ],
},
```
Resolve the evaluator model id from `judge_slate.json` at module import (with a safe fallback to the
first `JUDGE_TRIO` entry if the file is absent) so the estimate and the run cannot drift.

**`tests/test_spend_m7.py`** (new, `gate_m7`):
- `estimate("m7a")["est_usd"] > 0` and `realized_usd() + est <= cap_usd()`;
- every `milestone_tag == "m7a"` ledger row names a model in
  `{WORKHORSE_MODEL} ∪ {verified judges} ∪ {slate models}` — **no router endpoints**
  (`openrouter/auto`, `openrouter/fusion` explicitly excluded);
- every non-case-run m7a row carries a `purpose` in `{"probe", "synthetic_query", "deepeval"}`;
- `realized_usd() <= 6.00` (global, ledger-asserted);
- m7a's own tag total `<= 2.50`, and the report records whether the $1.00 soft target held;
- all of these `pytest.skip` cleanly when no m7a rows exist yet, matching `test_spend_m6.py`.

Plus the §0.3 scoping edit to `tests/test_spend_m6.py`.

---

## 9. Tests — `gate_m7`, all offline, all fast

New files, each `pytestmark = pytest.mark.gate_m7` (add `gate_m4` alongside only in
`test_readme_results.py`):

**`tests/test_rag_lab_adapters.py`**
- `gold_bearing_chunk_ids` agrees with the tournament's `_first_hit_rank` on synthetic chunks/spans:
  a chunk overlapping by exactly 50 chars is a hit, 49 is not, and a case with no gold spans yields
  `[]`.
- `chunk_to_node` preserves `chunk_id` as `id_` and carries start/end/section_ref in metadata.
- The adapter returns nodes in the retriever's order with monotonically non-increasing scores, driven
  by a stub retriever (no Qdrant, no network) — mirroring `test_braintrust_adapter.py`'s
  `StubRetriever`.
- `pytest.importorskip("llama_index.core")` guards only the tests that genuinely need the framework;
  the mapping tests run unconditionally.

**`tests/test_rag_lab_eval.py`**
- Ranking-order agreement: Spearman over two hand-built rank vectors returns the known value (reuse
  `dealpoint.eval.agreement.spearman`).
- Disagreement classification: one synthetic example per `cause` label lands on the right label.
- Version assertion raises when `chunk_version`/`index_version` disagree with `versions.json`.
- Report renders from a fixture payload with no I/O beyond a `tmp_path`.

**`tests/test_synthetic_queries.py`**
- The generation plan targets **only** gold-bearing dev chunks and caps at 2 per chunk.
- The frozen JSONL round-trips and carries generator id/version/prompt hash/cost per row.
- **A test asserting the module never reads `test.jsonl`** — mirror `tournament.main`'s refusal:
  assert `TEST_JSONL_PATH` is not opened (monkeypatch `builtins.open` or assert the module's source
  contains no reference to the test path constant, as `tournament.py`'s docstring promises).

**`tests/test_deepeval_adapter.py`**
- `resolve_evaluator_model` picks a non-candidate family from `judge_slate.json`; raises when every
  candidate is in `CANDIDATE_FAMILIES`; honours `DEEPEVAL_MODEL` but still records the policy check.
- `row_to_test_case` mapping (the pure dict form) over a fixture row: input/actual_output/
  retrieval_context/tools_called/expected_tools all correct, including the `finding is None` path.
- `expected_tools` derivation from `required_evidence` for a defined-term case and a `None` case.
- Comparison tables handle `None` without coercion, and human comparisons report `n: 0, "pending"`.
- Classification is one of the three allowed strings.
- All of this **without deepeval installed** — the mapping is framework-free by design.

**`tests/test_braintrust_sync.py`** — the spec's "offline tests with a fake client cover the mapping":
- A `FakeBraintrustClient` recording `init_dataset` / `init_experiment` / `start_span` / `log` calls.
- `sync(fake, dry_run=True)` creates every expected dataset, experiment and prompt exactly once, and
  a second call creates nothing new (**idempotency**).
- **Score-budget assertion**: ≤ 12 scores/case on a ≤ 60-case subset passes; 13 raises; an M4/M6
  sweep plan with 7 scores raises.
- Every logged row's metadata carries all ten common keys (§5.1) — the §0.5 `metadata: {}` trap.
- Score names are namespaced: every name matches `^(obj|li|judge|deepeval|procedure)/`.
- `log_hierarchy` produces the `case → agent → search_agreement → … → final_answer → scoring` nesting
  from a fixture row **and makes no model call** (assert the fake LLM client was never touched).
- `representative_cases` selects one per category by the documented rule, deterministically, and the
  selection is stable across two invocations.
- `review_set(12)` satisfies: ≥1 abstention, ≥2 variants, spans reasoning types; and `review_set(30)`
  works with the **same** rule (the extensibility requirement).

**`tests/test_btql_queries.py`**
- All six queries are present, each with `from:` and one of `select:`/`dimensions:`/`measures:`.
- Result parsing against a recorded fixture response.
- The live execution test is marked `needs_network`.

**`tests/test_m7_artifacts.py`**
- Every artifact the DoD names exists once generated, and each has a `!data/reports/...` line in
  `.gitignore` (a direct guard against §0.1).
- `data/reports/framework_versions.json` records the installed framework versions.
- **Frozen-integrity assertions**: sha256 of `data/eval/test_subset_v1.json`,
  `data/eval/judged_subset.json` and `data/reports/tournament.json` match the values recorded in
  `li_rag_eval.json`'s `frozen_assertions`; `data/reports/versions.json` still reports
  `index_version e2b4a2b97561` and `chunk_version 8e5e8ba56765`; `ARM_C_RETRIEVER["name"] ==
  "hybrid_rrf"`.
- `docs/demo-walkthrough.md` exists, contains the role diagram, and **every artifact path it names
  resolves on disk** (the spec's "walkthrough reproducibility" reviewer check, mechanised).

All tests must skip cleanly rather than fail when a metered artifact has not been produced yet —
the established convention in `test_readme_results.py` and `test_spend_m6.py`.

---

## 10. Docs

### `docs/demo-walkthrough.md` (draft)

5–10 minute technical walkthrough in the engineer's order:
**Datasets → RAG Lab → Agent Systems → Logs trace → Scorers → Review → Eval of Evals → Loop/SQL →
Debugger → Model Economics → Dashboard.**

Use **actual** experiment names, case ids, scorer/prompt versions and observed results — e.g.
`rag-m3-hybrid`, `judged-A-anthropic_claude-haiku-4.5-e2b4a2b97561-e3ee9cc`,
`contract_39__redacted_q05`, `skill_version f8d255cc169b`, `rubric_version cfda9f8cc401`,
`index_version e2b4a2b97561`, the `hybrid_rrf` hit@5 of 0.9138 vs dense 0.8103.

Mark sections needing M7b with `[M7b]`: custom trace view, saved views, dashboard, Loop investigation
thread, Review scoring session, Playground, Topics/Patterns/Debugger.

**Role diagram** (required, present in the file):

```
custom Python  ──▶ benchmark truth      (parser, canonical offsets, MAUD gold spans, scorers)
LlamaIndex     ──▶ RAG lab / RAG eval   (retriever composition, RetrieverEvaluator, synthetic queries)
DeepEval       ──▶ independent agent-eval cross-check (task completion, tool correctness, step efficiency)
Braintrust     ──▶ traces / experiments / comparison (surface, not source of truth)
local reports + Git ──▶ permanent evidence (data/reports/, data/results/, specs/)
```

### `docs/braintrust-queries.md`

The six BTQL investigations: question, exact query, how to run it (`curl` + `bt sql` forms),
and the observed results with the date. See §6.

### `docs/milestones/m7a.md`

Follow the shape of `docs/milestones/m6.md`: goal, implementation summary, an acceptance-criteria
table mapping each DoD checkbox to its evidence (file + line), tests executed, decisions, and
brief-vs-spec differences.

### README

Beyond the results block (§7): add a short "Framework roles" paragraph near "How it was built"
carrying the same role diagram, and link the new reports from "Where to look next"
(`data/reports/li_rag_eval.md`, `data/reports/deepeval_crosscheck.md`, `docs/demo-walkthrough.md`,
`docs/braintrust-queries.md`).

---

## 11. Brief-vs-spec differences to record (do not resolve silently)

The brief is the requirements document and wins on product decisions; where the milestone spec
departs from it under explicit engineer authority, **record the difference in the reports**. Put
these in a `brief_differences` array in both `li_rag_eval.json` and `deepeval_crosscheck.json`,
following the shape `judges.json` and `pareto.json` already use (`{id, topic, difference}`):

1. **DeepEval was a brief exclusion.** Brief §3 lists DeepEval under "Not used" and §4 excludes it
   outright. M7a's authority line amends §4 as of 2026-09-05, and the milestone's own conclusion
   field (`KEEP_CORE_DIAGNOSTIC` / `KEEP_OPTIONAL_ANALYSIS` / `REMOVE_NO_ADDED_SIGNAL`) is the
   mechanism by which that amendment is tested rather than assumed.
2. **LlamaIndex scope widened.** Brief §3 allows LlamaIndex "only behind the `Retriever` interface";
   §4 excludes LlamaIndex agents/workflows. M7a additionally uses it for framework-native evaluation
   and synthetic-query generation. Agents/workflows remain excluded — state that they were not used.
   Note also that M3 measured LlamaIndex and *declined* it (`tournament.json`'s `llamaindex.adopted:
   false`, with the recorded 178 MB reason); M7a adopts it for a different purpose, and the earlier
   decision is superseded for that purpose only, not reversed for retrieval.
3. **Score budget raised.** Brief §2.7 caps Braintrust at ≤ 6 scores/case; M7a permits ≤ 12 on
   subsets ≤ 60 cases (ceiling, not target), with M4/M6 sweeps still at six.
4. **Dual-path tracing risk.** Brief §4 excludes "dual tracing paths". DeepEval ships
   OpenTelemetry and posthog. Record explicitly that **no second trace store is created** — Braintrust
   remains primary observability, DeepEval telemetry is opted out, and DeepEval results land in local
   JSON plus `deepeval/`-namespaced Braintrust scores only.
5. **Scale, inherited.** The judged subset is 18 cases × 6 variants, not brief §2.5's 40 × 6; human
   calibration is n = 0 ("pending"), not the brief's 30 hand-scored traces. Already recorded in
   `judges.json`; restate wherever a DeepEval↔human comparison is reported.
6. **The M6 envelope test scoping** (§0.3) — record as a decision with its reasoning.

---

## 12. Order of work

1. §0.2 `SWEEP_DEFS["m7a"]` + §8 config constants — unblocks the factory's spend gate.
2. §0.1 `.gitignore` allowlist; §0.3 M6 test scoping; §2 dependency groups + disk guard +
   `framework_versions.json`.
3. §7 README block + test rewrite; `just report`. (Cheap, offline, independently verifiable.)
4. §3.1–3.2 LlamaIndex adapters + native evaluation on dev (offline — no metered calls).
5. §5 `braintrust_sync` mapping + §9 fake-client tests (offline).
6. §3.3 synthetic queries — **first metered step**: estimate → ≤3-chunk calibration → cap check → run.
7. §3.4 `li_rag_eval.json`/`.md`.
8. §4 DeepEval — second metered step, same estimate/calibrate/cap discipline.
9. §5 live `just braintrust-sync` (unmetered — replays only, no model calls).
10. §6 BTQL investigations against the freshly synced data.
11. §10 docs, walkthrough, `docs/milestones/m7a.md`.
12. Full verification (§13).

Metered work is steps 6 and 8 only. Everything else is offline or unmetered.

---

## 13. Verification

```bash
uv run pytest -m "gate_m7 and not needs_network" -q        # the milestone gate
uv run pytest -m "not needs_network and not needs_model" -q # full offline suite
uv run ruff check .
uv run pyright
just report                                                 # README block regenerated, not hand-edited
just braintrust-sync                                        # idempotent: run twice, second is a no-op
uv run python -m dealpoint.eval.budget m7a                  # must print JSON and exit 0
```

Judge every command by its **exit status**. Text like `error` or `not found` inside otherwise-passing
output is text, not a failure.

Then confirm each DoD box by hand:

- [ ] Optional groups installed within the disk guard; `framework_versions` recorded
      (`data/reports/framework_versions.json`, `m7a_disk_guard.json`).
- [ ] `li_rag_eval.json`: per-retriever LI hit_rate/mrr vs `obj/` hit@k/mrr on all 58 dev cases;
      ranking-order agreement; disagreement cases with causes; synthetic set frozen with provenance
      and evaluated; frozen winner and test set untouched (hash assertions present and passing).
- [ ] `deepeval_crosscheck.json`: resolved evaluator recorded; metrics on the judged subset ×
      6 variants; comparisons vs deterministic / judge / human(n=0, pending) / required-tool;
      disagreement cases; one classification string.
- [ ] `just braintrust-sync` recreates datasets, experiments (tags/namespaces), replayed
      representative traces, scorers, prompts, parameters, the 12-trace review set; idempotent;
      offline fake-client tests cover the mapping; score-budget assertion (≤12 on ≤60, six elsewhere).
- [ ] `docs/braintrust-queries.md` + `btql_investigations.json` with executed results.
- [ ] `docs/demo-walkthrough.md` draft present, role diagram present, every named artifact resolves;
      README updated.
- [ ] Spend: M7a realised ≤ $1.00 target / ≤ $2.50 absolute; global ≤ $6.00, ledger-asserted.
- [ ] Suite, ruff, pyright green.
- [ ] `claude mcp list` output recorded verbatim; `braintrust` present; **not** re-added; no
      MCP-dependent work attempted.

Reviewer will additionally check: frozen-test integrity, framework boundaries, no duplicate canonical
truth, no duplicate tracing architecture, score-budget compliance, spend compliance, and walkthrough
reproducibility.

---

## 14. Out of scope

M7b (after the Claude Code restart, with MCP): custom trace view, saved views, dashboard, Loop
investigation thread, Review scoring session, Playground, Topics/Patterns/Debugger use, walkthrough
finalisation. Also out: any later milestone, the product UI, the MCP layer, and **any edit to
`specs/grilled-product-brief.md` or `specs/milestones/m7a.md`** — both are read-only requirements.

M7a must checkpoint cleanly so M7b resumes from it and never reruns it: every artifact on disk,
every experiment name and case id recorded locally, nothing depending on a live trace id.

## 15. No-bloat rule

A feature survives only if it strengthens evaluation, clarifies provenance, diagnoses failures,
improves reproducibility, exposes a useful control, or materially improves the demo.
**Tested-and-rejected features are documented, not hidden** — this applies squarely to the Braintrust
tools decision (§5.4) and to a `REMOVE_NO_ADDED_SIGNAL` DeepEval verdict (§4.4), both of which are
legitimate passing outcomes. No new agent runtime; no second tracing backend; no metric weaker than
gold labels unless it adds a distinct diagnostic; no hardcoded judge-model stack.
