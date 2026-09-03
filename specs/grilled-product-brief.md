# Deal-Point Eval — Grilled Product Brief

**Status:** implementation-ready. Settled in a grilling session on 2026-09-03. SSSF builds from this document milestone by milestone; it should not need to make product decisions.

**One-line:** An M&A deal-point review agent over real public merger agreements, built to demonstrate serious modern agent evaluation — deterministic ground truth first, calibrated model judging second, deployment cost/quality evidence third.

**Spine claim:** *Grounded accuracy* — the agent gives the expert-labelled ABA answer **and** cites the provision the experts cited — measured across four system versions that differ by exactly one variable each, and across five models on the best version.

Throughout this brief: **objective truth** = deterministic Python scorers over MAUD expert labels and aligned gold spans. **Secondary** = calibrated LLM-judge panel. The two are never merged into one number.

---

## 1. Product requirements

### 1.1 Task
Given one full public merger agreement (~70–100k tokens) and one of 12 ABA deal-point questions, produce a **finding**: the ABA answer option, 1–3 verbatim citations, and a short rationale — or abstain if the provision is genuinely absent.

This is a **harder task than the published MAUD benchmark** (which supplies the relevant span). README must say so and must state that scores are not comparable to MAUD leaderboard numbers.

### 1.2 Finding schema (public output contract — all arms, all models)
```json
{
  "answer":    "<exactly one ABA option string for this question> | \"ABSTAIN\"",
  "evidence":  [ { "section_ref": "6.3(b)", "quote": "<verbatim from the agreement>" } ],   // 1..3 items; [] only when ABSTAIN
  "rationale": "<= 80 words"
}
```
Internal execution record (never exposed as an answer; keeps operational failure out of legal metrics):
```
status: ANSWERED | ABSTAINED | CAP_HIT | EXECUTION_FAILED
failure_reason: null | "schema_invalid_after_retry" | "api_error" | "tool_error" | ...
trajectory: [ {tool, args, result_ref, chunk_ids/char_ranges, t_ms} ... ]
usage: {input_tokens, output_tokens, cost_usd, wall_ms, tool_calls}
```
Retry policy: schema-invalid output → one retry; API errors → 3 retries with backoff (infra); anything else → `EXECUTION_FAILED`.

### 1.3 The agent (one agent, three tools, hard cap 8 tool calls)
| Tool | Purpose | Implementation |
|---|---|---|
| `search_agreement(query, k=5)` | find candidate chunks in the *current* agreement; callable repeatedly | retriever interface (§3.3); arm-dependent config |
| `get_section(section_ref)` | follow cross-references ("as set forth in Section 6.3(b)") | deterministic lookup in the section map |
| `lookup_defined_term(term)` | return the `"Term" means …` block | deterministic regex over canonical text; fuzzy on term casing/quotes |

Loop exit = structured finding (native tool-calling / structured output via OpenRouter). Hitting the cap → `CAP_HIT`, no answer. Market comparison is **not** a tool.

### 1.4 User workflow (Next.js UI, final milestone)
**Deal page** (one page, read-only, cached by default):
1. Select agreement (real deal name parsed from line 1), question (12), arm (A–D), and model (for arm D results from the Pareto run).
2. Finding: answer, rationale, status badge.
3. Evidence: agreement text with model quotes highlighted; toggle "reveal expert gold span".
4. Trajectory: ordered tool calls; skill-adherence rule checks ticked/crossed.
5. Scores for this case (the six deterministic headline scores + judge scores if this case was judged) and a Braintrust trace link (may be expired; link is supplementary).
6. Market comparison: answer distribution across **all 152** MAUD agreements (expert labels), this deal's gold answer and the model's finding pinned, 3–5 example gold spans from other deals. Labelled **"MAUD expert annotations — not model output."**
7. "Run live (arm D, default model)" button → runs the agent for the selected case, streams/returns the finding, scores it against gold on the spot.

**Experiments page** (`/experiments`, static, generated from `data/reports/`): four-arm comparison table, retrieval tournament table, judge calibration/agreement table, model cost/quality Pareto chart. Must work with Braintrust offline or purged.

### 1.5 API contract (FastAPI, consumed by Next.js)
```
GET  /api/agreements                         -> [{id, name, parties, dated, approx_tokens, in_eval_split}]
GET  /api/agreements/{id}/text               -> {canonical_text, sections:[{ref, title, start, end}]}
GET  /api/questions                          -> [{id, text, category, span_type, options[], reasoning_type}]
GET  /api/cases/{agreement_id}/{question_id}?arm=D&model=<id>
     -> {finding, status, evidence:[{section_ref, quote, start, end}], trajectory, scores, judge_scores|null,
         gold:{answer, spans:[{start,end}]}, braintrust_url|null}
POST /api/run   {agreement_id, question_id}  -> same shape as /api/cases (arm D, default model), <= 90 s
GET  /api/market/{question_id}               -> {distribution:{option:count}, examples:[{agreement, answer, quote}]}
GET  /api/reports                            -> {four_arm, tournament, judges, pareto}   (contents of data/reports/*.json)
```
All offsets are into the **canonical text** (§6.2).

### 1.6 Portable skill
`skills/ma-deal-point-review/SKILL.md` (generic procedure, ≤ ~800 tokens) + `playbooks/<question_id>.md` (senior-associate playbook per deal point, 3–5 lines each, loaded only for the applicable question). Injected as a system-prompt block in arm D only. Same files are what the optional MCP layer exposes later. Every generic rule must be phrased so it maps to a deterministic trajectory check (§2.5, skill adherence).

---

## 2. Evaluation requirements

### 2.1 Case sets (canonical = versioned JSONL in repo, generated deterministically)
| Set | Source | Size | Use |
|---|---|---|---|
| `dev` | 5 agreements × 12 questions | ≤ 60 | iteration, tournament, smoke gate |
| `test` | 15 agreements × 12 questions | ≤ 180 | reported results only; frozen |
| `counterfactual` | 30 redacted-span + 10 out-of-scope | 40 | abstention suite; frozen |

Cases exist only where MAUD has a non-null gold answer for (agreement, question) — hence "≤". Split is at **agreement level**, seeded (`seed=42`), documented. Each case: `case_id, agreement_id, question_id, gold_answer, gold_spans:[{start,end}], span_type, majority_answer, required_evidence` (§A).

Redacted cases: delete the aligned gold fragments cleanly from the canonical text (no marker), rebuild the affected chunks, expected `answer="ABSTAIN"`. Metadata `source_case_id`, `redacted_ranges`. Out-of-scope cases: 10 hand-written questions no merger agreement answers, asked against test agreements, expected `ABSTAIN`.

### 2.2 System versions (arms) — same model, same chunks, one variable per step
| Arm | Loop | `search_agreement` backend | Skill | Isolates |
|---|---|---|---|---|
| A `pipeline-dense` | none: dense top-5 → one LLM call | dense | no | the RAG baseline |
| B `agent-dense` | 3-tool agent | dense | no | **agency** (A→B) |
| C `agent-hybrid-rerank` | 3-tool agent | tournament winner, frozen `index_version` | no | **retrieval quality** (B→C) |
| D `agent-hybrid-rerank-skill` | 3-tool agent | as C | yes | **skill** (C→D) |
Plus a zero-cost **majority-class baseline** row (predict each question's majority answer) reported alongside every accuracy figure.

### 2.3 Retrieval tournament (dev only, LLM-free, before arm C is frozen)
Fixed: section-aware chunks (§3.4), one hand-written **canonical query** per question (§A), optional pre-generated frozen alternate queries for multi-query. Varied: `dense`, `bm25`, `hybrid-rrf`, `hybrid-rrf+rerank`, `hybrid+multiquery-fusion(+rerank)`. Metrics: gold-span hit@5, hit@10, MRR over dev cases (qrels = aligned gold spans). Intermediate stages are instrumented so the report says which component found / promoted the gold span. Winner → arm C; `index_version` = hash of (chunking params, embedding model, retriever config).

### 2.4 Deterministic metrics (objective truth) — all pure Python, run offline, unit-tested
| Group | Metric | Definition |
|---|---|---|
| Retrieval | `gold_seen` | any chunk returned by any tool call overlaps a gold span (≥ 50 chars) |
| | `gold_first_rank` | rank of first gold-overlapping chunk in the first search |
| Classification | `answer_correct` | `answer == gold_answer` (exact string), on `ANSWERED` only; overall + per question; vs majority baseline |
| Citation | `citation_verbatim` | every quote, after canonical normalisation, is a substring of the canonical text |
| | `citation_gold_overlap` | ≥ 1 quote's char range overlaps a gold span by ≥ 50 chars |
| | `fabrication` | any quote fails verbatim |
| **Headline** | **`grounded_accuracy`** | `answer_correct AND citation_gold_overlap` |
| Abstention | `abstain_recall` | on counterfactual set: `status == ABSTAINED` |
| | `false_abstain` | on dev/test: `status == ABSTAINED` where gold exists |
| | `redacted_fabrication` | on redacted cases: any non-verbatim quote (worst failure) |
| Trajectory | `tool_calls`, `cap_hit`, `execution_failed` | from the execution record |
| | `required_evidence_met` | trajectory retrieved the required block (§A), by any tool |
| Skill adherence | `skill_adherence` | fraction of *applicable* SKILL.md rules satisfied by the trajectory (rules listed in §B); computed for all arms |
| Efficiency | `input_tokens, output_tokens, cost_usd, wall_ms` | per case, per arm, per model |

"Correct **No**" ≠ abstention: a question whose gold answer is "No" (e.g. no pandemic carve-out) is an `ANSWERED` case with a citation to the relevant clause.

### 2.5 Judge panel (secondary, calibrated, clearly labelled model-judged)
- **Subset:** 40 stratified test cases (≈3 per question + 4 counterfactual).
- **Variants judged:** arm A @ default model, plus arm D @ each of the 5 Pareto models (default model counted once) = 6 variants → 240 traces.
- **Judges:** 3 heterogeneous models from families **not** in the candidate-agent slate (e.g. GPT + Grok + Qwen if the open-weight agent candidate is DeepSeek). Blinded: arm and model names stripped from the trace.
- **Call shape:** one call per trace returning all four dimensions → 720 calls. Est. $15–40.
- **Dimensions** (five-level anchored rubrics **frozen before any judge runs**, stored in `eval/judges/rubrics.md`): (1) reasoning/analysis quality; (2) evidence sufficiency; (3) trajectory quality *beyond* deterministic adherence (search strategy, wasted calls, stopping); (4) professional answer quality.
- **Calibration (mandatory):** 30 traces hand-scored by the author on the same rubric, blinded, ~5 per variant. Report Spearman ρ and weighted Cohen's κ judge↔human per dimension, pairwise judge agreement, and correlation of judged quality with `grounded_accuracy`.
- **Logged to Braintrust:** one aggregate (mean-of-judges) score per dimension; per-judge scores in metadata.

### 2.6 Model cost/quality experiment (first-class)
Held fixed: arm D, frozen index, skill, test + counterfactual sets. Varied: model only. Slate (frozen at run time, via OpenRouter): Sonnet 5 (default), Opus 5 (ceiling, **run last**), Haiku 4.5, current Gemini Flash-class, one reliable open-weight tool-calling model chosen at implementation time. Report per model: `grounded_accuracy`, `abstain_recall`, `execution_failed`, `cap_hit`, judged quality (2.5), median/p90 latency, realised $/case. Output: Pareto chart (grounded accuracy vs $/case; latency encoded) in `data/reports/pareto.json` + image. Est. $100–200 total.

### 2.7 Braintrust usage (surface, not source of truth)
- Project `dealpoint-eval`. Dataset pushed as `maud-dealpoint-{dev,test,counterfactual}-v1` for viewing; never edited there.
- Runner: `braintrust.Eval()`; experiment name `{arm}-{model}-{index_version}-{git_sha7}`; metadata `{arm, model, index_version, skill_version, git_sha, case_set}`.
- Tracing: `wrap_openai` on the OpenRouter client; `@traced` on each tool. Trajectory metrics come from the harness's own execution record, not from spans.
- **Score budget (free tier: 10k scores/month, 14-day retention):** ≤ 6 scores per case — `grounded_accuracy, answer_correct, citation_gold_overlap, citation_verbatim, abstain_correct, skill_adherence` (+ 4 judge aggregates on the judged subset). Everything else → metadata + local JSONL. Smoke gate does not log to Braintrust.
- `just report` regenerates all tables/charts from local JSONL; README embeds those. Braintrust links are supplementary and expected to expire.

---

## 3. Architecture decisions
| Decision | Choice | Rationale |
|---|---|---|
| Language / tooling | Python 3.12, `uv`, `just`, `pytest` | matches repo |
| LLM access | `openai` SDK → OpenRouter, native tool calling | one key, model swap = string change |
| Agent loop | hand-rolled (~100–150 lines) | must be legible for trajectory evals |
| Retrieval | **LlamaIndex allowed only behind `Retriever` interface** for `search_agreement`: dense, BM25, hybrid, fusion, rerank | richer RAG experimentation; parser/offsets/agent never depend on it |
| Vector store | Qdrant embedded/local mode | zero infra, native hybrid |
| Embeddings / reranker | `BAAI/bge-small-en-v1.5`; `BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2` | CPU-only VM (2 vCPU, 7 GB, ~3.5 GB free disk) |
| Section parser & alignment | custom, framework-free, char offsets into canonical text | gold-span eval must never depend on a framework |
| Scorers | plain Python, unit-tested, offline | objective truth |
| Eval runner / tracing / comparison | Braintrust SDK | surface only (§2.7) |
| Judges | OpenRouter, 3 non-candidate families, one call per trace | §2.5 |
| Regression gates | pytest markers per milestone; SSSF `quality.py` runs them | benchmark thresholds as gates |
| UI | FastAPI (§1.5) + Next.js (`web/`), cached results + one live button | polished surface, last milestone |
| MCP | thin server over the same three functions, post-MVP | portability demo, not part of measured claims |
| Not used | DeepEval, OpenTelemetry/OpenInference, LlamaIndex agents, CUAD, ACORD, multi-agent | see §4 |

### 3.4 Canonical document model & chunking
- Canonical text = raw `.txt` → NFKC → curly quotes/dashes/nbsp normalised → whitespace collapsed. **All offsets, quotes, spans, chunks refer to canonical text.** Model quotes are normalised identically before the verbatim check.
- Section map: multiple heading patterns (`Section 1.1`, `1.1 Title`, `ARTICLE IV`, inline variants); each chunk gets `section_ref` = nearest preceding heading; parser coverage (% of text under a recognised heading) reported per agreement.
- Chunks: section-bounded, sub-split at ~400 tokens / 50 overlap, with `{agreement_id, chunk_id, section_ref, start, end}`. Fixed across all arms and the tournament.

### 3.5 Suggested layout
```
dealpoint/            data/{download,align,sections,select,cases}.py  corpus/{document,chunks,retrievers}.py
                      agent/{tools,loop,schema,prompts}.py  eval/{scorers,run,tournament,judges,report}.py  api/app.py
skills/ma-deal-point-review/{SKILL.md,playbooks/*.md}
data/raw/ (gitignored)  data/eval/*.jsonl (committed)  data/results/*.jsonl (committed, gitattributes if large)  data/reports/
web/ (Next.js)   tests/   specs/
```

---

## 4. MVP exclusions (explicit)
- CUAD (no supporting claim). ACORD (different corpus; optional post-v1 generalisation benchmark).
- LlamaIndex agents/workflows; any multi-agent design; corpus-comparison as an agent tool; web/calculator tools.
- DeepEval; OpenTelemetry/OpenInference; dual tracing paths.
- LLM judging on dev runs or on all arms; judges without frozen rubrics or without calibration.
- Chunk-size sweeps in the tournament; LLM-generated tournament queries.
- `confidence` field / calibration curves; rationale-faithfulness judge.
- Full-context (no-retrieval) arm; all 92 questions; all 152 agreements in the eval index (market panel uses labels only).
- Auth, persistence, multi-user, editing, uploads in the UI. Absolute accuracy thresholds set before first results.

---

## 5. Datasets (facts verified 2026-09-03)
- **MAUD v1** — Zenodo record 7500064, `maud_v1.zip` (29 MB), CC BY 4.0. Contains `data/contracts/contract_{0..151}.txt` (full agreements, ~250–500 KB / ~70–100k tokens each) and `MAUD_train.csv` (25.8k rows; HF mirror `theatticusproject/maud` has train/validation/test). Fields: `data_type, contract_name, text, answer, label, question, subquestion, text_type, id, category`. Use `data_type == main` only. 144 effective (question, subquestion) items across 22 `text_type` span types and 7 categories.
- **Deal identity:** party names are in line 1 of each contract (e.g. "AGREEMENT AND PLAN OF MERGER by and among ADAMAS PHARMACEUTICALS, INC., SUPERNUS …"). Parse, don't look up.
- **Span alignment is not verbatim:** 0/400 exact matches; 46% of `main` spans contain `<omitted>`. Procedure: normalise (§3.4) → split on `<omitted>` → fragments ≥ 20 chars → exact find, else `rapidfuzz` partial alignment ≥ 90 → char ranges. Measured ~99% alignable. Gold span for a case = union of aligned fragment ranges for that (contract, `text_type`). Coverage reported; cases with unaligned gold are excluded and listed.
- **Answer imbalance:** 60+ of 144 items have ≥ 90% majority answers. The 12 selected questions all have majority ≤ 0.72 (§A). Majority baseline always reported.
- **Agreement selection (20):** exclude agreements with parser coverage below the M0 threshold or > 2 unalignable gold spans across the 12 questions; then seeded greedy selection maximising answer diversity across the 12 questions, stratified by Type of Consideration; 5 dev / 15 test.
- Market panel uses gold labels from all 152 agreements; null answers are skipped.

---

## 6. Acceptance criteria (project-level)
1. `just data` reproduces canonical texts, section maps, aligned spans, the 20-agreement selection, and all three case sets bit-for-bit from seed.
2. `uv run pytest` green offline (no network) for parser, alignment, tools, scorers, case builder; smoke gate needs only `OPENROUTER_API_KEY`.
3. Four-arm test sweep + counterfactual sweep complete at the default model; `just report` writes `data/reports/four_arm.json` with all §2.4 metrics per arm and per question, next to the majority baseline; arms differ only by the stated variable (asserted by config diff in the report).
4. Retrieval tournament report exists; arm C's `index_version` is pinned in config and stamped on every experiment.
5. Judge panel run on the 40-case subset with frozen rubrics; calibration table (ρ, weighted κ per dimension, pairwise judge agreement, correlation with grounded accuracy) in `data/reports/judges.json`.
6. Model experiment complete for ≥ 4 of 5 models; `data/reports/pareto.json` + chart.
7. Every result in the README is regenerated from local files by `just report`; no README number depends on a Braintrust link.
8. README states the MAUD non-comparability caveat, the dataset licence/attribution, and the objective-vs-judged distinction.
9. (v1) Deal page and `/experiments` render entirely from cached data; live button runs arm D and scores the result against gold.

---

## 7. Build milestones (SSSF: one `just sdlc`/`simple-sdlc` request per milestone, referencing this brief section; gates as pytest markers `-m gate_mN`)
| # | Milestone | Deliverables | Gate |
|---|---|---|---|
| 0 | **Data foundation** | download, canonical text, section parser + coverage, span alignment, agreement selection, `dev/test/counterfactual` JSONL, question spec module (§A) | alignment ≥ 95% of candidate cases; 20 agreements selected; case counts within §2.1; determinism test (rebuild == committed) |
| 1 | **Tools + dense index + agent loop** | chunks, Qdrant dense index, three tools, hand-rolled loop, schema validation, execution record, arm A pipeline mode | 5 dev cases end-to-end on Haiku: 0 `EXECUTION_FAILED`, ≥ 1 `grounded_accuracy` hit; tool unit tests |
| 2 | **Scorers + Braintrust + smoke gate** | all §2.4 scorers, `braintrust.Eval` runner, metadata/score budget, `just eval --arm --set --model`, smoke eval | scorer unit tests incl. synthetic fabricated/redacted cases; smoke eval (5 dev, Haiku) passes and does not log; one real dev run visible in Braintrust |
| 3 | **Retrieval tournament → arm C** | `Retriever` interface, LlamaIndex-backed configs, tournament script, report, `index_version` | tournament report written; hit@5 of winner ≥ dense baseline on dev |
| 4 | **Four-arm run + report** | skill files (§B), arm D, test + counterfactual sweeps on default model, `just report`, README results section | reports generated; **no arm regresses vs its previously recorded dev result** (relative gate); README caveats present |
| 5 | **Judge panel + calibration** | rubrics frozen, judge harness, blinding, 40-case subset, calibration CLI for hand scoring, agreement stats | `judges.json` with ρ/κ per dimension |
| 6 | **Model Pareto** | model slate config, sweeps (Opus last), Pareto report + chart | `pareto.json`; ≥ 4 models |
| 7 | **API + Next.js UI** | FastAPI (§1.5), deal page, `/experiments`, live button | pages render from cache; `POST /api/run` returns a scored finding ≤ 90 s |
| 8 | *(optional)* **MCP wrapper** | MCP server over the three tools + SKILL.md | tools callable from Claude Code |

**Core eval MVP = 0–6.** 7 = product surface. 8 = optional. Order is fixed; do not start 7 before 4 is green.

SSSF notes: the builder must not edit `data/eval/*.jsonl` by hand (regenerate via script); `adw_modules/quality.py` should run `uv run pytest -m "gate_m{N} and not needs_model"` by default and the `needs_model` smoke gate explicitly at M2+; reviewer checks that each arm differs from its predecessor by exactly one config key.

---

## 8. Unresolved / optional items
- ACORD as a post-v1 retrieval-generalisation suite (BEIR format, 1–5 graded).
- Rationale-faithfulness judge; `confidence` field + calibration curves.
- OTLP export to Braintrust in place of SDK tracing (vendor-neutral tracing claim).
- Full-context (no-retrieval, 100k-token prompt) comparison arm.
- Expanding to more questions/agreements (harness is parameterised; only the question spec and selection config change).
- Pairwise (A vs D) judging instead of pointwise, if pointwise agreement is weak.
- Exact open-weight agent model and Gemini Flash version — pick at M6 by checking OpenRouter tool-calling reliability on 5 dev cases.
- Braintrust key currently in `.braintrust.json` (gitignored) — rotate if the file ever leaves the VM.

---

## Appendix A — Question spec (12 questions; exact MAUD strings; gold = `data_type=main`, `subquestion=<NONE>`)
`required_evidence` = a deterministic trajectory check that the named block was retrieved by **any** tool; listed only where the answer genuinely cannot be read from the operative clause alone. Arm D's skill additionally says *how* (via `lookup_defined_term`), which is scored as adherence, not as a requirement.

| id | reasoning | MAUD `question` | span type (`text_type`) | options (majority first) | canonical query | required_evidence |
|---|---|---|---|---|---|---|
| q01 | direct | `Type of Consideration-Answer` | Type of Consideration | All Cash (.59) · All Stock · Mixed Cash/Stock · Mixed Cash/Stock: Election | "merger consideration per share cash stock conversion of company common stock" | — |
| q02 | direct | `Liability standard for no-shop breach by Target Non-D&O Representatives` | No-Shop | Strict liability (.58) · Reasonable standard | "no solicitation breach by representatives deemed breach by the company" | — |
| q03 | numeric | `Initial matching rights period (COR)-Answer` | Agreement provides for matching rights in connection with COR | 4 business days (.51) · 5 business days · 3 business days · 4 calendar days · 3 calendar days · 2 business days or less · Greater than 5 business days | "change of recommendation notice period parent match negotiate business days" | — |
| q04 | structured | `Ordinary course efforts standard-Answer` | Ordinary course covenant | Flat covenant (no efforts standard) (.53) · Commercially reasonable efforts · Reasonable best efforts | "conduct of business prior to closing ordinary course consistent with past practice efforts" | — |
| q05 | defined-term | `Knowledge Definition-Answer` | Knowledge Definition | Constructive knowledge (.55) · Actual knowledge | "definition of knowledge of the company actual knowledge after due inquiry" | defined term **Knowledge** (or "Knowledge of the Company") |
| q06 | defined-term | `FLS (MAE) Standard-Answer` | MAE Definition | "Would" (reasonably) be expected to (.62) · No · "Would" · Other forward-looking standard · "Could" (reasonably) be expected to | "material adverse effect definition would reasonably be expected to have" | defined term **Material Adverse Effect** (or Company MAE) |
| q07 | defined-term | `Definition includes asset deals-Answer` | Superior Offer Definition | Greater than 50% but not "all or substantially all" (.48) · 50% · "All or substantially all" · Less than 50% | "superior proposal definition assets of the company percentage consolidated" | defined term **Superior Proposal** |
| q08 | defined-term | `Definition contains knowledge requirement - answer` | Intervening Event Definition | Known, but consequences unknown or not reasonably foreseeable, at signing (.52) · Not known and not reasonably foreseeable at signing · Known, but consequences unknown, at signing · Not known at signing | "intervening event definition not known to the board reasonably foreseeable as of the date" | defined term **Intervening Event** |
| q09 | cross-ref | `Acquisition Proposal required to be publicly disclosed-Answer (Y/N)` | Tail Period & Acquisition Proposal Details | Yes (.59) · No | "termination fee tail acquisition proposal publicly disclosed made known within twelve months" | — |
| q10 | cross-ref | `Fiduciary exception:  Board determination standard-Answer (no-shop)` | Fiduciary exception:  Board determination (no-shop) | "Inconsistent" with fiduciary duties (.42 of non-null) · "Reasonably likely/expected to be inconsistent" with fiduciary duties · Other specified standard · "Reasonably likely/expected violation" of fiduciary duties · "Reasonably likely/expected breach" of fiduciary duties · "Required to comply" with fiduciary duties · "Breach" of fiduciary duties | "board determines in good faith failure to take action would be inconsistent with fiduciary duties unsolicited proposal" | — (47/151 gold answers null → fewer cases) |
| q11 | carve-out | `Target stockholder proceedings-Answer (Y/N)` | MAE Definition | Yes (.59) · No | "material adverse effect carve-out stockholder litigation arising from the merger agreement" | defined term **Material Adverse Effect** |
| q12 | carve-out | `Negative Interim Covenant includes carveout for pandemic responses-Answer (Y/N)` | Negative interim operating covenant | No (.56) · Yes | "interim operating covenants except as required by COVID-19 pandemic measures" | — |

Note: two questions (q06, q11) share the MAE Definition span; all others have distinct span types. Question prompts to the model use the MAUD question text plus a one-line plain-English gloss and the exact option list; the gloss is part of the question spec (all arms), not the skill.

## Appendix B — Skill structure and adherence rules
`SKILL.md` (generic; each rule is an applicable/satisfied trajectory check):
1. Locate the operative clause with `search_agreement` before answering (≥ 1 search).
2. If the question turns on a defined term (q05–q08, q11), call `lookup_defined_term` for it before answering.
3. When the operative clause cross-references a section, call `get_section` on it before answering.
4. Read carve-outs/exceptions to the end (a retrieved chunk that ends mid-list is followed by the next chunk/section).
5. Quote verbatim; ≤ 3 citations; each citation from a chunk actually retrieved in the trajectory.
6. Before abstaining, run ≥ 2 searches with different phrasing.
7. Abstain only when the provision is genuinely absent; otherwise answer "No"-type options with a citation.
8. Rationale ≤ 80 words, states the option chosen and the clause it rests on.

`playbooks/q##.md` (loaded only for that question): where the term lives, what distinguishes the options, the trap case. Skill total ≤ ~1,500 tokens including one playbook.

## Appendix C — Judge rubric skeleton (anchors written and frozen at M5, before any judge call)
Each dimension scored 1–5 with a one-sentence anchor per level; judges receive: question spec, blinded trace (tools, args, retrieved text), finding, and — for evidence sufficiency only — the gold span. Output JSON `{reasoning:int, evidence:int, trajectory:int, professional:int, notes:str}`. Human calibration form uses the identical rubric and blinding.

## Appendix D — Hidden assumptions surfaced during grilling
- MAUD's published task is span→answer; the demo's task is document→(answer, citation). Not comparable; say so.
- Gold spans need fuzzy alignment; alignment coverage is a reported data-quality metric and an M0 gate.
- Section headings are inconsistent across EDGAR texts; `section_ref` quality varies and is reported.
- Many MAUD questions are near-degenerate; majority baseline must accompany every accuracy.
- Braintrust free tier meters *scores* (10k/mo) and purges after 14 days; local files are canonical.
- Arm D may not beat arm C on grounded accuracy; skill adherence is measured independently so the C→D result is informative either way.
- Open-weight tool calling on OpenRouter may fail; `EXECUTION_FAILED`/`CAP_HIT` are first-class results, not noise to hide.
- The VM is CPU-only with ~3.5 GB free disk: small embedding/reranker models, no full-152 vector index by default, `node_modules` budgeted.
