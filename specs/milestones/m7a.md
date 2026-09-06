# M7a — Framework roles made explicit: LlamaIndex RAG lab, DeepEval cross-check, Braintrust sync (scriptable layer)

**Authority:** `specs/grilled-product-brief.md` (§2.7 Braintrust as surface, §3 framework decisions, §4
exclusions as amended by the engineer on 2026-09-05), the engineer's M7 decisions of 2026-09-05
(final clarification round, no further questions). M7 is **additive**: M1–M6 results, frozen subsets,
skill, arm configs, scorers and the M3 winner are not rewritten, retuned or rerun except for a
bounded compatibility operation that is genuinely required and recorded.

M7 is two factory jobs. **M7a (this spec)** is everything reproducible from code/SDK/`bt` CLI.
**M7b** (after a Claude Code restart, separate spec) is the interactive Braintrust cockpit layer.
M7a must checkpoint cleanly; M7b resumes from it, never reruns it.

## Budget and disk
- Global OpenRouter cap is **$6.00** (`MAX_OPENROUTER_SPEND_USD`). M7 total: soft ≤ **$1.00**, absolute
  **$2.50**. Before any bulk metered call: estimate calls and spend; run a ≤ 3-item calibration sample;
  compare projected vs realised; proceed only within both caps. Every metered call carries
  `milestone_tag: m7a` and, when not a case run, `purpose` (`probe`, `synthetic_query`, `deepeval`).
  No router endpoints; models only from the slate, the M5 judge trio, or the configured evaluator.
- Disk: ~1.3 GB free. Measured in isolation on 2026-09-05: `llama-index-core` + `llama-index-retrievers-bm25`
  ≈ 178 MB standalone; `deepeval` ≈ 92 MB (no torch). **Re-measure with `uv pip install --dry-run` (or a
  scratch venv) before touching the live environment**; install as optional groups `rag-lab` and `deepeval`;
  if free space would drop below 500 MB, scope down (document what is omitted). Keep the visualizer
  `node_modules`. Never destabilise the M1–M6 environment.

## 1. LlamaIndex — RAG composition + framework-native evaluation (`dealpoint/rag_lab/`)
LlamaIndex owns: retriever composition where useful, framework-native RAG evaluation, the synthetic-query
study. It does **not** own the parser, canonical sections, MAUD offsets, agent loop, tools, answer schema or
canonical scorers. MAUD gold-span overlap remains benchmark truth; LlamaIndex node ids or qrels never replace it.

A. **Native evaluation on canonical dev retrieval.** Wrap each frozen M3 candidate (dense, BM25, hybrid-RRF,
   hybrid+rerank, fusion variants) behind LlamaIndex `BaseRetriever` adapters over the *same* chunks and
   indexes (`chunk_version`/`index_version` asserted), or compose native equivalents where LlamaIndex has
   them; run `RetrieverEvaluator` (hit_rate, mrr) over the canonical dev queries with `expected_ids` =
   chunk ids overlapping gold spans. Compare with project `obj/gold_span_hit@k` and `obj/gold_span_mrr`
   per retriever. Report: per-retriever both metric sets; **ranking-order agreement** (Spearman over
   retriever ranks); **disagreement cases** in both directions (LI hit / MAUD miss and MAUD hit / LI miss)
   with a cause label — chunk identity, partial overlap, duplicate relevant chunks, section boundary,
   reranking — and 1–2 worked examples with case ids.
B. **Synthetic-query robustness (secondary).** Generate ≤ 2 questions per **gold-bearing dev chunk only**
   with LlamaIndex's generator, model from `RAG_SYNTH_MODEL` (default: the dev workhorse
   `z-ai/glm-5.3-flash`), cost estimated first; freeze as `data/eval/synthetic_dev_queries.jsonl` with
   generator id/version/prompt hash/cost. Evaluate every M3 candidate on it. It never touches the test set,
   never changes the frozen M3 winner, never becomes benchmark truth, never drives selection. Answer one
   question: does the frozen winner stay strong under a different query distribution?
Output: `data/reports/li_rag_eval.json` + `.md`.

## 2. DeepEval — independent agent evaluator (`dealpoint/eval/deepeval_adapter.py`)
Thin adapter from canonical result rows (finding + execution record + trajectory) to DeepEval test cases.
No second trace store; Braintrust remains primary observability. Evaluator model via `DEEPEVAL_MODEL`;
default resolves the first available model that satisfies the M5 non-candidate-family policy (read from
the existing judge configuration, never hardcoded); the resolved provider/model/version is recorded in
results and report. Subset: the M5 judged subset × already-judged variants (A@Haiku, D@Haiku, D@GLM, plus
M6's judged variants) so DeepEval, objective, judge and human align trace-for-trace. Metrics: task
completion, tool correctness (expected tools from `required_evidence`), argument correctness where
applicable, step efficiency (DeepEval-native if present, else a documented GEval definition). No generic
RAG metrics weaker than MAUD truth. Compare: DeepEval ↔ deterministic (`grounded_accuracy`,
`required_evidence_met`, tool calls/cap); ↔ calibrated judge dimensions; ↔ human (n reported; "pending"
if absent); step efficiency ↔ human trajectory quality. Surface disagreement cases. Conclude with one of
`KEEP_CORE_DIAGNOSTIC` / `KEEP_OPTIONAL_ANALYSIS` / `REMOVE_NO_ADDED_SIGNAL`, from evidence.
Output: `data/reports/deepeval_crosscheck.json` + `.md`.

## 3. Braintrust synchronisation — `just braintrust-sync` (SDK / `bt` CLI, public capabilities only)
Starter tier: 14-day log retention. Everything below must be **re-creatable from Git/local sources by one
command**; the walkthrough never depends on permanent trace ids (record stable case ids and experiment
names alongside any trace id).
- **Namespaces** for score names: `obj/`, `li/`, `judge/`, `deepeval/`, `procedure/`. Common metadata:
  `stage, arm, reasoning_type, retriever, case_type, model, index_version, skill_version, git_sha,
  framework_versions`. M7 experiments may log ≤ **12 scores/case on subsets ≤ 60 cases** (ceiling, not
  target); M4/M6 sweeps keep six. Secondary diagnostics go in metadata.
- **Experiments**, named/tagged by stage: RAG (`rag-m3-dense`, `rag-m3-bm25`, `rag-m3-hybrid`,
  `rag-m3-hybrid-rerank`, fusion where retained, `rag-m7-li-crosscheck`, `rag-m7-synthetic`); agent
  systems (existing A–D names, tagged `stage=agent`); evaluation systems (`judge-…`, `deepeval-…`);
  economics (M6 Pareto, tagged). Existing experiments are re-logged from canonical rows only if needed for
  tags — never re-run.
- **Logs with a readable hierarchy** for representative M3–M7 traces, **replayed from stored trajectories**
  (no model calls): `case` → `agent` → `search_agreement` (retrieval stages as child spans) →
  `lookup_defined_term` / `get_section` → `final_answer` → `scoring` spans carrying provenance
  (`obj/`, `judge/`, `deepeval/`). Select and record representative cases: successful direct;
  retrieval-rescue; defined-term/cross-reference; inefficient trajectory; wrong answer;
  abstention/counterfactual — chosen by rule (documented), not for convenience.
- **Datasets** (mirrors, never hand-edited): dev, test, counterfactual, judged/calibration subset,
  synthetic-query set; metadata `question_id, agreement_id, reasoning_type, case_type, source hashes`.
- **Scorers** as Braintrust functions that import the canonical implementations (`grounded_accuracy`,
  `answer_correct`, `citation_verbatim`, `citation_gold_overlap`, `required_evidence_met`,
  `skill_adherence`); judge/DeepEval provenance kept distinct; pass/fail thresholds only where meaningful.
- **Prompts**: base agent system instructions, arm-D skill injection, calibrated judge rubric/prompt —
  with source hashes; canonical ownership stays in Git.
- **Parameters**: one compact versioned schema of non-secret runtime parameters (arm, retriever config,
  top_k, max_tool_calls, skill version, model alias, dataset/split, mode) exposing existing config.
- **Tools**: expose `search_agreement` / `get_section` / `lookup_defined_term` as Braintrust tools only if
  thin adapters can run in Braintrust's function environment without duplicating the index/data layer;
  otherwise document the decision — no placeholder stubs.
- **Human review preparation**: a deliberately diverse **12-trace blinded calibration set** (rule: span
  reasoning types, successes/failures, ≥ 1 abstention, ≥ 2 judged variants) synced so the engineer can
  score it in Braintrust Review against the frozen M5 rubric fields; methodology must extend to ~30 later
  unchanged; existing local human scores (if any) remain canonical and are displayed, never re-labelled.
- **BTQL investigations**: implement and actually run the six queries (retrieval rescue; failure
  attribution; reasoning slices; DeepEval disagreement; model economics; trajectory inefficiency) via the
  API; save where the product allows, and put exact reproducible queries + results in
  `docs/braintrust-queries.md` and `data/reports/btql_investigations.json`.
- **MCP**: verify `claude mcp list` shows `braintrust` (already configured by the operator; needs
  authentication after restart). Do not re-add, do not attempt MCP-dependent work.

## 4. Permanent local artifacts and walkthrough draft
Update `data/reports/` (li_rag_eval, deepeval_crosscheck, btql_investigations, representative cases) and
README. Write `docs/demo-walkthrough.md` **draft**: a 5–10 minute technical walkthrough in the engineer's
order (Datasets → RAG Lab → Agent Systems → Logs trace → Scorers → Review → Eval of Evals → Loop/SQL →
Debugger → Model Economics → Dashboard), using actual experiment names, case ids, scorer/prompt versions and
observed results; sections that need M7b (views, dashboard, Loop, Review scoring) are marked `[M7b]`.
Include the role diagram: custom Python = benchmark truth; LlamaIndex = RAG lab/eval; DeepEval =
independent agent-eval cross-check; Braintrust = traces/experiments/comparison; local reports/Git =
permanent evidence.

## README results block (engineer's instruction, 2026-09-05)
`regenerate_readme` writes the block between the README markers. Change what it writes: per model, a
table of arms with `grounded_accuracy` shown as "x% (k of n scored)", plus the one-line majority-baseline
note. Nothing else in the README block: no execution-failure, cap-hit, or v1-vs-v2 tables and no
per-arm verdict sentences. Those diagnostics stay, complete and honest, in `data/reports/four_arm.md`
and `four_arm.json`, which the README links to. Update `tests/test_readme_results.py` to the new block
shape. The README is an introduction for readers who do not know the project; the reports are the record.
The same rule applies to the `<!-- BEGIN PARETO -->` block M6 added: per-model table of grounded accuracy
"x% (k of n scored)" and $/case only; move "Not run", "Partially run", "Comparability note", "Spend",
"Recorded decisions", "Brief-vs-spec differences" and "Caveats" out of the README into `data/reports/pareto.md`
(they stay complete there); place both generated blocks under the README's "Latest numbers" heading; update
`tests/test_readme_pareto.py` to the new shape.

## Definition of done (`gate_m7`; add the marker)
- [ ] Optional dependency groups installed within the disk guard; `framework_versions` recorded.
- [ ] `li_rag_eval.json`: per-retriever LI hit_rate/mrr vs `obj/` hit@k/mrr on all dev cases,
      ranking-order agreement, disagreement cases with causes; synthetic set frozen with provenance and
      evaluated; frozen winner and test set untouched (hash assertions).
- [ ] `deepeval_crosscheck.json`: resolved evaluator recorded; metrics on the judged subset × variants;
      comparisons vs deterministic / judge / human(n) / required-tool; disagreement cases; classification.
- [ ] `just braintrust-sync` recreates datasets, experiments (tags/namespaces), replayed representative
      traces, scorers, prompts, parameters, the 12-trace review set; idempotent; offline tests with a fake
      client cover the mapping; score budget assertion (≤ 12 on ≤ 60, six elsewhere).
- [ ] `docs/braintrust-queries.md` + `btql_investigations.json` with executed results.
- [ ] `docs/demo-walkthrough.md` draft; README updated; role diagram present.
- [ ] Spend: M7a realised ≤ $1.00 target / ≤ $2.50 absolute; global ≤ $6.00 (ledger-asserted).
- [ ] Suite, ruff, pyright green. Reviewer verifies: frozen-test integrity, framework boundaries, no
      duplicate canonical truth, no duplicate tracing architecture, score-budget compliance, spend
      compliance, walkthrough reproducibility (every name/id resolves to a local artifact).

## No-bloat rule
A feature survives only if it strengthens evaluation, clarifies provenance, diagnoses failures, improves
reproducibility, exposes a useful control, or materially improves the demo. Tested-and-rejected features
are documented, not hidden. No new agent runtime; no second tracing backend; no metric weaker than gold
labels unless it adds a distinct diagnostic; no hardcoded judge-model stack.

## Out of scope (M7b)
Custom trace view, saved views, dashboard, Loop investigation thread, Review scoring session, Playground,
Topics/Patterns/Debugger use, walkthrough finalisation — all after the restart, with MCP.
