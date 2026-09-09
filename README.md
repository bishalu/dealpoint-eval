# dealpoint-eval

A worked demo of how to evaluate an AI agent on real professional work.

The agent reads a merger agreement and pulls out its deal points. That part is ordinary. The interesting part is everything around it. The case sets, the scorers and the comparison decide, with evidence, whether one version of the agent is better than another.

## The task

When two public companies merge, the deal lives in a merger agreement. It is a 200-page contract that says what happens if a better offer arrives, what counts as a material adverse change, how long the buyer has to match a rival bid, and a few hundred other things. Lawyers call these terms deal points, and reading them out of a fresh agreement is slow, expert work.

That makes it a good test case. The document is too long to read in one pass. The answer is one clause buried somewhere in it. A confident wrong answer costs real money. Most knowledge work worth automating looks like this, which is why a toy benchmark would not tell you much.

## Why the results are checkable

Most evaluations of AI legal work use a second AI to grade the answers. This one mostly does not have to.

MAUD, from the Atticus Project, is a public dataset where experienced M&A lawyers answered 92 standard deal-point questions across 152 real merger agreements from SEC filings. They also recorded the exact passage each answer came from. So code, not opinion, can check every answer the agent gives:

1. Did it pick the answer the lawyers picked?
2. Did it cite the passage the lawyers relied on?
3. Is the text it quoted really in the document, word for word?

An answer that is right but cites the wrong clause scores nothing. That is deliberate. A lawyer who cannot show you the clause cannot be trusted with the deal.

## What the agent does

It gets one real agreement, about 90,000 tokens, and one of twelve deal-point questions. It has three tools: search the agreement, open a numbered section, and look up a defined term such as "Material Adverse Effect". It gets at most eight tool calls. Then it either answers with quotes or says the provision is not there.

The twelve questions cover different kinds of reasoning. A direct lookup (what form does the consideration take?). A number (how many business days does the buyer get to match a rival bid?). A defined term (does "Knowledge" cover what someone should have known?). A cross-reference. A carve-out buried inside a definition.

## What gets compared

Four versions of the system. Each differs from the one before it by exactly one thing, so a change in score has one candidate cause.

| version | what it is |
|---|---|
| A | one search, one answer. The simplest thing that could work |
| B | the three-tool agent |
| C | the same agent, with a better search engine underneath, picked in a tournament |
| D | the same agent, following a written review procedure, the way a senior associate briefs a junior |

A second experiment holds version D fixed and swaps only the language model, from expensive to very cheap, to see what quality costs.

Three model judges also score a blinded sample against a fixed rubric. They sit in their own tier on purpose. Rather than treating their verdicts as truth, the report measures how far they agree with each other and with the code-checked scores. The repo also ships a scoring form for the same sample, so a person can grade it by hand and see where the judges drift.

## Latest numbers

<!-- BEGIN RESULTS -->
**v2 after harness repair.** These numbers are budget-scaled. Each model ran a frozen 32-case slice of MAUD's test set (18 cases for `anthropic/claude-haiku-4.5`), not the full 167-case design in the brief. Every score is objective. Deterministic Python checks each answer against the lawyers' own labels, and no model does any judging. The scores are not comparable to MAUD leaderboard numbers, because MAUD hands the model the passage to read and this task makes the agent find it.

For reference, always giving the most common answer scores 46.1% on the full 167-case test set. The subset selection rule prefers cases where that common answer is wrong, so the subset has no meaningful baseline of its own. Compare the arms against each other instead. Both baseline figures are in the full report.

**Model `anthropic/claude-haiku-4.5`:**

| arm | grounded_accuracy |
|---|---|
| A | 50.0% (10 of 18 scored) |
| D | 62.5% (8 of 18 scored) |

**Model `z-ai/glm-5.3-flash`:**

| arm | grounded_accuracy |
|---|---|
| A | 76.5% (17 of 32 scored) |
| B | 58.8% (17 of 32 scored) |
| C | 66.7% (21 of 32 scored) |
| D | 64.7% (17 of 32 scored) |

Full diagnostics are in [`data/reports/four_arm.md`](data/reports/four_arm.md) and [`data/reports/four_arm.json`](data/reports/four_arm.json): execution-failure and cap-hit rates, the v1-vs-v2 comparison, residual-failure attribution, and the per-arm verdicts.
<!-- END RESULTS -->

`just report` rebuilds this section from the result files, so the numbers here cannot drift from the ones in `data/reports/`. The sample is small on purpose. The whole evaluation ran on a few dollars of API credit, so read the direction of a difference, not its second decimal.

### Model cost/quality (M6)

<!-- BEGIN PARETO -->
These numbers are budget-scaled: a frozen 32-case subset per model, 18 at Haiku. grounded_accuracy is objective, computed by deterministic Python over the expert labels. Judged quality is secondary, comes from model judges, and is never treated as truth. Scores are not comparable to the MAUD leaderboard.

Arm D is held fixed (frozen index, skill, tools, cases); only the model varies.

| model | grounded_accuracy | $/case |
|---|---|---|
| anthropic/claude-haiku-4.5 | 62.5% (8 of 18 scored) | $0.04580 |
| deepseek/deepseek-v4-flash | 69.2% (13 of 32 scored) | $0.00226 |
| google/gemini-3.1-flash-lite | 63.2% (19 of 32 scored) | $0.00553 |
| qwen/qwen3.7-flash | 63.6% (11 of 32 scored) | $0.00110 |
| z-ai/glm-5.3-flash | 64.7% (17 of 32 scored) | $0.00278 |

**Frontier:** `qwen/qwen3.7-flash`, `deepseek/deepseek-v4-flash`. One of those two beats every other model in the table on both price and accuracy. The exact dominance rule is in the full report.

Full diagnostics are in [`data/reports/pareto.md`](data/reports/pareto.md) and [`data/reports/pareto.json`](data/reports/pareto.json): models not run or only partly run, the comparability note, spend, recorded decisions, brief-vs-spec differences, and caveats.
<!-- END PARETO -->

## How the benchmark was assembled

MAUD gives the lawyers' passages as loose text, not as positions inside the agreements, so the pipeline has to find them again. It downloads the 152 agreements, turns each into one canonical text, finds the section headings, and locates every expert passage in its document by fuzzy matching. The published excerpts differ from the originals in quotes, page markers and line joins, so exact matching would miss most of them. The pipeline reports coverage per question.

From the 139 agreements the parser handles cleanly, a seeded rule picks 20: five for development (58 cases) and fifteen for the frozen test set (167 cases). Forty more cases test whether the agent knows when to say no. Thirty of those had the lawyers' clause deleted from the document. The other ten ask reasonable diligence questions that no merger agreement answers, such as the target's cyber-insurance deductible.

Search runs locally on Qdrant, with small open embeddings and BM25 keyword matching combined by rank. The tournament also tried a reranker and a multi-query variant. Neither earned its latency.

## How it was built

The requirements brief, `specs/grilled-product-brief.md`, came first and has been the authority ever since. A small software factory under `adws/` then built the project one milestone at a time: a planning model writes a plan, a coding model implements it, deterministic checks run the tests, a reviewing model rules on every acceptance item, and a documenter writes the milestone up in `docs/milestones/`. An outer loop moves to the next milestone and stops for a person only when there is a real decision to make.

Braintrust holds traces of the factory's own runs under `sssf-dealpoint`, and the evaluation experiments under `dealpoint-eval`. The result files in `data/results/` and the reports in `data/reports/` are the permanent record.

### Framework roles

Four tools do four different jobs, and none of them stands in for another.

```
custom Python  --> benchmark truth      (parser, canonical offsets, MAUD gold spans, scorers)
LlamaIndex     --> RAG lab / RAG eval   (retriever composition, RetrieverEvaluator, synthetic queries)
DeepEval       --> independent agent-eval cross-check (task completion, tool correctness, step efficiency)
Braintrust     --> traces / experiments / comparison (a view, not the source of truth)
local reports + Git --> permanent evidence (data/reports/, data/results/, specs/)
```

Overlap with the MAUD gold spans is the only thing that ever decides a winner. LlamaIndex evaluates retrieval and generates synthetic queries against that same truth (`data/reports/li_rag_eval.md`). DeepEval cross-checks the agent traces the judged subset saw (`data/reports/deepeval_crosscheck.md`). Braintrust holds the traces and experiment comparisons, and `just braintrust-sync` rebuilds them from Git and local files.

## Try it

You need `uv` and an `OPENROUTER_API_KEY` in `.env`.

```
just data          # download MAUD, align the expert passages, build the case sets
just test          # offline test suite, no network, no spend
just index         # chunk and embed the selected agreements
just tournament    # retrieval tournament on the dev set, no model calls
just eval D z-ai/glm-5.3-flash test --limit 5
just report
just braintrust-sync # rebuild Braintrust datasets/experiments/traces from local sources
```

Running `just data` twice changes no committed file, and a test checks that. Any run that costs money refuses to start if it would push spending past `MAX_OPENROUTER_SPEND_USD`.

## Where to look next

Start with `docs/demo-walkthrough.md`, a guided tour through the datasets, the RAG lab, the agent versions, the traces, the scorers and the economics. After that:

- `docs/milestones/` says what each milestone set out to do and what it delivered
- `data/reports/four_arm.md` has the full comparison of the four versions, with every diagnostic
- `data/reports/tournament.md` has the search-engine tournament
- `data/reports/judges.md` has the judge panel and its agreement statistics
- `data/reports/li_rag_eval.md` has the LlamaIndex retrieval evaluation and the synthetic-query study
- `data/reports/deepeval_crosscheck.md` has the DeepEval cross-check
- `docs/braintrust-queries.md` has the six BTQL investigations, with the exact queries and their results
- `specs/grilled-product-brief.md` is the requirements document

## Scope

This is an evaluation project, not a product. There is no interface beyond the command line, and the sample sizes are the ones a few-dollar budget buys.

The task is also harder than the published MAUD benchmark, which hands the model the relevant passage. Here the agent has to find it in the whole agreement, so the scores do not compare to MAUD leaderboard numbers in either direction.

## Layout

```
dealpoint/      parser, alignment, corpus, agent loop, tools, scorers, reports
data/eval/      committed case sets and the frozen test subset
data/results/   result rows and the spend ledger
data/reports/   generated reports and version stamps
docs/           milestone records and walkthroughs
skills/         the review procedure the agent follows in version D
specs/          the brief and the milestone specs
adws/           the software factory
```

MAUD is by The Atticus Project, CC BY 4.0, Zenodo record 7500064.
