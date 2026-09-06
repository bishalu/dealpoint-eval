# dealpoint-eval

When two public companies merge, the deal lives in a merger agreement: a 200-page contract that says what happens if a better offer shows up, what counts as a material adverse change, how long the buyer has to match a rival bid, and a few hundred other things. Lawyers call these terms deal points, and reading them out of a fresh agreement is slow, expert work.

This repo asks whether an AI agent can do that reading, and, more to the point, how you would know if it could.

## Why this dataset makes the question answerable

Most AI evaluations of legal work rely on another AI to grade the answers. This one mostly doesn't have to. MAUD, from the Atticus Project, is a public dataset in which experienced M&A lawyers answered 92 standard deal-point questions for 152 real merger agreements from SEC filings, and recorded the exact passage each answer came from. So for every question the agent answers, three things can be checked mechanically:

1. Did it pick the answer the lawyers picked?
2. Did it cite the passage the lawyers relied on?
3. Is the text it quoted really in the document, word for word?

A right answer with a wrong citation scores zero here. In practice that is the failure that matters most: a lawyer who can't show you the clause can't be trusted with the deal.

## What the agent does

Give it one real agreement (roughly 90,000 tokens, far too long to read in one go) and one of twelve deal-point questions. It has three tools: search the agreement, open a numbered section, and look up a defined term such as "Material Adverse Effect". It gets at most eight tool calls, then it must either answer with quotes or say the provision isn't there.

Twelve questions were chosen to cover different kinds of reasoning: a direct lookup (what form does the consideration take?), a number (how many business days does the buyer get to match a rival bid?), a defined term (does "Knowledge" include what someone should have known?), a cross-reference, and a carve-out buried in a definition.

## What gets compared

Four versions of the system, each differing from the last by exactly one thing:

| version | what it is |
|---|---|
| A | one search, one answer. The simplest thing that could work. |
| B | the three-tool agent |
| C | the same agent with a better search engine underneath, chosen in a tournament |
| D | the same agent following a written review procedure, the way a senior associate would brief a junior |

A separate experiment keeps version D fixed and swaps only the language model, from expensive to very cheap, to see what quality actually costs.

Alongside the mechanical checks there is a panel of three model judges scoring a blinded sample on a fixed rubric. They are kept in a separate tier on purpose: the report measures how well they agree with each other and with the mechanical results, rather than treating their scores as truth. A form for human scoring of the same sample is included, so the judges themselves can be calibrated.

## Latest numbers

<!-- BEGIN RESULTS -->
**v2 after harness repair.** Results below are from a **budget-scaled** 32-case frozen discriminative subset of MAUD's test set (18 cases at `anthropic/claude-haiku-4.5`), not the brief's full 167-case design; the task here is document -> (answer, citation) while MAUD's published task is span -> answer, so scores are **not comparable** to MAUD leaderboard numbers, and every metric below is **objective** (deterministic Python over expert labels) -- M4 has no model judging.

Majority-baseline overall accuracy: 0.0%.

Majority-baseline accuracy on this subset is 0.0% by construction: data/eval/test_subset_v1.json's selection rule prefers cases where gold_answer != majority_answer (the majority already gets these wrong), so a near-zero subset baseline is not evidence the majority baseline is generally weak. For context, the majority baseline on the full 167-case MAUD test set is 46.1%.

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

Full diagnostics -- execution-failure/cap-hit rates, the v1-vs-v2 comparison, residual-failure attribution, per-arm verdicts -- are in [`data/reports/four_arm.md`](data/reports/four_arm.md) and [`data/reports/four_arm.json`](data/reports/four_arm.json).
<!-- END RESULTS -->

`just report` regenerates this section from the result files. The test subset deliberately favours questions where the most common answer is wrong, so the majority baseline reads 0% by construction. The sample is small, deliberately, because the whole evaluation ran on a few dollars of API credit: read the direction of a difference, not its second decimal. Full tables with every metric are in `data/reports/`.

### Model cost/quality (M6)

<!-- BEGIN PARETO -->
This report is **budget-scaled** (32-case frozen subset per model, 18 at Haiku) -- grounded_accuracy is **objective** (deterministic Python over expert labels); judged quality is **secondary and model-judged, never objective truth**; scores are **not comparable** to the MAUD leaderboard.

Arm D is held fixed (frozen index, skill, tools, cases); only the model varies.

| model | grounded_accuracy | $/case |
|---|---|---|
| anthropic/claude-haiku-4.5 | 62.5% (8 of 18 scored) | $0.04580 |
| deepseek/deepseek-v4-flash | 69.2% (13 of 32 scored) | $0.00226 |
| google/gemini-3.1-flash-lite | 63.2% (19 of 32 scored) | $0.00553 |
| qwen/qwen3.7-flash | 63.6% (11 of 32 scored) | $0.00110 |
| z-ai/glm-5.3-flash | 64.7% (17 of 32 scored) | $0.00278 |

**Frontier** (Minimise realised $/case, maximise grounded_accuracy. A point is dominated iff another has x<=x' and y>=y' with at least one strict inequality. Exact ties on both axes keep the lexicographically-smallest model id.): ['qwen/qwen3.7-flash', 'deepseek/deepseek-v4-flash']

Full diagnostics -- not-run/partially-run models, the comparability note, spend, recorded decisions, brief-vs-spec differences, caveats -- are in [`data/reports/pareto.md`](data/reports/pareto.md) and [`data/reports/pareto.json`](data/reports/pareto.json).
<!-- END PARETO -->

## How the benchmark was assembled

MAUD provides the lawyers' passages as text, not as positions inside the agreements. The pipeline downloads the 152 agreements, turns each into one canonical text, finds the section headings, and locates every expert passage inside its document with fuzzy matching, since the published excerpts differ from the originals in quotes, page markers, and joins. Coverage is reported per question.

From the 139 agreements the parser handles well, a seeded rule selects 20: five for development (58 cases) and fifteen for the frozen test set (167 cases). Forty more cases test whether the agent knows when to say no: thirty where the lawyers' clause was deleted from the document, and ten plausible diligence questions no merger agreement answers, such as the target's cyber-insurance deductible.

Search runs locally on Qdrant with small open embeddings and BM25 keyword matching, combined by rank. A reranker and a multi-query variant were tried in the tournament and did not earn their latency.

## How it was built

A requirements brief, `specs/grilled-product-brief.md`, was written first and has been the authority since. A small software factory under `adws/` then built the project milestone by milestone: a planning model writes a plan, a coding model implements it, deterministic checks run the tests, and a reviewing model rules on every acceptance item before a documenter writes up the milestone in `docs/milestones/`. An outer loop moves from one milestone to the next and stops for a person only when there is a real decision to make.

Traces of the factory's own runs are in Braintrust under `sssf-dealpoint`; the evaluation experiments are under `dealpoint-eval`. The local result files in `data/results/` and the reports in `data/reports/` are the permanent record.

### Framework roles

Four different tools do four different jobs, and none of them stand in for another:

```
custom Python  --> benchmark truth      (parser, canonical offsets, MAUD gold spans, scorers)
LlamaIndex     --> RAG lab / RAG eval   (retriever composition, RetrieverEvaluator, synthetic queries)
DeepEval       --> independent agent-eval cross-check (task completion, tool correctness, step efficiency)
Braintrust     --> traces / experiments / comparison (surface, not source of truth)
local reports + Git --> permanent evidence (data/reports/, data/results/, specs/)
```

MAUD gold-span overlap is the only thing that ever decides a winner. LlamaIndex evaluates and generates synthetic queries against that same truth (`data/reports/li_rag_eval.md`); DeepEval cross-checks the agent traces the same fixed judged subset saw (`data/reports/deepeval_crosscheck.md`); Braintrust holds the traces and experiment comparisons, recreated from Git/local sources by `just braintrust-sync`. See `docs/demo-walkthrough.md` for a guided tour and `docs/braintrust-queries.md` for the BTQL investigations run against it.

## Try it

You need `uv` and an `OPENROUTER_API_KEY` in `.env`.

```
just data          # download MAUD, align the expert passages, build the case sets
just test          # offline test suite, no network, no spend
just index         # chunk and embed the selected agreements
just tournament    # retrieval tournament on the dev set, no model calls
just eval D z-ai/glm-5.3-flash test --limit 5
just report
just braintrust-sync # recreate Braintrust datasets/experiments/traces from local sources
```

Running `just data` twice changes no committed file, and a test checks that. Any run that costs money refuses to start if it would push spending past `MAX_OPENROUTER_SPEND_USD`.

## Where to look next

- `docs/milestones/` explains what each milestone set out to do and what it delivered
- `data/reports/four_arm.md` has the full comparison of the four versions
- `data/reports/tournament.md` has the search-engine tournament
- `data/reports/judges.md` has the judge panel and its agreement statistics
- `data/reports/li_rag_eval.md` has the LlamaIndex RAG lab's native evaluation and synthetic-query study
- `data/reports/deepeval_crosscheck.md` has the DeepEval independent agent-eval cross-check
- `docs/demo-walkthrough.md` is a guided tour through datasets, RAG lab, agent systems, traces, scorers and economics
- `docs/braintrust-queries.md` has the six BTQL investigations, exact queries and results
- `specs/grilled-product-brief.md` is the requirements document

## Scope

This task is harder than the published MAUD benchmark, which gives the model the relevant passage; here the model has to find it in the whole agreement, so scores are not comparable to MAUD leaderboard numbers. This is an evaluation project, not a product: there is no interface beyond the command line, and the sample sizes are those of a few-dollar budget.

## Layout

```
dealpoint/      parser, alignment, corpus, agent loop, tools, scorers, reports
data/eval/      committed case sets and the frozen test subset
data/results/   result rows and the spend ledger
data/reports/   generated reports and version stamps
docs/           milestone records
skills/         the review procedure the agent follows in version D
specs/          the brief and the milestone specs
adws/           the software factory
```

MAUD is by The Atticus Project, CC BY 4.0, Zenodo record 7500064.

