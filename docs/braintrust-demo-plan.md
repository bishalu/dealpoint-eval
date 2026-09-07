# Braintrust demo project: plan

Goal: a clean Braintrust project you can walk tab to tab, where every tab shows a real Braintrust strength
on DealPoint data, built by one reproducible command, for the least money possible.

## What is wrong with the current project, with causes

| symptom | cause (verified) | fix |
|---|---|---|
| Logs tab has 1 trace | Everything was logged as *experiments* (offline evals). Logs are for traces; we never sent any. | Replay the 108 judged traces plus the six representative ones into Logs, score-free (free; they enable Topics, Patterns, Debugger, Loop). |
| Dashboard charts are empty | Two causes. (1) Custom dashboards are a Pro feature; Starter shows only the built-in "Cost and quality". (2) Row-level metadata is null (`metadata.arm`, `metadata.config` live on the experiment, not the rows), so every `group by metadata.*` is empty. | Put arm/model/config/case_type on every row. Use the built-in dashboard plus the Experiments comparison views, which is where Braintrust's value is anyway. |
| Experiments tab is a mess, many blank | Three re-syncs on the old project minted 58 suffixed stub copies (1 row, no scores) after the quota hit, plus probe experiments. | New project. One experiment per system. Same 18 cases everywhere so case-by-case comparison lines up. |
| "What am I comparing to what?" | Experiments were never grouped or named for comparison. | Name by question: `arm-A` .. `arm-D` (same model), `model-<name>` (same arm), `rag-<retriever>`, `judge-<variant>`. Tag `stage`. Pin the four comparison views. |
| Prompts tab is cluttered | 3 real prompts plus 12 `*-probe`/`*-testrun` leftovers. | New project gets exactly: base agent prompt, arm-D skill injection, judge rubric, and three arm-A prompt variants for the Playground A/B. |

Correction to something I said earlier: experiments keep up to 365-day retention on every plan. Only logs and
playground data expire after 14 days on Starter. The scored experiments are not about to vanish.

## Answers to your seven questions

1. **One log**: see table. Logs = traces. We will fill them, free.
2. **Empty dashboards**: Pro-only feature plus null row metadata. The demo's comparison surface is the Experiments page (case-level diff, Summary and Grid layouts, "Comparison grade"), not dashboards.
3. **Messy experiments**: new project, one copy each, 18 shared cases, four saved comparison views.
4. **Why dataset first, and what to say**: the dataset is the unit everything else joins on. Every experiment is a run over the same dataset rows, which is what lets Braintrust line up arm A vs B vs C vs D on the *same question* and grade improvement vs regression per case. The line: "One versioned set of real contract questions with the expert-marked answer span. Every system we build is a run over it. When production produces a stupid answer, that trace becomes a new row here in one click, and it stays a regression test forever." That last sentence is the feature customers buy.
5. **Prompts**: Playground A/B/C/D on prompt variants is a natural stop: three variants of the arm-A prompt (terse, cite-first, abstain-first) against the 18 cases with the cheap model, scored live. Cost: about $0.10 of OpenRouter, a few hundred scores.
6. **Scorers and parameters**: today scoring is code in Git, run locally, results logged. In the new project: (a) the six deterministic scorers pushed as real Braintrust code functions via `bt functions push` (the SDK cannot, the CLI can), so Playground and online scoring can run them; (b) the judge rubric as a Braintrust **LLM scorer** (`bt scorers create`, choice scores 1..5 per dimension), so a judge is a versioned object you can point at any run; (c) the four arm configs as **Parameters** objects (retriever, top_k, max_tool_calls, skill on/off, model), which is the "controlled experiment" story: change one parameter, run, compare.
7. **Tools, SQL, Loop**: Tools: no. Ours need the 68 MB local index; documented as a deliberate non-goal. SQL: yes, free, the six BTQL investigations become saved queries. Loop: yes, worth two threads (an investigation and a chart), paid from the $10/month model credit.

## The demo slice (cost control by design)

Everything runs on the **18 judged cases** (`subset_hash 5918ef10a7e6`) except the RAG lab, which is over the 58 dev queries. Same rows everywhere is what makes comparison work, and it is also what makes it cheap.

| object | count | scores each | scores |
|---|---|---|---|
| arm-A/B/C/D @ GLM | 4 x 18 | 6 | 432 |
| model-* (arm D, 5 models) | 5 x 18 | 6 | 540 |
| rag-* (6 retrievers) | 6 x 58 | 2 (hit@5, mrr) | 696 |
| judge-* (6 variants) | 6 x 18 | 4 | 432 |
| human/* | 24 packets | 4 | 96 |
| deepeval-crosscheck | 108 | 2 | 216 |
| hero case | 2 trees | 4 | 8 |
| playground A/B/C (later, optional) | 3 x 18 | 6 | 324 |
| **total** | | | **~2,400 (~2,750 with Playground) = about $6 to $7** |

Zero-score items: 114 traces into Logs, datasets (5), prompts (6), parameters (4), scorers (7), views (8),
SQL queries (6), one Loop thread, Topics on the logs (from the $10 credit; needs 100+ traces, we have 114).
Processed data: well under 100 MB against the 1 GB month.

Hard rules carried over: dry-run default, `--live` opt-in, exact score count printed before the first write,
cap 3,000 for this project, ledger so nothing is written twice, a tripwire on the old project's full sync.

## Tab-by-tab: what each stop proves

| tab | what you show | Braintrust strength |
|---|---|---|
| Datasets | `judged-18` with expected gold span; click one row | shared, versioned test cases; trace to dataset |
| Experiments | select arm-A..D, Summary table, then Grid on the Knowledge question | case-level comparison, improvement/regression grading |
| Experiments | select model-*; sort by $/case | same view, different axis: model economics |
| Experiments | select rag-*; hit@5 and MRR side by side | retriever bake-off with two scorers agreeing |
| Logs | 114 traces; open the inefficient one; open the hero one | nested agent tracing: query, tools, evidence, scoring |
| Scorers | six code scorers plus the LLM judge; invoke one on a log | code + LLM + human scores on the same cases |
| Review | the 12-trace review set flagged; the one Starter review score configured | SME labels feeding the loop |
| Prompts and Parameters | versioned prompts; the four arm parameter sets | first-class objects, not files |
| Playground | arm-A prompt x 3 variants over judged-18, scored live | iterate before coding |
| SQL | the six investigations | serious querying of nested data |
| Topics, Patterns, Debugger | topics over the logs; one Pattern (trajectory inefficiency); Debugger on the hero trace | discover, diagnose, without knowing the query in advance |
| Loop | "which arm/model combos hit the cap most, and why?" | AI-native investigation |
| Dashboard | built-in Cost and quality | monitoring (custom dashboards are Pro) |

## Vehicle: me, directly, with the CLI and MCP, as code in the repo

Not an SSSF milestone. This is a curated, human-in-the-loop configuration job where the verification is
"does the tab look right", which I can check through the MCP after every step; the factory's reviewer
cannot see the UI and would re-derive everything from files. It stays reproducible because it lands as code:
`dealpoint/eval/braintrust_demo.py` (one module, `--project dealpoint-demo`, dry-run default, `--live`, cap,
ledger) plus `just braintrust-demo`. SSSF builds the product; this builds the showroom.

Steps, in order (each verified via MCP before the next):
1. `bt projects create dealpoint-demo`; AI provider (OpenRouter key) set in org settings so Playground/judge scorer run on your key, not credits.
2. Datasets: judged-18 (with expected = gold span), dev-58, review-set-12, counterfactual-40 (rows only, no scores).
3. Experiments with row-level metadata, from stored results; one `--live` run; ledger; exact count printed first.
4. Logs: 108 judged + 6 representative traces, score-free.
5. Scorers: `bt functions push` the six code scorers; `bt scorers create` the judge; Parameters x4; Prompts x6.
6. Views: four comparison views, review flag on the 12, saved SQL x6; Topics enabled on logs; one Pattern; one Loop thread and one Loop chart (with you, 5 minutes).
7. `just demo-walkthrough` regenerated for the new project; old project left as is (or emptied of stubs if you say so).

Time: about 3 hours of my work plus 10 minutes of yours in the browser.
