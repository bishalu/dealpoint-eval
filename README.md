# dealpoint-eval

Can an AI agent read a 90,000-token merger agreement and tell you, with a citation, whether the deal has a go-shop? And when it says yes, is the clause it quotes the one an M&A lawyer would have pointed at?

This repo is a working answer to a narrower question I care about more: what does it take to *know* whether such an agent works, rather than to feel that it does. The ground truth here is unusually good. MAUD (Atticus Project, CC BY 4.0) has expert lawyers answering 92 deal-point questions across 152 public merger agreements, and it records the exact passage they relied on. So every run can be checked three ways at once, deterministically: the answer, the clause, and whether the agent's quote is really in the document. No LLM judge is needed to score correctness. I still built a judge panel, but it sits in a clearly separate tier, and the report measures how much it agrees with the numbers that don't need it.

Everything below was produced for about three dollars of API credit, by a small software factory that I mostly watched.

## The one number, and the ones around it

Grounded accuracy: the agent gave the expert's answer *and* quoted a passage overlapping the expert's clause. A right answer with a wrong citation scores zero, because in a law firm that is the failure that gets you fired.

Around it: citation verbatim, fabrication, abstention on cases where the evidence was surgically removed from the document, whether the agent bothered to look up the defined term the question turns on, tool calls, cap hits, cost, latency. All plain Python over expert labels.

Then the second tier. Three model judges from families outside the candidate slate score a blinded subset on a five-level rubric that was frozen before any judging happened. They agree with each other at κ ≈ 0.78. They agree with the objective numbers at ρ ≈ 0.3 to 0.5. That gap is the interesting part, and it is exactly why judges are secondary here.

## Four arms, one variable each

| arm | what changes |
|---|---|
| A | one retrieve-then-answer call. The RAG everyone builds first. |
| B | a three-tool agent loop: search the agreement, open a section, look up a defined term. 8 calls max. |
| C | same agent, with the retriever that won a tournament on the dev set |
| D | same agent, plus a written review procedure it has to follow |

Then a fifth experiment holds D fixed and swaps only the model.

The finding I did not expect: the single-call pipeline is competitive with the agent on the cases it answers. What the agent buys is coverage, and it pays for it by hitting the tool cap on a third of cases. The written procedure did not raise accuracy, but it halved the cap hits. Agency isn't free, and a skill document is a leash before it is a brain.

## Results

<!-- BEGIN RESULTS -->
**v2 after harness repair.** Results below are from a **budget-scaled** 32-case frozen discriminative subset of MAUD's test set (18 cases at `anthropic/claude-haiku-4.5`), not the brief's full 167-case design; the task here is document -> (answer, citation) while MAUD's published task is span -> answer, so scores are **not comparable** to MAUD leaderboard numbers, and every metric below is **objective** (deterministic Python over expert labels) -- M4 has no model judging.

Majority-baseline overall accuracy: 0.0%.

Majority-baseline accuracy on this subset is 0.0% by construction: data/eval/test_subset_v1.json's selection rule prefers cases where gold_answer != majority_answer (the majority already gets these wrong), so a near-zero subset baseline is not evidence the majority baseline is generally weak. For context, the majority baseline on the full 167-case MAUD test set is 46.1%.

**Model `anthropic/claude-haiku-4.5`:**

| arm | n_cases | n_scored | grounded_accuracy |
|---|---|---|---|
| A | 18 | 10 | 50.0% (10/18) |
| D | 18 | 8 | 62.5% (8/18) |

Overall EXECUTION_FAILED rate by arm: A=0.0%, D=5.6%. Overall CAP_HIT rate by arm: A=0.0%, D=50.0%.

**Model `z-ai/glm-5.3-flash`:**

| arm | n_cases | n_scored | grounded_accuracy |
|---|---|---|---|
| A | 32 | 17 | 76.5% (17/32) |
| B | 32 | 17 | 58.8% (17/32) |
| C | 32 | 21 | 66.7% (21/32) |
| D | 32 | 17 | 64.7% (17/32) |

Overall EXECUTION_FAILED rate by arm: A=25.0%, B=0.0%, C=0.0%, D=15.6%. Overall CAP_HIT rate by arm: A=0.0%, B=43.8%, C=31.2%, D=15.6%.

**v1 -> v2 EXECUTION_FAILED rate by (model, arm):**

| model | arm | v1 | v2 | delta |
|---|---|---|---|---|
| anthropic/claude-haiku-4.5 | A | 5.6% | 0.0% | -5.6% |
| anthropic/claude-haiku-4.5 | D | 88.9% | 5.6% | -83.3% |
| z-ai/glm-5.3-flash | A | 65.6% | 25.0% | -40.6% |
| z-ai/glm-5.3-flash | B | 18.8% | 0.0% | -18.8% |
| z-ai/glm-5.3-flash | C | 12.5% | 0.0% | -12.5% |
| z-ai/glm-5.3-flash | D | 34.4% | 15.6% | -18.8% |

- Residual failure **z-ai/glm-5.3-flash/A** (provider): 8/32 (25.0%) EXECUTION_FAILED; most common failure_detail class: 'empty response content' (4/8). 8/8 of these end with finish_reason == 'length' on both the initial attempt and the schema retry -- the model spends its whole MAX_TOKENS_FINAL=1200 budget (already doubled from 600 per this milestone's fix) on prose reasoning before ever emitting the JSON object, so raw_final_text is a truncated reasoning preamble with no JSON in it at all. This is the model's own verbosity, not a request-shape defect the harness can fix without an unbounded token ceiling (out of scope: CAP_HIT-style caps are deliberate, not to be raised without limit).
- Residual failure **z-ai/glm-5.3-flash/D** (provider): 5/32 (15.6%) EXECUTION_FAILED; most common failure_detail class: 'empty response content' (4/5). 5/5 of these end with finish_reason == 'length' on both the initial attempt and the schema retry -- the model spends its whole MAX_TOKENS_FINAL=1200 budget (already doubled from 600 per this milestone's fix) on prose reasoning before ever emitting the JSON object, so raw_final_text is a truncated reasoning preamble with no JSON in it at all. This is the model's own verbosity, not a request-shape defect the harness can fix without an unbounded token ceiling (out of scope: CAP_HIT-style caps are deliberate, not to be raised without limit). 5/5 of these failed cases (and 27/32 of this leg's cases overall) finalised immediately after a tool-calling turn whose finish_reason was 'length' -- MAX_TOKENS_TOOL_TURN (300) cut that turn off before the model could keep searching, so the loop moved to finalisation with fewer tool calls used than it would otherwise have made. This is left-at-300, unfixed for this v2 report (no budget for another sweep); it is a contributing cause alongside, not instead of, final-answer verbosity.

- At z-ai/glm-5.3-flash, arm D does NOT improve grounded_accuracy over arm C (66.7% -> 64.7%, delta -2.0%). See adherence, fabrication, abstention, trajectory and efficiency deltas below instead.
<!-- END RESULTS -->

`just report` regenerates every number above from local result files. The 32-case test subset deliberately prefers questions where the majority answer is wrong, so the majority baseline reads 0% by construction. Read directions, not decimals: most percentages rest on 10 to 20 scored cases.

## How the benchmark was made

MAUD ships the expert spans as text, not as offsets into the agreements, and the text is not verbatim. Page markers, joined excerpts, curly quotes. So the pipeline downloads the 152 agreements, normalises each into one canonical text (every offset in the project points into that text), parses section headings, and fuzzy-aligns each expert span back into place. Alignment holds on over 97% of spans; the misses are reported per question rather than smoothed over.

Twenty agreements were selected by a seeded rule from the 139 the parser handles well. Twelve questions were chosen for reasoning diversity: direct lookups, numeric terms, defined-term dependencies, cross-references, carve-outs. Five agreements make the dev set (58 cases), fifteen the frozen test set (167 cases). Forty counterfactual cases test abstention: thirty where the expert's clause was deleted from the document, ten plausible diligence questions ("what is the target's cyber-insurance deductible?") that no merger agreement answers.

Retrieval is Qdrant in local mode, `bge-small` embeddings, BM25, fused by reciprocal rank. The tournament also tried a cross-encoder reranker and multi-query fusion. Hybrid fusion won at the same latency as dense alone. The reranker bought nothing for 13 times the time, which I found satisfying. `data/reports/tournament.md` has the table.

## How it was built

A grilling session turned a one-page idea into `specs/grilled-product-brief.md`, and that brief has been the law since. A software factory under `adws/` built each milestone from it. An Opus planner writes a plan. A Sonnet builder implements it. Code, not an agent, runs the suite, ruff, pyright, and the milestone's own tests. An Opus reviewer rules on every definition-of-done item with a machine-readable verdict, and a documenter writes the milestone record from the diff and the evidence. An outer loop carries the project milestone by milestone and only stops for a human on a real decision.

It was not smooth. The reviewer sent work back on most milestones. One escalation ended with my spec being corrected rather than a threshold being tuned to a measured number, which is the right way round. The first four-arm sweep was mostly execution failures in my harness, not model quality; the report keeps that run as v1 and shows the failure rates before and after the repair. `docs/milestones/` tells the whole story milestone by milestone.

Factory traces go to Braintrust under `sssf-dealpoint`, evaluation experiments under `dealpoint-eval`. The local JSONL in `data/results/` and the reports in `data/reports/` are the permanent record.

## Run it

You need `uv` and an `OPENROUTER_API_KEY` in `.env`.

```
just data          # download MAUD, align spans, select agreements, build case sets
just test          # offline suite, no network, no spend
just index         # chunk and embed the selected agreements
just tournament    # LLM-free retrieval tournament on dev
just eval D z-ai/glm-5.3-flash test --limit 5
just report
```

`just data` twice changes no committed byte, and a test asserts it. Any metered run refuses to start if the projected spend would cross `MAX_OPENROUTER_SPEND_USD`.

## Where to look

- `docs/milestones/` for what each milestone set out to do, what shipped, and what it cost
- `data/reports/four_arm.md` for the arm comparison with scored counts next to every percentage
- `data/reports/judges.md` for the judge panel and its agreement statistics
- `data/eval/calibration/` for the blinded human calibration form
- `specs/grilled-product-brief.md` for the requirements, still authoritative

## What this is not

Not comparable to the MAUD leaderboard: MAUD hands the model the relevant span, this task makes it find the span in a 90,000-token document. Not a product. Not a large-n study. It is a small, honest, reproducible evaluation of an agent against expert ground truth, and a record of how much of the apparent signal was really harness bugs.

## Layout

```
dealpoint/      parser, alignment, corpus, agent loop, tools, scorers, reports
data/eval/      committed case sets and the frozen test subset
data/results/   canonical result rows and the spend ledger
data/reports/   generated reports and version stamps
docs/           milestone records
skills/         the review procedure the agent follows in arm D
specs/          the brief and the milestone specs
adws/           the software factory
```

MAUD is by The Atticus Project, CC BY 4.0, Zenodo record 7500064.
