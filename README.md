# dealpoint-eval

An M&A deal-point review agent, built to show what a serious evaluation of an agent looks like when the ground truth is strong enough to make LLM judges optional.

The agent reads a real public merger agreement (70k to 100k tokens), answers one of 12 deal-point questions from the ABA Public Target Deal Points Study, and cites the clause it relied on. Expert lawyers have already answered the same questions for 152 agreements (MAUD, CC BY 4.0). That gives every run three checkable facts at once: the answer, the clause the experts cited, and whether the agent's quote is really in the document.

## What gets measured

The headline number is grounded accuracy: the agent gave the expert's answer and quoted a passage that overlaps the expert's clause. A right answer with a wrong citation does not count. Around it sit a dozen deterministic metrics, all plain Python over the expert labels: citation verbatim, fabrication, abstention on cases where the evidence was removed, whether the agent looked up the defined term it needed, tool calls, cap hits, cost, and latency.

Model judges exist too, but as a second tier that is never mixed with the first. Three judges from families outside the candidate slate score a blinded subset on a rubric frozen before any judging. The report shows how well they agree with each other and with the objective numbers. A human calibration form for the same traces lives in `data/eval/calibration/`.

The comparison itself has four arms that differ by one variable each. A is a single retrieve-and-answer call. B adds a three-tool agent loop. C swaps in the retriever that won an LLM-free tournament on the dev set. D adds a written review procedure the agent must follow. A fifth experiment holds arm D fixed and varies only the model.

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

Every number above is regenerated from local result files by `just report`. The frozen 32-case subset deliberately prefers questions the majority answer gets wrong, so the majority baseline is 0% by construction. The retrieval tournament, the judge calibration, and the model comparison have their own reports under `data/reports/`.

## How the benchmark is built

MAUD ships expert annotations as text spans, not as full agreements, and the spans are not verbatim copies of the documents. The pipeline downloads the 152 agreements from Zenodo, normalises them into one canonical text per agreement so every offset means the same thing everywhere, parses section headings, and aligns each expert span back into the document with fuzzy matching. Alignment succeeds on more than 97% of spans. Coverage and misses are reported per question rather than hidden.

The 20 agreements in the benchmark come from a seeded selection over the 139 that the section parser handles well. The 12 questions were chosen for reasoning diversity and answer balance. Five agreements form the dev set (58 cases), fifteen the frozen test set (167 cases), and 40 counterfactual cases test abstention: 30 with the expert's clause deleted from the document, 10 plausible diligence questions no merger agreement answers.

Retrieval runs on Qdrant in local mode with `bge-small` embeddings and BM25, fused by reciprocal rank. The tournament measured a reranker and multi-query fusion as well. Hybrid fusion won on canonical queries at the same latency as dense alone. The reranker added nothing for 13 times the time. Details are in `data/reports/tournament.md`.

## How it was built

A grilling session turned a one-page idea into `specs/grilled-product-brief.md`. From there a software factory (SSSF, under `adws/`) built each milestone: an Opus planner writes a plan, a Sonnet builder implements it, deterministic gates run the suite, ruff, pyright, and the milestone's own tests, an Opus reviewer rules on every definition-of-done item, and a documenter writes the record under `docs/milestones/` from the diff and the evidence. A thin outer loop carries the project milestone by milestone and stops for a human only on a real decision. The reviewer sent work back on most milestones, and one escalation ended with a spec being corrected rather than a number being tuned.

The factory's own traces are in Braintrust under `sssf-dealpoint`. The evaluation experiments are under `dealpoint-eval`. Local JSONL under `data/results/` and the reports under `data/reports/` remain the permanent record.

## Reproduce

Install `uv` and set `OPENROUTER_API_KEY` in `.env`. Then:

```
just data          # download MAUD, align spans, select agreements, build case sets
just test          # offline suite: no network, no model spend
just index         # chunk and embed the selected agreements
just tournament    # LLM-free retrieval tournament on dev
just eval D z-ai/glm-5.3-flash test --limit 5   # one arm, one model, a few cases
just report        # regenerate every table above from data/results
```

`just data` run twice changes no committed byte. A test asserts it. Metered runs refuse to start when the projected spend would exceed `MAX_OPENROUTER_SPEND_USD`.

## Reading the evidence

- `docs/milestones/*.md` says what each milestone set out to do, what shipped, which acceptance items were met, and what it cost.
- `data/reports/four_arm.md` is the arm comparison with the scored count next to every percentage.
- `data/reports/tournament.md` is the retrieval tournament.
- `data/reports/judges.md` is the judge panel and its agreement statistics.
- `specs/grilled-product-brief.md` is the authoritative requirements document.

## Caveats

This is a harder task than the published MAUD benchmark, which hands the model the relevant span. Scores are not comparable to MAUD leaderboard numbers. The whole evaluation ran on a few dollars of OpenRouter credit, so the test subset is 32 cases and most percentages rest on 10 to 20 scored cases. Read directions, not decimals. The first four-arm sweep was dominated by execution failures in the harness. The report keeps that run as v1 and shows the failure rates before and after the repair.

## Layout

```
dealpoint/      parser, alignment, corpus, agent loop, tools, scorers, reports
data/eval/      committed case sets and the frozen test subset
data/results/   canonical result rows and the spend ledger
data/reports/   generated reports and version stamps
docs/           milestone records and walkthroughs
skills/         the M&A review procedure the agent follows in arm D
specs/          the brief and the milestone specs
adws/           the software factory
```

MAUD is by The Atticus Project, CC BY 4.0, Zenodo record 7500064.
