# The DealPoint tour

Ten minutes, eleven stops, one contract question you will get sick of hearing. Everything you click was made by `just braintrust-cockpit` from files in Git; nothing here exists because someone clicked it once. Numbers come from `data/reports/`, Braintrust is the window we look through.

Open two tabs before you start:

- the dashboard: https://www.braintrust.dev/app/Vibeset%20Technologies/p/dealpoint-eval/dashboards/b3122524-c555-4599-8544-f302bc284b52
- the project: https://www.braintrust.dev/app/Vibeset%20Technologies/p/dealpoint-eval

Say this once, at the top: "Custom Python decides who wins. LlamaIndex, DeepEval and Braintrust are there to check its work and to show it off."

## Stop 1: the question everyone gets asked

Open the dataset `maud-dealpoint-dev-v1`.

Every row is a real merger agreement and a question a lawyer actually cares about, with the expert-marked span of the contract that answers it (that is MAUD, the public benchmark underneath all this). 58 dev cases, 167 test cases, 40 counterfactuals where the honest answer is "this contract does not say". The counterfactuals are the trap: a system that never abstains looks great right up until it invents a clause.

Pick one row and read the question aloud. "Does the definition of Knowledge include constructive knowledge?" You will see this question again. Several times. That is the point.

## Stop 2: the retriever bake-off

Open the view `RAG tournament`, then the chart "RAG tournament: hit@5, hit@10 and MRR by retriever" on the dashboard.

Six ways of finding the right paragraph, all frozen from milestone 3 and wrapped in LlamaIndex so its own evaluator can score them next to ours:

| retriever | hit@5 | hit@10 | MRR |
|---|---|---|---|
| dense | 81.0% | 84.5% | 0.626 |
| bm25 | 86.2% | 94.8% | 0.673 |
| hybrid (rrf), the winner | 91.4% | 96.6% | 0.732 |
| hybrid + rerank | 91.4% | 94.8% | 0.727 |

The fun part is the agreement: LlamaIndex's hit rate and our gold-span hit rate line up perfectly (Spearman 1.0) for the six wrapped configs, and the one genuinely native LlamaIndex config that disagreed with ours did so for a boring, real reason (duplicate relevant chunks), which we wrote down. Cross-checks that agree are only interesting when they can disagree.

One-liner: "Hybrid wins, reranking does not buy anything here, and two independent scorers say the same thing."

## Stop 3: four ways to build the agent

Open the experiments that start with `A-`, `B-`, `C-`, `D-` (the unsuffixed ones, for example `A-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc`).

Same question set, same retriever, four architectures: A answers in one shot from the top hits, B and C add structure, D is the full agent loop with tools (`search_agreement`, `get_section`, `lookup_defined_term`). Every experiment carries the same six scores, so the "obj/grounded_accuracy by arm, grouped by model" chart is the whole comparison in one picture.

Point at the surprise: on the cheap workhorse model, arm A (0.76 grounded accuracy) is not embarrassed by arm D. The agent loop wins when it finds the defined term and loses when it goes searching in circles. Hold that thought for stop 5.

## Stop 4: the model shootout

Open the `pareto-*` experiments and the "$/case by model" chart.

Arm D fixed, model swapped: GLM 5.3 flash, Haiku 4.5, DeepSeek v4 flash, Qwen 3.7 flash, Gemini 3.1 flash lite. Two land on the frontier (best accuracy for the money): Qwen 3.7 flash and DeepSeek v4 flash (0.69 grounded accuracy). Every metered call is in a local ledger, so the dollar axis is real money, not list price.

One-liner: "The expensive model is not on the frontier. That is why we keep the cheap ones in the race."

## Stop 5: watch one agent think

Open the experiment `m7-representative-traces`.

Six traces, chosen by a rule (not by us cherry-picking), one per failure shape: a clean success, a retrieval rescue (the second search found what the first missed), a defined-term cross-reference, an inefficient trajectory, a wrong answer, and an abstention on a counterfactual. Expand one: `case > agent > search_agreement > lookup_defined_term > final_answer > scoring`, replayed from the stored trajectory with zero model calls.

Open the inefficient one and count the tool calls. Then say: "This is the whole bug. It found section 12.1 on call two and kept searching until it hit the cap." That sentence is milestone 8.

## Stop 6: three cheap judges

Open `judge-D@haiku` (and its siblings `judge-A@haiku`, `judge-D@glm`, and so on).

Deterministic scores tell you whether the answer matched the gold span. They cannot tell you whether the reasoning was any good. So three small models from three families that are not in the candidate slate (Mistral, NVIDIA, ByteDance) each read the same blinded packet, the question, the trajectory, the finding, and for the evidence dimension the gold span, and score four things from 1 to 5: reasoning, evidence, trajectory, professional quality. One call per trace, all four numbers back as JSON. 108 traces, 324 calls, about a dollar.

Show the "judge/<dimension> mean by variant" chart. The judges are not the truth. They are cheap, consistent readers whose blind spots we are about to measure.

## Stop 7: the lawyer versus the judges

Open the experiment `m7b-hero-case`, and the view `Judge disagreement`.

The hero case is `contract_32__q04`, picked by rule: arm A and arm D disagree on it (A wrong, D right), and it is where the three judges disagree most with each other (a spread of 4 on a 5-point scale). Expand `scoring` and you get the three judge spans and the aggregate, with the lawyer's scores on the same rows.

Now the table that makes the whole exercise worth it. A lawyer (24 packets, blind) against the three judges on the packets where legal judgment mattered:

| packet | what happened | lawyer | Mistral | NVIDIA | ByteDance |
|---|---|---|---|---|---|
| `32fc075d8413` | answered a definition question from clauses that merely use the term | 2/2/3/1 | 4/1/4/4 | 4/5/4/4 | 4/3/5/4 |
| `790521a3adab` | said "no carve-out" from the intro to the list, without the list | 2/2/3/2 | 4/4/4/4 | 5/5/4/5 | 4/3/5/4 |
| `01de6203baeb` | quoted the actual Knowledge definition, clean | 5/5/4/5 | 5/5/4/5 | 5/5/4/5 | 5/5/4/4 |

All three judges gave 4s and 5s to two confidently wrong legal answers. They reward a fluent, well-cited answer without checking that the citation proves the specific claim. Agreement overall is high (Spearman 0.99 on reasoning, 0.95 on evidence), but that is inflated by fourteen cap-hit packets everyone scores 1. Trajectory is where the judges genuinely cannot do it (weighted kappa 0.55; Mistral alone 0.18), because the rubric gives credit for targeted progress even when the run failed, and only the human noticed.

Say it plainly: "Three judges agreeing is not evidence. One lawyer disagreeing for a reason you can write down is."

## Stop 8: the second opinion

Open `deepeval-crosscheck`.

DeepEval reads the same 108 traces with its own metrics (task completion, tool correctness, step efficiency as a written-down GEval). Correlation with grounded accuracy is weak (0.21), correlation with the judges is high (0.84) but contaminated: its evaluator is one of the three judge models. Verdict, recorded in the report: keep it as optional analysis, not as a gate. Frameworks that get tested and demoted on evidence are the healthy kind.

## Stop 9: ask the logs a question

Open the views `Failure attribution`, `Retrieval rescue`, `DeepEval vs judge disagreement`, `Trajectory inefficiency`.

Each is a saved BTQL query, the same text as in `docs/braintrust-queries.md`, so the doc and the view cannot drift. `Failure attribution` groups cap-hits and execution failures by model and arm; it is the numeric version of stop 5.

Three things need a human in the browser, and this is the moment to do them (each takes a minute; record it with the command so the walkthrough doc picks up the URL):

1. Loop: on the hero case, ask "which model and arm combinations hit CAP_HIT or EXECUTION_FAILED, and how often?" and save the thread. `just demo-manifest-record --step 2 --url <thread-url> --note "<does it match Failure attribution>"`
2. Playground: the hero packet against the published prompt `judge-calibrated-rubric`, the three judge models side by side. `just demo-manifest-record --step 3 --url <playground-url> --note "<match stored judge JSON?>"`
3. Custom trace view: ask Loop for "the three judge spans as a 3 by 4 grid of scores with the human row beneath", save it project-wide. `just demo-manifest-record --step 4 --tv <tv-param> --note "<what it showed>"`

Then `just demo-walkthrough` and the generated doc carries the links.

## Stop 10: the dashboard

Back to the first tab. Six charts, top to bottom: accuracy by arm and model, judge dimensions by variant, judge versus human on the review set, the retriever bake-off, dollars per case, DeepEval agreement. If you only get one screen, this is it.

## Stop 11: what it cost

Braintrust's monthly score quota went in a day, and not on this demo: every re-sync minted a copy of every experiment and re-logged every score. The fix is in the code now (dry run by default, `--live` to write, a hard cap of 600 scores per run, and a ledger so a score is never written twice). The entire milestone that built this demo wrote 104 scores, about 26 cents, and the re-run that added the six trace trees wrote zero.

## If someone asks

- "Why isn't Braintrust the source of truth?" Fourteen-day retention on this plan; the reports in Git are forever, and one command rebuilds the Braintrust side.
- "Why three judges instead of one good one?" Because one judge's blind spot looks like a fact. Three disagreeing is a signal; three agreeing with each other and not with the lawyer is a better one.
- "Why so many cap-hits?" Retrieval control flow, not model IQ. The agent finds the definitions section and keeps searching. That is the next milestone, and the lawyer's review is its spec.
- "Is the lawyer's scoring real?" AI-drafted, reviewed and adopted by the lawyer, provenance recorded next to the scores. Say that, do not hide it.

Dates: the scored experiments were created on 2026-09-06; on this plan they expire around 2026-09-20. Record the demo before then, or re-sync (a few hundred scores) after.
