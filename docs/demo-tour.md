# The DealPoint tour

Ten minutes, nine stops, three acts, one contract question. The story: how Braintrust took an agentic
legal application from a baseline to a system we can trust, with objective evaluation, a multi-judge panel,
human calibration, failure diagnosis and model economics. DealPoint is the case study; Braintrust is the
quality cockpit. Nothing you click was made by hand: two commands rebuilt this project in a fresh org
(`just braintrust-sync --live`, `just braintrust-showroom --live`), and every number below comes from
`data/reports/` in Git.

Project: https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval

**Eight dashboards, one question each.** Braintrust's dashboards are the monitor over project logs, so
they cannot read experiment scores and cannot put anything but time on a time-series axis. The showroom
therefore mirrors every number into log metadata (free; scores are the metered thing), every chart is a
ranked bar list with the groups on the axis, the dashboard's name is the question, and the time range is
pinned to 30 days so batch-ingested logs never fall out of a sliding window. Each dashboard carries a
description saying what it compares and how to read it top to bottom; rows are named for what they are
(Mistral Small, "lawyer: evidence"), groups for what they group (system, model, system@model).

| dashboard | what it ranks | link |
|---|---|---|
| DealPoint eval overview | the six headline charts, one per question below | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/58a7278a-71c8-4a71-9a5b-40ab019ee0c2 |
| Which system? | A pipeline+dense, B agent+dense, C agent+hybrid, D agent+hybrid+skill on the same 32 cases: correct outcome, cap-hits, fabrication, abstention, the loop on GLM vs Haiku, dollars, seconds, tool calls, correct outcomes per dollar | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/8e0003f4-5840-45b3-bc9a-86fcb686ec1a |
| Which model? | arm D on the same 18 cases, five models: correct outcome, dollars, seconds, p90, cap-hits, execution failures, per dollar, dollars per correct outcome, the Pareto list across every system@model, and the judge panel's view of the same race | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/7c13d289-97cb-4db6-8340-6494e85608a3 |
| Which retriever? | six retrievers on the 58 dev queries: hit@5, hit@10, MRR | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/32c5a4be-7898-4630-af11-ca40e743e9e4 |
| Judges and the lawyer | seven charts about judges only: can the panel be trusted (its mean next to the lawyer per dimension), which judge for which dimension (signed bias, closest to zero wins), what a call costs, and DeepEval as one second-opinion number | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/54eae7f9-0815-4497-8166-725929ddd416 |
| DeepEval | only what it uniquely adds: tool correctness, argument correctness and step efficiency by system@model for the agent arms (the baseline calls no tools, so it is excluded), how the agent worked, which nothing of ours grades | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards (DeepEval) |
| LlamaIndex | its own hit rate and MRR by retriever on the same 58 queries, including its native bm25 that our scorer cannot see | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards (LlamaIndex) |
| Which prompt? | the four arm-A prompts, judged on professional quality, evidence and reasoning | https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/dashboards/9c9b9fa9-4a54-47d9-9c6a-57e60dd009c5 |

Every ranking chart compares one case pool at a time (the five arm-D models on the same 18 judged cases,
the four systems on the same 32) and only canonical traces (`comparable = 1`: no representative picks,
live replays or partial runs). Accuracy there is "correct outcome over every case": an unanswered case
is wrong, a correct abstention on a counterfactual is right. It reads lower than the "of scored cases"
figures in the reports; both are true, say which one you mean. If you only get one screen, the overview
is it; every chart has a stop below that explains it.

**The spine.** One question, MAUD's `q05`: *Does the agreement's definition of Knowledge include
constructive knowledge?* Two options, "Constructive knowledge" or "Actual knowledge", plus "ABSTAIN" when
the agreement does not say. The majority of agreements say constructive, so a system that pattern-matches
gets it wrong on the ones that say actual. The case we follow is `contract_144__q05`: the agreement's
Section 9.03 limits knowledge to the *actual* knowledge of six named officers. Its twin is
`contract_39__redacted_q05`: the same question on an agreement whose Knowledge definition was redacted, so the
only right answer is to abstain.

**How to read the Experiments tab.** Every experiment carries the same metadata schema: `axis` (the question
it belongs to), `varies` (the one variable that changes along that axis), `holds` (what is fixed), plus
`arm`, `loop`, `retriever`, `skill`, `model`, `cases`. The **Arms A to D** view pins those columns; in any
other view turn them on once from the column picker and the table reads as a grid: SYSTEM (A to D, one config key per step), MODEL (arm D, five models), RETRIEVAL (six retrievers, one
scorer cross-check, one query-distribution check), JUDGE (six variants under three judges and one lawyer),
CROSSCHECK (DeepEval), PROMPT (four arm-A prompts), TRACES. The saved views below are those axes.

Say this once, at the top: "Custom Python decides who wins. Braintrust is where I look at the evidence,
compare case by case, and let a lawyer and three judges argue about it."

---

## Act I. Build a measurable agent

### Stop 1. The question, and the trap

Open the dataset `maud-dealpoint-playground-armA`, row `contract_144__q05`.
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/datasets/maud-dealpoint-playground-armA

This row is the exact packet the baseline system saw: the question block (MAUD question, plain-English
gloss, the two allowed answers plus ABSTAIN) and the five passages dense retrieval pulled for it. `expected`
is the expert answer, "Actual knowledge", followed by the gold span from Section 9.03. Every comparison in the
next nine minutes joins on rows like this one.

Then open `maud-dealpoint-counterfactual` and find `contract_39__redacted_q05`: same question, the
definition removed, `expected` = ABSTAIN. Forty of these. They are the trap: a system that never abstains
looks fine right up until it invents a clause.

Sizes, for the record: 58 dev cases, 167 test cases (32-case frozen subset for the sweeps), 40
counterfactuals, 18 judged cases, 106 synthetic queries, 12-trace review set.

One-liner: "Truth here is a character span a lawyer marked, not a model's opinion."

### Stop 2. Can it retrieve the right evidence?

Experiments tab, view **RAG tournament**.
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/experiments?v=c9ab8582-0d44-4381-a7a9-c203a1727d82

Six retrievers, same 58 dev queries, same index, two independent scorers on every row: our `obj/*`
(gold-span hit) and LlamaIndex's `li/*`.

| retriever | hit@5 | hit@10 | MRR |
|---|---|---|---|
| dense | 81.0% | 84.5% | 0.626 |
| bm25 | 86.2% | 94.8% | 0.673 |
| **hybrid_rrf** (winner) | **91.4%** | **96.6%** | **0.732** |
| hybrid_rrf + rerank | 91.4% | 94.8% | 0.727 |
| multi-query fusion | 82.8% | | 0.690 |
| fusion + rerank | 87.9% | | 0.718 |

On the Knowledge question specifically (five dev contracts), dense puts the definition at rank 1 on two of
them and at ranks 4, 7 and 2 on the rest; hybrid puts it at rank 1 on four of five. `rag-m7-synthetic`
re-asks the winner 106 LlamaIndex-generated questions (94.3% hit rate: it holds). `rag-m7-li-crosscheck`
is the honesty check: LlamaIndex's native scorer and ours agree at Spearman 0.99, and the three rows where
they disagree have a written cause (duplicate relevant chunks).

One-liner: "Hybrid wins, reranking buys nothing here, and two scorers that could disagree don't."

### Stop 3. Does agency help? Does better RAG help? Does the skill help?

Experiments tab, view **Arms A to D (same 32 cases)**. Select the four `*-z-ai_glm-5.3-flash-*-e3ee9cc`
experiments, open the Summary table, then the Grid, then search `contract_144__q05`.
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/experiments?v=693c6c36-ad62-4bde-9b14-87d4314f24aa

Four systems, one config key changed per step, all on the cheap workhorse model and the same 32 cases:

| arm | loop | retriever | skill | grounded accuracy (scored cases) | cap-hit | fabrication | execution failed |
|---|---|---|---|---|---|---|---|
| A | pipeline | dense | off | 76.5% (17/32 scored) | 0% | 0% | 25.0% |
| B | agent | dense | off | 58.8% | 43.8% | 11.1% | 0% |
| C | agent | hybrid_rrf | off | 66.7% | 31.2% | 19.0% | 0% |
| D | agent | hybrid_rrf | on | 64.7% | 15.6% | 35.0% | 15.6% |

The Grid row for `contract_144__q05` is the whole argument in one line: **A abstained** (it had the passages
and still said the agreement does not address it), **B, C and D answered "Actual knowledge"** with the
Section 9.03 quote. The loop finds the defined term the pipeline could not commit to.

Then say the honest aggregate. On this model, A's scored accuracy is highest but a quarter of its runs fail
to produce a finding at all; the loop trades execution failures for cap-hits (B) and fabrication (D). Better
retrieval (C) recovers accuracy; the skill (D) halves the cap-hits and raises fabrication. Each step is a
trade, and the trade is visible per case, not just in a mean. The four `arm-*-config` Parameters objects
are these rows as first-class, versioned configs.

One-liner: "Does agency help? On this question, yes. In aggregate it's a trade, and now the trade has a
grid."

---

## Act II. Correctness is not enough

### Stop 4. Watch one agent think

Logs tab, filter `metadata.case_id = 'contract_144__q05'`, open `contract_144__q05 | D@glm`.
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/logs

Expand the tree: `agent > lookup_defined_term("Knowledge") > search_agreement("Knowledge means the actual
knowledge of the individuals listed") > lookup_defined_term("knowledge") > search_agreement(...) >
search_agreement("due inquiry diligent search should have known reasonable inquiry knowledge") >
final_answer`. Five tool calls, 20.9k input tokens, 40 seconds, $0.0019. It found the definition on call
one and spent four more calls looking for a constructive-knowledge carve-out that is not there. That is
diligence; it is also the bill.

Now the twin: filter `metadata.case_id = 'contract_39__redacted_q05'` and open any `D@*` trace. Status
`CAP_HIT`: eight calls searching for a definition that was redacted, then the cap. Every arm-D model did
this; only the single-shot pipeline abstained. Hold that thought for stop 8.

266 traces live here (108 judged, 151 from the sweeps, six representative, one live replay), every one
replayed from stored trajectories with zero model calls, and zero scores: Logs are for looking, Experiments
are for scoring.

One-liner: "Same loop that verifies a definition loops forever when the definition is missing."

### Stop 5. What deterministic truth captures, and scoring in production

Scorers tab, then Configuration.
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/scorers

Six objective scores sit on every experiment row: `obj/grounded_accuracy`, `answer_correct`,
`citation_gold_overlap`, `citation_verbatim`, `abstain_correct`, `skill_adherence`. They are deterministic
Python over MAUD's expert spans, computed in Git, and they decide who wins. They cannot run inside Braintrust
(they need the corpus and the index), and that is fine: Braintrust displays them, Python owns them.

What deterministic truth cannot capture is whether the reasoning was any good. So the Scorers tab holds four
LLM judges, `judge-reasoning`, `judge-evidence`, `judge-trajectory`, `judge-professional`, each the frozen
M5 rubric for one dimension, on the same OpenRouter models the calibration used (Mistral Small for three,
ByteDance Seed for trajectory).

The payoff, under Configuration: the online scoring rule **"online: judge-professional on new logs"** runs
`judge-professional` on every new root log as it lands. Live action (30 seconds):

```
just braintrust-showroom --live --replay contract_144__q05:D@glm
```

One new trace appears in Logs with category `live-replay-<timestamp>`; within a minute a `Judge: professional`
score attaches to it. Fallback if OpenRouter's Mistral endpoint is rate-limited that minute (it was during
setup): the 266 existing logs already carry the score where it succeeded (average 0.65), and the trace from
the previous replay is scored, `779b4ef5-cde7-4211-b988-10bf7fb5a3c2`.

One-liner: "Offline the scorers grade experiments; online the same scorer grades production as it lands."

### Stop 6. Human review, three judges, and where they disagree (the peak)

Three views: **Review set (12)**, **Judges and the lawyer**, **Judge disagreement**.
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/experiments?v=f21728ce-ac6b-46c9-a88c-c35cf074b031
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/experiments?v=173420b9-ee31-4c91-83bd-f0dc5a358490
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/experiments?v=54edf525-6932-41b2-a8a6-8f2e35505c8b

Make the four roles explicit before showing any number:

- **Deterministic MAUD metrics** are benchmark truth: did the answer and citation match the expert span.
- **Human review** is the calibration reference: a lawyer scored 24 blinded packets on four dimensions,
  1 to 5. The Human review page has the same four sliders (`reasoning`, `evidence`, `trajectory`,
  `professional`), so a reviewer scores a flagged trace exactly the way the lawyer did; the 12 flagged
  traces are the review set.
- **The multi-judge panel** is the calibrated qualitative evaluator: three small models from three families
  not in the candidate slate (Mistral, NVIDIA, ByteDance), reading the same blinded packet, 108 traces,
  324 calls, about a dollar. `judge/*` on the `judge-*` experiments is their mean; `human/*` on the same
  rows is the lawyer.
- **DeepEval** is the independent framework cross-check (stop 6, last paragraph).

Open `judge-A@haiku` and `judge-D@haiku` side by side in the Grid on `contract_144__q05`:

| packet | system | lawyer (reasoning/evidence/trajectory/professional) | Mistral | NVIDIA | ByteDance |
|---|---|---|---|---|---|
| `0f0ddb6b0896` | A@haiku, abstained | 4 / 1 / 3 / 3 | 3 / 1 / 3 / 4 | 1 / 1 / 3 / 1 | 5 / 5 / 4 / 4 |
| `257db09a639b` | D@haiku, "Actual knowledge" | 5 / 5 / 3 / 5 | 5 / 5 / 4 / 5 | 5 / 5 / 4 / 5 | 5 / 4 / 4 / 4 |

On the good answer everyone agrees. On the abstention the panel splits by four points: ByteDance rewards a
tidy, well-hedged refusal; NVIDIA punishes it; the lawyer's read is "reasonable, but the evidence was right
there". A spread like that is information, not noise: it tells you which judge is credulous about fluent
hedging before you deploy that judge.

Which judge, then? The "Judges and the lawyer" dashboard shows each family's signed bias against the
lawyer per dimension (judge minus lawyer, 0 to 1 scale). Every bar is positive: all three judges run
high, everywhere. How high is the decision. On reasoning and evidence the bias is a rounding error
(0.01 to 0.06). On professional quality Mistral runs highest (0.07). On trajectory every judge
over-credits and Mistral most of all (0.29, versus 0.14 for NVIDIA and 0.16 for ByteDance): it gives
points for "targeted progress" on runs that produced nothing. Cost is a wash (all three within fifteen
thousandths of a cent per call; the whole calibration was fourteen cents). So the panel to deploy is a
per-dimension pick, not a best judge: Mistral for reasoning, evidence and professional quality, NVIDIA or
ByteDance for trajectory, which is how the four LLM scorers on the Scorers tab are assigned, with the
standing caveat that no judge is trustworthy on trajectory without the lawyer.

Then the twin, `contract_39__redacted_q05`: every arm-D run hit the cap, the lawyer gave them 1s, and the
judges agreed, except on trajectory, where Mistral gives 3s for "targeted progress" on a run that produced
nothing. Trajectory is where the judges cannot do the job (weighted kappa 0.55 against the lawyer); on
reasoning, evidence and professional quality they can (0.94, 0.95, 0.90 over 24 packets).

Now the reveal, the two packets where all three judges were wrong and the lawyer was right:

| packet | what happened | lawyer | Mistral | NVIDIA | ByteDance |
|---|---|---|---|---|---|
| `32fc075d8413` | A@haiku answered a redacted definition question from clauses that merely *use* the term | 2 / 2 / 3 / 1 | 4 / 1 / 4 / 4 | 4 / 5 / 4 / 4 | 4 / 3 / 5 / 4 |
| `790521a3adab` | A@haiku said "no carve-out" from the introduction to a list, without reading the list; the answer happened to be right | 2 / 2 / 3 / 2 | 4 / 4 / 4 / 4 | 5 / 5 / 4 / 5 | 4 / 3 / 5 / 4 |

Three judges gave 4s and 5s to two confidently wrong pieces of legal reasoning because the prose was fluent and
cited something. Only the lawyer checked whether the citation proved the claim. That is why the panel is
calibrated against a human and not trusted on its own, and why the human review page exists in the loop.

Close with DeepEval, on the **DeepEval** dashboard: open `deepeval-crosscheck` and the view **DeepEval vs judge disagreement**
(https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/experiments?v=db7fbd81-8c86-4a24-b048-a0fe27cac616).
DeepEval read the same 108 traces with its own metrics. Its task-completion score correlates weakly with
grounded accuracy (Spearman 0.21) and well with our judges, but its evaluator model is one of the three judge
models, so that agreement is partly contamination, and the report says so. Verdict on the record: keep it as
a second opinion, never as a gate.

Say it plainly: "Three judges agreeing is not evidence. One lawyer disagreeing for a reason you can write
down is."

### Stop 7. Iterate on the prompt before touching code

Playground. Prompts `arm-a-prompt-base`, `arm-a-prompt-terse`, `arm-a-prompt-cite-first`,
`arm-a-prompt-abstain-first`; dataset `maud-dealpoint-playground-armA`; scorers `judge-evidence`,
`judge-professional`, `judge-reasoning`; model GLM 5.3 flash via your OpenRouter key.
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/playgrounds

This is the PROMPT axis with everything else pinned: same 18 packets (the exact passages arm A retrieved),
same model, same judges, only the system prompt changes. Four prompts side by side, judged live, about ten
cents. Point at `contract_144__q05`: does "abstain first" make the baseline abstain more on the redacted
twin without abstaining here, where the definition is in the passages?

A second Playground, for the judges: the dataset `maud-dealpoint-judge-packets` holds the 24 packets the
lawyer scored, rendered exactly as the M5 judges saw them, with the lawyer's four scores as `expected` and
each judge family's scores in metadata. Open the prompt `judge-panel-rubric` (the frozen rubric) over it
with three model columns, Mistral Small, NVIDIA Nemotron and ByteDance Seed on OpenRouter, and you watch
the panel argue with the lawyer live on `contract_144__q05`; no scorer needed, `expected` is the answer
key. About two cents.

Pre-run fallback: the same four runs are saved as experiments `playground-arm-A-base`, `-terse`,
`-cite-first`, `-abstain-first` (axis PROMPT in the Experiments tab) and mirrored into the dashboard's two
*Which prompt?* charts. Result: cite-first wins on evidence (0.71 vs 0.61 for terse), abstain-first
costs evidence (0.60) for no professional gain, terse loses on every dimension. The base run is partial
(13 of 18 rows; Mistral's endpoint was rate-limiting the judges); to complete it, delete
`playground-arm-A-base` and run `just braintrust-showroom --live --run-playground --variants base`
(54 scores), then `--only promptlogs`.

One-liner: "The prompt is a versioned object, the packet is a dataset row, the judges are scorers: iterating
is a table, not a notebook."

---

## Act III. Turn evidence into a deployment decision

### Stop 8. Diagnose the real failure

Views **Failure attribution** and **Trajectory inefficiency**, then Logs on the twin.
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/experiments?v=a82535ee-8eb8-4bd4-83b0-9a183ab2a041
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/experiments?v=3fbc27c3-463f-4ff7-9b4b-60f11413d446

Failure attribution is the numeric version of stop 4: across the 266 logs, arm D hit the tool cap 54 times
in 149 runs, arm B 14 in 32, the pipeline never (it cannot). The "Which system?" dashboard's cap-hit chart
is the same picture over the Logs.

One investigation, three tools, in this order:

1. **Debugger** on `contract_39__redacted_q05 | D@glm` (Logs): step through the eight calls. Every one is a
   search for a definition that is not in the document; the loop has no notion of "I have looked
   everywhere".
2. **Topics** over the Logs (facet `DealPoint agent traces`, already configured): the clusters separate
   "found the defined term and answered", "searched in circles until the cap", "abstained". Cap-hit is a
   trajectory shape, not a model property.
3. **Loop**, one thread, saved: *"Across the project logs, which arm and model combinations hit CAP_HIT most,
   and what did those trajectories search for on their last three calls? Is there a stopping rule that
   would have ended them early?"* The answer is the spec for the next milestone: a "definition absent"
   stopping rule for the loop.

One Pattern records it so the next batch of logs gets matched against it automatically. Paste-ready
(Patterns has no REST path; create it in the UI or with the `new_pattern` MCP tool once `braintrust-demo`
is authorised):

- **Title:** Agent loops on a defined term the agreement does not contain, then hits the tool cap
- **Description:** On the counterfactual cases where the Knowledge (or another) definition was redacted,
  the arm-D agent keeps issuing `lookup_defined_term` and `search_agreement` calls for a definition that is
  not in the document until it hits the 8-call cap. Across the five arm-D models on the six redacted or
  out-of-scope judged cases, 25 of 30 traces ended in CAP_HIT, 4 abstained, 1 answered anyway (a
  fabrication); the single-shot pipeline abstained on 5 of 6. The loop has no "I have looked everywhere"
  stopping rule. Fix: a definition-absent stopping rule after two empty lookups.
- **Supporting traces (Logs ids):** `e9194ae8-2506-4101-b7ee-352cfef65d4a` (contract_39__redacted_q05,
  D@gemini), `69790cfb-4268-4dc5-8fdb-06d78033aba5` (contract_39__redacted_q05, D@qwen),
  `c164bda3-8235-4f45-a52a-476afc801f9a` (contract_99__redacted_q06, D@gemini),
  `ac2f94ad-5ad8-47d9-93b9-2fa53cf46ca5` (contract_99__redacted_q06, D@qwen),
  `d945f9c6-9962-4932-9164-0908f9c3a620` (contract_103__redacted_q08, D@qwen),
  `27ae52f0-0ca8-4a42-a212-f6286c9c21fa` (contract_75__redacted_q07, D@deepseek).

One-liner: "The bug is retrieval control flow, not model IQ, and the fix has a name before anyone writes
code."

### Stop 9. What should we deploy?

Experiments tab, view **Models (arm D fixed)**, sort by `$/case`.
https://www.braintrust.dev/app/bishal.ai/p/dealpoint-eval/experiments?v=8ea97ad4-ef60-40fd-a84c-7d5c444b93f1

The MODEL axis: arm D held fixed, five models, one row per case with realized cost and latency from the
local ledger, not list price. Two accountings exist and the demo names both: the milestone report's
"accuracy of scored cases" (a cap-hit or execution failure is dropped from the denominator) and the
dashboard's "correct outcome over every case" (an unanswered case is wrong, a correct abstention on a
counterfactual is right). The second is the deployment number.

| model | correct outcome, every case | accuracy of scored cases | cap-hit | execution failed | p50 latency | $/case | correct outcomes per $ |
|---|---|---|---|---|---|---|---|
| Gemini 3.1 flash lite | **46.9%** | 63.2% | 21.9% | **0.0%** | 13.4 s | $0.0055 | 85 |
| GLM 5.3 flash | 37.5% | 64.7% | **15.6%** | 15.6% | 33.1 s | $0.0028 | 135 |
| DeepSeek v4 flash | 28.1% | **69.2%** | 50.0% | 6.2% | 18.5 s | $0.0023 | 125 |
| Claude Haiku 4.5 (18 cases) | 27.8% | 62.5% | 50.0% | 5.6% | 25.8 s | $0.0458 | 6 |
| Qwen 3.7 flash | 22.6% | 63.6% | 48.4% | 16.1% | 13.9 s | $0.0011 | **206** |

The two accountings disagree on the winner, and that is the point of the stop. Scored accuracy crowns
DeepSeek because it only counts the runs that finished; counting every case, DeepSeek caps out half the
time and lands third. Gemini answers the most cases correctly, never fails to produce a finding, and is
as fast as anything on the slate, at twice Qwen's cost. Qwen is the cost floor: the most correct outcomes
per dollar, with the worst cap-hit and failure rates. Haiku is the same quality as GLM and DeepSeek at
sixteen times the cost, and the dashboard shows it at six correct outcomes per dollar.

The accuracy point is **safe accuracy**, the biggest number that survives scrutiny, read with one check
and one tie-breaker. **Safe accuracy**: the share of
cases where the system did not mislead, a correct answer, a correct abstention, or a silent failure
(abstained wrongly, hit the tool cap, failed to produce a finding). **Precision when it answers**: of the
cases it answered, how many were right; silence cannot inflate this one, which is why it always sits next
to safe rate. A misleading answer is a wrong answer, or any answer when the agreement does not address the
question. **Net accuracy**, correct minus misleading over every case, is the tie-breaker. In a legal tool
a silent failure costs a lookup; a misleading answer costs a client. The overview dashboard's first three
charts rank every system@model by these on the same 18 cases (D@gemini's safe accuracy is 83%; the agents run 83 to 89%):

| system@model, same 18 cases | safe accuracy | precision when answering | answered | quotes verbatim | net | $/case |
|---|---|---|---|---|---|---|
| **D@gemini** | 83% | 67% | 50% | **100%** | **33%** | $0.0055 |
| D@deepseek | **89%** | **71%** | 39% | 71% | 17% | $0.0024 |
| D@glm | 83% | 62% | 44% | 62% | 17% | $0.0028 |
| D@haiku | 83% | 62% | 44% | 88% | 11% | $0.0458 |
| D@qwen | 83% | 57% | 39% | 43% | 6% | $0.0011 |
| A@haiku | 67% | 45% | 61% | 100% | 22% | $0.0034 |

The decision, written down: **D@gemini, the agent on Gemini 3.1 flash lite**: safe on 83% of cases,
right two times in three when it speaks, every quote verbatim, and it speaks on half the cases. DeepSeek
is safer and more precise but answers only 39% of the time; the single-shot baseline on Haiku answers the
most and is the least safe, wrong one case in three. Silence is the failure we can afford. On GLM, the four systems net A 28%, C 25%, D 9%, B 6%:
the skill as written is the worst step in the ladder (it raises misleading answers to 28%), so the agent
to ship is C's shape on Gemini, with the stopping rule from stop 8 to convert cap-hits into answers, and
the online `judge-professional` score from stop 5 as the monitor after we ship. Haiku inside the loop is
out at sixteen times the cost.

One-liner: "Safe on five cases in six, right two times in three when it speaks, never a made-up quote. That is a tool a lawyer can use."

---

## Close (thirty seconds)

"Every object you saw was rebuilt from Git into a fresh Braintrust org today by two commands, inside the
free tier: 31 experiments, 7 datasets, 266 traces, about 4,000 scores, zero dollars. The implementation and
the evaluation infrastructure were built milestone by milestone through my SSSF software factory; Braintrust
is the quality cockpit the factory feeds."

---

## Before Wednesday (you, about twenty minutes)

Everything below needs a browser session or the `braintrust-demo` MCP OAuth; the code-side work is done.

1. Restart Claude Code and approve the `braintrust-demo` MCP OAuth for `bishal.ai` (the `braintrust` MCP
   entry is still bound to the old org).
2. Settings, bishal.ai: **Allow built-in models** (Topics clustering, Loop, Debugger draw on the $10 model
   credit). The Human review score `professional` (1 to 5 slider) and the online scoring rule already exist.
3. Patterns (enabled 2026-09-07): its automatic discovery is a scheduled Loop run on model credit and it
   already minted one misleading Pattern (see "If someone asks"). Delete or rewrite that one, create the
   stop-8 Pattern, then set discovery to manual after Wednesday so credit is not spent unattended.
4. Topics (enabled 2026-09-07): the facet `trace-outcome-summary` runs through the project's default
   preprocessor `dealpoint-trace-preprocessor`, which renders system, model, question, status, answer and
   objective grounding for every log. Look at the clusters once they materialise, then pause the daily
   job after Wednesday (model credit is the one meter no code guards).
5. One Loop thread with the stop-8 prompt, saved.
6. Playground: open the four `arm-a-prompt-*` prompts over `maud-dealpoint-playground-armA` with the three
   judges, run once (about $0.10 OpenRouter), save the session. The `playground-arm-A-*` experiments are the
   fallback.
7. Optional: a threshold alert on `Judge: professional` below 0.5 over the last day (MCP
   `create_threshold_alert`), configured, not triggered.
8. Rehearse once with the fallbacks: stop 5's replay may 429 on Mistral; stop 7's live run may be slow;
   both have saved results.
9. After recording: rotate the API key (it passed through chat) and re-run `just braintrust-showroom --live`
   only if something was deleted; the ledger under `data/reports/orgs/bishal-ai/` keeps every re-run
   idempotent.

Retention: logs written 2026-09-07 expire around 2026-09-21 on Starter; experiments keep for a year.

## If someone asks

- "Why isn't Braintrust the source of truth?" Fourteen-day log retention on this plan; the reports in Git are
  forever, and two commands rebuild the Braintrust side in a fresh org.
- "Why three cheap judges instead of one strong one?" One judge's blind spot looks like a fact. Three
  disagreeing is a signal; three agreeing with each other and not with the lawyer is a better one.
- "Why so many cap-hits?" Retrieval control flow, not model IQ. The agent finds the definitions section and
  keeps searching; when the definition is absent it never stops. That is the next milestone, and the
  lawyer's review is its spec.
- "Is the lawyer's scoring real?" AI-drafted, reviewed and adopted by the lawyer, provenance recorded next
  to the scores (`data/eval/calibration/human_scores.provenance.json`). Say that, do not hide it.
- "Patterns said Haiku was a cost outlier with no quality gain, 26% versus 31 to 47%." Half right. The
  cost is real (sixteen times GLM). The accuracy comparison was not: Braintrust's pattern discovery ranked
  Haiku's 18 judged cases (a third of them counterfactuals, where the metric then counted a correct
  abstention as wrong) against other models' 32-case sweeps, and treated one partial luna-pro trace as a
  variant. On the same 18 cases Haiku, GLM and DeepSeek each get 5 right. The mirror now carries
  `correct_all`, `comparable` and `case_pool` so neither the dashboard nor the pattern engine can mix
  pools again; the Pattern itself should be deleted or rewritten to "same quality as GLM and DeepSeek at
  sixteen times the cost".
- "Why does the dashboard say 46% for arm D when the report says 65%?" The dashboard counts every
  answerable test case, so a cap-hit or an execution failure counts as wrong; the report's headline is
  over the cases that produced a finding. Both are on the row (`ga_all` vs `ga_scored`); the honest number
  for a deployment decision is the dashboard's.
- "Where do the dashboard numbers come from if the logs have no scores?" From the same stored result rows
  the experiments were built from, mirrored onto each log as metadata by `just braintrust-showroom --only
  mirror`. Same source, second surface, zero scores.
- "Where do the deterministic scorers run?" In Git, in `dealpoint/eval/scorers.py`, over the corpus and index
  that cannot live inside Braintrust's function runtime. The LLM judges run in Braintrust. Honest split.
