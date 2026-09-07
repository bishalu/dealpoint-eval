# Braintrust demo: plan (as of 2026-09-07)

Goal: a clean Braintrust project you can walk tab to tab, where every tab shows a real Braintrust strength on
DealPoint data, rebuilt from Git by one command, inside a fresh org's free tier.

## Where we are

**The showroom is the new org `bishal.ai`, project `dealpoint-eval`.** Projects are free, scores are what
cost, and a fresh Starter org has its own 10k scores, 1 GB data and $10 model credit. The old org
(`Vibeset Technologies`) was cleaned of 78 stub experiments and 24 probe objects and stays as the historical record.

Done in the new org today:

| item | state |
|---|---|
| Auth | API key in `.env.braintrust` (old org's key kept as `.env.braintrust.legacy`); `bt` profile `demo`; Claude Code MCP entry `braintrust-demo` (needs your OAuth on restart); pi MCP entry; OpenRouter provider secret created |
| Data | `just braintrust-sync --live` once: 30 experiments, 6 datasets, 986 rows, **3,755 scores** (3,986 planned; nulls are not counted), 6 replayed traces, 3 prompts, 1 parameters object. Verified by read-back. |
| Free tier used (Braintrust's own meter, 2026-09-07) | scores **3,755 of 10,000**; logs (processed data) **0.0023 GB of 1 GB**; model credits **$0 of $10** |
| Code | `braintrust-sync`, `braintrust-cockpit`, `braintrust-showroom`: org-switchable (`BRAINTRUST_PROJECT`, `BRAINTRUST_ENV_FILE`, `BRAINTRUST_LEDGER_FILE`), dry-run default, `--live` opt-in, exact score counts before writing, per-org ledgers, no hardcoded org ids |

Not yet done (all free unless marked): showroom steps (experiment tags and descriptions, row-level metadata,
the 114 log traces, review flags, arm parameters, arm-A prompt variants, four judge scorers, comparison views);
Topics and one Pattern (model credit); a Playground run (~$0.10 OpenRouter, a few hundred scores if saved);
the tour regenerated with `bishal.ai` links; Claude Code restart for the MCP OAuth.

## Decisions taken

- **Models on your OpenRouter key, never Braintrust built-ins, for anything that is yours**: judge scorers
  (`mistralai/mistral-small-3.2-24b-instruct` for reasoning, evidence, professional; `bytedance-seed/seed-2.0-mini`
  for trajectory), Playground prompt variants (`z-ai/glm-5.3-flash`). Built-in models only for Braintrust's own
  analysis features (Topics, Loop, Debugger), from the included credit, as long as it stays cheap.
- **Deterministic scorers stay in Git.** They import the corpus layer and cannot run server-side; the Scorers tab
  carries the LLM judges. Honest split, and the one the brief already made.
- **No custom dashboards** (Pro-only). The comparison surface is the Experiments page: Summary table, Grid,
  Comparison grade. Dashboards were also empty for a second reason: metadata lived on experiments, not rows;
  the showroom's row step fixes that.
- **Same cases everywhere** is what makes case-by-case comparison work: arms and models on the 32-case test
  subset, judges/DeepEval/human on the 18 judged cases, RAG on the 58 dev queries.
- **The scheduled Pattern-discovery Loop job stays paused** (it ran a high-reasoning agent every weekday); one
  on-demand Pattern is enough for the demo.

## The story, tab by tab

| # | tab | what you show | Braintrust strength it proves |
|---|---|---|---|
| 1 | Datasets | `maud-dealpoint-judged_calibration`: real contract questions with the expert-marked span as `expected` | one versioned set of test cases everything joins on; a bad trace becomes a row here in one click |
| 2 | Experiments (view *RAG tournament*) | six retrievers, `li/*` next to `obj/*`; hybrid 91.4% hit@5 | two independent scorers on the same rows |
| 3 | Experiments (view *Arms A to D*) | select four, Summary table, then Grid on the Knowledge question | case-level diff, improvement/regression grading |
| 4 | Experiments (view *Models, arm D*) | five models, sort by `$/case` | same view, economic axis |
| 5 | Logs | 114 traces; open the inefficient one; count the tool calls | nested agent tracing |
| 6 | Scorers | four LLM judges on OpenRouter models; invoke one on a log | code + LLM + human scoring in one place |
| 7 | Experiments (*Judges and the lawyer*), hero case | judge/* vs human/* on the same rows; the two packets where all three judges were wrong and the lawyer was right | human review feeding the loop |
| 8 | Review | the 12 flagged traces, one Starter review score | SME labels |
| 9 | Prompts, Parameters | base prompt + three arm-A variants; `arm-A..D-config` | first-class, versioned objects |
| 10 | Playground | arm-A variants over the 18 cases, judged live | iterate before coding |
| 11 | SQL | the six saved investigations | serious querying of nested data |
| 12 | Topics, Patterns, Debugger, Loop | clusters over the logs; one Pattern; Debugger on the hero trace; one Loop thread | discover and diagnose without knowing the query first |
| 13 | close | "everything above came from `just braintrust-sync` in a fresh org for $0" | reproducibility |

## The three limits, and what each remaining step costs against them

Starter meters three things separately: **scores** (10k/month; each score written on a row), **processed data**
(1 GB/month; every byte ingested: rows, spans, datasets; measured at ingestion, deleting does not refund),
and **model credit** ($10/month; Braintrust's own built-in models: Topics, Loop, Debugger, Playground on built-ins).
OpenRouter calls hit none of them; they hit your OpenRouter ledger.

| step | scores | processed data | model credit | OpenRouter |
|---|---|---|---|---|
| sync (done) | 3,755 | ~2 MB | 0 | 0 |
| tags, row metadata, review flags, parameters, prompts, views | 0 | < 1 MB (metadata merges) | 0 | 0 |
| 114 trace trees into Logs | 0 | ~10-20 MB (full span trees with retrieved text) | 0 | 0 |
| 4 judge scorers (create) | 0 | ~0 | 0 | 0 |
| judge scorer calls in the demo | 1 per call | ~0 | 0 | cents |
| Playground: 3 variants x 18 cases, saved | ~216 (108 rows x 2 if judged) | ~1 MB | 0 | ~$0.10 |
| Topics over the 114 logs | 0 | 0 | ~$1-2 (facet + cluster tokens) | 0 |
| one Pattern, one Loop thread, Debugger on one trace | 0 | 0 | ~$1-3 | 0 |
| re-sync after a rotation or a new milestone | up to 3,755 again | ~2 MB | 0 | 0 |
| **projected end state** | **~4,000 of 10,000** | **< 0.05 of 1 GB** | **~$3-5 of $10** | **< $1** |

Guard rails already in code: dry-run default, `--live` opt-in, exact score count printed before the first write,
per-run cap, per-org ledger so nothing is re-logged. The one limit no code guards is model credit: Topics runs
daily once enabled, so enable it once for the demo window and pause it after recording.

## Costs

| item | scores | money |
|---|---|---|
| sync (done) | 3,755 (Braintrust's count; 3,986 planned, the difference is null scores) | $0 (inside free tier) |
| showroom steps | 0 | $0 |
| judge scorer invocations during the demo | 1 per call | OpenRouter cents |
| Playground run, 3 variants x 18 cases, saved | ~216 | $0 scores (inside tier) + ~$0.10 OpenRouter |
| Topics, one Pattern, one Loop thread, Debugger | 0 | model credit, inside $10 |
| Pro plan | | not needed |

## Sequence from here

1. **Revisit** (you): story, tab order, what deserves credit, factory vs. me. This document is the input.
2. Me: showroom steps in the new org (~15 minutes), then tour regeneration.
3. You: restart Claude Code, approve the `braintrust-demo` OAuth for `bishal.ai`; in the new org's Settings:
   Allow built-in models; Human review -> one score `professional` 1-5.
4. Me: Topics on the logs, one Pattern, the Playground run; you: one Loop thread (2 minutes).
5. Record within 14 days of the sync (logs expire; experiments keep 365 days). Rotate the API key afterwards
   (it passed through chat).

## Answers to the seven questions, kept for reference

1. One log: logs are traces; everything had been logged as experiments. The showroom fills Logs, free.
2. Empty dashboards: custom dashboards are Pro-only, and row metadata was missing. Use Experiments views.
3. Messy experiments: quota-era re-sync copies. New org, one copy each, consistent names.
4. Dataset first: it is the join key for every comparison, and the "trace to regression test" story.
5. Prompts: three arm-A variants for a Playground A/B/C.
6. Scorers and parameters: LLM judges as Braintrust scorers on OpenRouter; code scorers in Git; arm configs as Parameters.
7. Tools no (need the local index); SQL yes, free; Loop yes, one thread, from credit.

## Open questions for the revisit

- Do you want the Playground stop live (a real run during the demo, ~$0.10) or pre-run and saved?
- Should the factory (SSSF) own a "demo maintenance" milestone later (re-sync, tour regen), or stays this a manual, documented command?
- Any tab to drop? Thirteen stops is at the edge of ten minutes.
