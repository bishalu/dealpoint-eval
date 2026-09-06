# Model cost/quality Pareto experiment (M6)

This report is **budget-scaled** (32-case frozen subset per model, 18 at Haiku) -- grounded_accuracy is **objective** (deterministic Python over expert labels); judged quality is **secondary and model-judged, never objective truth**; scores are **not comparable** to the MAUD leaderboard.

Arm D is held fixed (frozen index, skill, tools, cases); only the model varies.

## Per-model results

| model | n_cases | grounded_accuracy | abstain_recall | execution_failed | cap_hit | median latency (ms) | p90 latency (ms) | $/case | total $ | frontier |
|---|---|---|---|---|---|---|---|---|---|---|
| z-ai/glm-5.3-flash | 32 | 64.7% (17/32) | 12.5% | 15.6% | 15.6% | 30604.0 | 54579.0 | $0.00278 | $0.0888 | no |
| anthropic/claude-haiku-4.5 | 18 | 62.5% (8/18) | 0.0% | 5.6% | 50.0% | 18195.0 | 63379.0 | $0.04580 | $0.8243 | no |
| deepseek/deepseek-v4-flash | 32 | 69.2% (13/32) | 0.0% | 6.2% | 50.0% | 17671.0 | 24678.0 | $0.00226 | $0.0722 | yes |
| qwen/qwen3.7-flash | 32 | 63.6% (11/32) | 0.0% | 15.6% | 50.0% | 13890.0 | 17228.0 | $0.00110 | $0.0352 | yes |
| google/gemini-3.1-flash-lite | 32 | 63.2% (19/32) | 37.5% | 0.0% | 21.9% | 11361.0 | 18954.0 | $0.00553 | $0.1770 | no |

**Frontier** (Minimise realised $/case, maximise grounded_accuracy. A point is dominated iff another has x<=x' and y>=y' with at least one strict inequality. Exact ties on both axes keep the lexicographically-smallest model id.): ['qwen/qwen3.7-flash', 'deepseek/deepseek-v4-flash']

## Not run

- `anthropic/claude-opus-5`: not run (budget) -- spec deliverable 1 names Opus as out of budget for this milestone; at $5/M in, $25/M out it would cost more alone than this milestone's entire remaining envelope.
- `openai/gpt-5.6-luna-pro`: started, not completed (stop-floor) -- see partial_runs
- `meta-llama/llama-4-maverick`: not run (budget/stop-floor)
- `xiaomi/mimo-v2.5`: not run (budget/stop-floor)
- `minimax/minimax-m2.5`: not run (budget/stop-floor)
- `moonshotai/kimi-k2.5`: not run (budget/stop-floor)
- `anthropic/claude-sonnet-5`: not run (budget/stop-floor)
- `x-ai/grok-4.3`: not run (budget/stop-floor)

## Partially run (metered, not completed)

| model | n_cases_completed/n_cases_attempted | realized_usd | reason |
|---|---|---|---|
| google/gemini-3.1-flash-lite | 0/14 | $0.062627 | superseded by the completed 32-case pass |
| openai/gpt-5.6-luna-pro | 0/21 | $0.283390 | stopped (stop-floor) |

## Comparability note

Haiku's arm-D result is 18 cases (tranche_1); every other model's is 32. Any Haiku-vs-other comparison is over the 18 cases they share and is never compared to a 32-case mean without saying so.

## Spend

```json
{
  "by_milestone": {
    "m1": 0.41912,
    "m2": 0.170014,
    "m4": 0.751879,
    "m4_1": 1.420234,
    "m5": 0.110038,
    "m6": 0.742655,
    "m7a": 0.14189
  },
  "envelope_headroom": 0.24417,
  "envelope_usd": 4.0,
  "m6_judge_usd": 0.077565,
  "m6_probe_usd": 0.034754,
  "m6_realized_usd": 0.742655,
  "m6_sweep_usd": 0.630336,
  "target_usd": 1.0,
  "total_realized_usd": 3.75583
}
```

## Recorded decisions

- **PD1**: Haiku and z-ai/glm-5.3-flash are reused from M4/M4.1's frozen arm-D results, never re-run. -- Spec deliverable 1: reuse, do not re-run.
- **PD2**: The open-weight slot is deepseek/deepseek-v4-flash, not the spec's named deepseek/deepseek-v3.2. -- The engineer's 2026-09-04 'Slate and judges' instruction ranks deepseek/deepseek-v4-flash ahead of deepseek-v3.2 in the candidate slate, and it is cheaper per case: deepseek-v4-flash is $0.08358/M input, $0.16716/M output vs deepseek-v3.2's $0.269/M input, $0.400/M output (live prices, 2026-09-05) -- roughly 3x cheaper per token both ways.
- **PD3**: No ceiling anchor (anthropic/claude-sonnet-5, x-ai/grok-4.3) was run. -- Neither anchor fit the remaining envelope even on the reduced 18-case judged subset: claude-sonnet-5 is estimated at ~$0.869 for 18 cases and x-ai/grok-4.3 at ~$0.478 for 18 cases (2026-09-05 live prices), against a total M1-M6 envelope of $4.00 with a $3.90 stop-floor -- headroom the sweep and judging legs of the cheap tier had already largely consumed by the time either anchor would have started.

## Brief-vs-spec differences

- **scale**: Brief section 2.6 designs the full benchmark over the complete test + counterfactual sets; this milestone runs the 32-case frozen subset (18 at Haiku), budget-scaled, per the engineer's 2026-09-04 instruction.
- **model_count**: Brief section 6 acceptance 6 asks for the model experiment to be complete for >= 4 of the brief's 5 named models (Sonnet 5 default, Opus 5 ceiling, Haiku, a Gemini Flash-class model, one open-weight model). This run covers 3 of those 5 (Haiku, a Gemini Flash-class model, an open-weight model) -- Sonnet 5 and Opus 5 are both absent -- while completing 5 models overall by adding qwen/qwen3.7-flash and z-ai/glm-5.3-flash to the cheap tier.
- **ceiling**: The reported Pareto frontier is a frontier of the cheap tier only: no ceiling model (Sonnet 5, Opus 5, or Grok 4.3) was measured this run, so a stronger, more expensive model could sit above every point on this frontier. The frontier and every dominance/tie judgement in this report are scoped to the models actually completed.

## Caveats

- Judged quality is secondary and model-judged; it is never objective truth.
- The reported frontier is a frontier of the cheap tier; no ceiling model was run.
- Scores are not comparable to the MAUD leaderboard (this task is document -> (answer, citation); MAUD's published task is span -> answer).
