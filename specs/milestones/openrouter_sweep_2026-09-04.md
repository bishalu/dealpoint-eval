# OpenRouter model sweep — 2026-09-04 (operator, for the Pareto slate and the judge trio)

Filter: tool calling + structured outputs supported, no `:free`/`:batch`/preview variants, prompt ≤ $2.5/M.
`$/case` = estimated cost of one **arm-D agent case** at M1's measured shape (≈ 19.2k input tokens, 0.6k
output, ~4 calls). **Verify ids, prices and tool-calling on 2 dev cases at run time; record the result.**

## Candidate slate (arm D on the frozen 32-case subset unless noted) — ranked; run in this order

| # | model | family | in $/M | out $/M | $/case | 32-case est. | role |
|---|---|---|---|---|---|---|---|
| 1 | `anthropic/claude-haiku-4.5` | Anthropic | 1.000 | 5.000 | 0.0222 | reuse M4 | default (already run) |
| 2 | `z-ai/glm-5.3-flash` | Zhipu | 0.075 | 0.250 | 0.0016 | reuse M4 replication | **mandatory**, dev workhorse |
| 3 | `deepseek/deepseek-v4-flash` | DeepSeek | 0.088 | 0.177 | 0.0018 | $0.06 | open-weight |
| 4 | `qwen/qwen3.7-flash` | Alibaba | 0.030 | 0.130 | 0.0007 | $0.02 | cheapest credible |
| 5 | `google/gemini-3.1-flash-lite` | Google | 0.250 | 1.500 | 0.0057 | $0.18 | cross-family flash |
| 6 | `openai/gpt-5.6-luna-pro` | OpenAI | 0.200 | 1.200 | 0.0046 | $0.15 | strong cheap OpenAI (fallback `openai/gpt-5-mini`, 0.0060) |
| 7 | `meta-llama/llama-4-maverick` | Meta | 0.200 | 0.696 | 0.0043 | $0.14 | open-weight, large ctx |
| 8 | `xiaomi/mimo-v2.5` | Xiaomi | 0.140 | 0.280 | 0.0029 | $0.09 | optional |
| 9 | `minimax/minimax-m2.5` | MiniMax | 0.270 | 1.080 | 0.0058 | $0.19 | optional |
| 10 | `moonshotai/kimi-k2.5` | Moonshot | 0.450 | 2.250 | 0.0100 | $0.32 | optional |
| 11 | `anthropic/claude-sonnet-5` | Anthropic | 2.000 | 10.000 | 0.0444 | $0.53 on the **12-case** judged subset | ceiling anchor, budget permitting |
| 12 | `x-ai/grok-4.3` | xAI | 1.250 | 2.500 | 0.0255 | $0.31 on the 12-case subset | second anchor, budget permitting |

Rows 1–7 are the core (≈ $0.55 of new spend); 8–10 are added in order while the M6 allocation holds;
11–12 run last, on the 12-case subset only, and only if the envelope still has headroom after judging.
Opus 5 (0.0555/case) is out of budget: report as "not run (budget)".

## Judge trio (families that appear NOWHERE in the slate above; cheap; JSON-capable; one call per trace)

| model | family | in $/M | out $/M | note |
|---|---|---|---|---|
| `mistralai/mistral-small-3.2-24b-instruct` | Mistral | 0.075 | 0.200 | |
| `nvidia/nemotron-3-super-120b-a12b` | NVIDIA | 0.085 | 0.400 | |
| `bytedance-seed/seed-2.0-mini` | ByteDance | 0.100 | 0.400 | |
| spare: `amazon/nova-lite-v1` | Amazon | 0.060 | 0.240 | no structured-output flag — JSON via prompt only |

Judge input ≈ 8k tokens → ≈ $0.001 per call; 12 cases × 3 judges × ~8 variants ≈ 290 calls ≈ $0.35.
Judge the default, GLM, and every model on the Pareto frontier first; the rest only if headroom remains.

## Other notable options (not selected, for the record)
`google/gemini-2.5-flash-lite` (0.0022), `openai/gpt-5-nano` (0.0012), `openai/gpt-oss-120b` (0.0008, open-weight),
`nvidia/nemotron-3-super-120b-a12b` (0.0019 — reserved as a judge), `qwen/qwen3.7-plus` (0.0069, stronger Qwen),
`deepseek/deepseek-v4-pro` (0.0211), `z-ai/glm-5.3` (0.0295), `google/gemini-3.6-flash` (0.0167).
