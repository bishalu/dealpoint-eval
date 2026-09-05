# Operations Reference

What breaks a long autonomous run that is not the work itself, and what the factory does about it. Every item here was learned on a real multi-day run; none is theoretical.

## Provider timeouts

A coding-agent turn on a large context (150k+ tokens, high thinking) can take longer than five minutes. Two independent five-minute defaults kill it:

- **pi's per-request timeout** — `retry.provider.timeoutMs` in `~/.pi/agent/settings.json` (SDK default ≈ 5 min); also `httpIdleTimeoutMs` (default 300000) for stream idle.
- **pi-claude-code-provider's idle/total timeouts** — `PI_CLAUDE_CODE_PROVIDER_IDLE_TIMEOUT_MS` (5 min) and `PI_CLAUDE_CODE_PROVIDER_TOTAL_TIMEOUT_MS` (30 min), read from the environment every ADW hands to pi (`.env`).

The provider takes the **minimum** of its own setting and what pi passes per request, so raising only the env var changes nothing. Raise both:

```json
// ~/.pi/agent/settings.json
{ "retry": { "provider": { "timeoutMs": 1200000 } }, "httpIdleTimeoutMs": 1200000 }
```
```
# .env
PI_CLAUDE_CODE_PROVIDER_IDLE_TIMEOUT_MS=1200000
PI_CLAUDE_CODE_PROVIDER_TOTAL_TIMEOUT_MS=5400000
```

The symptom in the trace: `stopReason: "error"` turns with `errorMessage: "Claude Code request exceeded 300000ms"` in `raw_output.jsonl`, then `never produced valid <Type> JSON: no JSON object found` from the harness. It masquerades as an agent that "ran out of turns" or "only did reconnaissance".

## Context budgets

Same-session retries accumulate context. A builder resumed for a third attempt at 247k tokens will time out on every turn. Rule: **resume for gate corrections and JSON retries; start fresh for a new attempt.** `run.forget_agent(name)` does that; the plan, evidence and recorded failure must be on disk and named in the prompt.

## Failure triage

`adw_modules/triage.py` decides infra vs product from the raised error plus the agent's raw stream since the phase started:

| class | signal | remedy |
|---|---|---|
| `infra:provider_timeout` | `request exceeded Nms`, `timed out` | fresh session; raise the timeouts above |
| `infra:provider_error` | 429/529, overloaded, usage limit, ECONNRESET | fresh session after a pause |
| `infra:context_overflow` | `prompt is too long`, `context_length_exceeded` | fresh session; narrower reads |
| `infra:permission_breach` | `PermissionBreach`, `limited to … but modified` | fresh session; nothing else touching the tree |
| `infra:empty_response` | no JSON and no assistant text | fresh session |
| `product` | anything else | the corrective cycle |

An outer loop should keep two budgets: corrective cycles (product) and infra retries (bounded separately, e.g. 3), and stop with the triage evidence — class, error count, most frequent provider message — when either is exhausted.

## Operator concurrency

`permissions.py` fingerprints the working tree before and after every agent call. **Anything the operator changes in the repo while an agent phase is running is attributed to that agent**: rolled back if outside its allowlist (the phase dies), or swept into its commit if inside. Edit the repo only during a code phase or when no agent is running; a watcher polling `phases` for a `code`/`running` row is a practical window.

## Stopping a run

`just kill <adw_id>` signals the pids the trace recorded, agents first. Children launched by an outer loop should run in their own process group (`start_new_session=True`) so killing the parent takes the subtree; a signal-killed child (`143`) is an interruption, not a defect — reset it to pending rather than launching a correction.
