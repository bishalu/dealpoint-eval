# Headroom in the factory: MCP-only, behind one flag

Headroom ([headroom-ai](https://pypi.org/project/headroom-ai/)) is a context
optimization layer for LLM agents. This note records how it is wired into the
existing SSSF path, why it is wired that way, what the first measurement showed,
and how to take it out again.

## The path, and where Headroom joins it

```
agent-sandbox (this exe.dev VM)
  └─ Claude Code  ──/sssf──▶  adws/adw_*.py  (Python ADW: phases, gates, trace)
                                 └─ pi -p --tools … --session-id …     (one per agent call)
                                      ├─ pi-mcp-extension   ← Headroom joins HERE
                                      │    reads ~/.pi/agent/mcp.json + <cwd>/.pi/mcp.json
                                      │    spawns `headroom mcp serve` (stdio)
                                      │    registers mcp_headroom_headroom_{compress,retrieve,stats} as pi tools
                                      └─ pi-claude-bridge  (provider claude-bridge/*)
                                           bridges every allowed pi tool into an in-process MCP server
                                           runs the Claude worker (Agent SDK, subscription auth,
                                           --strict-mcp-config) which sees exactly pi's tools
```

Two facts decide the design:

1. **The outer Claude Code's MCP configuration does not reach the workers.**
   pi-claude-bridge runs the Claude Agent SDK with `strictMcpConfig: true`
   (`~/.pi/agent/claude-bridge.json`), so `~/.claude.json` / `.mcp.json`
   servers are ignored on purpose. Registering Headroom with `claude mcp add`
   or `headroom mcp install` would do nothing for a worker.
2. **Every pi tool the agent is allowed is bridged into the worker.** The
   bridge's `resolveMcpTools` takes `context.tools` wholesale — built-ins,
   extension tools, MCP-client tools alike — and serves them to Claude as
   `mcp__custom-tools__<pi tool name>`. So the way to give a Claude worker a
   Headroom tool is to give pi a Headroom tool.

pi's MCP client is `pi-mcp-extension` (already installed globally; it is how
the roster's `mcp_braintrust_*` tools work). It names tools
`<toolPrefix>_<server>_<tool>`; Headroom's own tools are already prefixed
`headroom_`, hence the doubled `mcp_headroom_headroom_compress`. That doubling
is MCP namespacing, not a bug — Headroom's own docs say the same for Claude
Code's `mcp__headroom__headroom_retrieve`.

## What the flag does

`defaults.headroom.enabled` in `adws/adw_sssf_config/sssf.config.yaml`
(default `false`); `SSSF_HEADROOM=1|0` in the environment overrides it for one
launch, in either direction. The module is `adws/adw_modules/headroom.py`, with
three call sites:

| where | when | what |
|---|---|---|
| `agents.load_config` | every config load | `headroom.apply`: append the three tool names to every agent that has a `tools` list. `pi --tools` filters extension tools too, so an unnamed tool is silently absent (references/config.md). Agents with `tools: None` already see everything. |
| `agents.validate` | every ADW start | `headroom.preflight`: flag on + `headroom` not on PATH, or pi-mcp-extension not in `~/.pi/agent/settings.json` packages → config validation fails with the install command. |
| `session.ensure` | every run, before any pi spawns | `headroom.prepare` → `materialize`: write the `headroom` server (stdio, `headroom mcp serve`, `lifecycle: eager`) into `<repo>/.pi/mcp.json` when on; remove exactly that key when off, deleting the file only if nothing else is in it. Other servers and `settings` in that file are never touched. A flag-on run registers `headroom.release` with `atexit`, so the entry exists only while that run is alive. |

`.pi/mcp.json` is gitignored: it is derived from the flag per run, not
configuration to track. `eager` rather than `lazy` because a `pi -p` run can
never type `/mcp:start`.

**Off means absent.** With the flag off no server entry exists, no tool name is
added, and the roster, the prompts, the gates, the phases and the
`claude-bridge/*` models run exactly as before. Stock plan/build/review/repair/
document behaviour is unchanged by construction: the flag is read, nothing is
branched on it elsewhere.

**Known limit: the project file is per checkout.** Two flag-on runs in the same
checkout share `.pi/mcp.json`, and the first to finish removes the entry under
the second. Any pi that starts in the checkout while a flag-on run is alive
(an interactive session, a flag-off ADW) also spawns `headroom mcp serve`; its
tools stay invisible to that agent because they are not in its `tools` list,
so the cost is one idle subprocess. Run flag-on work one at a time per
checkout, or in a worktree.

### Installation (sandbox)

```
uv tool install 'headroom-ai[mcp]'      # → ~/.local/bin/headroom (0.37.0 at the time of writing)
pi install npm:pi-mcp-extension          # already present here
```

Note `pip install headroom` is a different, unrelated project ("Max
Headroom", a chat CLI). The package is `headroom-ai`.

### Turning it on

```
SSSF_HEADROOM=1 just sdlc "…"            # one launch
# or, for the roster:  defaults.headroom.enabled: true
```

### Taking it out

Delete `adws/adw_modules/headroom.py`, its call sites (`agents.py` ×2,
`session.py` ×1), `HeadroomConfig` and the `headroom:` field in
`data_types.py`, the `headroom:` block in `sssf.config.yaml`, the `.pi/mcp.json`
gitignore line, `tests/test_headroom_flag.py`, the `headroom-probe` recipe, and
`adws/headroom_integration_test.py`. `uv tool uninstall headroom-ai` if the CLI
is unwanted. Nothing else references it.

## What is deliberately not done

**No HTTP proxy in front of the workers.** Headroom's automatic savings come
from `headroom proxy` sitting on `ANTHROPIC_BASE_URL` and compressing tool
outputs before the model sees them (Compress-Cache-Retrieve). The workers here
run on the operator's Claude subscription through the Agent SDK; the first
implementation does not put a third-party proxy in that path. The MCP-only mode
is the removable, observable first step; the proxy is a separate decision with
its own risks (auth headers, billing, session state, cache keys) and is not
gated by this flag.

Consequence, worth being blunt about: **MCP-only Headroom cannot reduce the
input tokens of a single-session task.** `headroom_compress` takes `content` as
an argument — the model must already hold the content and must write it back
out through the tool call — so on a read-then-answer task it *adds* output
tokens and then adds the compressed copy to the context on top of the original.
The measurement below shows exactly that. Where it can pay is across
boundaries: an agent compressing a large finding before writing it to
`context_handoff/`, a subagent handing back a compressed digest with a hash, or
`headroom_retrieve` recovering exact detail after pi's own compaction has
summarized it away. Whether to build any of that is a follow-up product
decision; the plumbing this note describes is what makes it possible.

Also not done: no Braintrust/SSSF gate depends on Headroom; no prompt in
`prompt_engineering/` mentions it. The roster does not know the tools exist
unless the flag is on.

## The probe

`just headroom-probe` (`adws/headroom_integration_test.py`, disposable; stdlib
only) runs the same task twice through `adws/adw_prompt.py` with a generated
one-agent roster (`claude-bridge/claude-haiku-4-5`, tools `read, ls`): once
with `SSSF_HEADROOM=0`, once with `=1`. The task: read a generated 300-record
JSON log (≈31 KB), report the exact count of `level == "ERROR"` records and the
`host` of one record deep in the file. The Headroom arm's prompt additionally
asks the worker to compress the content, answer from the compressed form, and
retrieve by hash if needed. The script then reads the factory's own record —
`raw_output.jsonl` (`tool_execution_end` events name every tool the worker
called, `message_end` usage carries pi's token accounting), `envelope.json`,
and `sssf.db` — and writes `headroom_probe_report.{json,md}` into the Headroom
arm's session directory.

"Proven" means: the run exited 0, `.pi/mcp.json` was materialized, at least
one `mcp_headroom_*` tool appears in the worker's `tool_execution_end` events
with `isError: false`. The worker is Claude (through the bridge), so a tool
call in that stream is Claude choosing and invoking the tool.

Run it in a quiet checkout. SSSF's permission check compares the working tree
before and after each agent call, so another ADW editing the same checkout at
the same time is attributed to the probe agent and rolled back (the first run
of this probe did exactly that to a concurrent milestone build). A worktree
(`git worktree add … HEAD`, copy `.env` and `.env.braintrust`) isolates it.

### Results

<!-- headroom-probe-results:start -->
Isolated worktree at commit 1268eed, 2026-09-08, `claude-bridge/claude-haiku-4-5`,
thinking `low`, 300 records (31 KB). All twelve checks passed: both arms exited
0; the flag-off arm never had a `.pi/mcp.json` and saw only `read`; the flag-on
arm had the entry while alive and not after; the worker invoked all three
Headroom tools with no errors; both arms answered both questions exactly.

| measure | baseline | headroom |
|---|---|---|
| adw_id | hrb3b87f3 | hrh319423 |
| exit code | 0 | 0 |
| phase failure | — | — |
| answer read from | envelope.json | envelope.json |
| wall time (s) | 18.3 | 106.9 |
| assistant turns | 2 | 5 |
| tool calls | 1 | 4 |
| tools seen | read | mcp_headroom_headroom_compress, mcp_headroom_headroom_retrieve, mcp_headroom_headroom_stats, read |
| Headroom tools invoked | none | mcp_headroom_headroom_compress, mcp_headroom_headroom_retrieve, mcp_headroom_headroom_stats |
| tokens, sum over turns (pi total) | 20957 | 163748 |
|   input (uncached) | 17 | 41 |
|   cache read | 4641 | 102447 |
|   cache write | 15493 | 49090 |
|   output | 806 | 12170 |
| context occupancy after last turn | 15978 | 49488 |
| context occupancy (sssf.db) | 15978 | 49488 |
| cost USD (pi) | 0.0 | 0.0 |
| cost USD (sssf.db) | 0.0 | 0.0 |
| read tool output truncated | False | False |
| answer error_count correct | True | True |
| answer needle host correct | True | True |
| summary | `error_count=14; host_234=node-0` | `error_count=14; host_234=node-0` |

Headroom calls in the flag-on arm:

- `mcp_headroom_headroom_compress` ok=True {"result_chars": 14331, "original_tokens": 9609, "compressed_tokens": 5128, "tokens_saved": 4481, "savings_percent": 46.6, "transforms": ["router:mixed:0.35"]}
- `mcp_headroom_headroom_retrieve` ok=True {"result_chars": 37698, "source": "local"}
- `mcp_headroom_headroom_stats` ok=True {"result_chars": 1363, "savings_percent": 46.6, "compressions": 1, "retrievals": 1, "total_tokens_saved": 4481}

Reading: **the mechanism works end to end, and MCP-only mode is a net cost on
this task.** The worker reads the file (≈15 K tokens into context), then writes
the whole content back out as the `content` argument of `headroom_compress`
(that is the ≈12 K output tokens — output is the expensive kind), receives a 5 K
compressed copy on top, then a 38 K-character retrieval on top of that. Final
context is 3× the baseline's, summed tokens 8×, wall time 6×, and the answer was
already right without any of it. Headroom's own accounting (46.6% saved,
4 481 tokens) is true of the payload it was handed and says nothing about the
session, which is what the proxy path would change and this flag does not.

What this buys is the plumbing, verified: a Claude worker under SSSF can call a
Headroom tool, and the flag can be turned off with nothing left behind. Where a
compress-then-hand-off pattern (see above) would pay is a separate experiment
on a multi-agent chain; this probe deliberately measures the simplest case.
<!-- headroom-probe-results:end -->
