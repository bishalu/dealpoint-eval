"""Headroom context tools for the agents, MCP-only and feature-flagged.

The path is:

    ADW  ->  pi -p  ->  pi-mcp-extension  ->  `headroom mcp serve` (stdio)
                 \\->  pi-claude-bridge   ->  Claude worker (Agent SDK)

pi-mcp-extension is pi's MCP *client*: it reads `~/.pi/agent/mcp.json` merged
with `<cwd>/.pi/mcp.json`, spawns each server, and registers every discovered
tool as a pi tool named `<toolPrefix>_<server>_<tool>`. pi-claude-bridge then
hands every pi tool the agent is allowed to the Claude worker through an
in-process MCP server of its own, and runs the worker with strict MCP config —
so the worker sees exactly pi's tools and nothing from the outer Claude Code's
own MCP configuration. Headroom therefore reaches the Claude workers by being a
pi tool, which is the only thing this module arranges.

Three pieces, all keyed on one flag (`defaults.headroom.enabled`, overridden by
`SSSF_HEADROOM=1|0` in the environment):

  * `materialize()` writes the `headroom` server into the repo's project-level
    `.pi/mcp.json` when the flag is on, and removes exactly that key when it is
    off. The file is gitignored; other entries in it are never touched. pi runs
    with cwd = repo root, which is where pi-mcp-extension looks. An enabled run
    removes the entry again when it exits (`release()`), so the file is only
    ever present while a flag-on run is alive. Known limit: the file is per
    checkout, so two flag-on runs in the same checkout share it and the first to
    finish removes it under the second — run one at a time, or use a worktree.
  * `apply()` appends the Headroom tool names to every agent that carries a
    `tools` allowlist, because `pi --tools` filters extension tools too — an
    unnamed tool is silently unavailable (references/config.md).
  * `preflight()` fails validation fast when the flag is on but the pieces are
    missing: the `headroom` CLI, or pi-mcp-extension in pi's packages.

Off is the default, and off means absent. Nothing here changes a run unless the
flag is on; removing the feature is deleting this module, its three call sites,
and the `headroom:` block in the config.

Deliberately NOT here: routing the Claude workers' model traffic through
Headroom's HTTP proxy (`ANTHROPIC_BASE_URL`). The workers run on the operator's
subscription via the Agent SDK; the first implementation keeps Headroom to
on-demand MCP tools only. See docs/headroom.md.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from .data_types import SSSFConfig

ENV_FLAG = "SSSF_HEADROOM"
SERVER_NAME = "headroom"
PI_AGENT_DIR = Path.home() / ".pi" / "agent"
PI_MCP_EXTENSION = "pi-mcp-extension"
DEFAULT_TOOL_PREFIX = "mcp"       # pi-mcp-extension's default `settings.toolPrefix`

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


# ── the flag ─────────────────────────────────────────────────────────────────

def enabled(cfg: SSSFConfig, env: Mapping[str, str] | None = None) -> bool:
    """Config says; the environment overrides for one launch, either way."""
    env = os.environ if env is None else env
    raw = env.get(ENV_FLAG)
    if raw is not None:
        value = raw.strip().lower()
        if value in _TRUE:
            return True
        if value in _FALSE:
            return False
        raise ValueError(f"{ENV_FLAG}={raw!r} is not a boolean (use 1/0)")
    return cfg.defaults.headroom.enabled


# ── tool names ───────────────────────────────────────────────────────────────

def tool_prefix(repo_root: Path | None = None) -> str:
    """pi-mcp-extension's `settings.toolPrefix`, project over global, else default."""
    prefix = DEFAULT_TOOL_PREFIX
    paths = [PI_AGENT_DIR / "mcp.json"]
    if repo_root is not None:
        paths.append(Path(repo_root) / ".pi" / "mcp.json")
    for path in paths:
        data = _read_json(path)
        found = (data.get("settings") or {}).get("toolPrefix")
        if isinstance(found, str) and found:
            prefix = found
    return prefix


def pi_tool_names(cfg: SSSFConfig, tool_prefix: str = DEFAULT_TOOL_PREFIX) -> list[str]:
    """The pi-side names: `<prefix>_<server>_<tool>`, exactly as the client registers them."""
    return [f"{tool_prefix}_{SERVER_NAME}_{tool}" for tool in cfg.defaults.headroom.tools]


def apply(cfg: SSSFConfig, env: Mapping[str, str] | None = None,
          tool_prefix: str = DEFAULT_TOOL_PREFIX) -> None:
    """When on, name the Headroom tools in every agent's allowlist (in place).

    An agent with `tools: None` already gets every tool and is left alone.
    Idempotent: a name already present is not repeated.
    """
    if not enabled(cfg, env):
        return
    names = pi_tool_names(cfg, tool_prefix)
    for agent in cfg.agents:
        if agent.tools is None:
            continue
        agent.tools.extend(name for name in names if name not in agent.tools)


# ── the project mcp.json ─────────────────────────────────────────────────────

def server_entry(cfg: SSSFConfig) -> dict:
    """pi-mcp-extension's per-server config for `headroom mcp serve` over stdio.

    `eager`, not `lazy`: a lazy server is started by the `/mcp:start` command,
    which a `pi -p` run never gets to type.
    """
    hr = cfg.defaults.headroom
    entry: dict = {
        "transport": "stdio",
        "command": hr.command,
        "args": list(hr.args),
        "lifecycle": "eager",
        "requestTimeoutMs": hr.request_timeout_ms,
    }
    if hr.env:
        entry["env"] = dict(hr.env)
    return entry


def materialize(cfg: SSSFConfig, repo_root: Path, on: bool) -> Path | None:
    """Add or remove the `headroom` server in `<repo_root>/.pi/mcp.json`.

    Only that one key is ever written or deleted; settings and other servers
    survive untouched. Off with nothing to remove is a no-op that creates no
    file. Off after we were the file's only content deletes the file, so a
    disabled run leaves no `.pi/` behind it did not find. Returns the path
    written, or None when nothing changed.
    """
    path = Path(repo_root) / ".pi" / "mcp.json"
    data = _read_json(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    has = SERVER_NAME in servers

    if on:
        wanted = server_entry(cfg)
        if has and servers[SERVER_NAME] == wanted:
            return None
        servers[SERVER_NAME] = wanted
        data["mcpServers"] = servers
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
        return path

    if not has:
        return None
    del servers[SERVER_NAME]
    data["mcpServers"] = servers
    if not servers and set(data) <= {"mcpServers"}:
        path.unlink()
        return path
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def prepare(cfg: SSSFConfig, repo_root: Path, env: Mapping[str, str] | None = None) -> bool:
    """Make the project's pi MCP config agree with the flag, before any pi spawns.

    Returns True when the flag is on — the caller should then `release()` at
    exit, so an enabled run does not leave the server configured for every pi
    that starts in this checkout afterwards (an interactive one, or another
    ADW with the flag off that happens to run before the next off-launch).
    """
    on = enabled(cfg, env)
    materialize(cfg, repo_root, on=on)
    return on


def release(cfg: SSSFConfig, repo_root: Path) -> None:
    """Undo `prepare(on=True)`: drop the server entry again. Safe to call twice."""
    materialize(cfg, repo_root, on=False)


# ── preflight ────────────────────────────────────────────────────────────────

def preflight(cfg: SSSFConfig, env: Mapping[str, str] | None = None,
              pi_packages: list[str] | None = None) -> list[str]:
    """Problems that would make an enabled run fail late; empty when off or fine."""
    if not enabled(cfg, env):
        return []
    problems = []
    command = cfg.defaults.headroom.command
    if shutil.which(command) is None:
        problems.append(f"headroom: {command!r} is not on PATH — "
                        "install with `uv tool install 'headroom-ai[mcp]'`, "
                        f"or set {ENV_FLAG}=0")
    packages = pi_packages if pi_packages is not None else _pi_packages()
    if not any(PI_MCP_EXTENSION in package for package in packages):
        problems.append(f"headroom: pi's MCP client ({PI_MCP_EXTENSION}) is not installed — "
                        f"run `pi install npm:{PI_MCP_EXTENSION}`, or set {ENV_FLAG}=0")
    return problems


# ── internals ────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _pi_packages() -> list[str]:
    packages = _read_json(PI_AGENT_DIR / "settings.json").get("packages") or []
    return [str(p) for p in packages]
