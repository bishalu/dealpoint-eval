"""Headroom MCP-only integration: the feature flag, the project mcp.json
materialization, and the tool-name injection (adws/adw_modules/headroom.py).

Offline. Nothing here spawns pi, Claude, or Headroom; the live proof is the
disposable probe in adws/headroom_integration_test.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ADWS = Path(__file__).resolve().parents[1] / "adws"
if str(ADWS) not in sys.path:
    sys.path.insert(0, str(ADWS))

from adw_modules import headroom
from adw_modules.data_types import (
    AgentConfig,
    ConfigDefaults,
    HeadroomConfig,
    PromptEngineering,
    SSSFConfig,
)


def _cfg(enabled: bool = False, tools=None) -> SSSFConfig:
    return SSSFConfig(
        defaults=ConfigDefaults(headroom=HeadroomConfig(enabled=enabled)),
        agents=[AgentConfig(name="builder",
                            prompt_engineering=PromptEngineering(system="s.md", user="u.md"),
                            tools=tools)],
    )


# ── flag resolution ──────────────────────────────────────────────────────────

def test_flag_defaults_off():
    assert headroom.enabled(_cfg(), env={}) is False


def test_flag_reads_config():
    assert headroom.enabled(_cfg(enabled=True), env={}) is True


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("on", True), ("yes", True),
    ("0", False), ("false", False), ("off", False), ("", False),
])
def test_env_overrides_config_both_ways(value, expected):
    assert headroom.enabled(_cfg(enabled=not expected), env={"SSSF_HEADROOM": value}) is expected


# ── tool names ───────────────────────────────────────────────────────────────

def test_pi_tool_names_follow_pi_mcp_extension_convention():
    # pi-mcp-extension registers <prefix>_<server>_<tool>; Headroom's tools
    # are themselves prefixed "headroom_", so the doubling is expected.
    assert headroom.pi_tool_names(_cfg(), tool_prefix="mcp") == [
        "mcp_headroom_headroom_compress",
        "mcp_headroom_headroom_retrieve",
        "mcp_headroom_headroom_stats",
    ]


def test_apply_extends_tool_lists_only_when_enabled():
    cfg = _cfg(enabled=True, tools=["read", "bash"])
    headroom.apply(cfg, env={}, tool_prefix="mcp")
    assert cfg.agents[0].tools == ["read", "bash",
                                   "mcp_headroom_headroom_compress",
                                   "mcp_headroom_headroom_retrieve",
                                   "mcp_headroom_headroom_stats"]

    off = _cfg(enabled=False, tools=["read", "bash"])
    headroom.apply(off, env={}, tool_prefix="mcp")
    assert off.agents[0].tools == ["read", "bash"]


def test_apply_is_idempotent_and_leaves_unrestricted_agents_alone():
    cfg = _cfg(enabled=True, tools=["read", "mcp_headroom_headroom_compress"])
    headroom.apply(cfg, env={}, tool_prefix="mcp")
    headroom.apply(cfg, env={}, tool_prefix="mcp")
    assert (cfg.agents[0].tools or []).count("mcp_headroom_headroom_compress") == 1

    unrestricted = _cfg(enabled=True, tools=None)     # None = every tool, already
    headroom.apply(unrestricted, env={}, tool_prefix="mcp")
    assert unrestricted.agents[0].tools is None


# ── project mcp.json materialization ─────────────────────────────────────────

def test_materialize_on_writes_only_the_headroom_server(tmp_path: Path):
    path = headroom.materialize(_cfg(enabled=True), tmp_path, on=True)
    assert path == tmp_path / ".pi" / "mcp.json"
    data = json.loads((tmp_path / ".pi" / "mcp.json").read_text())
    server = data["mcpServers"]["headroom"]
    assert server["transport"] == "stdio"
    assert server["command"] == "headroom"
    assert server["args"] == ["mcp", "serve"]
    assert server["lifecycle"] == "eager"       # -p mode cannot /mcp:start a lazy server
    assert list(data["mcpServers"]) == ["headroom"]


def test_materialize_preserves_other_project_servers(tmp_path: Path):
    pi_dir = tmp_path / ".pi"
    pi_dir.mkdir()
    (pi_dir / "mcp.json").write_text(json.dumps({
        "settings": {"toolPrefix": "mcp"},
        "mcpServers": {"other": {"transport": "stdio", "command": "other-mcp"}},
    }))
    headroom.materialize(_cfg(enabled=True), tmp_path, on=True)
    data = json.loads((pi_dir / "mcp.json").read_text())
    assert data["settings"] == {"toolPrefix": "mcp"}
    assert set(data["mcpServers"]) == {"other", "headroom"}

    headroom.materialize(_cfg(), tmp_path, on=False)
    data = json.loads((pi_dir / "mcp.json").read_text())
    assert data["settings"] == {"toolPrefix": "mcp"}
    assert list(data["mcpServers"]) == ["other"]


def test_materialize_off_removes_the_file_it_alone_populated(tmp_path: Path):
    headroom.materialize(_cfg(enabled=True), tmp_path, on=True)
    headroom.materialize(_cfg(), tmp_path, on=False)
    assert not (tmp_path / ".pi" / "mcp.json").exists()


def test_materialize_off_is_a_noop_without_a_file(tmp_path: Path):
    assert headroom.materialize(_cfg(), tmp_path, on=False) is None
    assert not (tmp_path / ".pi").exists()


# ── preflight ────────────────────────────────────────────────────────────────

def test_preflight_is_silent_when_disabled():
    assert headroom.preflight(_cfg(enabled=False), env={}) == []


def test_preflight_names_a_missing_binary(monkeypatch):
    monkeypatch.setattr(headroom.shutil, "which", lambda _cmd: None)
    problems = headroom.preflight(_cfg(enabled=True), env={},
                                  pi_packages=["npm:pi-mcp-extension"])
    assert len(problems) == 1
    assert "headroom" in problems[0] and "uv tool install" in problems[0]


def test_preflight_names_a_missing_pi_mcp_extension(monkeypatch):
    monkeypatch.setattr(headroom.shutil, "which", lambda _cmd: "/usr/bin/headroom")
    problems = headroom.preflight(_cfg(enabled=True), env={}, pi_packages=[])
    assert len(problems) == 1
    assert "pi-mcp-extension" in problems[0]
