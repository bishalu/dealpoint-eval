"""Factory robustness: the orchestration checkpoint is written by the factory itself (and by a parallel
ADW in the same checkout); it must never be attributed to the agent whose phase happens to be open."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adws"))

from adw_modules import permissions


def test_state_json_is_factory_owned_and_never_a_breach(monkeypatch):
    before = {"specs/mvp/state.json": "100644,aaaa", "dealpoint/x.py": "100644,bbbb"}
    after = {"specs/mvp/state.json": "100644,cccc", "dealpoint/x.py": "100644,bbbb"}
    assert permissions.factory_owned("specs/mvp/state.json")
    assert not permissions.factory_owned("dealpoint/x.py")
    monkeypatch.setattr(permissions, "snapshot", lambda run: after)
    agent = type("A", (), {"name": "planner", "writes": ["specs/????????_*.md"]})()
    cfg = type("C", (), {"defaults": type("D", (), {"data_dir": "adws/adw_data", "protected_files": []})()})()
    run = type("R", (), {"cfg": cfg, "repo_root": "."})()
    touched = permissions.enforce(run, None, agent, before)  # type: ignore[arg-type]
    assert touched == [], "the factory's own checkpoint write is neither a breach nor a touched path"
