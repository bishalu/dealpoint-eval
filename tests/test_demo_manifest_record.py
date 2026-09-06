"""M7b gate: `just demo-manifest-record` appends/replaces one cockpit-session
step without touching other manifest keys (spec `m7b.md` section 7).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gate_m7b


def test_record_step_appends_and_replaces_by_step_number():
    from dealpoint.eval.demo_manifest_record import record_step

    manifest = {"project_id": "p1", "cockpit_session": []}
    m1 = record_step(manifest, step=1, note="scored 12", url=None, tv=None)
    assert len(m1["cockpit_session"]) == 1
    assert m1["cockpit_session"][0]["note"] == "scored 12"
    assert m1["project_id"] == "p1"

    m2 = record_step(m1, step=2, note="loop thread", url="https://example.com", tv=None)
    assert len(m2["cockpit_session"]) == 2

    m3 = record_step(m2, step=1, note="rescored 12", url=None, tv=None)
    assert len(m3["cockpit_session"]) == 2  # step 1 replaced, not duplicated
    assert next(e for e in m3["cockpit_session"] if e["step"] == 1)["note"] == "rescored 12"


def test_main_writes_to_demo_manifest_path(tmp_path, monkeypatch):
    import json

    import dealpoint.config as config_mod
    import dealpoint.eval.demo_manifest_record as record_mod

    fake_path = tmp_path / "demo_manifest.json"
    fake_path.write_text(json.dumps({"project_id": "p1"}), encoding="utf-8")
    monkeypatch.setattr(config_mod, "DEMO_MANIFEST_PATH", fake_path)
    monkeypatch.setattr(record_mod, "DEMO_MANIFEST_PATH", fake_path)

    rc = record_mod.main(["--step", "3", "--note", "playground checked", "--url", "https://x"])
    assert rc == 0
    written = json.loads(fake_path.read_text(encoding="utf-8"))
    assert written["cockpit_session"][0]["step"] == 3
    assert written["cockpit_session"][0]["url"] == "https://x"
