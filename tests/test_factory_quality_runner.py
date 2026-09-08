"""Factory robustness: a quality check must report, never crash the phase (M9c/M9d both died on
`can't concat str to bytes` when the offline suite exceeded the runner's timeout)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adws"))

from adw_modules import quality


class _Console:
    def __init__(self):
        self.notes: list[str] = []

    def note(self, text):
        self.notes.append(text)


class _Tracer:
    def event(self, record):
        pass


def _fake_run(tmp_path):
    return SimpleNamespace(phases=[SimpleNamespace(phase_id="p1")], repo_root=str(tmp_path), adw_id="t",
                           console=_Console(), tracer=_Tracer(), session_dir=tmp_path)


def test_as_text_decodes_bytes_and_tolerates_none():
    assert quality._as_text(b"x\xff") == "x�" and quality._as_text(None) == "" and quality._as_text("s") == "s"


def test_timeout_is_a_failed_result_not_an_exception(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    monkeypatch.setattr(quality, "_check_dir", lambda run, name: tmp_path)
    spec = quality.QualityCheckSpec(name="slow", area="backend", operation="test",
                                    argv=[sys.executable, "-c", "import time; time.sleep(5)"], timeout_seconds=1)
    result = quality._run(spec, run)
    assert result.passed is False and result.returncode == 124 and "Timed out after 1s" in result.output_tail


def test_timeout_with_bytes_streams_is_decoded(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    monkeypatch.setattr(quality, "_check_dir", lambda run, name: tmp_path)

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1, output=b"partial out", stderr=b"partial err")

    monkeypatch.setattr(quality.subprocess, "run", _raise)
    spec = quality.QualityCheckSpec(name="t", area="backend", operation="test", argv=["x"], timeout_seconds=1)
    result = quality._run(spec, run)
    assert result.returncode == 124 and "partial err" in result.output_tail and "Timed out" in result.output_tail


def test_unexpected_runner_error_is_a_failed_result(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    monkeypatch.setattr(quality, "_check_dir", lambda run, name: (_ for _ in ()).throw(RuntimeError("boom")))
    spec = quality.QualityCheckSpec(name="t", area="backend", operation="test", argv=["x"], timeout_seconds=1)
    result = quality._run(spec, run)
    assert result.passed is False and result.returncode == 1 and "boom" in result.output_tail
