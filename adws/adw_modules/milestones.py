"""Milestone registry, orchestration state, and child-run plumbing.

The project-level MVP loop is two ADWs over this module: `adw_milestone.py`
runs ONE milestone end to end and records what happened; `adw_mvp.py` decides
which milestone is next and launches it as its own session. Neither script
holds state — this module does, in `specs/mvp/state.json`, a checkpoint small
enough to read cold and honest enough to resume from.

The registry is static on purpose. Milestone order, gate markers, and which
milestones touch the spend guard are factory facts, versioned with the
factory; an agent never gets to decide what the next milestone is.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Optional

from . import git_helper
from .data_types import (ChildLaunch, MilestoneRecord, MilestoneSpec, MVPState)
from .utils import now_iso, operator_env

STATE_PATH = Path("specs/mvp/state.json")
BRIEF_PATH = "specs/grilled-product-brief.md"
REPORTS_DIR = Path("data/reports")

# The core eval MVP, in order. UI (M7) and MCP (M8) are deliberately absent:
# the autonomous run ends at M6 by the engineer's decision.
SEQUENCE: list[MilestoneSpec] = [
    MilestoneSpec(id="m0_1", title="Data foundation repair: parser coverage, selection, abstention set",
                  spec_path="specs/milestones/m0_1.md", gate_marker="gate_m0"),
    MilestoneSpec(id="m1", title="Agent foundation: chunks, dense index, three tools, bounded loop",
                  spec_path="specs/milestones/m1.md", gate_marker="gate_m1", needs_model=True),
    MilestoneSpec(id="m2", title="Eval harness: deterministic scorers, Braintrust adapter, smoke eval",
                  spec_path="specs/milestones/m2.md", gate_marker="gate_m2", needs_model=True),
    MilestoneSpec(id="m3", title="Retrieval tournament on dev; freeze arm C",
                  spec_path="specs/milestones/m3.md", gate_marker="gate_m3"),
    MilestoneSpec(id="m4", title="Four-arm experiment on the frozen test set",
                  spec_path="specs/milestones/m4.md", gate_marker="gate_m4", needs_model=True,
                  spend_gate=True,
                  budget_argv=["uv", "run", "python", "-m", "dealpoint.eval.budget", "four_arm"],
                  absolute_usd=3.00),
    MilestoneSpec(id="m4_1", title="Execution-reliability repair and four-arm re-run",
                  spec_path="specs/milestones/m4_1.md", gate_marker="gate_m4", needs_model=True,
                  spend_gate=True,
                  budget_argv=["uv", "run", "python", "-m", "dealpoint.eval.budget", "four_arm"],
                  absolute_usd=1.50),
    MilestoneSpec(id="m5", title="Calibrated multi-judge evaluation",
                  spec_path="specs/milestones/m5.md", gate_marker="gate_m5", needs_model=True,
                  spend_gate=True,
                  budget_argv=["uv", "run", "python", "-m", "dealpoint.eval.budget", "judges"],
                  absolute_usd=1.20),
    MilestoneSpec(id="m6", title="Model cost/quality Pareto experiment",
                  spec_path="specs/milestones/m6.md", gate_marker="gate_m6", needs_model=True,
                  spend_gate=True,
                  budget_argv=["uv", "run", "python", "-m", "dealpoint.eval.budget", "pareto"],
                  absolute_usd=2.00),
    MilestoneSpec(id="m7a", title="Framework roles: LlamaIndex RAG lab, DeepEval cross-check, Braintrust sync",
                  spec_path="specs/milestones/m7a.md", gate_marker="gate_m7", needs_model=True,
                  spend_gate=True,
                  budget_argv=["uv", "run", "python", "-m", "dealpoint.eval.budget", "m7a"],
                  absolute_usd=2.50),
]

# Exit codes shared by both ADWs, so the parent can branch without parsing prose.
EXIT_PASSED = 0
EXIT_DEFECT = 1          # bounded loops exhausted on an ordinary defect — correctable
EXIT_ESCALATE = 3        # a human decision is required — the loop stops cleanly
EXIT_INFRA = 4           # the infrastructure failed, not the work — retry fresh, don't count it
MAX_INFRA_RETRIES = 3    # per milestone, across the whole loop, before a human is asked


def spec(milestone_id: str) -> MilestoneSpec:
    for item in SEQUENCE:
        if item.id == milestone_id:
            return item
    raise SystemExit(f"unknown milestone {milestone_id!r} — known: {[m.id for m in SEQUENCE]}")


# ── state ────────────────────────────────────────────────────────────────────

def load_state(path: Path = STATE_PATH) -> MVPState:
    if path.exists():
        return MVPState.model_validate_json(path.read_text())
    return MVPState()


def save_state(state: MVPState, path: Path = STATE_PATH) -> None:
    state.updated_at = now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.model_dump(), indent=2, sort_keys=True) + "\n")


def record(state: MVPState, milestone_id: str) -> MilestoneRecord:
    if milestone_id not in state.milestones:
        state.milestones[milestone_id] = MilestoneRecord(id=milestone_id)
    return state.milestones[milestone_id]


def next_milestone(state: MVPState) -> Optional[MilestoneSpec]:
    """The first milestone that has not passed. Blocked ones are returned too —
    the caller decides whether a blocked milestone stops the loop."""
    for item in SEQUENCE:
        if record(state, item.id).status != "passed":
            return item
    return None


def brief_sha() -> str:
    """Content hash of the authoritative brief, so a run records what it built against."""
    result = subprocess.run(["git", "hash-object", BRIEF_PATH], capture_output=True, text=True)
    return result.stdout.strip()[:12] if result.returncode == 0 else ""


def describe(state: MVPState) -> str:
    """One line per milestone — what a fresh session prints before doing anything."""
    lines = []
    for item in SEQUENCE:
        rec = record(state, item.id)
        extra = f"  sessions={','.join(rec.adw_ids)}" if rec.adw_ids else ""
        if rec.blocker:
            extra += f"  BLOCKER: {rec.blocker}"
        lines.append(f"  {item.id:5} {rec.status:8} attempts={rec.attempts}{extra}")
    return "\n".join(lines)


# ── evidence collection (deterministic, read-only) ───────────────────────────

def collect_metrics() -> dict:
    """Pull the headline numbers a milestone left in data/reports/ and data/eval/.

    Deterministic and read-only: every JSON report's top-level scalar fields, plus
    case counts. Nothing here interprets a number; the documenter and the
    reviewer read the same files and draw their own conclusions.
    """
    out: dict = {}
    if REPORTS_DIR.exists():
        for path in sorted(REPORTS_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                scalars = {k: v for k, v in data.items()
                           if isinstance(v, (int, float, str, bool)) and not isinstance(v, bool)}
                if scalars:
                    out[path.stem] = scalars
    eval_dir = Path("data/eval")
    if eval_dir.exists():
        counts = {}
        for path in sorted(eval_dir.glob("*.jsonl")):
            try:
                counts[path.stem] = sum(1 for line in path.open() if line.strip())
            except OSError:
                pass
        if counts:
            out["case_counts"] = counts
    return out


def collect_versions() -> dict[str, str]:
    """Frozen identifiers the product exposes, if it exposes them yet."""
    versions: dict[str, str] = {}
    for name in ("index_version", "skill_version", "dataset_version", "parser_version"):
        path = Path("data/reports") / f"{name}.txt"
        if path.exists():
            versions[name] = path.read_text().strip()
    frozen = Path("data/reports/versions.json")
    if frozen.exists():
        try:
            versions.update({k: str(v) for k, v in json.loads(frozen.read_text()).items()})
        except (OSError, ValueError):
            pass
    return versions


# ── child runs (the parent loop launching one milestone ADW) ─────────────────

def launch_child(run, launch: ChildLaunch) -> int:
    """Run one milestone ADW as a subprocess and return its exit code.

    Output streams to `log_path`, so the parent's console stays a summary while
    the child's full narrative is on disk. The child pid is recorded in the
    parent's `processes` row so `just kill <parent>` can reach it.
    """
    log_path = Path(launch.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        # Its own process group, so that when THIS process is stopped the whole
        # subtree — the milestone ADW and the coding agent under it — goes with
        # it. Otherwise `kill <parent>` orphans a builder mid-edit.
        proc = subprocess.Popen(launch.argv, cwd=run.repo_root, env=operator_env(),
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        run.tracer.process_start(run.adw_id, "adw", launch.label, proc.pid,
                                 " ".join(launch.argv))
        try:
            return proc.wait()
        except BaseException:
            _terminate_group(proc)
            raise
        finally:
            run.tracer.process_end(run.adw_id, proc.pid)


def _terminate_group(proc: subprocess.Popen, grace_seconds: float = 10.0) -> None:
    """SIGTERM the child's process group, then SIGKILL whatever ignored it."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def interrupted(returncode: int) -> bool:
    """A child that died of a signal was stopped, not defeated — never correct it."""
    return returncode < 0 or returncode in (130, 137, 143) or returncode >= 128


def current_short_sha() -> str:
    return git_helper.short_sha("HEAD")


def preflight() -> str:
    """Is the coding-agent provider usable right now? Empty string = yes, else the reason.

    The subscription provider silently drops out of pi's catalog when `claude` is not
    authenticated, and a milestone launched into that state burns its attempts on
    'Unknown provider'. Two cheap checks, both deterministic: the CLI's own auth
    status, and pi's catalog listing the provider the roster names.
    """
    try:
        auth = subprocess.run(["claude", "auth", "status"], capture_output=True, text=True,
                              timeout=30, env=operator_env())
        data = json.loads(auth.stdout[auth.stdout.find("{"):]) if "{" in auth.stdout else {}
        if not data.get("loggedIn"):
            return "claude is not logged in — run `claude login`, then `just mvp` to resume"
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return f"could not read `claude auth status` ({error}) — check the claude CLI, then resume"
    try:
        listing = subprocess.run(["pi", "--list-models"], capture_output=True, text=True,
                                 timeout=90, env=operator_env())
        if "pi-claude-code-provider" not in listing.stdout:
            return ("pi does not list the pi-claude-code-provider — the extension failed to load "
                    "(usually auth); run `claude login` and `pi --list-models`, then resume")
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"`pi --list-models` failed ({error}) — check pi, then resume"
    return ""
