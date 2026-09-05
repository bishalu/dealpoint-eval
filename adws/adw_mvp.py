#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#     "braintrust>=0.37.0",
#     "pydantic",
#     "python-dotenv",
#     "pyyaml",
#     "rich",
# ]
# ///
"""ADW MVP — the outer state machine that carries the project to the core eval MVP.

Usage:
    uv run adws/adw_mvp.py [--dry-run] [--only m1] [--max-corrections 2] [--adw-id X] [--config adws/adw_sssf_config/sssf.config.yaml]

Phases: engineer(request) -> code(<milestone>_run1 | <milestone>_fix2 ...) per milestone,
        each one launching adw_milestone.py in ITS OWN session and reading the
        checkpoint it wrote back

This script decides nothing about the product. It reads `specs/mvp/state.json`,
picks the first milestone that has not passed, launches `adw_milestone.py` for it
as a fresh SSSF session (fresh agent context — M1 never inherits M0's tokens),
and branches on the child's exit code: 0 advances, 1 launches one bounded
corrective cycle in the SAME child session (context intact) up to
`--max-corrections` times, 3 stops the loop for a human decision. It is
resumable: rerun it and it continues from the checkpoint.

Exit codes: 0 every milestone passed · 1 a milestone stayed failed · 3 stopped for a human.
"""

import argparse
import sys
from pathlib import Path

from adw_modules import agents, milestones, session, spend, utils
from adw_modules.data_types import ChildLaunch, PhaseParams

REQUIRED_AGENTS = ["planner", "builder", "reviewer", "documenter"]   # validated here, so a broken roster fails now, not hours in
DEFAULT_MAX_CORRECTIONS = 2


def dry_run(config: str) -> int:
    cfg = agents.load_config(config)
    agents.validate(cfg, REQUIRED_AGENTS)
    state = milestones.load_state()
    print(f"parent run: {state.parent_run_id or '(none yet)'}   brief: {milestones.brief_sha()}")
    print(milestones.describe(state))
    missing = [m.spec_path for m in milestones.SEQUENCE if not Path(m.spec_path).is_file()]
    print(f"spec files: {'all present' if not missing else 'MISSING ' + ', '.join(missing)}")
    cap = spend.cap_usd()
    print(f"spend cap: {'$%.2f' % cap if cap is not None else 'NOT SET (' + spend.CAP_ENV + ')'}   "
          f"realized: ${spend.realized_usd():.2f}")
    nxt = milestones.next_milestone(state)
    print(f"next action: {state.next_action or ('run ' + nxt.id if nxt else 'complete')}")
    return 0 if not missing else 1


def main(config: str, adw_id: str | None, only: str, max_corrections: int) -> int:
    cfg = agents.load_config(config)
    agents.validate(cfg, REQUIRED_AGENTS)
    run = session.ensure(cfg, adw_id)

    state = milestones.load_state()
    if not state.parent_run_id:
        state.parent_run_id = run.adw_id
    state.brief_sha = milestones.brief_sha()
    milestones.save_state(state)
    parent = state.parent_run_id

    with run.phase(PhaseParams(name="request", kind="engineer", owner=run.engineer,
                               description="Record the loop's inputs: parent id, brief hash, cap, and "
                                           "where the checkpoint says we are")) as ph:
        cap = spend.cap_usd()
        ph.log(input=f"core eval MVP loop — {only or 'all remaining milestones'}", parent=parent,
               brief_sha=state.brief_sha, max_corrections=max_corrections,
               spend_cap=f"${cap:.2f}" if cap is not None else "unset",
               status="\n" + milestones.describe(state))

    outcome = 0
    launched: dict[str, int] = {}          # per-milestone launches THIS run — the loop's own bound
    while True:
        state = milestones.load_state()
        ms = milestones.next_milestone(state)
        if ms is None:
            state.next_action = "core eval MVP complete"
            milestones.save_state(state)
            break
        rec = milestones.record(state, ms.id)
        if only and ms.id != only:
            break
        if rec.status == "blocked":
            outcome = milestones.EXIT_ESCALATE
            break
        correction = rec.status == "failed" and bool(rec.adw_ids)
        launched[ms.id] = launched.get(ms.id, 0) + 1
        product_attempts = rec.attempts - rec.infra_retries
        if rec.infra_retries > milestones.MAX_INFRA_RETRIES:
            rec.status = "blocked"
            rec.blocker = (f"{ms.id}: {rec.infra_retries} infrastructure failures "
                           f"({rec.last_failure_class}); last: {rec.last_failure[:400]}")
            state.next_action = f"HUMAN DECISION for {ms.id}: infrastructure keeps failing — {rec.blocker}"
            milestones.save_state(state)
            outcome = milestones.EXIT_ESCALATE
            break
        if launched[ms.id] > max_corrections + milestones.MAX_INFRA_RETRIES + 1 \
                or (correction and product_attempts > max_corrections):
            rec.status = "blocked"
            rec.blocker = (f"{ms.id} did not pass after {rec.attempts} attempt(s) "
                           f"({max_corrections} corrective cycle(s) allowed); last failure: "
                           f"{rec.last_failure[:400]}")
            state.next_action = f"HUMAN DECISION for {ms.id}: {rec.blocker}"
            milestones.save_state(state)
            outcome = milestones.EXIT_ESCALATE
            break

        child = rec.adw_ids[-1] if correction else utils.new_id(8)
        name = f"{ms.id}_{'fix' if correction else 'run'}{rec.attempts + 1}"
        with run.phase(PhaseParams(name=name, kind="code", owner="factory",
                                   description=f"Launch {ms.id} as its own SSSF session "
                                               f"{'to close its recorded failure' if correction else 'from a clean context'}, "
                                               f"and read back what it recorded")) as ph:
            argv = ["uv", "run", "adws/adw_milestone.py", "--milestone", ms.id, "--parent", parent,
                    "--adw-id", child, "--config", config] + (["--correction"] if correction else [])
            log_path = run.session_dir / f"{name}.log"
            ph.log(child=child, correction=correction, log=str(log_path))
            rc = milestones.launch_child(run, ChildLaunch(argv=argv, log_path=str(log_path), label=name))
            state = milestones.load_state()
            rec = milestones.record(state, ms.id)
            if milestones.interrupted(rc):
                # Killed, not failed: leave nothing that a resume would mistake for a defect.
                rec.status = "pending"
                rec.last_failure = ""
                state.next_action = (f"{ms.id} was interrupted (session {child} exit {rc}) — "
                                     f"resume with `just mvp`")
                milestones.save_state(state)
                ph.log(exit=rc, status="interrupted")
                outcome = milestones.EXIT_ESCALATE
                break
            if rc != milestones.EXIT_PASSED and rec.status not in ("failed", "blocked"):
                # The child died before its record phase — say so, in the checkpoint.
                rec.status = "failed"
                rec.last_failure = f"session {child} exited {rc} before recording — see {log_path}"
                if child not in rec.adw_ids:
                    rec.adw_ids.append(child)
                milestones.save_state(state)
            ph.log(exit=rc, status=rec.status, blocker=rec.blocker or "-",
                   commits=", ".join(rec.commits) or "-")

        if outcome == milestones.EXIT_ESCALATE or rc == milestones.EXIT_ESCALATE:
            outcome = milestones.EXIT_ESCALATE
            break
        if rc == milestones.EXIT_INFRA:
            run.console.note(f"{ms.id}: infrastructure failure ({rec.last_failure_class}) — "
                             f"retrying with fresh sessions, not counted as a correction")
        # 0 → the loop picks the next milestone; 1 → the same milestone comes back as a correction.

    state = milestones.load_state()
    done = milestones.next_milestone(state) is None
    if state.pending_human_input:
        run.console.note("pending human input: " + "; ".join(state.pending_human_input))
    run.console.note("next action: " + state.next_action)
    rc = run.finish(accepted=(outcome == 0 and (done or bool(only))),
                    reason=state.next_action)
    return milestones.EXIT_ESCALATE if outcome == milestones.EXIT_ESCALATE else rc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the checkpoint and the plan; launch nothing")
    parser.add_argument("--only", default="", help="run this one milestone and stop")
    parser.add_argument("--max-corrections", type=int, default=DEFAULT_MAX_CORRECTIONS,
                        help="bounded corrective cycles per milestone before escalating")
    parser.add_argument("--config", default="adws/adw_sssf_config/sssf.config.yaml")
    parser.add_argument("--adw-id", default=None, help="join or pin the parent session")
    args = parser.parse_args()
    if args.dry_run:
        sys.exit(dry_run(args.config))
    sys.exit(main(args.config, args.adw_id, args.only, args.max_corrections))
