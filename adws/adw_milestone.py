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
"""ADW Milestone — one bounded product milestone, end to end, under the factory's own gates.

Usage:
    uv run adws/adw_milestone.py --milestone m1 [--parent <mvp adw_id>] [--adw-id X] [--correction] [--config adws/adw_sssf_config/sssf.config.yaml]

Phases: engineer(request) -> planner -> git(commit_plan) [-> code(spend_gate)]
        -> builder -> [code(checks) -> builder(fix)] bounded
        -> reviewer [-> builder(revise) -> code(checks/fix) -> reviewer] bounded
        -> git(commit_build) -> code(changes) -> code(evidence) -> documenter -> git(commit_docs)
        -> code(record) -> git(commit_state)

The milestone spec (specs/milestones/<id>.md) is the ask; the brief is the
authority; `specs/mvp/state.json` is where the result lands. The reviewer's
typed disposition is what the outer loop branches on: PASS lands the work,
REVISE is closed here inside a bounded revision loop, ESCALATE ends the run
with exit 3 so a human can make the one decision it names.

`--correction` re-enters the SAME session: agents resume their context windows,
the planner is skipped, and the builder's spec is the recorded failure of the
previous attempt plus the unchanged milestone spec.

Exit codes: 0 passed · 1 defect after bounded loops (the parent may launch a
correction) · 3 a human decision is required (budget, ESCALATE) — the parent stops.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from adw_modules import (agents, changes, gates, git_helper, milestone_prompts,
                         milestones, quality, session, spend)
from adw_modules.data_types import (AgentCall, BuildOutput, ChangeCapture,
                                    DocumentOutput, PhaseParams, PlanOutput,
                                    ReviewOutput)
from adw_modules.utils import now_iso

REQUIRED_AGENTS = ["planner", "builder", "reviewer", "documenter"]
MAX_FIX_LOOPS = 3
MAX_REVISION_LOOPS = 2


def main(milestone_id: str, parent: str = "", config: str = "adws/adw_sssf_config/sssf.config.yaml",
         adw_id: str | None = None, correction: bool = False) -> int:
    cfg = agents.load_config(config)
    agents.validate(cfg, REQUIRED_AGENTS)
    ms = milestones.spec(milestone_id)
    if not Path(ms.spec_path).is_file():
        raise SystemExit(f"milestone spec missing: {ms.spec_path}")
    run = session.ensure(cfg, adw_id)
    baseline = git_helper.rev("HEAD")

    state = milestones.load_state()
    rec = milestones.record(state, ms.id)
    rec.status = "running"
    rec.attempts += 1
    rec.started_at = rec.started_at or now_iso()
    if run.adw_id not in rec.adw_ids:
        rec.adw_ids.append(run.adw_id)
    state.current_milestone = ms.id
    state.next_action = f"{ms.id} running in session {run.adw_id} (attempt {rec.attempts})"
    milestones.save_state(state)

    include_model = ms.needs_model and bool(os.environ.get("OPENROUTER_API_KEY"))
    spec_text = Path(ms.spec_path).read_text()
    prompt = milestone_prompts.build_prompt(ms, spec_text,
                                           correction=rec.last_failure if correction else "")
    evidence_path = run.context_handoff_dir / "milestone_evidence.json"
    doc_path = f"docs/milestones/{ms.id}.md"

    def commit(ph, envelope, fallback: str) -> str:
        message = getattr(envelope, "commit_message", "") or fallback
        sha = git_helper.commit_all(message)
        rec.commits.append(sha)
        ph.log(sha=sha, message=message)
        return sha

    def record_check(ph, result) -> None:
        passed = sum(1 for check in result.checks if check.passed)
        ph.log(passed=result.passed, checks=f"{passed}/{len(result.checks)}",
               artifacts=", ".join(result.artifacts))

    def verify(tag: str, build):
        """Checks → fix → checks, bounded. Returns (checks, build) — the last of each."""
        checks = None
        for i in range(1, MAX_FIX_LOOPS + 1):
            with run.phase(PhaseParams(name=f"checks_{tag}{i}", kind="code", owner="quality",
                                       description="Suite, ruff, pyright and the milestone gate in one "
                                                   "pass — known commands, so code runs them")) as ph:
                checks = quality.run_milestone_checks(run, ms.gate_marker, include_model)
                record_check(ph, checks)
            if checks.passed or i == MAX_FIX_LOOPS:
                break
            with run.phase(PhaseParams(name=f"fix_{tag}{i}", kind="agent", owner="builder", retries=1,
                                       description="Repair every failure the checks reported, from "
                                                   "their verbatim output")) as ph:
                build = ph.call(AgentCall(output_type=BuildOutput, prompt=prompt,
                                          previous=quality.as_envelope(checks, "checks"),
                                          gates=[gates.diff_matches_claims]))
        return checks, build

    escalate = ""
    checks = None
    review = None
    build = None
    agent_failure = ""          # an agent that declared its own failure, or blew its gates

    with run.phase(PhaseParams(name="request", kind="engineer", owner=run.engineer,
                               description="Record which milestone this session works, under which "
                                           "parent run, against which brief")) as ph:
        ph.log(input=f"{ms.id}: {ms.title}", spec=ms.spec_path, parent=parent or "-",
               brief_sha=milestones.brief_sha(), attempt=rec.attempts, correction=correction,
               gate=ms.gate_marker, model_tests=include_model,
               baseline=git_helper.short_sha(baseline))

    plan = None
    if not correction:
        with run.phase(PhaseParams(name="plan", kind="agent", owner="planner", retries=1,
                                   description="Turn the milestone spec into a plan the builder can "
                                               "execute without deciding product questions")) as ph:
            plan = ph.call(AgentCall(output_type=PlanOutput, prompt=prompt,
                                     gates=[gates.artifacts_exist, gates.files_non_empty]))
        with run.phase(PhaseParams(name="commit_plan", kind="code", owner="git",
                                   description="Put the plan on record before any code exists to blur it")) as ph:
            commit(ph, plan, f"{ms.id}: plan — {plan.summary}")

    if ms.spend_gate:
        with run.phase(PhaseParams(name="spend_gate", kind="code", owner="budget",
                                   description="Project this milestone's OpenRouter spend against the "
                                               "operator's cap before any metered sweep starts")) as ph:
            sc = spend.check(run, ms)
            ph.log(ok=sc.ok, reason=sc.reason)
            state.spend.cap_usd = sc.cap_usd
            state.spend.realized_usd = sc.realized_usd
            state.spend.estimates.append({"milestone": ms.id, "session": run.adw_id,
                                          "calls": sc.calls, "est_usd": sc.estimate_usd,
                                          "ok": sc.ok, "at": now_iso()})
            milestones.save_state(state)
            if not sc.ok:
                escalate = f"spend guard: {sc.reason}"
            # The product runner enforces the per-milestone absolute at sweep time; the factory
            # tells it the bound and the baseline through the environment every agent inherits.
            if ms.absolute_usd is not None:
                os.environ["DEALPOINT_MILESTONE_ID"] = ms.id
                os.environ["DEALPOINT_MILESTONE_ABSOLUTE_USD"] = f"{ms.absolute_usd:.2f}"
                os.environ["DEALPOINT_MILESTONE_SPEND_START_USD"] = f"{sc.realized_usd:.4f}"
                ph.log(runner_bound=f"{ms.id} absolute ${ms.absolute_usd:.2f} from baseline ${sc.realized_usd:.4f}")

    try:
        if not escalate:
            with run.phase(PhaseParams(name="build" if not correction else f"build_c{rec.attempts}",
                                       kind="agent", owner="builder",
                                       description="Implement the milestone spec, or close the recorded "
                                                   "failure of the previous attempt")) as ph:
                build = ph.call(AgentCall(output_type=BuildOutput, prompt=prompt, previous=plan,
                                          gates=[gates.diff_matches_claims]))

            checks, build = verify("", build)

            if checks is not None and checks.passed:
                for i in range(1, MAX_REVISION_LOOPS + 1):
                    with run.phase(PhaseParams(name=f"review_{i}", kind="agent", owner="reviewer",
                                               description="Rule on every definition-of-done item against the "
                                                           "spec, the brief and the recorded evidence")) as ph:
                        _write_evidence(run, ms, rec, parent, checks, None, evidence_path)
                        review = ph.call(AgentCall(output_type=ReviewOutput,
                                                   prompt=milestone_prompts.review_prompt(ms, str(evidence_path)),
                                                   previous=build,
                                                   gates=[gates.artifacts_exist, gates.verdict_consistent]))
                    disposition = review.resolved_disposition()
                    if disposition in ("PASS", "ESCALATE") or i == MAX_REVISION_LOOPS:
                        break
                    with run.phase(PhaseParams(name=f"revise_{i}", kind="agent", owner="builder", retries=1,
                                               description="Execute the reviewer's one bounded corrective task")) as ph:
                        build = ph.call(AgentCall(output_type=BuildOutput,
                                                  prompt=milestone_prompts.revise_prompt(ms, review),
                                                  previous=review, gates=[gates.diff_matches_claims]))
                    checks, build = verify(f"r{i}_", build)
                    if not checks.passed:
                        break
                if review is not None and review.resolved_disposition() == "ESCALATE":
                    escalate = f"reviewer: {review.escalation_reason or '; '.join(review.blocking)}"
    except (RuntimeError, agents.GateFailure) as error:
        # The phase already recorded itself as failed; keep going so the checkpoint
        # carries the agent's OWN words (a builder's "status=fail" summary, a gate's
        # violations) rather than a generic "exited before recording".
        agent_failure = str(error)[:1500]
        run.console.note(f"milestone stops at a failed agent phase: {agent_failure[:200]}")

    verified = (not escalate and not agent_failure and checks is not None and checks.passed
                and review is not None and review.resolved_disposition() == "PASS")

    if verified:
        with run.phase(PhaseParams(name="commit_build", kind="code", owner="git",
                                   description="Land the code only now: green checks, PASS review")) as ph:
            commit(ph, build, f"{ms.id}: {build.summary}")

        with run.phase(PhaseParams(name="changes", kind="code", owner="git",
                                   description="Diff the whole milestone against its pinned baseline, "
                                               "for the documenter")) as ph:
            changeset = changes.capture(run, ChangeCapture(base=baseline))
            ph.log(base=f"{changeset.base.label} @ {changeset.base.commit[:7]}",
                   files=len(changeset.files) + len(changeset.untracked),
                   lines=f"+{changeset.insertions} -{changeset.deletions}")
            if changeset.empty:
                raise RuntimeError("nothing changed since the baseline — there is nothing to document")

        with run.phase(PhaseParams(name="evidence", kind="code", owner="factory",
                                   description="Freeze the run's deterministic evidence so the record "
                                               "is written from facts, not memory")) as ph:
            _write_evidence(run, ms, rec, parent, checks, review, evidence_path)
            ph.log(path=str(evidence_path))

        with run.phase(PhaseParams(name="document", kind="agent", owner="documenter", retries=1,
                                   description="Write the milestone record from the diff and the evidence")) as ph:
            document = ph.call(AgentCall(
                output_type=DocumentOutput,
                prompt=milestone_prompts.document_prompt(ms, str(evidence_path), doc_path),
                previous=changes.as_envelope(changeset, "Read diff_path and the evidence file in full "
                                                        "before writing. Every number comes from them."),
                gates=[gates.artifacts_exist, gates.files_non_empty]))

        with run.phase(PhaseParams(name="commit_docs", kind="code", owner="git",
                                   description="Ship the milestone record beside the code it describes")) as ph:
            commit(ph, document, f"{ms.id}: milestone record")

    with run.phase(PhaseParams(name="record", kind="code", owner="factory",
                               description="Write the milestone's outcome, ids, metrics and versions "
                                           "into the orchestration checkpoint")) as ph:
        state = milestones.load_state()
        rec_now = milestones.record(state, ms.id)
        rec_now.adw_ids, rec_now.attempts, rec_now.commits = rec.adw_ids, rec.attempts, rec.commits
        rec_now.started_at = rec.started_at
        rec_now.models = {name: entry.get("model", "") for name, entry in run.agent_map.items()}
        rec_now.metrics = milestones.collect_metrics()
        rec_now.versions = milestones.collect_versions()
        rec_now.ended_at = now_iso()
        if verified:
            rec_now.status, rec_now.blocker, rec_now.last_failure = "passed", "", ""
            nxt = milestones.next_milestone(state)
            state.next_action = f"run {nxt.id}" if nxt else "core eval MVP complete"
        elif escalate:
            rec_now.status, rec_now.blocker = "blocked", escalate
            state.next_action = f"HUMAN DECISION for {ms.id}: {escalate}"
        else:
            rec_now.status = "failed"
            rec_now.last_failure = (agent_failure + "\n\n" if agent_failure else "") \
                + milestone_prompts.failure_text(checks, review)
            state.next_action = f"corrective cycle for {ms.id} in session {run.adw_id}"
        state.current_milestone = ms.id
        milestones.save_state(state)
        ph.log(status=rec_now.status, next=state.next_action)

    if verified:
        with run.phase(PhaseParams(name="commit_state", kind="code", owner="git",
                                   description="Checkpoint the orchestration state with the milestone it "
                                               "just closed, so a cold session resumes from facts")) as ph:
            commit(ph, None, f"{ms.id}: passed — orchestration state @ {git_helper.short_sha('HEAD')}")

    rc = run.finish(accepted=verified,
                    reason=escalate or agent_failure
                    or "checks or review never came back clean within the bounded loops")
    return milestones.EXIT_ESCALATE if escalate else rc


def _write_evidence(run, ms, rec, parent, checks, review, path: Path) -> None:
    evidence = {
        "milestone": {"id": ms.id, "title": ms.title, "spec": ms.spec_path,
                      "gate_marker": ms.gate_marker},
        "session": {"adw_id": run.adw_id, "parent": parent, "attempt": rec.attempts,
                    "brief_sha": milestones.brief_sha()},
        "git": {"head": git_helper.short_sha("HEAD"), "commits_this_milestone": rec.commits},
        "checks": [c.model_dump() for c in (checks.checks if checks else [])],
        "checks_passed": bool(checks and checks.passed),
        "review": (review.model_dump() if review else None),
        "metrics": milestones.collect_metrics(),
        "versions": milestones.collect_versions(),
        "models": {name: entry.get("model", "") for name, entry in run.agent_map.items()},
        "spend": {"cap_usd": spend.cap_usd(), "realized_usd": spend.realized_usd()},
        "written_at": now_iso(),
    }
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone", required=True, help="milestone id from adw_modules.milestones.SEQUENCE")
    parser.add_argument("--parent", default="", help="the MVP run's adw_id, for linkage")
    parser.add_argument("--config", default="adws/adw_sssf_config/sssf.config.yaml")
    parser.add_argument("--adw-id", default=None, help="join or pin an existing session")
    parser.add_argument("--correction", action="store_true",
                        help="re-enter the session to close the recorded failure; skips planning")
    args = parser.parse_args()
    sys.exit(main(args.milestone, args.parent, args.config, args.adw_id, args.correction))
