"""The prompts a milestone run hands its agents — built by code, from the spec.

Every agent in the chain reads these, so they are assembled once here, from
the milestone spec and the recorded evidence, rather than improvised per call.
The spec file is the ask; the brief is the authority; the evidence files are
what the reviewer and documenter judge from. Nothing here addresses the
harness — retries, loops and commits are the ADW's choices, not prose.
"""

from __future__ import annotations

from .data_types import MilestoneSpec, QualityResult, ReviewOutput
from .milestones import BRIEF_PATH


def build_prompt(spec: MilestoneSpec, spec_text: str, correction: str = "") -> str:
    head = (f"Implement milestone {spec.id} — {spec.title} — exactly as specified in "
            f"`{spec.spec_path}` (reproduced below).\n"
            f"Where: this repository; the spec names the modules and files.\n"
            f"Done means: every item under the spec's \"Definition of done\" holds on disk, "
            f"`uv run pytest -m \"{spec.gate_marker} and not needs_network\" -q` is green, and the "
            f"full offline suite, `uv run ruff check .` and `uv run pyright` are green.\n"
            f"Out of scope: any later milestone, the product UI, the MCP layer, and any edit to "
            f"`{BRIEF_PATH}` or `{spec.spec_path}` — both are read-only requirements.\n"
            f"Authority: `{BRIEF_PATH}` is the requirements document; the milestone spec bounds "
            f"this work. Where the spec and the brief differ on a product decision, the brief wins "
            f"and the difference is reported, not resolved silently.\n")
    if correction:
        head += (f"\nThis is a CORRECTIVE cycle in the same session. The previous attempt at this "
                 f"milestone failed; the recorded failure is reproduced below and is your spec, "
                 f"alongside the unchanged milestone spec. Fix the recorded failure fully — do not "
                 f"widen scope, do not re-plan.\n\n--- recorded failure ---\n{correction}\n")
    return head + f"\n--- {spec.spec_path} ---\n{spec_text}\n"


def review_prompt(spec: MilestoneSpec, evidence_path: str) -> str:
    return (f"Review milestone {spec.id} — {spec.title} — against `{spec.spec_path}` "
            f"(the bounded spec) and `{BRIEF_PATH}` (the authoritative requirements).\n"
            f"Where: the code on disk, the committed data and reports under `data/`, and the "
            f"deterministic evidence at `{evidence_path}` (gate results, metrics, versions) — read "
            f"the evidence rather than re-running suites.\n"
            f"Done means: one finding per \"Definition of done\" item and per brief requirement the "
            f"milestone touches, each with file:line or report-field evidence; then a disposition.\n"
            f"Disposition rules: PASS only when every item is met by what is on disk. REVISE when a "
            f"material gap is closable within this milestone — put ONE bounded corrective task in "
            f"`corrective_task`. ESCALATE only when a human must decide: the spec contradicts the "
            f"brief, satisfying the spec needs a destructive or irreversible action, credentials or "
            f"inputs are missing, benchmark integrity would need a post-hoc subjective choice, or "
            f"the milestone would tune against the frozen test set. Ordinary defects are REVISE, "
            f"never ESCALATE.\n"
            f"Out of scope: style, refactors, and work later milestones own.\n")


def revise_prompt(spec: MilestoneSpec, review: ReviewOutput) -> str:
    task = review.corrective_task.strip() or "\n".join(f"- {b}" for b in review.blocking)
    return (f"Close the reviewer's findings on milestone {spec.id} — {spec.title}.\n"
            f"Where: as named in the corrective task below and in `previous_envelope.findings`.\n"
            f"Done means: the corrective task is complete and every unmet finding is met, with "
            f"the suite, ruff, pyright and `pytest -m {spec.gate_marker}` still green.\n"
            f"Out of scope: anything the reviewer did not name.\n\n"
            f"--- corrective task ---\n{task}\n")


def document_prompt(spec: MilestoneSpec, evidence_path: str, doc_path: str) -> str:
    return (f"Write the milestone record for {spec.id} — {spec.title} — to `{doc_path}` "
            f"(create the directory if needed); this path replaces the default `app_docs/` "
            f"destination and is what `document_path` must name.\n"
            f"Where: the diff in `previous_envelope.diff_path`, the spec `{spec.spec_path}`, and the "
            f"run evidence at `{evidence_path}` — every number and identifier in the record comes "
            f"from those files, never from memory.\n"
            f"Done means: the record has these sections, in order — Goal; Implementation summary; "
            f"Acceptance criteria and result (one line per Definition-of-done item: met/not met + "
            f"evidence); Tests and quality gates executed (commands, exit codes, from the evidence); "
            f"Benchmark/eval metrics (from the evidence, or 'none for this milestone'); "
            f"Dataset/index/skill/model versions; Corrective cycles performed; Known limitations "
            f"and exclusions; Git SHA(s); SSSF session ids (this run and its parent); Braintrust "
            f"experiment/run identifiers where the evidence lists any; Next milestone.\n"
            f"Out of scope: roadmap, speculation, or anything the diff and evidence do not show.\n")


def failure_text(checks: QualityResult | None, review: ReviewOutput | None) -> str:
    """What the next corrective cycle is told, verbatim from the evidence."""
    parts: list[str] = []
    if checks is not None and not checks.passed:
        parts.append("Deterministic checks failed:\n" + "\n\n".join(checks.failures))
    if review is not None and review.resolved_disposition() != "PASS":
        unmet = [f"- {f.requirement}: {f.evidence}" for f in review.findings if not f.met]
        parts.append("Reviewer disposition " + review.resolved_disposition()
                     + (":\n" + "\n".join(unmet) if unmet else "")
                     + ("\nBlocking:\n" + "\n".join(f"- {b}" for b in review.blocking)
                        if review.blocking else "")
                     + ("\nCorrective task:\n" + review.corrective_task
                        if review.corrective_task else ""))
    return "\n\n".join(parts) or "the milestone did not verify (no evidence recorded)"
