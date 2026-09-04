# Review Task

## Variables

### prompt

{{prompt}}

### previous_envelope

{{previous_envelope}}

### context_handoff_dir

{{context_handoff_dir}}

## Task

Confirm that the work reported in `previous_envelope` is what was asked for.

1. Establish the spec: read `<context_handoff_dir>/plan.md` if it exists, else use `prompt`.
2. Read the code that was actually written, starting from `previous_envelope.changed_files`.
3. Rule on every requirement in the spec — one `findings` entry each, with evidence.
4. Write the review to `<context_handoff_dir>/review.md`, then emit your `Report` JSON.

## Report

Respond with ONLY valid JSON matching `ReviewOutput` — no prose before or after:

```json
{
  "status": "success",
  "approved": false,
  "summary": "<one sentence: N of M requirements met>",
  "findings": [
    { "requirement": "<the ask, in the requester's words>", "met": true, "evidence": "src/server.ts:42 — handler registered" }
  ],
  "blocking": ["<what must change before this can be approved>"],
  "disposition": "PASS | REVISE | ESCALATE",
  "corrective_task": "<REVISE only: one bounded task a builder executes as-is — what, where, how verified>",
  "escalation_reason": "<ESCALATE only: the single decision a human must make>",
  "artifacts": ["<context_handoff_dir>/review.md"],
  "notes_for_next_agent": "<what the builder must fix, or how to verify if approved>"
}
```

`status` is `success` when the review itself completed — it is not the verdict. The verdict is `approved`, and it is true only when `findings` has no unmet entry and `blocking` is empty. `disposition` restates the verdict for code: `PASS` iff approved; otherwise `REVISE` with a `corrective_task`, or `ESCALATE` with an `escalation_reason`.
