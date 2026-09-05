# M5 human calibration scoring form

These are **model-judged secondary scores**, budget-scaled, and never objective truth. Scoring here calibrates the judges against a human reader; it does not replace the deterministic scores in `four_arm.json`.

`rubric_version`: `cfda9f8cc401`

## Rubric (identical to `eval/judges/rubrics.md`)

# M5 judge rubrics

You are scoring one blinded trace of a merger-agreement question-answering system. You are
given the question, the trace of retrieval/lookup steps the system took, the finding it
produced (or a statement that it produced none), and -- for the evidence-sufficiency
dimension only -- an expert-annotated gold span from the underlying contract.

**Do not speculate about which system, model, vendor or arm produced this trace.** Score
only what is in front of you.

## Output contract

Return exactly one JSON object with these five keys and no others:

```json
{"reasoning": <int 1-5>, "evidence": <int 1-5>, "trajectory": <int 1-5>,
 "professional": <int 1-5>, "notes": "<string, your reasoning in a sentence or two>"}
```

Every one of `reasoning`, `evidence`, `trajectory`, `professional` must be an integer from 1
to 5 inclusive. `notes` is a short string.

---

## Dimension: reasoning

Reasoning / analysis quality: how well the system's rationale reasons from what it retrieved
to its answer, including whether it draws the correct legal distinctions the question calls
for.

1. No finding was produced (the system abstained, hit its tool-call cap, or failed to
   execute), so there is no rationale to assess; or the rationale present is incoherent,
   self-contradictory, or unrelated to the question asked.
2. The rationale restates retrieved text without connecting it to the question, or draws a
   conclusion the retrieved text does not support.
3. The rationale reaches a plausible conclusion but skips or blurs a distinction the question
   turns on (e.g. conflates two different standards, or asserts a conclusion beyond what the
   quoted text shows).
4. The rationale correctly reasons from the retrieved text to the answer, addressing the
   question's key distinction, with only minor imprecision.
5. The rationale is precise, directly ties each part of its conclusion to specific retrieved
   language, and correctly handles any nuance or edge case the question raises.

## Dimension: evidence

Evidence sufficiency: whether the evidence the system cites actually supports its answer,
judged **against the supplied gold span** (the expert-annotated span of the contract that
answers this question). This is the only dimension for which a gold span is supplied.

1. No finding was produced, so there is no evidence to assess; or the system cited no
   evidence, or its evidence has no relationship to the gold span at all.
2. The cited evidence is present in the trace but does not actually establish the answer
   given, or clearly conflicts with the gold span.
3. The cited evidence is adjacent to the gold span (same general area of the contract, same
   topic) but does not fully establish the answer -- a reader would need to look further to
   be sure.
4. The cited evidence substantially overlaps the gold span and, read on its own, supports the
   answer given.
5. The cited evidence precisely captures the operative language of the gold span and fully
   supports the answer given, with no gap a careful reader would need to fill in.

## Dimension: trajectory

Trajectory quality: how well the system searched -- its query and tool choices, whether it
wasted calls, and whether it stopped at the right point -- **judged independently of the
system's deterministic skill-adherence score**, which you are not shown and must not try to
infer.

1. No finding was produced. Judge the trajectory that IS shown on its own terms: a trace that
   never issued a single well-targeted search, or that hit its call cap while still searching
   unproductively, scores at this level; a trace that made a reasonable, efficient effort and
   still legitimately found nothing available (e.g. a redacted or out-of-scope case) may score
   higher on this dimension alone even though it produced no finding.
2. The trajectory is mostly repetitive or off-target queries with little sign of adapting to
   what came back.
3. The trajectory finds relevant material but includes clearly wasted or redundant steps, or
   stops before it plausibly should have (or continues well past the point where it had what
   it needed).
4. The trajectory is efficient and adapts its queries sensibly to what it retrieves, with at
   most one step that, in hindsight, was not needed.
5. The trajectory is a minimal, well-targeted sequence of searches/lookups that finds the
   relevant material efficiently and stops as soon as it has enough.

## Dimension: professional

Professional answer quality: whether the answer, as written, is something a professional
would be comfortable relying on -- clear, appropriately hedged, free of overclaiming.

1. No finding was produced and no answer text exists to assess; or the answer text present is
   unclear, overconfident given what was actually found, or would mislead a professional
   reader.
2. The answer is understandable but is either bare (no supporting context) or overstates what
   the evidence shows.
3. The answer is clear and reasonably supported, but a professional reader would want it
   tightened or better hedged in at least one respect.
4. The answer is clear, appropriately scoped to what the evidence supports, and would be
   usable by a professional with little editing.
5. The answer is precise, appropriately hedged, and reads as something a professional could
   hand to a client or colleague without any further editing.


## Output contract

For each packet you score, append one line to `human_scores.jsonl` with these keys:

```json
{
  "packet_id": "<12-hex>",
  "scorer": "<string>",
  "scored_at": "<ISO-8601>",
  "reasoning": 1,
  "evidence": 1,
  "trajectory": 1,
  "professional": 1,
  "notes": "<string>"
}
```

Worked example:

```json
{
  "packet_id": "0123456789ab",
  "scorer": "jane",
  "scored_at": "2026-09-05T12:00:00+00:00",
  "reasoning": 4,
  "evidence": 3,
  "trajectory": 4,
  "professional": 4,
  "notes": "Correctly ties the answer to the cited section; evidence is adjacent to but not identical to the gold span."
}
```

## Suggested minimum set (12 packets, deterministic: first 4 by subset rank per variant)

Scoring more than these 12 is welcome and there is no upper bound imposed; this is only a deterministic starting point, not a cap.

- `00ffcebe096a`
- `0f0ddb6b0896`
- `257db09a639b`
- `308fa879e656`
- `31aa8adabacf`
- `32ed6171807c`
- `32fc075d8413`
- `40ea3e574ac0`
- `790521a3adab`
- `aff954956d71`
- `b36644f0d8db`
- `ea3c64c7fa8d`

## Blinding limitations

- The packet blinds arm and model identity (the spec's requirement) but cannot blind the SHAPE of the trajectory: arm A is a single-shot pipeline with no tool calls, arm D is an agent loop, so a knowledgeable reader can often infer which family of system produced a trace from trajectory length and tool-call pattern alone.
- An empty `options` list (rendered as '(free-form; no fixed option list for this question)') hints that a case is an out-of-scope probe, even though the exact question_id and category are never shown.

## Statistics notes

- Weighted Cohen's kappa is defined on integer ratings; the mean of three judges is not one. Spearman rho uses the raw mean-of-judges; kappa uses round() of it (banker's rounding, Python's default) -- see D5.
- A statistic of null means 'not computable' (degenerate input: n<2, zero variance, or a single distinct rating from both raters) and is always paired with its n; it is never coerced to 0 or 1.

**Do not open `variant_key.json` until you have finished scoring.** It maps each `packet_id` back to its case, arm and model, and would unblind you mid-scoring.

Score packets from `packets.md`, in any order you like.
