# Cockpit session — operator procedure (M7b §7)

The factory (`just braintrust-cockpit`, `just demo-walkthrough`) produces every
object below except the interactive steps: no MCP tool creates a human review
score, a Loop thread, a Playground comparison, or a Loop-generated custom
trace view. Run this procedure with an MCP-enabled Braintrust session
(`claude mcp list` must show `braintrust`) after `gate_m7b` is green, and
record each result with:

```bash
uv run python -m dealpoint.eval.demo_manifest_record --step <n> --note "<what you saved>" [--url <url>] [--tv <tv-param>]
```

This appends to `data/reports/demo_manifest.json`'s `cockpit_session` list;
`just demo-walkthrough` (step 5) then embeds recorded URLs in the doc.

## 1. Score the review set

- Object: `maud-dealpoint-review-set` (see `demo_manifest.json`'s `views` ->
  `Review set (12)`), or, since Review-mode scoring needs configured human
  review scores this Starter-plan workspace cannot carry (see
  `demo_manifest.json`'s `human_scoring_probe`), score
  `data/eval/calibration/form.md` instead — the 12 packets it lists are the
  same 12 the review-set view flags.
- Prompt/question: follow `form.md`'s per-packet rubric (reasoning, evidence,
  trajectory, professional, 1-5, frozen rubric anchors from
  `eval/judges/rubrics.md`, hash `cfda9f8cc401`).
- Save: `data/eval/calibration/human_scores.jsonl` (append rows in `form.md`'s
  schema).
- Then run: `just calibration` (recomputes `data/reports/judges.json`'s
  agreement stats) and `just braintrust-cockpit` (pushes the new
  `human/<dimension>` scores onto the matching `judge-<variant_id>` rows).
- Record: `--step 1 --note "<n scored, scorer id>"`.

## 2. Loop investigation thread over the hero case

- Object: the hero case named in `demo_manifest.json`'s `hero_case.case_id`,
  on the `m7b-hero-case` experiment (`demo_manifest.json`'s `replay.experiment_name`).
- Prompt: ask Loop the same question BTQL query 2 answers in code —
  "Which model/arm combinations hit EXECUTION_FAILED or CAP_HIT, and how
  often, in the traces around this case?" (see `docs/braintrust-queries.md`
  query 2 for the exact BTQL it should agree with).
- Save: the Loop thread, project-wide.
- Record: `--step 2 --url <thread-url> --note "<one-sentence takeaway; does it match query 2's counts>"`
  (this also covers the Topics cluster-name/`failure_attribution` cross-check
  from §7's factory half — note that mapping in the same `--step 2` call if
  Topics has finished clustering by the time you run this).

## 3. Playground: hero case vs the calibrated judge prompt

- Object: the hero case's blinded packet (same `packet_id` `judge_spans()`
  used for the `m7b-hero-case` replay — look it up via
  `dealpoint.eval.braintrust_cockpit.judge_spans(hero_case_id, "A@haiku")["packet_id"]`
  or `"D@haiku"`).
- Prompt: the calibrated judge rubric/prompt Braintrust prompt
  `judge-calibrated-rubric` (published by `braintrust_sync.py`), run against
  the three judge models side by side: `mistralai/mistral-small-3.2-24b-instruct`,
  `nvidia/nemotron-3-super-120b-a12b`, `bytedance-seed/seed-2.0-mini`
  (`data/reports/judge_slate.json`).
- Save: the Playground session URL.
- Record: `--step 3 --url <playground-url> --note "<matches stored judge JSON: yes/no, and where it differs>"`.

## 4. Custom trace view from Loop

- Object: the `m7b-hero-case` experiment's spans (three `judge/<family>`
  children plus `judge/aggregate` under `scoring`, per variant).
- Prompt: ask Loop to generate a custom trace view — "show the three judge
  spans as a 3 x 4 grid of scores (rows: judges, columns: reasoning,
  evidence, trajectory, professional) with the human row beneath, when a
  human score exists."
- Save: project-wide, so it applies to every judged trace, not just the hero
  case.
- Record: `--step 4 --tv <tv-parameter-value> --note "<one line: what the grid showed for the hero case>"`.

## 5. Regenerate the walkthrough

```bash
just demo-walkthrough
```

This re-reads `data/reports/demo_manifest.json` (now carrying the four
recorded steps) and rewrites `docs/demo-walkthrough.md`, so the `[cockpit
session]` markers left by the factory carry the real URLs recorded above.
