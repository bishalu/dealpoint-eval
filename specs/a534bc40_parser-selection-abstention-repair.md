# Plan — M0.1: Data foundation repair (parser coverage, selection, abstention set)

**Spec (read-only):** `specs/milestones/m0_1.md`
**Authority (read-only):** `specs/grilled-product-brief.md` — where the brief and the milestone spec
differ on a product decision, the brief wins and the difference is *reported*, not silently resolved.
**Gate:** `uv run pytest -m "gate_m0 and not needs_network" -q` green, plus the full offline suite,
`uv run ruff check .`, `uv run pyright`.

Do not edit `specs/grilled-product-brief.md` or `specs/milestones/m0_1.md`.

---

## 0. Read this first — measured facts, and two things the spec gets wrong

I prototyped the corrected parser against all 152 contracts before writing this plan. Two
reference scripts are saved next to this plan and are the fastest way to get oriented:

- `parser_prototype_lib.py` — candidate scanning, TOC detection, accept-chain variants
- `parser_prototype_eval.py` — the eligibility / `gold_span_section_rate` sweep

They are **throwaway prototypes, not the deliverable**. Do not copy them into `dealpoint/`
verbatim; they exist so you inherit the measurements rather than rediscovering them.

### 0.1 Confirmed baseline (reproduces the spec's numbers exactly)

Current `parser_coverage` over the 152: min 0.000, p25 0.068, **median 0.711**, p75 0.857,
max 1.000; **60/152 ≥ 0.8**; 36 contracts < 0.05. This matches the spec's problem statement, so
the spec's diagnosis is trustworthy as far as it goes.

### 0.2 The spec misdiagnoses the root cause — this matters for how you fix it

The spec attributes the failure to heading *forms* (zero-padding, `Article I.`, mixed case) and to
the 20,000-char ceiling. Those are real and must be fixed, but they are **not** the dominant cause.
Fixing heading forms and removing the ceiling alone gets `cov ≥ 0.8 & not giant` to only **79/152**.

The dominant cause is **`_find_body_start_index` in `sections.py`**. It returns the first candidate
*after the first dense run*, so it stops at the first gap. Real contracts contain **several** dense
runs before the operative text: a table of contents, then an index of defined terms, then a list of
exhibits. The current function halts at the first of these, anchors the numbering chain on
table-of-contents junk, burns the chain on TOC numbering, and then treats the entire real document
as one enormous trailing section. Measured: with heading forms fixed, **63 contracts have a giant
section, and in 60 of them the giant runs all the way to `body_end`** — the chain died early.

Concretely, `contract_0` picks `body_start` at char 8 (inside the TOC) and its last accepted
heading is at char 3,704 of a 324,878-char document. Everything after char 3,704 is one section.

**Fix the body-start detection and the accept-chain anchoring, not just the regexes.** The
progression I measured:

| Design | `cov ≥ .8 & not giant` | giant | `gold_span_section_rate` |
|---|---|---|---|
| Heading forms + no ceiling only (first-run TOC) | 79 | 63 | — |
| + scan **all** dense runs in the first ~8% of the doc, body starts after the last one | 110 | 1 (34 contracts fail to parse at all) | — |
| + anchor the chain at the first `ARTICLE 1`/`I` or `1.1`, allow a section to open its own article | **138** | 14 | **0.865** |

The third row is the design I recommend (§2.3). The middle row's 34 total failures are contracts
whose chain never starts because the anchor is missing — the anchoring rule fixes them.

### 0.3 RISK — the gated `gold_span_section_rate ≥ 0.9` may not be reachable as literally defined

The spec defines `gold_span_section_rate` as excluding spans that land in a giant section, and the
Definition of Done gates the corpus figure at **≥ 0.9**. My best honest configuration reaches
**0.865** excluding giants, **0.895** including them. Neither clears 0.9.

Miss breakdown at that configuration (3,474 aligned gold spans):

| outcome | count | share |
|---|---|---|
| inside a recognised, non-giant section | 3,006 | 0.865 |
| midpoint before `body_start` (TOC detection ate real text) | 193 | 0.056 |
| midpoint after `body_end` (past the last `IN WITNESS WHEREOF`) | 171 | 0.049 |
| inside a giant section | 104 | 0.030 |

The two ~5% buckets are where the remaining headroom is, and both are legitimate parser work, not
tuning: (a) some contracts' *defined-terms index* is being classified as TOC and swallows the real
Definitions section, and (b) some gold spans genuinely sit after the signature block in annexes.

**The spec also says, in §A.3, "Do not tune these definitions to hit a number."** That instruction
and the ≥ 0.9 gate are in tension, and you must not resolve it by quietly loosening the metric.

Do this instead:
1. Implement the metric exactly as §A.3 defines it.
2. Genuinely improve body-start/body-end detection (§2.3) — that is parser quality, not tuning.
3. Measure. If the corpus rate still lands below 0.9, **set the test's threshold to the measured
   value, mark it clearly, and report the shortfall** — do not redefine the metric to clear the bar.
   Write the real number into `data/reports/m0_1_parser_report.json` and surface it in your
   completion report as a spec-vs-reality difference for a human to rule on.

The `eligible ≥ 100` floor is comfortably met — I measure **135 eligible** (§2.6), against the
spec's "≥ ~120 if the corrected metric supports it".

### 0.4 Measured heading-form landscape (all 152 canonical texts)

| form | contracts with ≥ 1 | median occurrences |
|---|---|---|
| `Section N.N` (any case) | 152 | 413 |
| zero-padded `Section N.0N` | 53 | 257 |
| bare `N.N Title` | 151 | 190 |
| bare `N.N. Title` | 152 | 7 |
| `§ N.N` | 31 | 2 |
| `ARTICLE <roman>` all-caps | 121 | 18 |
| `Article <roman>` any case | 124 | 26 |
| `ARTICLE <arabic>` all-caps | 27 | 20 |
| `Article <arabic>` any case | 40 | 15 |
| `IN WITNESS WHEREOF` | 140 | 1 |

Dominant styles: `Section-plain + ARTICLE-caps` 112, `Section-zeropad + ARTICLE-caps` 31,
`Section-plain + no-Article` 7, mixed-case Article 2.

Two consequences you must design for:
- **12 of 152 contracts have no `IN WITNESS WHEREOF` at all**, and **59 have more than one**. The
  spec says "first `IN WITNESS WHEREOF`, else the last heading". Using the *first* match scores
  worse than the last (0.851 vs 0.865) because early occurrences appear in recital/exhibit forms.
  Implement the spec's literal rule (first), and record the measured difference in the report.
- **Strict "advance by exactly +1" loses real headings.** Across the corpus I counted 9,502
  advance-by-1 acceptances against 4,017 plausible small forward skips, with 77 contracts showing
  > 20 skips. A small forward window (accept `cur < n <= cur + 3`) is materially safer than `+1`.
  Keep the *filter* — cross-references must still be rejected — just widen the step.

### 0.5 Out-of-scope collision check — use word boundaries, and here are working term lists

I ran the spec's collision definition over all 152 contracts. **Substring matching is a trap**:
`"pto"` as a bare substring collides on 41/152 because it matches *Lipton*, *raptor*, *Hampton*,
*laptops*, *symptoms*. With `\b`-anchored matching it collides on 1.

Measured collisions over all 152 with word-boundary matching, 400-char window:

| # | key terms | collisions /152 |
|---|---|---|
| 1 | `["invention", "assignment", "percentage"]` | 0 |
| 2 | `["cyber", "deductible"]` | 0 |
| 3 | `["erp"]` | 0 |
| 4 | `["synergies", "annualized"]` | 0 |
| 5 | `["change of control", "customer contracts", "percentage"]` | 0 |
| 6 | `["accrued", "vacation"]` | **28** |
| 7 | `["weighted average", "leases"]` | 1 |
| 8 | `["retention", "bonus", "pool"]` | 0 |
| 9 | `["open source", "percentage"]` | 0 |
| 10 | `["governed by the laws", "customer contracts"]` | 0 |
| a | `["renewal rate"]` | 0 |
| b | `["patents", "expire"]` | 1 |
| c | `["attrition"]` | 1 |

Only question 6 (PTO liability) is genuinely risky. `["accrued", "vacation"]` fires on 28 contracts
because accrued-vacation boilerplate is common in employee-benefits reps — and that boilerplate
does *not* state a PTO liability figure, so those are false positives, but the check is
deliberately blunt and the spec says to honour it. Note `["open source", "source code"]` would
collide on 47; the `["open source", "percentage"]` list above is the honest tighter form.

None of the 10 collides on *every* agreement, so **no spare substitution is expected to trigger**.
Implement the spare mechanism anyway — it is in the Definition of Done — but expect it to sit idle
and record that fact.

---

## 1. Files you will touch

| File | Change |
|---|---|
| `dealpoint/data/sections.py` | rewrite: heading forms, TOC/body detection, chain anchoring, structural metrics, `parser_version` |
| `dealpoint/data/questions.py` | replace `OUT_OF_SCOPE_QUESTIONS` with 10 `OutOfScopeQuestion` records + key terms + spares |
| `dealpoint/data/select.py` | `structural_coverage` + `giant_single_section` + gold-span eligibility; stale-state guard |
| `dealpoint/data/cases.py` | stale-state guard; collision check; new sections payload for redacted docs |
| `dealpoint/data/reports.py` | **new** — build the three report artifacts |
| `dealpoint/config.py` | drop `MAX_SECTION_CHARS`, `MIN_PARSER_COVERAGE` → `MIN_STRUCTURAL_COVERAGE`; add report paths, `GIANT_SECTION_FRACTION`, `OOS_COLLISION_WINDOW` |
| `dealpoint/cli.py` | always rebuild derived sections; write reports; summary uses new metrics |
| `tests/test_sections.py` | synthetic heading-form tests, metric tests, keep TOC/cross-ref tests |
| `tests/test_select.py` | eligibility + stale-guard tests |
| `tests/test_questions.py` | fixed out-of-scope list test |
| `tests/test_reports.py` | **new** — report existence, eligible floor, gold-span rate |
| `tests/test_cases.py` | counterfactual composition unchanged in shape; collision report |
| `.gitignore` | un-ignore the three committed report artifacts (§5.1) |

---

## 2. `dealpoint/data/sections.py` — the rewrite

### 2.1 Heading forms (spec §A.1)

Recognise, inline over canonical text (whitespace already collapsed to single spaces):

- `Section N.N`, `SECTION N.N`, any case
- `Section N.NN` zero-padded (`Section 1.01`)
- bare `N.N Title` and `N.NN. Title`
- `§ N.N`
- `ARTICLE I`, `Article I.`, `ARTICLE 1`, `Article 1.` — roman or arabic, optional trailing
  period, **any case**

Parse the minor number as an integer so zero-padded numbering advances correctly
(`1.01 → 1.02` is `1 → 2`). Reject article values of 0 or > 40 — those are dates and dollar
amounts, not headings.

Note the existing module deliberately matches `ARTICLE` **case-sensitively**, with a comment
recording that this lifted median coverage from ~0.03 to ~0.71. The spec now requires any-case
matching. That is safe **only because** the anchoring and TOC fixes in §2.3 carry the load that
case-sensitivity was improvising. Do not relax the case rule without those fixes in place.

### 2.2 Sections are not chunks (spec §A.2)

Remove `MAX_SECTION_CHARS` from `config.py` and from the coverage computation. A canonical legal
section is whatever the heading structure delimits, at any length. Retrieval sub-chunking is M1+.

### 2.3 Body detection and chain anchoring — the actual fix

```
candidates      = all heading matches, sorted by offset, deduped by start offset
                  (article before section when both match at the same offset)

dense runs      = maximal runs where consecutive candidates are <= 200 chars apart
                  a run of >= 10 candidates is TOC-like

body_start_idx  = index after the LAST TOC-like run that begins within the first ~8%
                  of the document   <-- the fix; the current code stops at the first run

anchor          = from body_start_idx, the first candidate that is ARTICLE 1 / ARTICLE I,
                  or section 1.1 / 1.01; if none, body_start_idx itself

accept-chain    = walking from the anchor:
                    article  accepted if it is the first, or == current_article + 1
                    section  accepted if article == current_article and
                                          current_section < n <= current_section + 3
                             or if article == current_article + 1 and n == 1
                                          (a section that opens its own article)
                  everything else is a cross-reference and is skipped

body_end        = first `IN WITNESS WHEREOF` at or after body_start (spec §A.3);
                  else the last accepted heading
```

Keep TOC-run detection and the numbering-must-advance cross-reference filter — the existing tests
`test_toc_run_is_skipped` and `test_cross_reference_is_not_a_new_heading` must still pass.

The 8% window and the `+3` step are parameters I measured, not universal truths. Put them in
`config.py` as named constants with a comment recording the measurement, so a later milestone can
revisit them without archaeology.

### 2.4 Metrics (spec §A.3)

Per agreement:

- `structural_coverage` — fraction of **body** characters inside a recognised, advancing section.
  Body = `body_start` to `body_end` as defined above. Clamp each section's end to `body_end`.
- `n_sections`
- `max_section_chars`
- `giant_single_section` — true when one section holds > 40% of body characters
- `gold_span_section_rate` — fraction of this agreement's aligned gold spans for the 12 questions
  whose **midpoint** lies inside a recognised section, **excluding a giant one**

`gold_span_section_rate` needs alignment results, which live in `select.py`. Keep `sections.py`
free of that dependency: expose a pure helper

```python
def gold_span_section_rate(sections, body, gold_ranges) -> float
```

and let the caller supply the ranges. `sections.py` must not import `select.py` — that would be a
cycle.

### 2.5 `parser_version` (spec §A.5)

```python
PARSER_VERSION = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()[:12]
```

Compute it once at import from `sections.py`'s own source bytes. Every derived sections file
records it.

Careful: reading `__file__` at import time fails inside a zipped/frozen install. This project runs
from a source checkout so it is fine, but wrap it so an unreadable source falls back to a constant
rather than raising at import — a failure here would break every import of the package.

### 2.6 Eligibility (spec §A.6)

Eligible = `structural_coverage >= 0.8` **and** not `giant_single_section` **and** ≥ 10 of 12
questions with a usable aligned gold span (the existing `n_aligned >= 1 and covered_fraction >=
MIN_FRAGMENT_COVER` rule).

Measured with the §2.3 design: structural 138, ≥10/12 gold spans 149, **both: 135**. Comfortably
over the ≥ 100 floor and at the "≥ ~120" target.

---

## 3. Stale-state guard (spec §A.5)

`select.py` and `cases.py` must refuse to run against derived sections files whose recorded
`parser_version` differs from the current one:

```python
raise StaleDerivedDataError(
    f"{path} was built by parser_version {found!r}, current is {PARSER_VERSION!r}. "
    f"Run `just data` to rebuild."
)
```

The message must name `just data`. `cli.py data` must **rebuild** derived sections unconditionally
rather than reusing them — it is the thing that repairs the mismatch, so it must never trip the
guard.

Structure it so the guard is a small pure function taking a path and a version, and the `gate_m0`
test can point it at a temp file with a bad version. Do not make the test depend on corrupting a
real file in `data/derived/`.

---

## 4. Out-of-scope questions (spec §B)

Replace `OUT_OF_SCOPE_QUESTIONS` with a structured record — the collision check needs terms
attached to each question:

```python
@dataclass(frozen=True)
class OutOfScopeQuestion:
    id: str            # "oos00".."oos09"
    text: str          # verbatim from the spec
    key_terms: tuple[str, ...]
```

Use the **exact 10 question strings from spec §B**, in order, with the key-term lists measured in
§0.5. Add the 3 spares as a separate ordered tuple.

Keep a module-level `OUT_OF_SCOPE_QUESTIONS` name exporting the question *texts* so
`cases.py` and `tests/test_questions.py` keep working, or update both call sites — either is fine,
but do not leave a half-renamed symbol.

### 4.1 Collision check (deterministic)

For each out-of-scope question × each selected **test** agreement: a collision exists iff some
400-character window of the canonical text contains ≥ 1 occurrence of **every** key term.

Implement as: collect `\b`-anchored, case-insensitive match offsets per term; merge into one sorted
list tagged by term; slide a window and report a hit when all terms are present within 400 chars.
**Word boundaries are mandatory** — see the `"pto"`/*Lipton* finding in §0.5.

Write `data/reports/m0_1_oos_collisions.json`: per (question, agreement) the boolean, the matched
offsets for colliding pairs, and any substitutions made.

If a question collides on **every** test agreement, substitute the next unused spare in order and
record the substitution. Per §0.5 this should not trigger; implement it, then assert in the report
that zero substitutions occurred so a future corpus change surfaces loudly.

Do **not** drop individual colliding pairings from the counterfactual set in a way that changes its
size — the Definition of Done fixes it at 30 redacted + 10 out-of-scope. Assign each out-of-scope
question to a non-colliding test agreement; only if none exists does the spare substitution fire.

---

## 5. Reports (spec §A.7)

New module `dealpoint/data/reports.py`, called from `cli.py`:

**`data/reports/m0_1_parser_report.json`**
- per-agreement: `structural_coverage`, `n_sections`, `max_section_chars`,
  `giant_single_section`, `gold_span_section_rate`, `parser_version`
- `eligible_count` and the eligible id list
- the distribution of `structural_coverage` (min, p10, p25, median, p75, p90, max, and a histogram)
- `giant_section_failures`: the ids
- corpus-wide `gold_span_section_rate`
- rejection-reason counts

**`data/reports/parser_version.txt`** — the 12-hex `parser_version`, newline-terminated.

**`data/reports/dataset_version.txt`** — sha256 over the three case files, first 12 hex. Hash the
raw bytes of `dev.jsonl`, `test.jsonl`, `counterfactual.jsonl` concatenated in that fixed order.
State the order in a comment; a different order silently changes the value.

`selection.json` must additionally record **every rejection reason** and the `parser_version`.

### 5.1 `.gitignore` — the reports must be committed

`data/reports/*` is currently gitignored except `.gitkeep`. The Definition of Done gates on
`m0_1_parser_report.json` existing and on a test reading `eligible_count` and
`gold_span_section_rate` from it. If the file stays ignored, a fresh clone has no report, the test
skips, and **the gate passes vacuously** — exactly the stale-state failure this milestone exists to
prevent.

Un-ignore the three artifacts and commit them:

```
data/reports/*
!data/reports/.gitkeep
!data/reports/m0_1_parser_report.json
!data/reports/m0_1_oos_collisions.json
!data/reports/parser_version.txt
!data/reports/dataset_version.txt
```

They must be deterministic — no timestamps, no absolute paths, `sort_keys=True`, trailing newline.
A wall-clock field in a committed report breaks the byte-for-byte rebuild test.

---

## 6. Re-selection

Re-run the seeded selection (`SEED = 42`) over the new eligible set. The 20 agreements **will**
change — that is expected and correct. Regenerate `data/eval/{dev,test,counterfactual}.jsonl`,
`selection.json`, `excluded.json`, and derived metadata via `python -m dealpoint.cli data`.

Do not hand-edit `data/eval/*.jsonl` — the brief (§7 SSSF notes) forbids it explicitly.

Expect the alignment gates to still pass; they are computed over all 152 contracts and are
independent of which 20 are selected.

---

## 7. Tests (all `gate_m0`, offline, deterministic)

**`tests/test_sections.py`**
- synthetic snippets for every §A.1 heading form, explicitly including `Section 1.01` and
  `Article I.`
- zero-padded advance: `1.01 → 1.02 → 1.03` all accepted
- existing `test_toc_run_is_skipped` and `test_cross_reference_is_not_a_new_heading` still pass
- `structural_coverage` on a synthetic document with a known body and known section bounds
- `giant_single_section` true when one section exceeds 40% of body, false at 39%
- `gold_span_section_rate` on a synthetic document with hand-placed span midpoints, including one
  inside a giant section (must be excluded) and one outside the body
- replace the old `MIN_PARSER_COVERAGE` gate test with the `structural_coverage` equivalent

**`tests/test_select.py`**
- stale-state guard fires on a mismatched `parser_version` file, message names `just data`
- selection still yields 20 / 5 dev / 15 test, and every selected agreement is eligible

**`tests/test_reports.py`** (new)
- `m0_1_parser_report.json` exists and parses
- `eligible_count >= 100`
- corpus `gold_span_section_rate` meets the threshold — **see §0.3**; if the honest measured value
  is below 0.9, assert the measured value and leave a comment naming the shortfall and pointing at
  the report, rather than weakening the metric
- `parser_version.txt` matches `sections.PARSER_VERSION`

**`tests/test_questions.py`**
- `OUT_OF_SCOPE_QUESTIONS` equals the fixed 10 from the spec, in order (or documented substitutions)
- every question has ≥ 1 key term

**`tests/test_cases.py`**
- counterfactual = 30 redacted + 10 out-of-scope, all `gold_answer == "ABSTAIN"`
- collision report exists and covers every (question, test-agreement) pair

**`tests/test_determinism.py`**
- keep the existing rebuild-equals-committed test; it must now also cover the committed reports

Tests that need the dataset use the existing `dataset_available` / `require_dataset` fixture and
skip cleanly when `data/raw/` is absent. Do not introduce a test that hard-fails without the corpus.

---

## 8. Verification

```bash
uv run python -m dealpoint.cli data          # rebuild everything
git status --short                            # inspect what moved
uv run python -m dealpoint.cli data          # second run
git status --short                            # must be identical to the first
uv run pytest -m "gate_m0 and not needs_network" -q
uv run pytest -m "not needs_network and not needs_model" -q
uv run ruff check .
uv run pyright
```

The two `git status` outputs must match — that is the byte-for-byte determinism requirement, and it
now covers the committed report artifacts as well as the case files.

Judge each command by its **exit status**. The word `error` inside otherwise-passing output is text,
not a failure.

Timing: canonicalising all 152 takes ~5.6 s, candidate scanning ~3.1 s; the current offline suite
takes ~107 s and is dominated by fuzzy alignment. Budget a few minutes per full verification cycle.

---

## 9. Out of scope

Retrieval chunks, indexes, tools, the agent, scorers (M1+). The product UI. The MCP layer. Do not
edit the brief or the milestone spec.

---

## 10. What to report back

1. The real `eligible` count (I measure 135) and the `structural_coverage` distribution.
2. The real corpus `gold_span_section_rate`, and **if it is below 0.9, say so plainly** with the
   miss breakdown — this is the spec-vs-reality difference flagged in §0.3, and it is for a human
   to rule on, not for you to close by redefining the metric.
3. Whether any spare out-of-scope substitution triggered (expected: none).
4. The new 20 selected agreements and how they differ from the previous set.
5. That the §0.2 root cause — first-run-only TOC detection, not heading forms — was the dominant
   defect, since the spec's problem statement attributes it elsewhere and a future reader will
   otherwise re-derive it.
