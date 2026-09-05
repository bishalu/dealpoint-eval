# M&A Deal-Point Review — Skill

A generic procedure for answering one MAUD-style question about one merger
agreement. Ground every answer only in the text of THIS agreement. Follow
these eight rules in order; each is checked mechanically against your
trajectory and final answer.

1. **Locate before you answer.** Call `search_agreement` at least once to
   find the operative clause before giving a final answer. Never answer from
   memory of "typical" merger agreements.

2. **Resolve defined terms.** If the question turns on a defined term (e.g.
   Knowledge, Material Adverse Effect, Superior Proposal, Intervening Event),
   call `lookup_defined_term` for that term before answering. Do not guess a
   definition from context alone when the tool is available.

3. **Follow cross-references.** If the clause you are reading points to
   another section ("as set forth in Section X.Y", "subject to Article Z"),
   call `get_section` on that reference before answering. The operative rule
   is often in the referenced section, not the one you started in.

4. **Read carve-outs to the end.** If a retrieved passage ends mid-list (a
   dangling ";", ",", " and", or " or"), the carve-outs or exceptions
   continue in the next chunk or section — read on and retrieve it before
   concluding what is and is not excluded.

5. **Quote verbatim, cite sparingly.** Every citation must be an exact quote
   (character for character, aside from straightening quotes/dashes) from a
   chunk you actually retrieved in this trajectory. Use at most 3 citations
   — quality over quantity.

6. **Search before abstaining.** Before answering ABSTAIN, run at least 2
   searches with meaningfully different phrasing. A single failed search is
   not evidence the provision is absent.

7. **Abstain only when genuinely absent.** Only answer ABSTAIN when the
   provision the question asks about is genuinely not addressed in this
   agreement — not merely hard to find. A correct "No"-type answer (the
   agreement addresses the point and the answer is negative) is not an
   abstention and must be cited like any other answer.

8. **State your reasoning concisely.** Your rationale must be at most 80
   words and must name the option you chose and the specific clause or
   section it rests on.
