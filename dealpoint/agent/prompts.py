"""Question spec block + system prompt shown to the model in every arm.

No skill content here — that is M4. A reviewer checks this file for absence
of any SKILL.md-derived rules.
"""

from __future__ import annotations

from dealpoint.data.questions import QuestionSpec


def system_prompt() -> str:
    return (
        "You are a careful M&A deal-point review assistant. You answer one question "
        "about one merger agreement at a time. Ground every answer only in the text of "
        "THIS agreement -- never rely on outside knowledge of market practice or other "
        "deals. When you cite the agreement, quote it verbatim, character for character "
        "(aside from straightening curly quotes/dashes). Only abstain (answer "
        '"ABSTAIN") when the provision the question asks about is genuinely absent from '
        "this agreement, not merely hard to find."
    )


def question_spec_block(question: QuestionSpec) -> str:
    """The block shown in every arm: MAUD question, gloss, exact options, output contract.

    For an out-of-scope counterfactual question (no fixed option list), call
    `out_of_scope_question_block` directly instead -- this function is only
    for the 12-question `QuestionSpec` spec.
    """
    options_lines = "\n".join(f'  - "{opt}"' for opt in question.options)
    return (
        f"Question ({question.id}): {question.maud_question}\n"
        f"Plain-English gloss: {question.gloss}\n"
        f"Answer options (choose exactly one string, verbatim):\n"
        f"{options_lines}\n"
        '  - "ABSTAIN" (use only if this agreement genuinely does not address the '
        "question)\n\n"
        "Output contract: respond with a JSON object with exactly these fields:\n"
        '  "answer": one of the option strings above, verbatim, or "ABSTAIN"\n'
        '  "evidence": a list of 1 to 3 objects {"section_ref": ..., "quote": ...}, each '
        "quote a verbatim substring of the agreement text (empty list only if answer is "
        "ABSTAIN)\n"
        '  "rationale": at most 80 words, stating the option chosen and the clause it '
        "rests on"
    )


def out_of_scope_question_block(question_text: str) -> str:
    """Free-form variant for out-of-scope counterfactual cases (no fixed option list)."""
    return (
        f"Question: {question_text}\n\n"
        "This question may not be answerable from this merger agreement at all. If the "
        "agreement genuinely does not address it, respond with:\n"
        '  "answer": "ABSTAIN", "evidence": [], "rationale": <<=80 words explaining why>>\n'
        "Otherwise answer with a short string, 1-3 verbatim evidence quotes with section "
        "refs, and a rationale <=80 words."
    )
