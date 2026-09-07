"""`just demo-walkthrough` -- regenerate `docs/demo-walkthrough.md` from
`data/reports/demo_manifest.json` (spec `m7b.md` section 6). Offline, no
model calls: every fact quoted comes from the manifest and already-written
local reports. Every `[M7b]` marker the M7a draft left is replaced with
either finished content or a `[cockpit session]` marker (spec section 7) --
never both, never left as `[M7b]`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dealpoint.config import (
    DEMO_MANIFEST_PATH,
    DEMO_WALKTHROUGH_DRAFT_PATH,
    DEMO_WALKTHROUGH_PATH,
    M7B_MAX_WORDS,
)

COCKPIT_MARKER = "[cockpit session]"


def _manifest() -> dict:
    return json.loads(Path(DEMO_MANIFEST_PATH).read_text(encoding="utf-8"))


# --- generated section bodies (one per `[M7b]`-marked paragraph) -----------


def _intro_paragraph() -> str:
    return (
        "Every name, id and number below resolves to `data/reports/demo_manifest.json`, "
        "which `just braintrust-cockpit` regenerates from Git/local sources. Sections "
        f"needing a human operator in an MCP-enabled session are marked **{COCKPIT_MARKER}**; "
        "everything else is produced by code, checked by the `gate_m7b` offline suite."
    )


def _logs_trace_section(manifest: dict) -> str:
    view_ids = ", ".join(f"`{v['name']}` ({v['id']})" for v in manifest["views"])
    return (
        f"Saved views over these spans, idempotent by name: {view_ids}. "
        f"{COCKPIT_MARKER} A Loop-generated custom trace view (judge spans as a 3x4 grid, "
        "human row beneath); its `tv` param via `just demo-manifest-record --step 4`."
    )


def _review_section(manifest: dict) -> str:
    probe = manifest["human_scoring_probe"]
    n_pushed = manifest["n_human_scores_pushed"]
    n_human_live = (manifest.get("ledger") or {}).get("n_human_scores", 0)
    n_planned = manifest["n_human_scores_planned"]
    return (
        f"Human-scoring path: **{probe['decision']}** -- Starter plan allows one configured "
        "review score, not the four rubric dimensions needed, so scoring stays at "
        f"`data/eval/calibration/form.md` (M5) and `just braintrust-cockpit` pushes "
        f"{n_planned} `human/<dimension>` scores onto the matching `judge-<variant_id>` rows "
        f"({n_human_live} live per the score ledger; {n_pushed} in this run). Shown in the experiment table and trace, not Review "
        f"mode.\n\n{COCKPIT_MARKER} Scoring the review set at `form.md`, then `just calibration` "
        "and `just braintrust-cockpit` (`--step 1`)."
    )


def _loop_sql_section() -> str:
    return (
        "The six BTQL investigations are saved as four of the seven cockpit views "
        "(`Retrieval rescue`, `Failure attribution`, `DeepEval vs judge disagreement`, "
        f"`Trajectory inefficiency`), matching `docs/braintrust-queries.md` verbatim.\n\n"
        f"{COCKPIT_MARKER} A Loop thread over the hero case asking query 2's question in plain "
        "words; URL via `just demo-manifest-record --step 2`."
    )


def _debugger_section(manifest: dict) -> str:
    topics = manifest["topics"]["config"]
    pattern = manifest["pattern"]["definition"]
    pattern_id = manifest["pattern"].get("id")
    body = (
        f"Topics (`{topics['facet_name']}` facet, clustering on) render each `case > agent` "
        "span as question/tools/answer/`obj/grounded_accuracy` text."
    )
    if pattern_id:
        body += (
            f" One Pattern, `{pattern['name']}`, names {len(pattern['supporting_trace_ids'])} "
            "supporting trace ids from the same predicate as BTQL query 6."
        )
    else:
        # No public REST route exists to create a Pattern on this platform
        # (PATTERN_REST_LIMITATION); the payload below is defined in code but
        # not yet a real Braintrust object -- say so, behind the marker, 
        # rather than asserting it exists.
        body += (
            f"\n\n{COCKPIT_MARKER} One Pattern, `{pattern['name']}` "
            f"({len(pattern['supporting_trace_ids'])} supporting trace ids from the same predicate "
            "as BTQL query 6) is defined in code but has no public REST creation route on this "
            "platform, so it is created during the cockpit session (`new_pattern`), not by "
            "`braintrust_cockpit.py`."
        )
    body += (
        f"\n\n{COCKPIT_MARKER} Cluster names surfaced by Topics, and whether any maps to a "
        "query-2 failure cause (`--step 2`)."
    )
    return body


def _dashboard_section(manifest: dict) -> str:
    dash = manifest["dashboard"]
    return (
        f"A saved dashboard, `{dash['name']}` ({dash['id']}), six charts: obj/grounded_accuracy "
        "by arm/model; judge/<dimension> mean by variant; judge vs human on the review set "
        "(captioned \"pending human calibration\" only while no human scores exist); RAG tournament hit@5/hit@10/MRR by retriever; $/case by model; "
        "DeepEval vs obj/ agreement rate."
    )


def _playground_paragraph() -> str:
    return (
        f"{COCKPIT_MARKER} Playground: the hero case's blinded packet against the calibrated "
        "judge prompt, three judge models side by side (`--step 3`)."
    )


def _hero_case_paragraph(manifest: dict) -> str:
    hero = manifest["hero_case"]
    if hero.get("case_id") is None:
        return "No case in the judged subset satisfies the hero disagreement rule this run."
    replay = manifest["replay"]
    return (
        f"**Hero case `{hero['case_id']}`** (arms A@haiku/D@haiku disagree on "
        f"`obj/grounded_accuracy`: `{hero['grounded_accuracy_a']}` vs "
        f"`{hero['grounded_accuracy_d']}`; max pairwise judge spread `{hero['max_pairwise_spread']}` "
        f"among {hero['n_candidates']} candidates). Replayed as "
        f"`{replay.get('experiment_name') or 'm7b-hero-case'}`: "
        "`case -> agent -> ... -> scoring -> {judge/mistral, judge/nvidia, judge/bytedance, "
        "judge/aggregate}` for both variants, zero model calls."
    )


# --- paragraph-level marker replacement -------------------------------------

_PARAGRAPH_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\*\*\[M7b\]\*\* Custom trace view / saved views over these spans\.", "_logs_trace_section"),
    (r"\*\*\[M7b\]\*\* The 12-trace blinded review set[\s\S]*?carries 24 scored rows\.", "_review_section"),
    (r"\*\*\[M7b\]\*\* Braintrust Loop investigation thread\.[\s\S]*?trajectory inefficiency 50 rows\.", "_loop_sql_section"),
    (r"\*\*\[M7b\]\*\* Topics/Patterns/Debugger use over the synced traces\.", "_debugger_section"),
    (r"\*\*\[M7b\]\*\* A saved dashboard over the above experiments/scores\.", "_dashboard_section"),
)

_SECTION_BUILDERS = {
    "_logs_trace_section": _logs_trace_section,
    "_review_section": _review_section,
    "_loop_sql_section": lambda manifest: _loop_sql_section(),
    "_debugger_section": _debugger_section,
    "_dashboard_section": _dashboard_section,
}


def generate_walkthrough(manifest: dict, draft_text: str) -> str:
    """Replace every `[M7b]` marker paragraph in the FROZEN M7a draft
    (`docs/templates/demo-walkthrough.draft.md`, never edited in place) with
    finished content (or a `[cockpit session]` marker) computed from
    `manifest`. Reading from the frozen draft rather than the previous
    output keeps a second `just demo-walkthrough` run idempotent.
    """
    text = draft_text.replace("# Demo walkthrough (draft) — M7a", "# Demo walkthrough")
    text = re.sub(
        r"A 5–10 minute technical walkthrough[\s\S]*?never reruns it\.",
        _intro_paragraph(),
        text,
        count=1,
    )
    for pattern, builder_name in _PARAGRAPH_REPLACEMENTS:
        builder = _SECTION_BUILDERS[builder_name]
        replacement = builder(manifest)
        new_text, n = re.subn(pattern, lambda _m, r=replacement: r, text, count=1, flags=re.DOTALL)
        if n != 1:
            raise ValueError(f"expected exactly one match for pattern {pattern!r}, found {n}")
        text = new_text

    # Playground has no [M7b] marker in the M7a draft -- append under Review.
    review_body = _review_section(manifest)
    text = text.replace(review_body, review_body + "\n\n" + _playground_paragraph())

    hero_line = "## The hero path\n\n" + _hero_case_paragraph(manifest) + "\n"
    text = text.replace("## 1. Datasets", hero_line + "\n## 1. Datasets")

    remaining_markers = re.findall(r"\[M7b\]", text)
    if remaining_markers:
        raise ValueError(f"{len(remaining_markers)} [M7b] marker(s) survived generation")

    text = text.rstrip("\n") + (
        "\n\n---\n\nEvery link and id above is one `data/reports/demo_manifest.json` can "
        "regenerate: re-run `just braintrust-cockpit` then `just demo-walkthrough`. This doc "
        "has no content that only exists because a person clicked it once.\n"
    )
    return text


def _word_count(text: str, *, exclude_appendix: bool = True) -> int:
    body = text.split("## Reproduce this walkthrough")[0] if exclude_appendix else text
    return len(body.split())


def main(argv: list[str] | None = None) -> int:
    manifest = _manifest()
    draft = Path(DEMO_WALKTHROUGH_DRAFT_PATH).read_text(encoding="utf-8")
    generated = generate_walkthrough(manifest, draft)
    n_words = _word_count(generated)
    if n_words > M7B_MAX_WORDS:
        print(f"warning: walkthrough is {n_words} words, over the {M7B_MAX_WORDS}-word cap", file=sys.stderr)
    Path(DEMO_WALKTHROUGH_PATH).write_text(generated, encoding="utf-8")
    print(f"wrote {DEMO_WALKTHROUGH_PATH} ({n_words} words excluding appendix)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
