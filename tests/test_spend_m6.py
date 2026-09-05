"""M6 spend extensions: the resized `pareto` sweep, the envelope assertion,
and the metered-call discipline (milestone_tag/purpose) read from the
ledger (spec DoD + operator instruction, 2026-09-05).
"""

from __future__ import annotations

import pytest

from dealpoint.eval.judge_slate import JUDGE_TRIO, SPARE_JUDGE
from dealpoint.eval.pareto_slate import PARETO_CANDIDATES
from dealpoint.eval.spend import (
    JUDGE_TRIO_MODELS,
    PARETO_NEW_MODELS,
    SWEEP_DEFS,
    estimate,
    read_ledger,
    realized_usd,
)

pytestmark = pytest.mark.gate_m6

SPEND_LEDGER_PATH = None  # resolved lazily below to avoid an import-time dotenv dependency


def _ledger_path():
    from dealpoint.config import SPEND_LEDGER_PATH as _P

    return _P


def test_pareto_sweep_has_three_legs_matching_pareto_new_models():
    sweep = SWEEP_DEFS["pareto"]
    assert "legs" in sweep
    assert len(sweep["legs"]) == 3
    probe_leg, sweep_leg, judge_leg = sweep["legs"]
    assert probe_leg["models"] == list(PARETO_NEW_MODELS)
    assert sweep_leg["models"] == list(PARETO_NEW_MODELS)
    assert sweep_leg["n_cases"] == 32
    assert judge_leg["models"] == list(JUDGE_TRIO_MODELS)
    assert judge_leg["n_cases"] == 18 * len(PARETO_NEW_MODELS)


def test_pareto_new_models_is_the_single_source_of_truth():
    """No drift: the sweep's models list and dealpoint.eval.pareto_slate's
    candidate list agree on the same ranked new-model set."""
    candidate_models = {c["model"] for c in PARETO_CANDIDATES}
    assert set(PARETO_NEW_MODELS) <= candidate_models


def test_estimate_pareto_is_nonzero_and_within_cap():
    from dealpoint.eval.spend import cap_usd

    data = estimate("pareto")
    assert data["est_usd"] > 0
    cap = cap_usd() or 6.00
    assert realized_usd() + data["est_usd"] <= cap


def test_realized_usd_within_envelope():
    """DoD (spec, verbatim): total realised OpenRouter spend across M1-M6 is
    <= $4.00, asserted from the ledger."""
    assert realized_usd() <= 4.00


def test_every_m6_ledger_row_names_an_in_slate_model():
    allowed = (
        {c["model"] for c in PARETO_CANDIDATES}
        | {"anthropic/claude-haiku-4.5", "z-ai/glm-5.3-flash"}
        | {j["model"] for j in JUDGE_TRIO}
        | {SPARE_JUDGE["model"]}
    )
    rows = [r for r in read_ledger(_ledger_path()) if r.get("milestone_tag") == "m6"]
    if not rows:
        pytest.skip("no m6 ledger rows yet")
    for row in rows:
        assert row.get("model") in allowed, f"off-slate model in m6 ledger row: {row.get('model')!r}"


def test_every_m6_probe_row_carries_purpose_and_probe_model():
    rows = [r for r in read_ledger(_ledger_path()) if r.get("milestone_tag") == "m6"]
    if not rows:
        pytest.skip("no m6 ledger rows yet")
    probe_like = [
        r
        for r in rows
        if not r.get("case_id") or str(r.get("case_set", "")).startswith("pareto_probe")
    ]
    for row in probe_like:
        assert row.get("purpose") == "probe"
        assert row.get("probe_model")


def test_m6_spend_buckets_sum_to_tag_total():
    rows = [r for r in read_ledger(_ledger_path()) if r.get("milestone_tag") == "m6"]
    if not rows:
        pytest.skip("no m6 ledger rows yet")
    total = round(sum(float(r.get("usd", 0.0) or 0.0) for r in rows), 6)
    probe = sum(float(r.get("usd", 0.0) or 0.0) for r in rows if r.get("purpose") == "probe")
    judge = sum(float(r.get("usd", 0.0) or 0.0) for r in rows if r.get("judge"))
    sweep = sum(
        float(r.get("usd", 0.0) or 0.0)
        for r in rows
        if r.get("purpose") != "probe" and not r.get("judge")
    )
    assert round(probe + judge + sweep, 6) == total


# --- corrective-cycle: a model that spent money must never be reported "not run" ---


def _pareto_json():
    from dealpoint.config import PARETO_JSON_PATH

    if not PARETO_JSON_PATH.exists():
        return None
    import json

    return json.loads(PARETO_JSON_PATH.read_text(encoding="utf-8"))


def test_no_partial_run_model_is_ever_described_as_not_run():
    report = _pareto_json()
    if report is None:
        pytest.skip("data/reports/pareto.json not generated yet; run `just pareto-report`")
    partial_run_models = {pr["model"] for pr in report.get("partial_runs", [])}
    if not partial_run_models:
        pytest.skip("no partial_runs entries in this report")

    from dealpoint.config import PARETO_MD_PATH, README_PATH

    md_text = PARETO_MD_PATH.read_text(encoding="utf-8") if PARETO_MD_PATH.exists() else ""
    readme_text = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    import re as _re

    match = _re.search(
        _re.escape("<!-- BEGIN PARETO -->") + r"(.*?)" + _re.escape("<!-- END PARETO -->"),
        readme_text,
        _re.DOTALL,
    )
    readme_block = match.group(1) if match else ""

    for nr in report.get("not_run", []):
        if nr.get("model") in partial_run_models:
            assert "not run" not in (nr.get("reason") or "").lower(), (
                f"{nr['model']} appears in partial_runs but pareto.json still calls it "
                f"'not run': {nr.get('reason')!r}"
            )

    for model in partial_run_models:
        for label, text in (("pareto.md", md_text), ("README PARETO block", readme_block)):
            # A per-model line containing both the model id and "not run" would
            # misdescribe it; the model's own entry text (its not_run bullet, if
            # any) is checked structurally above, so here just confirm no
            # "`<model>`: not run" style bullet survived the render.
            assert f"`{model}`: not run" not in text, (
                f"{label} still describes {model!r} as not run"
            )


def test_completed_plus_partial_runs_usd_equals_m6_sweep_usd():
    report = _pareto_json()
    if report is None:
        pytest.skip("data/reports/pareto.json not generated yet; run `just pareto-report`")
    completed_total = sum(
        (m.get("total_usd") or 0.0)
        for m in report.get("models", {}).values()
        if not m.get("reused")
    )
    partial_total = sum(pr["realized_usd"] for pr in report.get("partial_runs", []))
    sweep_usd = report["spend"]["m6_sweep_usd"]
    assert abs((completed_total + partial_total) - sweep_usd) <= 0.001
