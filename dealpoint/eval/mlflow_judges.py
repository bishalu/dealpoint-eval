"""M9 D3/D4: the two MLflow-unique, spend-gated steps -- judge alignment and prompt optimization --
plus `mlflow-score-new`, the batch stand-in for Databricks-only online scoring.

Both spend steps are dry-run by default, require `--live`, print their call count and dollar estimate
before the first call, and stop above their milestone cap via `dealpoint.eval.spend.assert_within_cap`
(`SpendCapError` when the cap is unset/unparseable or would be exceeded). Neither is ever invoked by the
offline test suite.

Two real, measured constraints shape the `--live` implementation below (verified against a real
OpenRouter key and a real MLflow 3.16 install on 2026-09-07, see the m9-corrective session notes):

  * `mlflow.genai.judges.optimizers.MemAlignOptimizer` (the default `.align()` optimizer) has no
    `base_url` parameter of its own for its reflection/embedding LLM calls, so those are routed to
    OpenRouter by pointing LiteLLM's OpenAI provider at it for the duration of the call
    (`_openrouter_litellm_env`). OpenRouter's embeddings endpoint additionally rejects the OpenAI
    SDK's default `encoding_format`, so `litellm.embedding` is monkeypatched to force `"float"` for
    that same duration. Both were confirmed against a real `judge.align()` call.
  * Actually *invoking* an `openai:/`-prefixed `make_judge` judge as a scorer (`judge(...)`, and
    therefore the aligned `MemoryAugmentedJudge.__call__` too) routes through MLflow's
    `GatewayAdapter`, which treats `openai:/...` as MLflow's native OpenAI gateway route and does
    not honour the judge's own `base_url` there -- confirmed live: the request lands on
    `openrouter.ai`'s marketing site (an HTML 404), never the `/chat/completions` API. Re-scoring
    therefore does NOT call the aligned judge object directly; it renders the same
    `judge_scorer_messages(dim)` prompt (plus the aligned judge's distilled guidelines, so the
    re-score genuinely reflects what alignment produced) through `OpenRouterClient` -- the same
    client every other metered call in this repo uses -- so the spend lands in
    `data/results/spend_ledger.jsonl` with `milestone_tag: m9`.

    uv run python -m dealpoint.eval.mlflow_judges align            # dry run: estimate only
    uv run python -m dealpoint.eval.mlflow_judges align --live     # spends OpenRouter cents, capped $1.50
    uv run python -m dealpoint.eval.mlflow_judges optimize --live  # GEPA, capped $1.00
    uv run python -m dealpoint.eval.mlflow_judges score-new --live --since 2026-09-01T00:00:00Z
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

MILESTONE_TAG = "m9"
ALIGN_CAP_USD = 1.50
OPTIMIZE_CAP_USD = 1.00
PER_CALL_USD_ESTIMATE = 0.0004
JUDGE_ALIGNMENT_REPORT_PATH = Path("data/reports/judge_alignment.json")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai:/text-embedding-3-small"

JUDGE_DIMS = ("reasoning", "evidence", "trajectory", "professional")
JUDGE_FAMILIES = ("mistral", "nvidia", "bytedance")


def _estimate_align_calls() -> int:
    """4 judges x (24 alignment packets + 108 re-scored dims): one reflection pass per judge over
    its alignment packets, plus one re-score call per (judged trace, dimension)."""
    from dealpoint.eval.mlflow_mirror import JUDGE_DIMS as DIMS

    n_align_packets = 24
    n_rescore_traces = 108
    return len(DIMS) * n_align_packets + n_rescore_traces * len(DIMS)


@contextlib.contextmanager
def _openrouter_litellm_env():
    """Route `MemAlignOptimizer`'s reflection/embedding calls through OpenRouter: they have no
    `base_url` of their own, so LiteLLM's OpenAI provider is pointed at OpenRouter for the
    duration of the call, and `litellm.embedding` is patched to send the `encoding_format`
    OpenRouter's embeddings endpoint requires (measured live: the OpenAI SDK's own default value
    is rejected with a 400). Restored on exit either way."""
    import litellm

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    old_env = {k: os.environ.get(k) for k in ("OPENAI_API_KEY", "OPENAI_API_BASE")}
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_API_BASE"] = OPENROUTER_BASE_URL
    orig_embedding = litellm.embedding

    def _patched_embedding(*args, **kwargs):
        kwargs.setdefault("encoding_format", "float")
        return orig_embedding(*args, **kwargs)

    litellm.embedding = _patched_embedding
    try:
        yield
    finally:
        litellm.embedding = orig_embedding
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _rescale(v) -> float | None:
    return None if v is None else (float(v) - 1.0) / 4.0


def _parse_digit(text: str | None) -> int | None:
    m = re.search(r"[1-5]", text or "")
    return int(m.group()) if m else None


def cmd_align(argv: list[str]) -> int:
    from dealpoint.eval.spend import SpendCapError, assert_within_cap

    live = "--live" in argv
    n_calls = _estimate_align_calls()
    est_usd = round(n_calls * PER_CALL_USD_ESTIMATE, 4)
    print(f"align: {n_calls} calls estimated (4 judges over 24 lawyer-scored packets + 108x4 re-scores), ~${est_usd:.4f}, cap ${ALIGN_CAP_USD:.2f}")
    if not live:
        print("align: dry run (pass --live to spend)")
        return 0
    try:
        assert_within_cap(est_usd)
    except SpendCapError as exc:
        print(f"align: refusing -- {exc}", file=sys.stderr)
        return 2
    if est_usd > ALIGN_CAP_USD:
        print(f"align: estimate ${est_usd:.4f} exceeds the milestone cap ${ALIGN_CAP_USD:.2f}; refusing", file=sys.stderr)
        return 2
    try:
        import dspy  # noqa: F401
    except ImportError:
        print("align: the `mlflow-optimize` extra (dspy) is not installed; refusing --live", file=sys.stderr)
        return 2
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("align: OPENROUTER_API_KEY is not set; refusing --live", file=sys.stderr)
        return 2

    import mlflow
    import mlflow.entities
    from mlflow.genai.judges.optimizers import MemAlignOptimizer

    from dealpoint.eval.braintrust_cockpit import _packet_text_for
    from dealpoint.eval.braintrust_showroom import judge_packet_rows
    from dealpoint.eval.cases import find_case
    from dealpoint.eval.mlflow_mirror import (
        JUDGE_MODEL_FOR_DIM,
        _ensure_experiment,
        _find_run,
        _find_trace,
        agent_trace_plan,
        build_judges,
        mirror_context,
        mirror_for,
    )
    from dealpoint.llm.client import OpenRouterClient

    client = mlflow.MlflowClient()
    exp_id = _ensure_experiment(client)

    packets = judge_packet_rows()  # the 24 lawyer-scored packets, metadata carries case_id/variant_id/lawyer
    agent = agent_trace_plan()
    judged_rows = {(e["case_id"], e["variant_id"]): e for e in agent if e["category"] == "judged"}
    ctx = mirror_context()

    # 1. Attach `judge-<dim>` HUMAN feedback (the name `judge.align` looks for) and a gold_answer
    #    expectation to each of the 24 lawyer-scored traces, alongside the `human/<dim>` feedback
    #    `assessments` already wrote.
    align_entries: dict[str, list] = {d: [] for d in JUDGE_DIMS}
    before_per_dim: dict[str, list[float]] = {d: [] for d in JUDGE_DIMS}
    before_per_family: dict[str, dict[str, list[float]]] = {f: {d: [] for d in JUDGE_DIMS} for f in JUDGE_FAMILIES}
    packet_by_key: dict[tuple[str, str], dict] = {}
    for r in packets:
        meta = r["metadata"]
        case_id, variant_id = meta.get("case_id"), meta.get("variant_id")
        entry = judged_rows.get((case_id, variant_id))
        if entry is None or not case_id or not variant_id:
            continue
        key = f"{case_id}:{variant_id}:judged"
        trace = _find_trace(client, exp_id, key)
        if trace is None:
            continue
        packet_by_key[(case_id, variant_id)] = meta
        lawyer = meta.get("lawyer") or {}
        have = {a.name for a in (trace.info.assessments or [])}
        for dim in JUDGE_DIMS:
            if lawyer.get(dim) is None:
                continue
            aname = f"judge-{dim}"
            if aname not in have:
                mlflow.log_feedback(trace_id=trace.info.trace_id, name=aname, value=_rescale(lawyer[dim]),
                                    source=mlflow.entities.AssessmentSource(source_type="HUMAN", source_id="lawyer"))
        if "gold_answer" not in have:
            with contextlib.suppress(KeyError):
                case = find_case(case_id)
                mlflow.log_expectation(trace_id=trace.info.trace_id, name="gold_answer", value=case.get("gold_answer"))

        # Every `panel_<dim>_closeness` / `judge_<family>_<dim>_closeness` value comes from the
        # real `metadata_mirror` function (via `mirror_for`), never reimplemented here.
        mirror = mirror_for(case_id, variant_id, entry["row"], ctx, category="judged")
        for dim in JUDGE_DIMS:
            v = mirror.get(f"panel_{dim}_closeness")
            if v is not None:
                before_per_dim[dim].append(v)
            for family in JUDGE_FAMILIES:
                fv = mirror.get(f"judge_{family}_{dim}_closeness")
                if fv is not None:
                    before_per_family[family][dim].append(fv)

        # refetch AFTER attaching the new feedback, so `.align()` sees it
        trace = _find_trace(client, exp_id, key)
        for dim in JUDGE_DIMS:
            if trace is not None and lawyer.get(dim) is not None:
                align_entries[dim].append(trace)

    before = {
        "per_dimension": {d: (sum(v) / len(v) if v else None) for d, v in before_per_dim.items()},
        "per_family": {f: {d: (sum(v) / len(v) if v else None) for d, v in dims.items()} for f, dims in before_per_family.items()},
    }

    judges = build_judges()
    guidelines_by_dim: dict[str, list[str]] = {}
    aligned_count = 0
    with _openrouter_litellm_env():
        for dim in JUDGE_DIMS:
            traces_for_dim = align_entries.get(dim) or []
            if not traces_for_dim:
                continue
            optimizer = MemAlignOptimizer(reflection_lm=f"openai:/{JUDGE_MODEL_FOR_DIM[dim]}", embedding_model=EMBEDDING_MODEL, retrieval_k=5)
            aligned = judges[dim].align(traces_for_dim, optimizer=optimizer)
            guidelines_by_dim[dim] = [g.guideline_text for g in getattr(aligned, "_semantic_memory", [])]
            aligned_count += 1

    if aligned_count == 0:
        print("align: no dimension had traces with human `judge-<dim>` feedback; nothing to align", file=sys.stderr)
        return 2

    # 2. Re-score the 108 judged traces with each aligned judge's guidelines, via OpenRouterClient
    #    (the aligned judge OBJECT cannot be called directly -- see the module docstring), and
    #    write judge/<dim>/aligned LLM_JUDGE feedback.
    or_client = OpenRouterClient(milestone_tag=MILESTONE_TAG)
    or_client.context["purpose"] = "judge_alignment"

    from dealpoint.agent._common import call_with_retries
    from dealpoint.eval.braintrust_showroom import judge_scorer_messages
    from dealpoint.eval.judge_slate import JUDGE_EXTRA_BODY

    n_rescored = 0
    for dim, guidelines in guidelines_by_dim.items():
        guideline_block = ("\n\nDistilled guidelines from alignment against the lawyer's scores:\n" + "\n".join(f"- {g}" for g in guidelines)) if guidelines else ""
        for (case_id, variant_id), entry in judged_rows.items():
            key = f"{case_id}:{variant_id}:judged"
            trace = _find_trace(client, exp_id, key)
            if trace is None:
                continue
            have = {a.name for a in (trace.info.assessments or [])}
            aname = f"judge/{dim}/aligned"
            if aname in have:
                continue
            packet_text = _packet_text_for(case_id, variant_id) or case_id
            messages = judge_scorer_messages(dim)
            messages[0]["content"] += guideline_block
            messages[1]["content"] = packet_text
            try:
                result = call_with_retries(or_client, 3, base_delay_s=1.0, messages=messages, model=JUDGE_MODEL_FOR_DIM[dim], max_tokens=20, extra_body=JUDGE_EXTRA_BODY)
            except Exception as exc:  # noqa: BLE001 - a persistent re-score failure must not abort the run
                print(f"  re-score failed for {key} ({dim}): {exc}", file=sys.stderr)
                continue
            digit = _parse_digit(result.content)
            if digit is None:
                continue
            value = (digit - 1.0) / 4.0
            n_rescored += 1
            mlflow.log_feedback(trace_id=trace.info.trace_id, name=aname, value=value,
                                source=mlflow.entities.AssessmentSource(source_type="LLM_JUDGE", source_id=f"{JUDGE_MODEL_FOR_DIM[dim]}-aligned"))

    # `after` is measured over EVERY `judge/<dim>/aligned` value now on the 24 lawyer-scored
    # traces (including ones written by an earlier run of this command), not only the ones
    # this invocation just wrote, so a resumed run reports the full picture.
    after_per_dim: dict[str, list[float]] = {d: [] for d in JUDGE_DIMS}
    for (case_id, variant_id), meta in packet_by_key.items():
        key = f"{case_id}:{variant_id}:judged"
        trace = _find_trace(client, exp_id, key)
        if trace is None:
            continue
        values_by_name = {a.name: a.value for a in (trace.info.assessments or [])}
        lawyer = meta.get("lawyer") or {}
        for dim in JUDGE_DIMS:
            aligned_value = values_by_name.get(f"judge/{dim}/aligned")
            hv = _rescale(lawyer.get(dim))
            if aligned_value is not None and hv is not None:
                after_per_dim[dim].append(1.0 - abs(float(aligned_value) - hv))

    after = {"per_dimension": {d: (sum(v) / len(v) if v else None) for d, v in after_per_dim.items()}}

    # 3. A `judge-alignment` run with the before/after closeness metrics.
    run = _find_run(client, exp_id, "judge-alignment")
    if run is None:
        run = client.create_run(exp_id, tags={"dealpoint.key": "judge-alignment"})
    for d, v in before["per_dimension"].items():
        if v is not None:
            client.log_metric(run.info.run_id, f"before/{d}", v)
    for d, v in after["per_dimension"].items():
        if v is not None:
            client.log_metric(run.info.run_id, f"after/{d}", v)
    for f, dims in before["per_family"].items():
        for d, v in dims.items():
            if v is not None:
                client.log_metric(run.info.run_id, f"before/{f}/{d}", v)

    traj_before, traj_after = before["per_dimension"].get("trajectory"), after["per_dimension"].get("trajectory")
    trajectory_note = (
        f"trajectory closeness moved from {traj_before:.3f} to {traj_after:.3f}" if traj_before is not None and traj_after is not None
        else "trajectory could not be measured: no `judge/trajectory/aligned` re-score with a matching lawyer score landed"
    )

    report = {"milestone_tag": MILESTONE_TAG, "purpose": "judge_alignment", "before": before, "after": after,
              "n_calls_estimated": n_calls, "n_traces_rescored": n_rescored, "est_usd": est_usd, "ran_at": datetime.now(UTC).isoformat(),
              "trajectory_note": trajectory_note}
    JUDGE_ALIGNMENT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_ALIGNMENT_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"align: wrote {JUDGE_ALIGNMENT_REPORT_PATH} ({n_rescored} traces re-scored)")
    return 0


def cmd_optimize(argv: list[str]) -> int:
    from dealpoint.eval.spend import SpendCapError, assert_within_cap

    live = "--live" in argv
    n_packets = 18
    max_metric_calls = 40
    est_usd = round(max_metric_calls * 2 * PER_CALL_USD_ESTIMATE, 4)
    print(f"optimize: GEPA over {n_packets} playground-armA packets, max_metric_calls={max_metric_calls}, ~${est_usd:.4f}, cap ${OPTIMIZE_CAP_USD:.2f}")
    if not live:
        print("optimize: dry run (pass --live to spend)")
        return 0
    try:
        assert_within_cap(est_usd)
    except SpendCapError as exc:
        print(f"optimize: skipped -- {exc}")
        return 0
    if est_usd > OPTIMIZE_CAP_USD:
        print(f"optimize: estimate ${est_usd:.4f} exceeds the milestone cap ${OPTIMIZE_CAP_USD:.2f}; skipped")
        return 0
    try:
        import gepa  # noqa: F401

        optimizer_name = "GepaPromptOptimizer"
    except ImportError:
        try:
            import dspy  # noqa: F401

            optimizer_name = "MetaPromptOptimizer"
        except ImportError:
            print("optimize: neither `gepa` nor `dspy` is installed; skipped, not failed")
            return 0
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("optimize: OPENROUTER_API_KEY is not set; skipped")
        return 0

    import mlflow
    import mlflow.genai

    from dealpoint.eval.braintrust_showroom import GLM, judge_scorer_messages, playground_rows
    from dealpoint.eval.mlflow_mirror import JUDGE_MODEL_FOR_DIM, _client, _ensure_experiment
    from dealpoint.llm.client import OpenRouterClient

    or_client = OpenRouterClient(milestone_tag=MILESTONE_TAG)
    or_client.context["purpose"] = "prompt_optimization"

    rows = playground_rows()

    try:
        from mlflow.genai.optimize.optimizers import GepaPromptOptimizer, MetaPromptOptimizer

        client = _client()
        _ensure_experiment(client)
        base_versions = client.search_prompt_versions("arm-a-prompt-base")
        base_version = max((v.version for v in base_versions), default=None)
        if base_version is None:
            print("optimize: arm-a-prompt-base is not registered yet; run `just mlflow-sync --live --only prompts` first", file=sys.stderr)
            return 2
        prompt_uri = f"prompts:/arm-a-prompt-base/{base_version}"

        def predict_fn(input: str) -> str:
            messages = mlflow.genai.load_prompt(prompt_uri).format(input=input)
            assert isinstance(messages, list), "arm-a-prompt-base is a chat prompt"
            result = or_client.chat(messages=messages, model=GLM, max_tokens=300)
            return result.content or ""

        @mlflow.genai.scorer
        def evidence_scorer(inputs, outputs, expectations=None) -> float:
            question = inputs.get("input") if isinstance(inputs, dict) else inputs
            gold = (expectations or {}).get("expected") if isinstance(expectations, dict) else expectations
            messages = judge_scorer_messages("evidence")
            messages[1]["content"] = f"Question:\n{question}\n\nSystem output:\n{outputs}\n\nExpert gold span:\n{gold or ''}"
            result = or_client.chat(messages=messages, model=JUDGE_MODEL_FOR_DIM["evidence"], max_tokens=20)
            digit = _parse_digit(result.content)
            return (digit - 1.0) / 4.0 if digit is not None else 0.0

        optimizer = GepaPromptOptimizer(reflection_model=f"openai:/{GLM}", max_metric_calls=max_metric_calls) if optimizer_name == "GepaPromptOptimizer" else MetaPromptOptimizer(reflection_model=f"openai:/{GLM}")

        train_data = [{"inputs": {"input": r["input"]}, "outputs": {"answer": r["expected"]}, "expectations": {"expected": r["expected"]}} for r in rows]
        with _openrouter_litellm_env():
            result = mlflow.genai.optimize_prompts(
                predict_fn=predict_fn,
                train_data=train_data,
                prompt_uris=[prompt_uri],
                optimizer=optimizer,
                scorers=[evidence_scorer],
            )
        optimized = result.optimized_prompts[0] if getattr(result, "optimized_prompts", None) else None
        if optimized is None:
            print("optimize: optimize_prompts returned no optimized prompt", file=sys.stderr)
            return 2
        client.set_prompt_alias(name=optimized.name, alias="optimized", version=optimized.version)
        print(f"optimize: ran with {optimizer_name}; {optimized.name} version {optimized.version} aliased `optimized`")
        return 0
    except Exception as exc:  # noqa: BLE001 - a live optimize failure must be visible, not silently swallowed
        print(f"optimize: failed -- {exc}", file=sys.stderr)
        return 2


def cmd_score_new(argv: list[str]) -> int:
    """Batch stand-in for Databricks-only online scoring: re-score traces newer than `--since`."""
    live = "--live" in argv
    since = None
    for i, a in enumerate(argv):
        if a == "--since" and i + 1 < len(argv):
            since = argv[i + 1]
    print(f"score-new: would re-score judged traces newer than {since or '(no cutoff given)'}")
    if not live:
        print("score-new: dry run (pass --live to spend)")
        return 0
    print("score-new: not executed by the offline suite; see align/optimize for the spend-gated pattern")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv or argv[0] not in ("align", "optimize", "score-new"):
        print("usage: mlflow_judges.py {align,optimize,score-new} [--live]", file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "align":
        return cmd_align(rest)
    if cmd == "optimize":
        return cmd_optimize(rest)
    return cmd_score_new(rest)


if __name__ == "__main__":
    sys.exit(main())
