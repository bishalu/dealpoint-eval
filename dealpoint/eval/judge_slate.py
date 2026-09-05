"""Run-time verification of the M5 judge trio (spec §5.1/§5.2).

The engineer's judge-trio instruction (`specs/milestones/openrouter_sweep_2026-09-04.md`)
names three heterogeneous, non-candidate model families. Because that sweep file is
dated and its ids may not exist on OpenRouter at run time, `verify_slate` confirms
each id is live, records its price, and makes one tiny JSON smoke call before any
judge is trusted -- with a deterministic, recorded fallback ladder if a named judge
is unavailable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from dealpoint.config import JUDGE_SLATE_PATH
from dealpoint.eval.rubric import rubric_text
from dealpoint.eval.spend import fetch_prices

# Families a judge may NEVER come from: these are the M6 candidate-agent slate
# (specs/milestones/openrouter_sweep_2026-09-04.md) plus the workhorse's family
# (Zhipu, since z-ai/glm-5.3-flash is itself a judged variant in M5).
CANDIDATE_FAMILIES: tuple[str, ...] = (
    "Anthropic",
    "Zhipu",
    "DeepSeek",
    "Alibaba",
    "Google",
    "OpenAI",
    "Meta",
    "Xiaomi",
    "MiniMax",
    "Moonshot",
    "xAI",
)

# The engineer's named trio, in try-order, plus the spare -- none from a
# candidate family (spec "Judge trio" instruction, 2026-09-04).
JUDGE_TRIO: tuple[dict, ...] = (
    {"model": "mistralai/mistral-small-3.2-24b-instruct", "family": "Mistral"},
    {"model": "nvidia/nemotron-3-super-120b-a12b", "family": "NVIDIA"},
    {"model": "bytedance-seed/seed-2.0-mini", "family": "ByteDance"},
)
SPARE_JUDGE: dict = {"model": "amazon/nova-lite-v1", "family": "Amazon"}

JUDGE_FAMILIES: tuple[str, ...] = tuple(
    dict.fromkeys([j["family"] for j in JUDGE_TRIO] + [SPARE_JUDGE["family"]])
)

JUDGE_DIMENSIONS: tuple[str, ...] = ("reasoning", "evidence", "trajectory", "professional")

# Sent on every judge call (spec section 5.3 request shape). Several
# judge-trio candidates are reasoning models that, left to their defaults,
# spend their entire max_tokens budget on hidden reasoning tokens and never
# emit the JSON body (measured: nvidia/nemotron-3-super-120b-a12b and
# bytedance-seed/seed-2.0-mini both returned content=None, finish_reason
# "length" at max_tokens=400 with reasoning left on). Disabling reasoning is
# what makes a uniform max_tokens=400 budget work across all three named
# judges plus the spare.
JUDGE_EXTRA_BODY: dict = {"reasoning": {"enabled": False}}

OUTPUT_CONTRACT_TEXT = (
    "\n\nReturn EXACTLY one JSON object with these five keys and no others: "
    '{"reasoning": <int 1-5>, "evidence": <int 1-5>, "trajectory": <int 1-5>, '
    '"professional": <int 1-5>, "notes": "<string>"}. Do not speculate about which '
    "system, model, vendor or arm produced this trace."
)

# Coarse provider-prefix -> family map, used only by the dynamic cheapest-
# fallback branch (spec §5.2 fallback ladder step 2) to estimate a live
# model's family from its OpenRouter id when no better information exists.
PROVIDER_PREFIX_FAMILY: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "deepseek": "DeepSeek",
    "qwen": "Alibaba",
    "alibaba": "Alibaba",
    "mistralai": "Mistral",
    "nvidia": "NVIDIA",
    "bytedance-seed": "ByteDance",
    "bytedance": "ByteDance",
    "meta-llama": "Meta",
    "x-ai": "xAI",
    "z-ai": "Zhipu",
    "minimax": "MiniMax",
    "moonshot": "Moonshot",
    "moonshotai": "Moonshot",
    "xiaomi": "Xiaomi",
    "amazon": "Amazon",
}


def parse_judge_json(raw: str | None) -> tuple[dict | None, str | None]:
    """Parse + validate a judge's raw response into the five-key output contract.

    Returns `(payload, None)` on success or `(None, error_message)`. Tolerant
    extraction (fenced blocks, prose-wrapped JSON, balanced-brace scan) via
    `dealpoint.agent.schema.extract_json_object`; then strict validation: all
    four dimensions present, each a plain `int` (not `bool`, not `float`, not
    a numeric string) in `1..5`; `notes` a string, truncated to 500 chars.
    """
    from dealpoint.agent.schema import extract_json_object

    if not raw or not raw.strip():
        return None, "empty response content"
    extracted = extract_json_object(raw)
    if extracted is None:
        return None, "no JSON object found in response"
    try:
        payload = json.loads(extracted)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "parsed JSON is not an object"
    for dim in JUDGE_DIMENSIONS:
        if dim not in payload:
            return None, f"missing dimension {dim!r}"
        value = payload[dim]
        if isinstance(value, bool) or not isinstance(value, int):
            return None, f"dimension {dim!r} is not an int: {value!r}"
        if not (1 <= value <= 5):
            return None, f"dimension {dim!r} out of range 1..5: {value!r}"
    notes = payload.get("notes", "")
    if not isinstance(notes, str):
        return None, "notes is not a string"
    return (
        {
            "reasoning": payload["reasoning"],
            "evidence": payload["evidence"],
            "trajectory": payload["trajectory"],
            "professional": payload["professional"],
            "notes": notes[:500],
        },
        None,
    )


def _smoke_packet_text() -> str:
    return (
        "## Question\n\nWhat form of consideration is used?\n\n"
        "## Options\n\n- All Cash\n- All Stock\n\n## Status\n\nANSWERED\n\n"
        "## Trajectory\n\n(no tool calls)\n\n## Finding\n\nanswer: All Cash\n\n"
        "## Gold span (evidence-sufficiency dimension ONLY)\n\nNo expert-annotated span is recorded for this case."
    )


def _family_for_model_id(model_id: str) -> str:
    prefix = model_id.split("/", 1)[0]
    return PROVIDER_PREFIX_FAMILY.get(prefix, prefix.title())


def _try_judge(client, candidate: dict, prices: dict) -> dict:
    model = candidate["model"]
    family = candidate["family"]
    price = prices.get(model)
    if price is None:
        return {"ok": False, "model": model, "family": family, "error": f"{model!r} not in live model list"}

    messages = [
        {"role": "system", "content": rubric_text() + OUTPUT_CONTRACT_TEXT},
        {"role": "user", "content": _smoke_packet_text()},
    ]
    try:
        result = client.chat(
            messages=messages,
            model=model,
            response_format={"type": "json_object"},
            max_tokens=120,
            temperature=0,
            extra_body=JUDGE_EXTRA_BODY,
        )
    except Exception as exc:  # noqa: BLE001 - any failure is a recorded verification failure
        return {"ok": False, "model": model, "family": family, "error": f"{type(exc).__name__}: {exc}"}

    payload, error = parse_judge_json(result.content)
    if payload is None:
        return {"ok": False, "model": model, "family": family, "error": error}

    return {
        "ok": True,
        "model": model,
        "family": family,
        "prompt_usd_per_token": price["prompt"],
        "completion_usd_per_token": price["completion"],
        "smoke_input_tokens": result.input_tokens,
        "smoke_output_tokens": result.output_tokens,
    }


def _cheapest_fallback(prices: dict, exclude_families: set[str], exclude_models: set[str]) -> dict | None:
    """Cheapest live model (by the assumed 8000-in/300-out judge shape) whose
    family is in neither `exclude_families` nor already tried in `exclude_models`.
    """
    candidates: list[tuple[float, str, str]] = []
    for model_id, price in prices.items():
        if model_id in exclude_models:
            continue
        family = _family_for_model_id(model_id)
        if family in exclude_families:
            continue
        est_cost = price["prompt"] * 8000 + price["completion"] * 300
        candidates.append((est_cost, model_id, family))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    _, model_id, family = candidates[0]
    return {"model": model_id, "family": family}


def verify_slate(client, *, prices: dict | None = None) -> dict:
    """Verify the judge trio at run time: live id check, price, one smoke call each.

    Applies the fallback ladder (spec §5.2) deterministically: named judge
    fails -> try the spare; spare also fails -> the cheapest available model
    from a family in neither `CANDIDATE_FAMILIES` nor an already-selected
    judge family; fewer than 2 distinct judge families surviving -> raise.

    `prices`, when given, is used directly (as `{model_id: {"prompt":...,
    "completion":...}}`) instead of calling `fetch_prices()` -- this is what
    lets the `gate_m5` test drive this function with a `FakeClient` and a
    synthetic price table, fully offline. Basis is then recorded as
    `"provided"` rather than `"live"`/`"pinned:..."`.
    """
    if prices is None:
        live_prices, price_basis = fetch_prices()
    else:
        live_prices, price_basis = prices, "provided"

    selected: list[dict] = []
    substitutions: list[dict] = []
    selected_families: set[str] = set()
    tried_models: set[str] = set()

    for candidate in JUDGE_TRIO:
        tried_models.add(candidate["model"])
        result = _try_judge(client, candidate, live_prices)
        if result["ok"] and result["family"] not in selected_families:
            selected.append(result)
            selected_families.add(result["family"])
            continue

        substitutions.append(
            {"failed": candidate["model"], "family": candidate["family"], "reason": result.get("error")}
        )

        # Fallback step 1: the spare.
        tried_models.add(SPARE_JUDGE["model"])
        spare_result = _try_judge(client, SPARE_JUDGE, live_prices)
        if spare_result["ok"] and spare_result["family"] not in selected_families:
            selected.append(spare_result)
            selected_families.add(spare_result["family"])
            substitutions.append(
                {"substituted_for": candidate["model"], "with": SPARE_JUDGE["model"], "step": "spare"}
            )
            continue
        if not spare_result["ok"]:
            substitutions.append(
                {"failed": SPARE_JUDGE["model"], "family": SPARE_JUDGE["family"], "reason": spare_result.get("error")}
            )

        # Fallback step 2: cheapest live model from a fresh family.
        exclude_families = set(CANDIDATE_FAMILIES) | selected_families
        fallback = _cheapest_fallback(live_prices, exclude_families, tried_models)
        if fallback is not None:
            tried_models.add(fallback["model"])
            fallback_result = _try_judge(client, fallback, live_prices)
            if fallback_result["ok"] and fallback_result["family"] not in selected_families:
                selected.append(fallback_result)
                selected_families.add(fallback_result["family"])
                substitutions.append(
                    {
                        "substituted_for": candidate["model"],
                        "with": fallback["model"],
                        "step": "dynamic_cheapest",
                    }
                )
            else:
                substitutions.append(
                    {
                        "failed": fallback["model"],
                        "family": fallback["family"],
                        "reason": fallback_result.get("error", "duplicate family"),
                    }
                )

    if len(selected_families) < 2:
        raise RuntimeError(
            f"fewer than 2 distinct judge families survived verification "
            f"(got {sorted(selected_families)}); refusing to proceed"
        )

    payload = {
        "verified_at": datetime.now(UTC).isoformat(),
        "price_basis": price_basis,
        "judges": selected,
        "substitutions": substitutions,
    }
    return payload


def write_judge_slate(payload: dict, path=JUDGE_SLATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
