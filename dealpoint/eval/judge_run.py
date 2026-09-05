"""The metered M5 judging run: one call per trace, four dimensions (spec §5.3).

`judge_one` is the pure-ish per-call unit (one client.chat call, one retry on
validation failure, never a fabricated score) that the `gate_m5` parsing test
drives with `FakeClient`. `main` is the disk-driven pipeline: verify the
rubric is frozen, verify the judge slate, build every packet for the frozen
judged subset, judge each `(packet_id, judge_model)` pair not already
recorded, resumably.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from dealpoint.config import (
    JUDGE_MAX_TOKENS,
    JUDGE_SCORES_PATH,
    JUDGES_MILESTONE_TAG,
)
from dealpoint.eval.judge_slate import JUDGE_EXTRA_BODY, OUTPUT_CONTRACT_TEXT, parse_judge_json
from dealpoint.eval.rubric import rubric_text, rubric_version

RETRY_INSTRUCTION = "\n\nYour previous response was invalid. Return ONLY the JSON object, nothing else."


class RubricNotFrozenError(RuntimeError):
    """Raised when `versions.json`'s `rubric_version` does not match the frozen rubric file."""


def assert_rubric_frozen() -> None:
    """Refuse to run when `versions.json`'s `rubric_version` is absent or stale.

    The freezing order (spec §3.3) is non-negotiable: `stamp_rubric_version()`
    must have already run, and the stamped value must equal a fresh hash of
    the committed rubric file, before the first judge call.
    """
    from dealpoint.config import VERSIONS_JSON_PATH

    if not VERSIONS_JSON_PATH.exists():
        raise RubricNotFrozenError(
            "data/reports/versions.json does not exist; run stamp_rubric_version() first"
        )
    payload = json.loads(VERSIONS_JSON_PATH.read_text(encoding="utf-8"))
    stamped = payload.get("rubric_version")
    current = rubric_version()
    if stamped != current:
        raise RubricNotFrozenError(
            f"versions.json rubric_version {stamped!r} != current rubric hash {current!r}; "
            "run dealpoint.eval.rubric.stamp_rubric_version() before judging"
        )


def _system_prompt() -> str:
    return rubric_text() + OUTPUT_CONTRACT_TEXT


def judge_one(
    client,
    packet_text: str,
    *,
    judge_model: str,
    judge_family: str,
    packet_id: str,
    variant_id: str,
    case_id: str,
    question_id: str,
) -> dict:
    """Judge one blinded packet with one judge model: one call, one retry on failure.

    Never fabricates a score: on a second validation failure the row carries
    `ok: false`, `failure_detail`, and null dimension scores.
    """
    system = _system_prompt()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": packet_text},
    ]

    result = client.chat(
        messages=messages,
        model=judge_model,
        response_format={"type": "json_object"},
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=0,
        extra_body=JUDGE_EXTRA_BODY,
    )
    payload, error = parse_judge_json(result.content)

    if payload is None:
        retry_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": packet_text + RETRY_INSTRUCTION},
        ]
        result2 = client.chat(
            messages=retry_messages,
            model=judge_model,
            response_format={"type": "json_object"},
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=0,
            extra_body=JUDGE_EXTRA_BODY,
        )
        payload2, error2 = parse_judge_json(result2.content)
        if payload2 is None:
            return {
                "packet_id": packet_id,
                "variant_id": variant_id,
                "case_id": case_id,
                "question_id": question_id,
                "judge_model": judge_model,
                "judge_family": judge_family,
                "rubric_version": rubric_version(),
                "ok": False,
                "reasoning": None,
                "evidence": None,
                "trajectory": None,
                "professional": None,
                "notes": None,
                "input_tokens": result.input_tokens + result2.input_tokens,
                "output_tokens": result.output_tokens + result2.output_tokens,
                "usd": result.cost_usd + result2.cost_usd,
                "failure_detail": (f"{error!r} then retry: {error2!r}")[:300],
                "ts": datetime.now(UTC).isoformat(),
            }
        payload = payload2
        input_tokens = result.input_tokens + result2.input_tokens
        output_tokens = result.output_tokens + result2.output_tokens
        usd = result.cost_usd + result2.cost_usd
    else:
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        usd = result.cost_usd

    return {
        "packet_id": packet_id,
        "variant_id": variant_id,
        "case_id": case_id,
        "question_id": question_id,
        "judge_model": judge_model,
        "judge_family": judge_family,
        "rubric_version": rubric_version(),
        "ok": True,
        "reasoning": payload["reasoning"],
        "evidence": payload["evidence"],
        "trajectory": payload["trajectory"],
        "professional": payload["professional"],
        "notes": payload["notes"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": usd,
        "failure_detail": None,
        "ts": datetime.now(UTC).isoformat(),
    }


def _load_jsonl(path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def measured_judge_call_usd() -> tuple[float, str]:
    """Mean `usd` over every ledger row carrying a `judge` field (M6 spec
    deliverable 3): the actual measured cost of one (packet, judge) call,
    falling back to the `SWEEP_DEFS["judges"]` `tokens_per_call` x live/pinned
    price shape when no such rows exist yet (e.g. before M5 ever ran).
    """
    from dealpoint.config import SPEND_LEDGER_PATH
    from dealpoint.eval.spend import SWEEP_DEFS, fetch_prices, read_ledger

    rows = [r for r in read_ledger(SPEND_LEDGER_PATH) if r.get("judge")]
    if rows:
        mean = sum(float(r.get("usd", 0.0) or 0.0) for r in rows) / len(rows)
        return mean, f"measured over {len(rows)} judge ledger rows"

    shape = SWEEP_DEFS["judges"]["tokens_per_call"]
    models = SWEEP_DEFS["judges"]["models"]
    prices, price_basis = fetch_prices()
    per_call = []
    for model in models:
        price = prices.get(model)
        if price is None:
            continue
        per_call.append(shape["input"] * price["prompt"] + shape["output"] * price["completion"])
    if not per_call:
        return 0.0, f"no pricing available ({price_basis})"
    return sum(per_call) / len(per_call), f"tokens_per_call x {price_basis} (no judge ledger rows yet)"


def _already_scored(path) -> set[tuple[str, str]]:
    done = set()
    for row in _load_jsonl(path):
        if row.get("ok"):
            done.add((row["packet_id"], row["judge_model"]))
    return done


def _append_row(path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
        fh.write("\n")
        fh.flush()


def build_all_packets() -> list[dict]:
    """Build (packet, packet_text, meta) for every judged trace in the frozen subset."""
    from dealpoint.corpus.document import Document, load_document
    from dealpoint.eval.blinding import build_packet, render_packet_text
    from dealpoint.eval.cases import find_case, resolve_document_id
    from dealpoint.eval.subset import load_subset_payload

    judged_subset = load_subset_payload("judged_subset")
    case_ids = judged_subset["case_ids"]
    variants = judged_subset["variants"]

    out: list[dict] = []
    doc_cache: dict[str, Document] = {}
    rows_cache: dict[str, dict[str, dict]] = {}

    for variant in variants:
        results_path = variant["results_path"]
        if results_path not in rows_cache:
            rows_by_case = {row["case_id"]: row for row in _load_jsonl_path(results_path)}
            rows_cache[results_path] = rows_by_case
        rows_by_case = rows_cache[results_path]

        for case_id in case_ids:
            row = rows_by_case.get(case_id)
            if row is None:
                continue
            case = find_case(case_id)
            document_id = resolve_document_id(case)
            if document_id not in doc_cache:
                doc_cache[document_id] = load_document(document_id)
            doc = doc_cache[document_id]

            packet = build_packet(row, case, doc, variant_id=variant["variant_id"])
            packet_text = render_packet_text(packet)
            out.append(
                {
                    "packet": packet,
                    "packet_text": packet_text,
                    "variant_id": variant["variant_id"],
                    "case_id": case_id,
                    "question_id": case["question_id"],
                }
            )
    return out


def _load_jsonl_path(path: str) -> list[dict]:
    from pathlib import Path

    return _load_jsonl(Path(path))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from dealpoint.eval.judge_slate import verify_slate
    from dealpoint.eval.run import _assert_within_milestone_absolute
    from dealpoint.eval.spend import assert_within_cap, realized_usd
    from dealpoint.llm.client import OpenRouterClient

    parser = argparse.ArgumentParser(prog="python -m dealpoint.eval.judge_run")
    parser.add_argument("--milestone-tag", default=JUDGES_MILESTONE_TAG)
    args = parser.parse_args(argv)

    assert_rubric_frozen()

    client = OpenRouterClient(milestone_tag=args.milestone_tag)

    # M6 operator instruction, 2026-09-05: the judge-slate smoke calls are not
    # case runs and must carry purpose=probe -- cleared to {} before the real
    # judging calls, which set their own per-call context below.
    client.context = {"purpose": "probe", "probe_model": "judge-slate"}
    slate = verify_slate(client)
    client.context = {}
    from dealpoint.eval.judge_slate import write_judge_slate

    write_judge_slate(slate)

    packets = build_all_packets()
    already = _already_scored(JUDGE_SCORES_PATH)

    # M6 spec deliverable 3: sized to the work actually pending, not the
    # static 54-trace shape (now wrong in both directions once M6 adds
    # variants) -- n_pending pairs x the M5-measured mean $/judge-call.
    pending = [
        (item, judge)
        for item in packets
        for judge in slate["judges"]
        if (item["packet"]["packet_id"], judge["model"]) not in already
    ]
    judge_call_usd, _basis = measured_judge_call_usd()
    est_usd = len(pending) * judge_call_usd
    assert_within_cap(est_usd)
    _assert_within_milestone_absolute(est_usd)

    before = realized_usd()
    n_calls = 0
    for item, judge in pending:
        client.context = {
            "case_id": item["packet"]["packet_id"],
            "variant": item["variant_id"],
            "judge": judge["model"],
        }
        row = judge_one(
            client,
            item["packet_text"],
            judge_model=judge["model"],
            judge_family=judge["family"],
            packet_id=item["packet"]["packet_id"],
            variant_id=item["variant_id"],
            case_id=item["case_id"],
            question_id=item["question_id"],
        )
        _append_row(JUDGE_SCORES_PATH, row)
        n_calls += 1
    after = realized_usd()

    print(json.dumps({"n_calls": n_calls, "realized_usd": round(after - before, 6)}))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
