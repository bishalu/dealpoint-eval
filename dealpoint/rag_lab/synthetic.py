"""Synthetic-query robustness study (secondary; spec section 1.B).

Generates <= 2 questions per gold-bearing DEV chunk only, using LlamaIndex's
question generator with a thin `CustomLLM` that delegates to the project's
own `OpenRouterClient` -- so every call lands in the spend ledger tagged
`milestone_tag="m7a", purpose="synthetic_query"`. A native LlamaIndex OpenAI
client pointed at OpenRouter would bypass the ledger; this module never does
that.

Guardrails (asserted here and in the report, spec section 1.B):
- only ever reads `load_case_set("dev")` -- never the test set;
- never changes the frozen M3 winner or `data/reports/tournament.json`;
- the frozen synthetic set never becomes benchmark truth and never drives
  retriever selection -- it answers exactly one question: does the frozen
  winner stay strong under a different query distribution?
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from dealpoint.config import RAG_SYNTH_MODEL_ENV, SYNTHETIC_DEV_QUERIES_PATH, WORKHORSE_MODEL
from dealpoint.corpus.chunks import Chunk, chunk_document
from dealpoint.corpus.document import load_document
from dealpoint.eval.cases import resolve_document_id
from dealpoint.rag_lab.evaluate import gold_bearing_chunk_ids

GENERATOR_ID = "llama_index.core.evaluation.dataset_generation.DatasetGenerator"
GENERATOR_VERSION = "num_questions_per_chunk=2"
MAX_QUESTIONS_PER_CHUNK = 2


def synth_model() -> str:
    import os

    return os.environ.get(RAG_SYNTH_MODEL_ENV) or WORKHORSE_MODEL


def prompt_text() -> str:
    """The literal question-generation prompt template used (for the recorded hash)."""
    from llama_index.core.evaluation.dataset_generation import DEFAULT_QUESTION_GENERATION_PROMPT

    return DEFAULT_QUESTION_GENERATION_PROMPT


def prompt_hash() -> str:
    return hashlib.sha256(prompt_text().encode("utf-8")).hexdigest()[:16]


def gold_bearing_dev_chunks() -> list[dict]:
    """One entry per DISTINCT gold-bearing dev chunk (deduped by chunk_id).

    Each entry carries the chunk plus one representative case_id/agreement_id
    (the first dev case whose gold span overlaps it) -- provenance only, the
    guardrail is that the *chunk selection* comes from
    `dealpoint.rag_lab.evaluate.gold_bearing_chunk_ids`, the same rule that
    defines `expected_ids` in the native evaluation.
    """
    from dealpoint.eval.cases import load_case_set

    cases = load_case_set("dev")  # NEVER the test set (spec section 1.B guardrail)
    doc_cache: dict[str, list[Chunk]] = {}
    by_chunk_id: dict[str, dict] = {}
    for case in cases:
        doc_id = resolve_document_id(case)
        if doc_id not in doc_cache:
            doc_cache[doc_id] = chunk_document(load_document(doc_id))
        chunks = doc_cache[doc_id]
        chunk_by_id = {c.chunk_id: c for c in chunks}
        for cid in gold_bearing_chunk_ids(case, chunks):
            if cid in by_chunk_id:
                continue
            chunk = chunk_by_id[cid]
            by_chunk_id[cid] = {
                "chunk_id": cid,
                "agreement_id": chunk.agreement_id,
                "text": chunk.text,
                "case_id": case["case_id"],
                "question_id": case["question_id"],
            }
    return [by_chunk_id[cid] for cid in sorted(by_chunk_id)]


def _make_llm(client):
    """A LlamaIndex `CustomLLM` delegating `complete()` to `client.chat(...)`,
    imported lazily so this module is importable with LlamaIndex absent.
    """
    from typing import Any

    from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
    from llama_index.core.llms import CustomLLM
    from pydantic import PrivateAttr

    model_id = synth_model()

    class _OpenRouterLLM(CustomLLM):
        model_id_: str = model_id
        _client: Any = PrivateAttr(default=None)

        def __init__(self, chat_client) -> None:
            super().__init__()
            self._client = chat_client

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(context_window=4096, num_output=256, model_name=self.model_id_)

        def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
            # 700 tokens, not a smaller default: measured (probe, 2026-09-05)
            # that z-ai/glm-5.3-flash frequently spends 250-500 output tokens
            # on preamble before the actual questions and gets cut off
            # (finish_reason="length", empty/partial text) at 256-512.
            result = self._client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_id_,
                max_tokens=700,
                temperature=0,
            )
            return CompletionResponse(text=result.content or "")

        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs):
            raise NotImplementedError("synthetic-query generation never streams")

    return _OpenRouterLLM(client)


def _looks_like_question(line: str) -> bool:
    """Filters `DatasetGenerator`'s naive newline-split output.

    `DatasetGenerator.generate_questions_from_nodes` (llama-index-core 0.14.x)
    simply splits the raw completion on `\\n` and strips a leading `"1) "`-
    style numeral -- it does not validate that a line is actually a
    question. The workhorse model (`z-ai/glm-5.3-flash`) reliably wraps its
    numbered questions in markdown (`# Quiz Questions`, `**Question 1:**`),
    which the naive split turns into spurious extra "questions". A simple,
    documented heuristic -- keep only lines containing `?` and longer than
    a trivial length -- filters those out without needing a second LLM call.
    """
    stripped = line.strip(" *#:")
    return "?" in stripped and len(stripped) > 10


def _generate_for_chunk(llm, chunk_entry: dict) -> list[str]:
    """<= MAX_QUESTIONS_PER_CHUNK questions for one chunk, via `DatasetGenerator`."""
    from llama_index.core.evaluation.dataset_generation import DatasetGenerator
    from llama_index.core.schema import TextNode

    node = TextNode(id_=chunk_entry["chunk_id"], text=chunk_entry["text"])
    generator = DatasetGenerator(
        nodes=[node],
        llm=llm,
        num_questions_per_chunk=MAX_QUESTIONS_PER_CHUNK,
        show_progress=False,
    )
    raw_questions = generator.generate_questions_from_nodes(num=MAX_QUESTIONS_PER_CHUNK * 3)
    questions = [q for q in raw_questions if _looks_like_question(q)]
    return questions[:MAX_QUESTIONS_PER_CHUNK]


def estimate_synthetic_cost(chunk_entries: list[dict]) -> dict:
    """Pre-run cost estimate: one generation call per chunk, workhorse-priced.

    Uses the same `tokens_per_call` shape as `SWEEP_DEFS["m7a"]`'s first leg
    (2200 in / 220 out) so the estimate and the actual sweep definition agree.
    """
    from dealpoint.eval.spend import fetch_prices

    prices, basis = fetch_prices()
    model = synth_model()
    price = prices.get(model)
    if price is None:
        raise KeyError(f"no pricing available for {model!r}")
    per_call = price["prompt"] * 2200 + price["completion"] * 220
    n = len(chunk_entries)
    return {
        "model": model,
        "n_chunks": n,
        "per_call_usd": per_call,
        "est_usd": round(per_call * n, 6),
        "basis": basis,
    }


def run_calibration(client, llm, chunk_entries: list[dict], n: int = 3) -> dict:
    """Generate on a <=3-chunk calibration sample; compare projected vs realised."""
    from dealpoint.eval.spend import realized_usd

    sample = chunk_entries[:n]
    before = realized_usd()
    rows = []
    for entry in sample:
        client.context = {
            "milestone_tag": "m7a",
            "purpose": "synthetic_query",
            "chunk_id": entry["chunk_id"],
            "case_id": entry["case_id"],
        }
        questions = _generate_for_chunk(llm, entry)
        rows.append({"chunk_id": entry["chunk_id"], "n_questions": len(questions)})
    after = realized_usd()
    realized_delta = round(after - before, 6)
    projected = estimate_synthetic_cost(sample)
    return {
        "n_sample": len(sample),
        "projected_usd": projected["est_usd"],
        "realized_usd": realized_delta,
        "rows": rows,
    }


def generate_synthetic_set(
    client,
    *,
    out_path: Path = SYNTHETIC_DEV_QUERIES_PATH,
    calibration_n: int = 3,
) -> dict:
    """Full metered pipeline: estimate -> calibration -> cap check -> generate -> freeze.

    Order is mandatory per the spec's "Budget and disk" section: estimate
    calls and spend, run a <=3-item calibration sample, compare projected vs
    realised, then `assert_within_cap` before any bulk call.
    """
    from dealpoint.eval.spend import assert_within_cap

    chunk_entries = gold_bearing_dev_chunks()
    full_estimate = estimate_synthetic_cost(chunk_entries)

    llm = _make_llm(client)
    calibration = run_calibration(client, llm, chunk_entries, n=calibration_n)

    assert_within_cap(full_estimate["est_usd"])

    generator_ph = prompt_hash()
    model = synth_model()
    ts = datetime.now(UTC).isoformat()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for entry in chunk_entries:
            client.context = {
                "milestone_tag": "m7a",
                "purpose": "synthetic_query",
                "chunk_id": entry["chunk_id"],
                "case_id": entry["case_id"],
            }
            questions = _generate_for_chunk(llm, entry)
            for q in questions:
                row = {
                    "query_id": hashlib.sha256(f"{entry['chunk_id']}|{q}".encode()).hexdigest()[:16],
                    "question": q,
                    "chunk_id": entry["chunk_id"],
                    "agreement_id": entry["agreement_id"],
                    "case_id": entry["case_id"],
                    "question_id": entry["question_id"],
                    "generator": GENERATOR_ID,
                    "generator_version": GENERATOR_VERSION,
                    "prompt_hash": generator_ph,
                    "model": model,
                    "ts": ts,
                }
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                rows_written += 1

    file_hash = hashlib.sha256(out_path.read_bytes()).hexdigest()

    return {
        "generator": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "prompt_hash": generator_ph,
        "model": model,
        "n_chunks": len(chunk_entries),
        "n_queries": rows_written,
        "calibration": calibration,
        "estimate": full_estimate,
        "file_sha256": file_hash,
        "out_path": str(out_path),
        "generated_at": ts,
    }


def main(argv: list[str] | None = None) -> int:
    from dealpoint.llm.client import OpenRouterClient

    client = OpenRouterClient(milestone_tag="m7a")
    t0 = time.time()
    result = generate_synthetic_set(client)
    result["wall_seconds"] = round(time.time() - t0, 3)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
