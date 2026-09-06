# M7a -- LlamaIndex RAG lab: native evaluation on canonical dev retrieval

Case set: `dev` (58 cases), chunk_version=`8e5e8ba56765`, index_version=`e2b4a2b97561`, git_sha7=`b9eac78`

LlamaIndex owns retriever composition/evaluation and the synthetic-query study; MAUD gold-span overlap remains benchmark truth. LlamaIndex node ids never replace it.

## Per-retriever: LI vs obj/

| config | li/hit_rate | li/mrr | obj/gold_span_hit@5 | obj/gold_span_mrr | n |
|---|---|---|---|---|---|
| dense | 81.0% | 0.621 | 81.0% | 0.621 | 58 |
| bm25 | 86.2% | 0.664 | 86.2% | 0.664 | 58 |
| hybrid_rrf | 91.4% | 0.724 | 91.4% | 0.724 | 58 |
| hybrid_rrf_rerank | 91.4% | 0.723 | 91.4% | 0.723 | 58 |
| multi_query_fusion | 82.8% | 0.678 | 82.8% | 0.678 | 58 |
| multi_query_fusion_rerank | 87.9% | 0.713 | 87.9% | 0.713 | 58 |

## Ranking-order agreement (Spearman over retriever ranks)

hit_rate vs hit@5: rho = 1.0

mrr vs mrr: rho = 1.0

## Disagreements

LI-hit/MAUD-miss: 0; MAUD-hit/LI-miss: 0

Near-zero disagreement is expected and reported plainly, not manufactured: expected_ids for the LlamaIndex RetrieverEvaluator ARE the project's gold-bearing chunk ids (dealpoint.rag_lab.evaluate.gold_bearing_chunk_ids uses the exact same overlap_chars/MIN_GOLD_OVERLAP_CHARS rule as dealpoint.eval.tournament._first_hit_rank), so the two metrics differ only where k-truncation or multi-hit averaging bites.

Cause rule:

> Deterministic disagreement-cause rule, applied in this priority order to a disagreeing (case, config, direction): 'reranking' when the config carries a rerank_model AND the pre-rerank fused ranking's hit/miss differs from the post-rerank ranking's hit/miss at the same k; 'partial_overlap' when the top-ranked retrieved chunk overlaps a gold span by 0 < overlap < MIN_GOLD_OVERLAP_CHARS (50) chars; 'duplicate_relevant_chunks' when >= 2 chunks in the case's full chunk list each overlap a gold span by >= 50 chars (expected_ids has >= 2 entries), so which one a retriever happens to surface first is underdetermined; 'section_boundary' when a single gold span is covered (any overlap > 0) by chunks carrying more than one distinct section_ref; 'chunk_identity' otherwise (the default: the two metrics disagree for a reason not captured by the other four labels -- typically the retrieved list and expected_ids simply differ).

## Synthetic-query robustness (secondary)

Generator: `llama_index.core.evaluation.dataset_generation.DatasetGenerator` (num_questions_per_chunk=2), model=`z-ai/glm-5.3-flash`, n_chunks=73, n_queries=106, prompt_hash=`93801419e0abfb5b`

Verdict: Frozen winner `hybrid_rrf` stays strong under the synthetic query distribution: li_hit_rate=0.9433962264150944 vs dense=0.8867924528301887.

| config | li/hit_rate (synthetic) | obj/hit@5 (synthetic) |
|---|---|---|
| dense | 88.7% | 93.4% |
| bm25 | 90.6% | 92.5% |
| hybrid_rrf | 94.3% | 97.2% |
| hybrid_rrf_rerank | 96.2% | 98.1% |
| multi_query_fusion | 94.3% | 97.2% |
| multi_query_fusion_rerank | 96.2% | 98.1% |

## Frozen-integrity assertions

```json
{
  "chunk_version": "8e5e8ba56765",
  "index_version": "e2b4a2b97561",
  "test_subset_v1_sha256": "37b9e61a55f96c00f23b192f826c86241151581b2383b5563665af06255bd19c",
  "tournament_json_sha256": "5277f1d9a795d9cc0632d5c4635ac8d4a203487b4b92a5773e2b208e5040d748"
}
```

## Decisions

- **m6_envelope_test_scoping**: tests/test_spend_m6.py's envelope assertion scoped to M1-M6 tags via realized_by_tag() rather than the whole ledger; see brief_differences[1].

## Brief-vs-spec differences

- **llamaindex_scope**: Brief section 3 allows LlamaIndex 'only behind the Retriever interface'; section 4 excludes LlamaIndex agents/workflows. M7a additionally uses it for framework-native evaluation (RetrieverEvaluator) and synthetic-query generation (DatasetGenerator). Agents/workflows remain excluded -- neither is used anywhere in this module. M3 measured LlamaIndex and declined it for retrieval composition (tournament.json's llamaindex.adopted: false, 178 MB reason); M7a adopts it for a different purpose (evaluation/synthesis, not composition) -- the earlier decision is superseded for that purpose only, not reversed for retrieval itself.
- **m6_envelope_test_scoping**: tests/test_spend_m6.py::test_realized_usd_within_envelope originally asserted realized_usd() (the WHOLE ledger) <= 4.00. Once M7a's own ledger rows exist that assertion would go false for a reason having nothing to do with M6. Repaired to sum only the M1-M6 milestone tags via the existing realized_by_tag() helper, which keeps M6's own claim exactly as strong as it was measured, rather than either weakening the envelope (raising 4.00, which the spec explicitly forbids) or leaving a spurious cross-milestone failure. M7a's own ledger discipline is asserted separately in tests/test_spend_m7.py against the $6.00 global cap.

