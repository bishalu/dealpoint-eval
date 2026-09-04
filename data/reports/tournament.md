# M3 retrieval tournament

Case set: `dev` (58 cases), chunk_version=`8e5e8ba56765`, index_version=`e2b4a2b97561`, git_sha7=`2afdce5`, total_wall_seconds=`774.955`

## canonical queries

| config | hit@5 | hit@10 | MRR | wall (s) |
|---|---|---|---|---|
| dense | 0.810 | 0.845 | 0.626 | 8.0 |
| bm25 | 0.862 | 0.948 | 0.673 | 0.0 |
| hybrid_rrf | 0.914 | 0.966 | 0.732 | 8.1 |
| hybrid_rrf_rerank | 0.914 | 0.948 | 0.727 | 106.4 |
| multi_query_fusion | 0.828 | 0.914 | 0.690 | 32.1 |
| multi_query_fusion_rerank | 0.879 | 0.914 | 0.718 | 130.6 |

## maud queries

| config | hit@5 | hit@10 | MRR | wall (s) |
|---|---|---|---|---|
| dense | 0.276 | 0.328 | 0.168 | 7.8 |
| bm25 | 0.328 | 0.500 | 0.189 | 0.0 |
| hybrid_rrf | 0.276 | 0.483 | 0.183 | 7.9 |
| hybrid_rrf_rerank | 0.414 | 0.603 | 0.283 | 106.5 |
| multi_query_fusion | 0.828 | 0.845 | 0.587 | 32.0 |
| multi_query_fusion_rerank | 0.586 | 0.810 | 0.379 | 128.4 |

## which component found it

| surfaced_by | count |
|---|---|
| bm25 | 20 |
| both | 71 |
| dense | 8 |
| none | 17 |

| rerank_effect | count |
|---|---|
| demoted | 17 |
| n/a | 22 |
| promoted | 39 |
| unchanged | 38 |

## Winner: `hybrid_rrf` (hit@5=0.914 vs dense 0.810, margin=+0.103, basis=canonical)

## LlamaIndex

adopted=False; measured standalone install 178 MB, marginal into this repo's venv ~100 MB. llama-index-core + llama-index-retrievers-bm25 measured (uv venv probe): 178 MB standalone, no torch/transformers/nvidia wheels, ~100 MB marginal into this repo's venv (numpy/pillow/pydantic already present). It fits the ~2 GB disk budget but was declined: bm25s gives BM25 in three calls, RRF is eight lines of pure Python, and fastembed already ships the named cross-encoder, so LlamaIndex would add install weight for no material capability this milestone needs.

