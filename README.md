# index-arena

Benchmarking approximate nearest neighbor (ANN) indexes for retrieval-augmented generation (RAG) systems using FAISS.

This project compares:

- `IndexFlatL2`
- `IndexIVFFlat`
- `IndexIVFPQ`
- `IndexHNSWFlat`


---

## Features

- Real embedding corpus benchmarking
- Ground-truth nearest neighbors via brute force
- IVF + HNSW parameter sweeps
- Filtered ANN benchmarks
- End-to-end RAG retrieval evaluation
- Pareto frontier analysis

---

## Benchmarked Indexes

| Index | Purpose |
|---|---|
| Flat | Exact search baseline |
| IVFFlat | Faster clustered search |
| IVFPQ | Memory-efficient ANN |
| HNSW | High-recall graph search |

---

## Metrics

Each benchmark run records:

- recall@10
- p50 / p95 / p99 latency
- QPS
- build time
- memory footprint
- filter selectivity
- index configuration

---

