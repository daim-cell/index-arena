# Codex Guidance

This file contains working guidance for Codex and other coding agents in this repository.


## Project Identity

The repository is named **index-arena**. It benchmarks FAISS approximate nearest neighbor
(ANN) index configurations for retrieval-augmented generation (RAG) workloads:

- `IndexFlatL2` as the exact-search baseline
- `IndexIVFFlat`
- `IndexIVFPQ`
- `IndexHNSWFlat`

Use `index-arena` in new paths and user-facing project text.

## Intended Delivery Order

Build work sequentially:

1. Flat baseline, synthetic data generation, and benchmark harness.
2. IVF, IVFPQ, and HNSW parameter sweeps on synthetic data.
3. Real corpus embeddings.
4. Filtered ANN strategies and selectivity evaluation.
5. Minimal RAG evaluation, Pareto plots, and a decision guide.

Keep the project a benchmark application, not a generic FAISS library or distributed
system. Do not introduce hosted vector databases, GPU code, or multi-machine execution.

## Environment And Resource Limits

- Target Python 3.11+ on an M1 MacBook Air with 8-16 GB RAM.
- Install FAISS on Apple Silicon through conda-forge (`faiss-cpu`), not pip.
- Estimate memory before increasing corpus sizes or sweep grids. A 1M by 768 float32
  embedding matrix uses roughly 3 GB before index overhead.
- Default to single-threaded runs. If FAISS thread count is changed, persist it in results.
- Keep dependencies narrow and ask before adding one.

## Benchmark Correctness Rules

- Cache brute-force ground truth in `data/ground_truth/` once per dataset/query pair.
- Key cached ground truth from the data contents, not just filenames.
- Treat a flat-index recall below 1.0 as a likely stale-cache or evaluation bug.
- Use `time.perf_counter()` for latency and discard at least 100 warmup queries.
- Calculate p50/p95/p99 from per-query measurements, and retain raw latency data for
  headline configurations.
- Raise rather than recording invalid rows such as NaN recall or zero latency.
- Make metric/index compatibility explicit: document L2 versus inner product and any
  embedding normalization.

## Code Conventions

- Use dataclasses for configurations with known schema.
- Put reproducible implementation in `src/` or executable workflows in `scripts/`;
  notebooks are exploratory only.
- Use the `logging` module at INFO level by default and emit progress during long runs.
- Format and lint Python using Black and Ruff with line length 100.
- Keep result schema stable unless the user agrees to a change.

## Repository Layout

The initial scaffold is intentionally small:

```text
configs/                 Versioned benchmark configuration files
data/
  raw/                   Local source corpora or downloaded inputs
  processed/             Derived vectors and prepared datasets
  ground_truth/          Content-keyed exact-neighbor caches
docs/                    Decision guide and written analysis
notebooks/               Non-critical-path exploration
results/
  runs/                  Benchmark result tables
  latencies/             Retained per-query latency artifacts
  plots/                 Pareto and comparison figures
scripts/                 Reproducible command-line entry points
src/index_arena/         Project implementation package
tests/                   Focused correctness and regression tests
```

Do not commit large local corpora, derived embeddings, or ground-truth caches. Results
and figures may be committed selectively when they form part of an analysis deliverable.

## Before Completing Work

- Test flat-index recall and cache invalidation behavior when evaluation logic changes.
- Validate index training for IVF-based indexes before search.
- Check that reported latency excludes warmup queries.
- State any test or benchmark step that was not run because it requires FAISS, a dataset,
  or substantial local compute.
