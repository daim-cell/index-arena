# CLAUDE.md

Guidance for AI coding agents working on this repo. Read this fully before making changes.

## What this project is

**vector-index-surgery** is a benchmark and analysis project comparing FAISS index types (`IndexFlatL2`, `IndexIVFFlat`, `IndexIVFPQ`, `IndexHNSWFlat`) across the recall / QPS / memory / build-time tradeoff space. It is deliberately scoped to run on an M1 MacBook Air (8–16 GB RAM) and produces:

1. A reproducible benchmark harness that sweeps index parameters
2. Pareto frontier plots across all configs
3. A filtered ANN evaluation across multiple filter strategies and selectivities
4. A minimal RAG pipeline using the winning config


## What this project is NOT

- **Not a FAISS wrapper library.** Do not generalize the harness into a reusable package. Keep it project-scoped.
- **Not a hosted-vector-DB comparison.** No Pinecone, Weaviate, Qdrant calls. FAISS only. (pgvector may be added later for one figure, but only if explicitly requested.)
- **Not a distributed system.** Single machine, single process. No Ray, no Dask, no multi-node anything.

## Goals and current state

The project is structured as five sequential goals:

1. **Foundation** — flat baseline + synthetic data + benchmark harness
2. **Sweeps** — IVF / IVFPQ / HNSW selected parameter sweeps on synthetic data
3. **Real corpus** — swap synthetic Gaussians for real embeddings on a real corpus
4. **Filtered ANN** — pre-filter vs post-filter vs inline filter across selectivities
5. **RAG + Pareto + decision guide** — end-to-end pipeline, all plots, written guide


## Hard constraints

### Environment

- **Python 3.11+**, FAISS installed via conda-forge (`conda install -c conda-forge faiss-cpu`). Do not suggest pip install for FAISS on M1 — the wheels are unreliable.
- **Target hardware: M1 MacBook Air, 8–16 GB RAM.** Before suggesting a sweep, estimate memory cost. A 1M × 768 float32 corpus is ~3 GB before the index — that's already tight.
- No GPU code. `faiss-gpu` is off the table.

### Ground truth

- Ground truth (true nearest neighbors via brute force) is computed once per (dataset, query set) pair and cached to disk under `data/ground_truth/`. Never recompute silently.
- If the dataset or query set changes, the cache key must change. Hash the data, not just the filename.
- Recall is computed against this ground truth. If recall numbers look suspicious (e.g., flat index showing recall < 1.0), the first suspect is a stale ground-truth cache.

### Timing

- Latency measurements use `time.perf_counter()`. Wall-clock only. No CPU time.
- Warm up before timing — at least 100 queries discarded before the measured run begins. FAISS indexes do lazy work on first queries.
- Single-threaded by default. If you set `faiss.omp_set_num_threads(n)` for any reason, log `n` in the results row.
- p50/p95/p99 come from per-query latencies, not averages. Store the raw latency array for at least the headline configs in case we want to re-aggregate later.

## Code conventions


### Style

- **Dataclasses for configs.** No dict-passing for parameters that have a known schema.
- **No notebooks in the critical path.** Notebooks are for exploration; anything reproducible lives in `src/` or `scripts/`.
- **Logging via `logging`**, not print. Default level INFO; benchmark loops log progress every N queries.
- **No silent failures.** If recall is NaN or latency is 0, raise. Don't write the row.
- **Black + ruff** for formatting / linting. Keep line length at 100.

### Dependencies

Keep the dependency list short. Do not use anything that is not actually required.

## Common pitfalls — read this before debugging

These are the failure modes that will cost you hours if you don't watch for them:

- **Recall = 1.0 for HNSW** usually means `efSearch` is way larger than k, or the query set is in the training set. Check.
- **Recall = 0.0 for IVF** usually means `nprobe` is 0 or the index wasn't trained. `IndexIVF*` requires `.train()` before `.add()`.
- **PQ recall is shockingly bad** — this is often correct. PQ trades recall for memory. If recall@10 is 0.6 at extreme compression, that's the finding, not a bug. But sanity-check by also running PQ with mild compression.
- **HNSW build hangs** — likely high `M` + high `ef_construction` on a large corpus. Estimated build times: 1M × 768d, M=64, ef_construction=400 → 30+ minutes on M1. This is expected. Use `tqdm` if FAISS exposes a callback, otherwise log start time and walk away.
- **Latency p99 looks insane** — first few queries are warmup. Discard them. See the timing section above.
- **Filtered HNSW recall collapses at low selectivity** — this is the actual finding of Goal 4, not a bug. Verify with a pre-filter brute force at the same selectivity to confirm the truth set is what you think it is.
- **Embedding model output is not L2-normalized** — many sentence-transformers outputs aren't normalized by default. Decide once whether you're using L2 or inner product, normalize accordingly, and document the choice. Mixing them silently gives garbage recall.

## When in doubt

- **Prefer paraphrased internal docs over re-deriving FAISS behavior from memory.** FAISS APIs change; check the installed version's docs if unsure.
- **Prefer fewer, well-chosen parameter points over exhaustive grids.** 3 well-spaced values per axis usually beats 7 closely-spaced ones for the M1 budget.
- **Ask the human before:** adding a dependency, expanding scope beyond the six goals, changing the results schema, or proposing a refactor that touches more than 3 files.
