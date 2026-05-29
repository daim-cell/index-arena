# FAISS Index Types

> Parameters, Trade-offs & Relationships

---

## Table of Contents

- [1. IndexFlatL2](#1-indexflatl2)
- [2. IndexIVFFlat](#2-indexivfflat)
- [3. IndexIVFPQ](#3-indexivfpq)
- [4. IndexHNSW](#4-indexhnsw)
- [5. Comparison Summary](#5-comparison-summary)

---

## 1. IndexFlatL2

IndexFlatL2 is the baseline brute-force index. It stores every vector at full precision and computes the exact L2 distance from a query to every single vector in the index. There is no approximation, no compression, and no training step.

Because of its exhaustive nature it always returns perfect recall (1.0), making it the standard ground-truth reference against which all approximate indexes are benchmarked.

### How it works

- **Add:** Insert a vector — it is appended to a flat array in RAM.
- **Query:** Compute L2 distance from the query to every stored vector.
- **Result:** Return the k smallest distances.

### Parameters

IndexFlatL2 has no tunable parameters. Search cost scales linearly with dataset size and cannot be adjusted.

### When to use

- Prototyping and small datasets (up to ~100k vectors)
- Establishing ground-truth recall for benchmarking other indexes
- When 100% recall is a hard requirement and the dataset fits in RAM

### Limitations

- **Memory:** entire index must reside in RAM — no on-disk option
- **Speed:** query time grows linearly — impractical beyond a few hundred thousand vectors
- **No compression** — each 128-dim float32 vector consumes 512 bytes

---

## 2. IndexIVFFlat

IndexIVFFlat adds a coarse clustering layer on top of flat storage. At build time the dataset is partitioned into `nlist` Voronoi cells using k-means. At query time only the `nprobe` nearest cells are searched, skipping the rest entirely. Vectors inside each cell are still stored at full precision — no compression.

### How it works

- **Train:** Run k-means to create `nlist` cluster centroids.
- **Add:** Assign each vector to its nearest centroid and store it in that cell.
- **Query:** Find the `nprobe` nearest centroids to the query, then search only those cells.

### Parameters

| Parameter | What it controls | Tuned at | Typical values |
|-----------|-----------------|----------|----------------|
| `nlist` | Number of clusters (Voronoi cells) | Build time | 64 – 1024 for 100k vectors |
| `nprobe` | How many clusters to search at query time | Runtime (no rebuild needed) | 4 – 64 (5–25% of nlist) |

### nlist and nprobe relationship

`nprobe` is always a subset of `nlist`. The ratio `nprobe / nlist` determines what fraction of the dataset is actually searched. A higher ratio means better recall but slower queries. The rule of thumb for a good balance is `nprobe ≈ 5–10% of nlist`.

- **Too few clusters** (small `nlist`): each cluster is large, even `nprobe=1` scans many vectors — slow
- **Too many clusters** (large `nlist`): clusters become tiny, recall collapses unless `nprobe` is also raised
- **Recommended starting point:** `nlist ≈ sqrt(dataset_size)`, `nprobe ≈ nlist * 0.05`

### When to use

- Datasets from 100k to several million vectors
- When full-precision vectors must be retained (no recall loss from compression)
- When you need a runtime speed/recall dial without rebuilding the index

---

## 3. IndexIVFPQ

IndexIVFPQ combines the cluster-based search of IVF with Product Quantization (PQ) compression inside each cluster. Instead of storing vectors at full precision, PQ splits each vector into `m` equal subvectors and replaces each subvector with a short integer code. This achieves dramatic memory reduction at the cost of approximate distance computation.

### Two-stage architecture

- **Stage 1 — IVF (coarse):** narrows the search to `nprobe` clusters, same as IVFFlat
- **Stage 2 — PQ (fine):** computes approximate distances using compressed codes inside those clusters

> Recall takes a double hit — once from IVF skipping clusters, and once from PQ distorting distances within the searched clusters.

### Parameters

| Parameter | What it controls | Tuned at | Typical values |
|-----------|-----------------|----------|----------------|
| `nlist` | Number of clusters (coarse IVF stage) | Build time | 64 – 1024 |
| `nprobe` | Clusters to search at query time | Runtime (no rebuild) | 4 – 64 |
| `m_pq` | Number of subvectors per vector | Build time | 8, 16, 32 (must divide `n_dims`) |
| `nbits` | Bits per subvector code (codebook size) | Build time | 8 (256 codewords, standard) |

### m_pq and nbits relationship

These two parameters jointly control compression quality. `m_pq` determines how finely the vector is sliced, and `nbits` determines how accurately each slice is represented.

```
Compressed vector size = m_pq × nbits bits
```

| Config | Compressed size | Compression ratio |
|--------|----------------|-------------------|
| Original 128-dim float32 | 512 bytes | 1x |
| m_pq=8, nbits=8 | 8 bytes | 64x |
| m_pq=16, nbits=8 | 16 bytes | 32x |
| m_pq=32, nbits=8 | 32 bytes | 16x (better recall) |

Higher `m_pq` preserves more information (better recall) but uses more memory and is slower to compute. `nbits=8` is almost universally the right choice — it gives 256 codewords per subvector which is the practical sweet spot. Lower values (`nbits=4`) hurt recall badly; higher values (`nbits=12`) are rarely worth the cost.

### Minimum subvector dimension

Each subvector must span at least 4 dimensions to encode meaningful geometry. For 128-dim vectors this means `m_pq=32` is the maximum safe value (128 / 32 = 4 dims per subvector). Going beyond this degrades recall significantly.

### When to use

- Datasets too large to fit in RAM at full precision (tens of millions of vectors)
- When memory is the primary constraint
- When approximate recall (0.85–0.95) is acceptable

---

## 4. IndexHNSW

Hierarchical Navigable Small World (HNSW) is a graph-based index. Vectors are inserted as nodes into a multi-layer proximity graph. Closer vectors are connected by edges. At query time a greedy traversal starts at the top layer and descends to the bottom layer, following edges toward the query vector at each step.

HNSW does not use clustering or compression. It stores vectors at full precision and achieves high recall through the graph structure alone.

### Parameters

| Parameter | What it controls | Tuned at | Typical values |
|-----------|-----------------|----------|----------------|
| `hnsw_m` | Connections per node (graph density) | Build time | 16 – 32 |
| `ef_construction` | Search width when inserting a new node | Build time | 100 – 200 |
| `ef_search` | Search width during query traversal | Runtime only | 16 – 128 |

### Parameter relationships

These three parameters have a clear separation of concerns across build time and query time:

- **`hnsw_m`** — controls graph structure. More connections means a denser graph, better recall, and more memory. Each node stores M connections per layer.
- **`ef_construction`** — controls graph quality. A wider build-time search finds better neighbors to connect to, producing a higher quality graph. Must always be `>= hnsw_m`.
- **`ef_search`** — controls query quality. A wider runtime search explores more of the graph, improving recall at the cost of latency. Must always be `>= k`.

A key asymmetry: `hnsw_m` and `ef_construction` are permanent decisions fixed at build time. `ef_search` is a free runtime dial — you can tune it without rebuilding the index.

### Dependency rules

```
ef_construction >= hnsw_m    # must hold, otherwise build fails to find enough neighbors
ef_search       >= k         # must hold, otherwise not enough results can be returned
```

> A poorly built graph (low `ef_construction`) cannot be fixed by raising `ef_search` later.

### When to use

- When recall must be very high (0.99+) with low latency
- When memory is available (no compression — full-precision storage)
- When the index will be queried frequently and query speed matters most

---

## 5. Comparison Summary

| Index | Search type | Compression | Recall | Memory | Tunable at runtime |
|-------|-------------|-------------|--------|--------|--------------------|
| `IndexFlatL2` | Exhaustive | None | Perfect (1.0) | Highest | None |
| `IndexIVFFlat` | Cluster-limited | None | Near-perfect | Same as Flat | `nprobe` |
| `IndexIVFPQ` | Cluster-limited | PQ (lossy) | Approximate | Lowest | `nprobe` |
| `IndexHNSW` | Graph traversal | None | Near-perfect | High | `ef_search` |

### Choosing the right index

| Use case | Recommended index |
|----------|------------------|
| Ground truth / tiny dataset | `IndexFlatL2` |
| Medium dataset, full precision, runtime tuning | `IndexIVFFlat` |
| Large dataset, memory constrained | `IndexIVFPQ` |
| Low latency, high recall, graph-based search | `IndexHNSW` |