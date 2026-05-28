from __future__ import annotations

import dataclasses
import gc
import logging
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import psutil

from data_gen import Dataset
from indexes import BaseIndex

logger = logging.getLogger(__name__)

WARMUP_COUNT = 100


@dataclass
class BenchmarkConfig:
    k: int = 10
    warmup_count: int = WARMUP_COUNT
    results_dir: Path = Path("results/runs")
    latencies_dir: Path = Path("results/latencies")
    embedding_source: str = "synthetic_gaussian"


@dataclass
class BenchmarkResult:
    run_id: str
    timestamp: str
    index_type: str
    dataset_size: int
    n_dims: int
    n_clusters: int
    query_count: int
    k: int
    build_time_s: float
    rss_before_mb: float
    rss_after_mb: float
    memory_mb: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    recall_at_10: float
    nlist: Optional[int]
    nprobe: Optional[int]
    m_pq: Optional[int]
    nbits: Optional[int]
    hnsw_m: Optional[int]
    ef_construction: Optional[int]
    ef_search: Optional[int]
    filter_selectivity: Optional[float]
    embedding_source: str
    omp_threads: int
    raw_latency_path: Optional[str]
    gt_cache_key: str


FIELDNAMES = [f.name for f in dataclasses.fields(BenchmarkResult)]


def _compute_recall(returned: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    hits = sum(len(set(returned[i].tolist()) & set(ground_truth[i].tolist())) for i in range(len(returned)))
    return hits / (len(returned) * k)


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def _measure_queries(
    index: BaseIndex,
    queries: np.ndarray,
    gt_indices: np.ndarray,
    config: BenchmarkConfig,
) -> tuple[list[float], float]:
    """Warm up then measure. Returns (latencies_ms, recall)."""
    k = config.k

    if len(queries) < config.warmup_count + 1:
        raise ValueError(
            f"Query set too small: {len(queries)} < warmup_count ({config.warmup_count}) + 1"
        )

    for q in queries[: config.warmup_count]:
        index.search(q[np.newaxis], k)

    latencies: list[float] = []
    returned_all = []
    for q in queries[config.warmup_count :]:
        t0 = time.perf_counter()
        result = index.search(q[np.newaxis], k)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        returned_all.append(result[0])

    if min(latencies) == 0.0:
        raise ValueError("Zero latency detected — clock resolution too coarse or warmup insufficient")

    returned_arr = np.array(returned_all)
    gt_measured = gt_indices[config.warmup_count :]
    recall = _compute_recall(returned_arr, gt_measured, k)

    if math.isnan(recall):
        raise ValueError("Recall is NaN — check ground truth and index correctness")

    return latencies, recall


def run_benchmark(
    index: BaseIndex,
    dataset: Dataset,
    gt_indices: np.ndarray,
    gt_cache_key: str,
    config: BenchmarkConfig,
    query_params: dict | None = None,
    index_params: dict | None = None,
    save_raw_latencies: bool = False,
    # Pre-built path: if provided, skip build and use these values
    prebuilt_build_time_s: float | None = None,
    prebuilt_rss_before_mb: float | None = None,
    prebuilt_rss_after_mb: float | None = None,
) -> BenchmarkResult:
    """Build (or reuse pre-built) index, then measure query performance."""
    query_params = query_params or {}
    index_params = index_params or {}

    if prebuilt_build_time_s is None:
        gc.collect()
        rss_before = _rss_mb()
        t0 = time.perf_counter()
        index.build(dataset.corpus)
        t1 = time.perf_counter()
        rss_after = _rss_mb()
        build_time_s = t1 - t0
        if rss_after - rss_before < 0:
            logger.warning(
                "Memory delta is negative (%.1f MB) — macOS page compression may skew this reading",
                rss_after - rss_before,
            )
    else:
        assert prebuilt_rss_before_mb is not None and prebuilt_rss_after_mb is not None, (
            "prebuilt_rss_before_mb and prebuilt_rss_after_mb must both be set when prebuilt_build_time_s is provided"
        )
        rss_before = prebuilt_rss_before_mb
        rss_after = prebuilt_rss_after_mb
        build_time_s = prebuilt_build_time_s

    if query_params:
        index.set_query_params(**query_params)

    latencies, recall = _measure_queries(index, dataset.queries, gt_indices, config)

    p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
    query_count = len(latencies)
    qps = query_count / (sum(latencies) / 1000.0)
    memory_mb = rss_after - rss_before  # type: ignore[operator]

    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    raw_latency_path: str | None = None
    if save_raw_latencies:
        config.latencies_dir.mkdir(parents=True, exist_ok=True)
        lat_path = config.latencies_dir / f"{index.index_type}_{run_id}.npy"
        np.save(lat_path, np.array(latencies, dtype=np.float32))
        raw_latency_path = str(lat_path)
        logger.info("Raw latencies saved to %s", lat_path)

    result = BenchmarkResult(
        run_id=run_id,
        timestamp=timestamp,
        index_type=index.index_type,
        dataset_size=dataset.config.n_vectors,
        n_dims=dataset.config.n_dims,
        n_clusters=dataset.config.n_clusters,
        query_count=query_count,
        k=config.k,
        build_time_s=build_time_s,
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        memory_mb=memory_mb,
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
        qps=qps,
        recall_at_10=recall,
        nlist=index_params.get("nlist"),
        nprobe=index_params.get("nprobe") or query_params.get("nprobe"),
        m_pq=index_params.get("m_pq"),
        nbits=index_params.get("nbits"),
        hnsw_m=index_params.get("hnsw_m"),
        ef_construction=index_params.get("ef_construction"),
        ef_search=query_params.get("ef_search"),
        filter_selectivity=None,
        embedding_source=config.embedding_source,
        omp_threads=faiss.omp_get_max_threads(),
        raw_latency_path=raw_latency_path,
        gt_cache_key=gt_cache_key,
    )

    logger.info(
        "%s | recall=%.4f | p50=%.2f ms | p99=%.2f ms | QPS=%.1f | mem=%.1f MB",
        index.index_type,
        recall,
        p50,
        p99,
        qps,
        memory_mb,
    )
    return result
