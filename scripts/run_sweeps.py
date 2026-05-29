"""Goal 2 entrypoint: parameter sweeps across IndexIVFFlat, IndexIVFPQ, IndexHNSWFlat."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import logging
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark import FIELDNAMES, BenchmarkConfig, BenchmarkResult, _compute_recall, _rss_mb
from data_gen import DataGenConfig, generate_dataset
from ground_truth import _cache_key, get_ground_truth
from indexes import HNSWFlatIndex, IVFFlatIndex, IVFPQIndex

import faiss
import uuid
import math
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Parameter grids ──────────────────────────────────────────────────────────

IVF_FLAT_GRID = {
    "nlist": [ 256, 1024],
    "nprobe": [ 16, 64],
}

IVF_PQ_GRID = {
    "nlist": [ 256, 1024],
    "m_pq": [32],
    "nbits": [8],
    "nprobe": [ 16, 64],
}

HNSW_GRID = {
    "hnsw_m": [16, 32],
    "ef_construction": [100, 200],
    "ef_search": [64, 128],
}

WARMUP_COUNT = 100


def _append_results(results: list[BenchmarkResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    logger.info("Wrote %d rows to %s", len(results), output)


def _make_result(
    index_type: str,
    dataset,
    gt_indices: np.ndarray,
    gt_key: str,
    bench_config: BenchmarkConfig,
    build_time_s: float,
    rss_before: float,
    rss_after: float,
    latencies: list[float],
    recall: float,
    index_params: dict,
    query_params: dict,
    raw_latency_path: str | None,
) -> BenchmarkResult:
    p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
    query_count = len(latencies)
    qps = query_count / (sum(latencies) / 1000.0)
    return BenchmarkResult(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        index_type=index_type,
        dataset_size=dataset.config.n_vectors,
        n_dims=dataset.config.n_dims,
        n_clusters=dataset.config.n_clusters,
        query_count=query_count,
        k=bench_config.k,
        build_time_s=build_time_s,
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        memory_mb=rss_after - rss_before,
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
        embedding_source=bench_config.embedding_source,
        omp_threads=faiss.omp_get_max_threads(),
        raw_latency_path=raw_latency_path,
        gt_cache_key=gt_key,
    )


def _run_queries(index, queries: np.ndarray, gt_indices: np.ndarray, bench_config: BenchmarkConfig):
    """Warmup then measure; return (latencies_ms, recall)."""
    k = bench_config.k
    for q in queries[:WARMUP_COUNT]:
        index.search(q[np.newaxis], k)

    latencies: list[float] = []
    returned_all = []
    for q in queries[WARMUP_COUNT:]:
        t0 = time.perf_counter()
        result = index.search(q[np.newaxis], k)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        returned_all.append(result[0])

    if min(latencies) == 0.0:
        raise ValueError("Zero latency — check warmup")

    returned_arr = np.array(returned_all)
    gt_measured = gt_indices[WARMUP_COUNT:]
    recall = _compute_recall(returned_arr, gt_measured, k)
    if math.isnan(recall):
        raise ValueError("Recall is NaN")
    return latencies, recall


# ── IVFFlat sweep ────────────────────────────────────────────────────────────

def sweep_ivf_flat(dataset, gt_indices, gt_key, bench_config, dry_run, output):
    from tqdm import tqdm

    combos = list(product(IVF_FLAT_GRID["nlist"], IVF_FLAT_GRID["nprobe"]))
    logger.info("IVFFlat sweep: %d combinations", len(combos))

    results = []
    for nlist, nprobe in tqdm(combos, desc="IVFFlat"):
        if dry_run:
            logger.info("DRY RUN: IVFFlat nlist=%d nprobe=%d", nlist, nprobe)
            continue

        gc.collect()
        rss_before = _rss_mb()
        t0 = time.perf_counter()
        index = IVFFlatIndex(nlist=nlist, nprobe=nprobe)
        index.build(dataset.corpus)
        build_time_s = time.perf_counter() - t0
        rss_after = _rss_mb()

        latencies, recall = _run_queries(index, dataset.queries, gt_indices, bench_config)
        r = _make_result(
            "IndexIVFFlat", dataset, gt_indices, gt_key, bench_config,
            build_time_s, rss_before, rss_after, latencies, recall,
            index_params={"nlist": nlist, "nprobe": nprobe},
            query_params={},
            raw_latency_path=None,
        )
        results.append(r)
        logger.info("IVFFlat nlist=%d nprobe=%d | recall=%.4f p50=%.2f ms", nlist, nprobe, recall, r.p50_ms)

    if not dry_run:
        _append_results(results, output)
    return results


# ── IVFPQ sweep ──────────────────────────────────────────────────────────────

def sweep_ivf_pq(dataset, gt_indices, gt_key, bench_config, dry_run, output):
    from tqdm import tqdm

    combos = list(product(IVF_PQ_GRID["nlist"], IVF_PQ_GRID["m_pq"], IVF_PQ_GRID["nbits"], IVF_PQ_GRID["nprobe"]))
    logger.info("IVFPQ sweep: %d combinations", len(combos))

    results = []
    for nlist, m, nbits, nprobe in tqdm(combos, desc="IVFPQ"):
        if dry_run:
            logger.info("DRY RUN: IVFPQ nlist=%d m=%d nbits=%d nprobe=%d", nlist, m, nbits, nprobe)
            continue

        gc.collect()
        rss_before = _rss_mb()
        t0 = time.perf_counter()
        index = IVFPQIndex(nlist=nlist, m=m, nbits=nbits, nprobe=nprobe)
        index.build(dataset.corpus)
        build_time_s = time.perf_counter() - t0
        rss_after = _rss_mb()

        latencies, recall = _run_queries(index, dataset.queries, gt_indices, bench_config)
        r = _make_result(
            "IndexIVFPQ", dataset, gt_indices, gt_key, bench_config,
            build_time_s, rss_before, rss_after, latencies, recall,
            index_params={"nlist": nlist, "m_pq": m, "nbits": nbits, "nprobe": nprobe},
            query_params={},
            raw_latency_path=None,
        )
        results.append(r)
        logger.info(
            "IVFPQ nlist=%d m=%d nbits=%d nprobe=%d | recall=%.4f mem=%.1f MB",
            nlist, m, nbits, nprobe, recall, r.memory_mb,
        )

    if not dry_run:
        _append_results(results, output)
    return results


# ── HNSW sweep ───────────────────────────────────────────────────────────────

def sweep_hnsw(dataset, gt_indices, gt_key, bench_config, dry_run, output):
    from tqdm import tqdm

    build_combos = list(product(HNSW_GRID["hnsw_m"], HNSW_GRID["ef_construction"]))
    ef_search_values = HNSW_GRID["ef_search"]
    total = len(build_combos) * len(ef_search_values)
    logger.info("HNSW sweep: %d builds × %d efSearch = %d measurements", len(build_combos), len(ef_search_values), total)

    results = []
    for M, ef_c in tqdm(build_combos, desc="HNSW builds"):
        if dry_run:
            for ef_s in ef_search_values:
                logger.info("DRY RUN: HNSW M=%d ef_construction=%d ef_search=%d", M, ef_c, ef_s)
            continue

        gc.collect()
        rss_before = _rss_mb()
        logger.info("Building HNSW M=%d ef_construction=%d ...", M, ef_c)
        t0 = time.perf_counter()
        index = HNSWFlatIndex(M=M, ef_construction=ef_c)
        index.build(dataset.corpus)
        build_time_s = time.perf_counter() - t0
        rss_after = _rss_mb()
        logger.info("HNSW M=%d ef_c=%d built in %.1f s", M, ef_c, build_time_s)

        for ef_s in ef_search_values:
            index.set_query_params(ef_search=ef_s)
            latencies, recall = _run_queries(index, dataset.queries, gt_indices, bench_config)
            r = _make_result(
                "IndexHNSWFlat", dataset, gt_indices, gt_key, bench_config,
                build_time_s, rss_before, rss_after, latencies, recall,
                index_params={"hnsw_m": M, "ef_construction": ef_c},
                query_params={"ef_search": ef_s},
                raw_latency_path=None,
            )
            results.append(r)
            logger.info(
                "HNSW M=%d ef_c=%d ef_s=%d | recall=%.4f p50=%.2f ms",
                M, ef_c, ef_s, recall, r.p50_ms,
            )

    if not dry_run:
        _append_results(results, output)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FAISS parameter sweeps (Goal 2)")
    parser.add_argument(
        "--index",
        choices=["ivf_flat", "ivf_pq", "hnsw", "all"],
        default="all",
        help="Which index family to sweep (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print grid without running benchmarks")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/runs/sweep_results.csv"),
        help="Output CSV path (appended)",
    )
    args = parser.parse_args()

    logger.info("Generating dataset...")
    data_config = DataGenConfig()
    dataset = generate_dataset(data_config)

    logger.info("Computing / loading ground truth...")
    gt_indices, _ = get_ground_truth(dataset.corpus, dataset.queries, k=10)
    gt_key = _cache_key(dataset.corpus, dataset.queries)

    bench_config = BenchmarkConfig()

    run_ivf_flat = args.index in ("ivf_flat", "all")
    run_ivf_pq = args.index in ("ivf_pq", "all")
    run_hnsw = args.index in ("hnsw", "all")

    if run_ivf_flat:
        sweep_ivf_flat(dataset, gt_indices, gt_key, bench_config, args.dry_run, args.output)
    if run_ivf_pq:
        sweep_ivf_pq(dataset, gt_indices, gt_key, bench_config, args.dry_run, args.output)
    if run_hnsw:
        sweep_hnsw(dataset, gt_indices, gt_key, bench_config, args.dry_run, args.output)

    if not args.dry_run:
        logger.info("Sweep complete. Results in %s", args.output)
    else:
        logger.info("Dry run complete — no benchmarks executed.")


if __name__ == "__main__":
    main()
