"""Step 2 — Run FAISS parameter sweeps on the cached Wikipedia embeddings.

Requires embed_corpus.py to have been run first.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import FIELDNAMES, BenchmarkConfig, run_benchmark
from embedder import EmbedderConfig, _cache_path, load_or_embed
from ground_truth import _cache_key, get_ground_truth
from indexes import FlatL2Index
from run_sweeps import (  # type: ignore[import]
    HNSW_GRID,
    IVF_FLAT_GRID,
    IVF_PQ_GRID,
    sweep_hnsw,
    sweep_ivf_flat,
    sweep_ivf_pq,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RealConfig:
    n_vectors: int
    n_dims: int
    n_clusters: int = 0


@dataclasses.dataclass
class RealDataset:
    corpus: np.ndarray
    queries: np.ndarray
    config: RealConfig


def _make_real_dataset(corpus: np.ndarray, queries: np.ndarray) -> RealDataset:
    cfg = RealConfig(n_vectors=corpus.shape[0], n_dims=corpus.shape[1])
    return RealDataset(corpus=corpus, queries=queries, config=cfg)


def _append_results(results, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    logger.info("Wrote %d rows to %s", len(results), output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FAISS sweeps on the cached Wikipedia embeddings (Goal 3)."
    )
    parser.add_argument(
        "--index",
        choices=["flat", "ivf_flat", "ivf_pq", "hnsw", "all"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config grid without benchmarking.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/runs/real_corpus_results.csv"),
    )
    parser.add_argument("--model", type=str, default="text-embedding-3-small")
    parser.add_argument("--n-corpus", type=int, default=100_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    args = parser.parse_args()

    embed_config = EmbedderConfig(
        model=args.model,
        n_corpus=args.n_corpus,
        n_queries=args.n_queries,
    )
    cache = _cache_path(embed_config)
    if not cache.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {cache}\nRun embed_corpus.py first."
        )

    if args.dry_run:
        logger.info("Dry run — config grid (trimmed from run_sweeps.py):")
        logger.info("IVFFlat: %s", IVF_FLAT_GRID)
        logger.info("IVFPQ  : %s", IVF_PQ_GRID)
        logger.info("HNSW   : %s", HNSW_GRID)
        logger.info("Output : %s", args.output)
        return

    logger.info("Loading cached embeddings from %s", cache)
    ec = load_or_embed(embed_config)

    dataset = _make_real_dataset(ec.corpus, ec.queries)
    embedding_source = f"wikipedia_{args.model.replace('-', '_')}"

    logger.info("Computing / loading ground truth...")
    gt_indices, _ = get_ground_truth(dataset.corpus, dataset.queries, k=10)
    gt_key = _cache_key(dataset.corpus, dataset.queries)

    bench_config = BenchmarkConfig(embedding_source=embedding_source)

    all_results = []

    if args.index in ("flat", "all"):
        logger.info("Running IndexFlatL2 baseline...")
        result = run_benchmark(
            index=FlatL2Index(),
            dataset=dataset,
            gt_indices=gt_indices,
            gt_cache_key=gt_key,
            config=bench_config,
            save_raw_latencies=True,
        )
        all_results.append(result)
        _append_results([result], args.output)

    if args.index in ("ivf_flat", "all"):
        results = sweep_ivf_flat(dataset, gt_indices, gt_key, bench_config, dry_run=False, output=args.output)
        all_results.extend(results)

    if args.index in ("ivf_pq", "all"):
        results = sweep_ivf_pq(dataset, gt_indices, gt_key, bench_config, dry_run=False, output=args.output)
        all_results.extend(results)

    if args.index in ("hnsw", "all"):
        results = sweep_hnsw(dataset, gt_indices, gt_key, bench_config, dry_run=False, output=args.output)
        all_results.extend(results)

    logger.info("Done. %d total rows written to %s", len(all_results), args.output)


if __name__ == "__main__":
    main()
