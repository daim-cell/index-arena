"""Goal 1 entrypoint: benchmark IndexFlatL2 on synthetic Gaussian data."""
from __future__ import annotations

import csv
import dataclasses
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark import FIELDNAMES, BenchmarkConfig, run_benchmark
from data_gen import DataGenConfig, generate_dataset
from ground_truth import _cache_key, get_ground_truth
from indexes import FlatL2Index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = Path("results/runs/flat_results.csv")


def _append_result(result) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row = dataclasses.asdict(result)
        writer.writerow(row)
    logger.info("Result appended to %s", RESULTS_FILE)


def main() -> None:
    data_config = DataGenConfig()
    logger.info("Generating dataset...")
    dataset = generate_dataset(data_config)

    logger.info("Computing / loading ground truth...")
    gt_indices, _ = get_ground_truth(dataset.corpus, dataset.queries, k=10)
    key = _cache_key(dataset.corpus, dataset.queries)

    bench_config = BenchmarkConfig()
    index = FlatL2Index()

    logger.info("Running IndexFlatL2 benchmark...")
    result = run_benchmark(
        index=index,
        dataset=dataset,
        gt_indices=gt_indices,
        gt_cache_key=key,
        config=bench_config,
        save_raw_latencies=True,
    )

    _append_result(result)
    logger.info(
        "Done — recall@10=%.4f, p50=%.2f ms, QPS=%.1f",
        result.recall_at_10,
        result.p50_ms,
        result.qps,
    )


if __name__ == "__main__":
    main()
