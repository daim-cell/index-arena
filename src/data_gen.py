from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataGenConfig:
    n_vectors: int = 100_000
    n_dims: int = 128
    n_clusters: int = 20
    cluster_spread: float = 1.0
    cluster_separation: float = 10.0
    n_queries: int = 1_000
    random_seed: int = 42


@dataclass
class Dataset:
    corpus: np.ndarray   # (n_vectors, n_dims) float32
    queries: np.ndarray  # (n_queries, n_dims) float32
    config: DataGenConfig


def generate_dataset(config: DataGenConfig = DataGenConfig()) -> Dataset:
    rng = np.random.default_rng(config.random_seed)

    centers = rng.normal(0.0, config.cluster_separation, size=(config.n_clusters, config.n_dims))

    assignments = rng.integers(0, config.n_clusters, size=config.n_vectors)
    noise = rng.normal(0.0, config.cluster_spread, size=(config.n_vectors, config.n_dims))
    corpus = (centers[assignments] + noise).astype(np.float32)

    # Queries use a fresh rng draw from the same distribution — never sampled from corpus.
    q_assignments = rng.integers(0, config.n_clusters, size=config.n_queries)
    q_noise = rng.normal(0.0, config.cluster_spread, size=(config.n_queries, config.n_dims))
    queries = (centers[q_assignments] + q_noise).astype(np.float32)

    logger.info(
        "Generated dataset: %d vectors, %d queries, %d dims, %d clusters",
        config.n_vectors,
        config.n_queries,
        config.n_dims,
        config.n_clusters,
    )
    return Dataset(corpus=corpus, queries=queries, config=config)
