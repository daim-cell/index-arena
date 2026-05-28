from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/ground_truth")


def _cache_key(corpus: np.ndarray, queries: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(corpus.tobytes())
    h.update(queries.tobytes())
    return h.hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"gt_{key}.npz"


def _compute_brute_force(corpus: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    d = corpus.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(corpus)
    logger.info("Running brute-force search: %d queries, k=%d", len(queries), k)
    distances, indices = index.search(queries, k)
    return indices, distances


def get_ground_truth(
    corpus: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, distances) of exact top-k neighbors, loading from cache or computing."""
    key = _cache_key(corpus, queries)
    path = _cache_path(key)

    if path.exists():
        data = np.load(path)
        stored_corpus_shape = tuple(int(x) for x in data["corpus_shape"])
        stored_query_shape = tuple(int(x) for x in data["query_shape"])
        stored_k = int(data["k"])
        if stored_corpus_shape != corpus.shape:
            raise ValueError(
                f"GT cache shape mismatch: stored corpus {stored_corpus_shape} != {corpus.shape}"
            )
        if stored_query_shape != queries.shape:
            raise ValueError(
                f"GT cache shape mismatch: stored queries {stored_query_shape} != {queries.shape}"
            )
        if stored_k != k:
            raise ValueError(f"GT cache k mismatch: stored {stored_k} != {k}")
        logger.info("Loaded ground truth from cache: %s", path)
        return data["indices"], data["distances"]

    logger.info("Cache miss — computing brute-force ground truth (key=%s)", key)
    indices, distances = _compute_brute_force(corpus, queries, k)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indices=indices,
        distances=distances,
        corpus_shape=np.array(corpus.shape),
        query_shape=np.array(queries.shape),
        k=np.array(k),
    )
    logger.info("Ground truth cached to %s", path)
    return indices, distances
