from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EmbedderConfig:
    model: str = "text-embedding-3-small"
    n_corpus: int = 100_000
    n_queries: int = 1_000
    batch_size: int = 500
    normalize: bool = True
    cache_dir: Path = Path("data/processed")
    wikipedia_config: str = "20220301.en"


@dataclass
class EmbeddedCorpus:
    corpus: np.ndarray    # (n_corpus, n_dims) float32
    queries: np.ndarray   # (n_queries, n_dims) float32
    corpus_texts: list[str]
    query_texts: list[str]
    model: str
    n_dims: int


def _cache_path(config: EmbedderConfig) -> Path:
    slug = config.model.replace("/", "_").replace("-", "_")
    return config.cache_dir / f"wikipedia_{slug}_{config.n_corpus}c_{config.n_queries}q.npz"


def _load_cache(path: Path) -> EmbeddedCorpus:
    data = np.load(path, allow_pickle=True)
    return EmbeddedCorpus(
        corpus=data["corpus"],
        queries=data["queries"],
        corpus_texts=data["corpus_texts"].tolist(),
        query_texts=data["query_texts"].tolist(),
        model=str(data["model"]),
        n_dims=int(data["n_dims"]),
    )


def _save_cache(path: Path, ec: EmbeddedCorpus) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        corpus=ec.corpus,
        queries=ec.queries,
        corpus_texts=np.array(ec.corpus_texts, dtype=object),
        query_texts=np.array(ec.query_texts, dtype=object),
        model=np.array(ec.model),
        n_dims=np.array(ec.n_dims),
    )
    logger.info("Embeddings cached to %s", path)


def _load_wikipedia_texts(config: EmbedderConfig) -> tuple[list[str], list[str]]:
    from datasets import load_dataset  # type: ignore[import]

    need = config.n_corpus + config.n_queries
    logger.info(
        "Streaming Wikipedia (%s), collecting %d paragraphs...", config.wikipedia_config, need
    )
    ds = load_dataset("wikipedia", config.wikipedia_config, split="train", streaming=True)
    texts: list[str] = []
    skipped = 0
    for row in ds:
        paras = [p.strip() for p in row["text"].split("\n") if len(p.strip()) > 50]
        if paras:
            texts.append(paras[0])
        else:
            skipped += 1
        if len(texts) >= need:
            break

    if len(texts) < need:
        raise RuntimeError(
            f"Wikipedia stream exhausted after {len(texts)} paragraphs (need {need})"
        )
    logger.info("Collected %d paragraphs (%d articles skipped for short content)", len(texts), skipped)
    return texts[: config.n_corpus], texts[config.n_corpus : need]


def estimate_tokens(texts: list[str]) -> int:
    import tiktoken  # type: ignore[import]

    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(t)) for t in texts)


def _embed_texts(texts: list[str], config: EmbedderConfig) -> np.ndarray:
    from openai import OpenAI  # type: ignore[import]

    api_key_env = "OPENAI_API_KEY"
    import os
    if not os.environ.get(api_key_env):
        raise EnvironmentError(
            f"{api_key_env} is not set. Copy .env.example → .env and add your key."
        )

    client = OpenAI()
    all_embeddings: list[list[float]] = []

    for i in tqdm(range(0, len(texts), config.batch_size), desc="Embedding batches"):
        batch = texts[i : i + config.batch_size]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=config.model, input=batch)
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                wait = 2**attempt
                logger.warning(
                    "API error (attempt %d/5): %s — retrying in %ds", attempt + 1, exc, wait
                )
                time.sleep(wait)
        all_embeddings.extend([item.embedding for item in resp.data])

    arr = np.array(all_embeddings, dtype=np.float32)
    if config.normalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norms, 1e-10, None)
    return arr


def load_or_embed(config: EmbedderConfig = EmbedderConfig()) -> EmbeddedCorpus:
    """Return cached embeddings if available; otherwise embed Wikipedia and cache."""
    path = _cache_path(config)

    if path.exists():
        logger.info("Cache hit — loading embeddings from %s", path)
        return _load_cache(path)

    logger.info("No cache found at %s — will embed now.", path)
    corpus_texts, query_texts = _load_wikipedia_texts(config)
    all_texts = corpus_texts + query_texts

    token_count = estimate_tokens(all_texts)
    cost = token_count / 1_000_000 * 0.02
    logger.info(
        "Token estimate: %d tokens (~$%.3f at $0.02/1M) for model=%s",
        token_count,
        cost,
        config.model,
    )
    logger.info("Starting embedding in 5 seconds — Ctrl-C to abort.")
    time.sleep(5)

    logger.info("Embedding %d corpus texts...", len(corpus_texts))
    corpus_arr = _embed_texts(corpus_texts, config)
    logger.info("Embedding %d query texts...", len(query_texts))
    query_arr = _embed_texts(query_texts, config)

    n_dims = corpus_arr.shape[1]
    ec = EmbeddedCorpus(
        corpus=corpus_arr,
        queries=query_arr,
        corpus_texts=corpus_texts,
        query_texts=query_texts,
        model=config.model,
        n_dims=n_dims,
    )
    _save_cache(path, ec)
    return ec
