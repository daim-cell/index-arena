"""Step 1 — Download Wikipedia paragraphs and embed via OpenAI API.

Run once. The resulting .npz is cached under data/processed/ and never re-created.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env from repo root if present

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedder import EmbedderConfig, _cache_path, _load_wikipedia_texts, estimate_tokens, load_or_embed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed Wikipedia corpus via OpenAI API and cache to disk."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count tokens and estimate cost without making any API calls.",
    )
    parser.add_argument("--n-corpus", type=int, default=100_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    parser.add_argument("--model", type=str, default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    config = EmbedderConfig(
        model=args.model,
        n_corpus=args.n_corpus,
        n_queries=args.n_queries,
        batch_size=args.batch_size,
    )
    cache = _cache_path(config)

    if cache.exists():
        logger.info("Already cached at %s — nothing to do.", cache)
        return

    if args.dry_run:
        logger.info("Dry run: loading Wikipedia texts to estimate token count...")
        corpus_texts, query_texts = _load_wikipedia_texts(config)
        all_texts = corpus_texts + query_texts
        token_count = estimate_tokens(all_texts)
        cost = token_count / 1_000_000 * 0.02
        logger.info("Corpus paragraphs: %d, Query paragraphs: %d", len(corpus_texts), len(query_texts))
        logger.info("Estimated tokens : %d", token_count)
        logger.info("Estimated cost   : $%.3f (at $0.02 / 1M tokens)", cost)
        logger.info("Model            : %s", config.model)
        logger.info("Output cache     : %s", cache)
        return

    load_or_embed(config)
    logger.info("Done. Cache written to %s", cache)


if __name__ == "__main__":
    main()
