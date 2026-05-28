from __future__ import annotations

import abc
import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class BaseIndex(abc.ABC):
    @abc.abstractmethod
    def build(self, corpus: np.ndarray) -> None:
        """Allocate, train (if required), and add vectors."""

    @abc.abstractmethod
    def set_query_params(self, **kwargs) -> None:
        """Apply query-time parameters (nprobe, efSearch, etc.)."""

    @abc.abstractmethod
    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return indices array of shape (n_queries, k)."""

    @property
    @abc.abstractmethod
    def index_type(self) -> str: ...


class FlatL2Index(BaseIndex):
    def __init__(self) -> None:
        self._index: faiss.IndexFlatL2 | None = None

    @property
    def index_type(self) -> str:
        return "IndexFlatL2"

    def build(self, corpus: np.ndarray) -> None:
        d = corpus.shape[1]
        self._index = faiss.IndexFlatL2(d)
        self._index.add(corpus)
        logger.info("FlatL2: added %d vectors (d=%d)", corpus.shape[0], d)

    def set_query_params(self, **kwargs) -> None:
        pass

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        assert self._index is not None, "Call build() first"
        _, indices = self._index.search(queries, k)
        return indices


class IVFFlatIndex(BaseIndex):
    def __init__(self, nlist: int, nprobe: int) -> None:
        self.nlist = nlist
        self.nprobe = nprobe
        self._index: faiss.IndexIVFFlat | None = None

    @property
    def index_type(self) -> str:
        return "IndexIVFFlat"

    def build(self, corpus: np.ndarray) -> None:
        n, d = corpus.shape
        if n < self.nlist * 39:
            logger.warning(
                "IVFFlat: corpus size %d < nlist*39=%d — index may be undertrained",
                n,
                self.nlist * 39,
            )
        quantizer = faiss.IndexFlatL2(d)
        self._index = faiss.IndexIVFFlat(quantizer, d, self.nlist)
        self._index.train(corpus)
        self._index.add(corpus)
        self._index.nprobe = self.nprobe
        logger.info(
            "IVFFlat: trained+added %d vectors (nlist=%d, nprobe=%d)", n, self.nlist, self.nprobe
        )

    def set_query_params(self, **kwargs) -> None:
        if "nprobe" in kwargs:
            assert self._index is not None
            self._index.nprobe = kwargs["nprobe"]
            self.nprobe = kwargs["nprobe"]

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        assert self._index is not None, "Call build() first"
        _, indices = self._index.search(queries, k)
        return indices


class IVFPQIndex(BaseIndex):
    def __init__(self, nlist: int, m: int, nbits: int, nprobe: int) -> None:
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self._index: faiss.IndexIVFPQ | None = None

    @property
    def index_type(self) -> str:
        return "IndexIVFPQ"

    def build(self, corpus: np.ndarray) -> None:
        n, d = corpus.shape
        if d % self.m != 0:
            raise ValueError(
                f"IVFPQIndex: n_dims ({d}) must be divisible by m ({self.m})"
            )
        if n < self.nlist * 39:
            logger.warning(
                "IVFPQ: corpus size %d < nlist*39=%d — index may be undertrained",
                n,
                self.nlist * 39,
            )
        quantizer = faiss.IndexFlatL2(d)
        self._index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, self.nbits)
        self._index.train(corpus)
        self._index.add(corpus)
        self._index.nprobe = self.nprobe
        logger.info(
            "IVFPQ: trained+added %d vectors (nlist=%d, m=%d, nbits=%d, nprobe=%d)",
            n,
            self.nlist,
            self.m,
            self.nbits,
            self.nprobe,
        )

    def set_query_params(self, **kwargs) -> None:
        if "nprobe" in kwargs:
            assert self._index is not None
            self._index.nprobe = kwargs["nprobe"]
            self.nprobe = kwargs["nprobe"]

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        assert self._index is not None, "Call build() first"
        _, indices = self._index.search(queries, k)
        return indices


class HNSWFlatIndex(BaseIndex):
    def __init__(self, M: int, ef_construction: int) -> None:
        self.M = M
        self.ef_construction = ef_construction
        self._index: faiss.IndexHNSWFlat | None = None

    @property
    def index_type(self) -> str:
        return "IndexHNSWFlat"

    def build(self, corpus: np.ndarray) -> None:
        n, d = corpus.shape
        if self.M >= 32 and self.ef_construction >= 200:
            logger.warning(
                "HNSW M=%d ef_construction=%d on %d vectors may take several minutes",
                self.M,
                self.ef_construction,
                n,
            )
        self._index = faiss.IndexHNSWFlat(d, self.M)
        # efConstruction must be set before .add()
        self._index.hnsw.efConstruction = self.ef_construction
        self._index.add(corpus)
        logger.info(
            "HNSWFlat: added %d vectors (M=%d, ef_construction=%d)", n, self.M, self.ef_construction
        )

    def set_query_params(self, **kwargs) -> None:
        if "ef_search" in kwargs:
            assert self._index is not None
            self._index.hnsw.efSearch = kwargs["ef_search"]

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        assert self._index is not None, "Call build() first"
        _, indices = self._index.search(queries, k)
        return indices
