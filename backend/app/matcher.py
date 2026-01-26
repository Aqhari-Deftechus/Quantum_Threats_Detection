from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class MatcherStatus:
    mode: str
    index_status: str


class Matcher:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.dimension = self.settings.matcher_dimension
        self.faiss_available = False
        self.faiss_index = None
        self.mode = "degraded_cosine"
        self.index_status = "complete"
        self._load_faiss()
        self._self_test()

    def _load_faiss(self) -> None:
        if not self.settings.matcher_faiss_enabled:
            return
        try:
            import faiss

            self.faiss_available = True
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.mode = "faiss_exact"
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAISS unavailable, degraded to cosine: %s", exc)
            self.faiss_available = False
            self.mode = "degraded_cosine"

    def _self_test(self) -> None:
        rng = np.random.default_rng(42)
        vectors = rng.random((10, self.dimension)).astype("float32")
        vectors = self._normalize(vectors)
        query = vectors[0:1]
        brute_scores = np.dot(vectors, query.T).reshape(-1)

        if self.faiss_available and self.faiss_index is not None:
            self.faiss_index.reset()
            self.faiss_index.add(vectors)
            scores, _ = self.faiss_index.search(query, 10)
            if not np.allclose(scores.reshape(-1), brute_scores, atol=1e-4):
                logger.error("FAISS self-test mismatch, falling back")
                self.mode = "degraded_cosine"

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def status(self) -> MatcherStatus:
        return MatcherStatus(mode=self.mode, index_status=self.index_status)

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        self.index_status = "in-progress"
        if self.faiss_available and self.faiss_index is not None:
            normalized = self._normalize(embeddings.astype("float32"))
            self.faiss_index.add(normalized)
            self.index_status = "complete"
        else:
            self.index_status = "degraded"
