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
        self._embeddings = np.empty((0, self.dimension), dtype="float32")
        self._id_map: list[int] = []
        self._name_map: dict[int, str] = {}
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

    def reset(self) -> None:
        self.index_status = "in-progress"
        self._embeddings = np.empty((0, self.dimension), dtype="float32")
        self._id_map = []
        self._name_map = {}
        if self.faiss_available and self.faiss_index is not None:
            self.faiss_index.reset()
        self.index_status = "complete"

    def rebuild(self, embeddings: np.ndarray, identity_ids: list[int], identity_names: dict[int, str]) -> None:
        self.reset()
        self._name_map = dict(identity_names)
        if embeddings.size == 0:
            return
        self.add_embeddings(embeddings, identity_ids, identity_names)

    def add_embeddings(self, embeddings: np.ndarray, identity_ids: list[int], identity_names: dict[int, str]) -> None:
        self.index_status = "in-progress"
        if embeddings.shape[0] != len(identity_ids):
            raise ValueError("Embeddings and identity_ids must be the same length")
        self._name_map.update(identity_names)
        if self.faiss_available and self.faiss_index is not None:
            normalized = self._normalize(embeddings.astype("float32"))
            self.faiss_index.add(normalized)
            self._embeddings = np.vstack([self._embeddings, normalized])
            self._id_map.extend(identity_ids)
            self.index_status = "complete"
        else:
            normalized = self._normalize(embeddings.astype("float32"))
            self._embeddings = np.vstack([self._embeddings, normalized])
            self._id_map.extend(identity_ids)
            self.index_status = "degraded"

    def search(self, embeddings: np.ndarray, top_k: int = 2) -> tuple[np.ndarray, np.ndarray]:
        if self._embeddings.size == 0:
            return np.empty((embeddings.shape[0], 0), dtype="float32"), np.empty((embeddings.shape[0], 0), dtype="int64")
        normalized = self._normalize(embeddings.astype("float32"))
        if self.faiss_available and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(normalized, top_k)
            return scores, indices
        scores = np.dot(normalized, self._embeddings.T)
        indices = np.argsort(-scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(scores, indices, axis=1)
        return sorted_scores, indices

    def resolve_identity_ids(self, indices: np.ndarray) -> list[list[int]]:
        resolved: list[list[int]] = []
        for row in indices:
            ids: list[int] = []
            for idx in row:
                if idx < 0 or idx >= len(self._id_map):
                    ids.append(-1)
                else:
                    ids.append(self._id_map[int(idx)])
            resolved.append(ids)
        return resolved

    def resolve_identity_names(self, identity_id: int) -> str | None:
        return self._name_map.get(identity_id)
