from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np

from .face_alignment import FaceAligner
from .face_recognizer import ArcFaceRecognizer
from ..scrfd_detector import ScrfdDetector

logger = logging.getLogger(__name__)


@dataclass
class WatchlistEntry:
    name: str
    embedding: np.ndarray


class WatchlistManager:
    def __init__(
        self,
        watchlist_dir: Path,
        detector: ScrfdDetector,
        recognizer: ArcFaceRecognizer,
        aligner: FaceAligner,
        similarity_threshold: float,
    ) -> None:
        self.watchlist_dir = watchlist_dir
        self.detector = detector
        self.recognizer = recognizer
        self.aligner = aligner
        self.similarity_threshold = similarity_threshold
        self.entries: list[WatchlistEntry] = []
        self._load_error: str | None = None
        self.reload()

    def reload(self) -> None:
        self.entries = []
        self._load_error = None
        self.watchlist_dir.mkdir(parents=True, exist_ok=True)
        if self.detector.session is None or self.recognizer.session is None:
            self._load_error = "Models not ready for watchlist enrollment."
            logger.warning(self._load_error)
            return
        for person_dir in sorted(self.watchlist_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            embeddings: list[np.ndarray] = []
            for image_path in sorted(person_dir.glob("*.jpg")):
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning("Watchlist image unreadable: %s", image_path)
                    continue
                faces = self.detector.detect(image)
                if not faces:
                    logger.warning("No face detected in watchlist image: %s", image_path)
                    continue
                face = max(faces, key=lambda f: f.score)
                if face.landmarks is None:
                    logger.warning("No landmarks for watchlist image: %s", image_path)
                    continue
                aligned_result = self.aligner.align(image, face.landmarks)
                if aligned_result is None:
                    logger.warning("Alignment failed for watchlist image: %s", image_path)
                    continue
                embedding = self.recognizer.embed(aligned_result.aligned)
                if embedding is None:
                    logger.warning("Embedding failed for watchlist image: %s", image_path)
                    continue
                embeddings.append(embedding)
            if embeddings:
                centroid = np.mean(np.stack(embeddings), axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
                self.entries.append(WatchlistEntry(name=person_dir.name, embedding=centroid))
                logger.info("Watchlist enrolled: %s (%d images)", person_dir.name, len(embeddings))

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        if not self.entries:
            return "Unknown", 0.0
        scores = [float(np.dot(entry.embedding, embedding)) for entry in self.entries]
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        if best_score < self.similarity_threshold:
            return "Unknown", best_score
        return self.entries[best_idx].name, best_score

    @property
    def load_error(self) -> str | None:
        return self._load_error
