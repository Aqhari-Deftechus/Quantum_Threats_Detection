from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np

from .face_alignment import FaceAligner
from .face_recognizer import ArcFaceRecognizer
from .detector_factory import DetectorProtocol

logger = logging.getLogger(__name__)


@dataclass
class WatchlistEntry:
    name: str
    embedding: np.ndarray


class WatchlistManager:
    def __init__(
        self,
        watchlist_dir: Path,
        detector: DetectorProtocol,
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
        self._failure_count = 0
        self.reload()

    def reload(self) -> None:
        self.entries = []
        self._load_error = None
        self._failure_count = 0
        self.watchlist_dir.mkdir(parents=True, exist_ok=True)
        if not self.detector.status().ready or self.recognizer.session is None:
            self._load_error = "Models not ready for watchlist enrollment."
            logger.warning(self._load_error)
            return
        image_patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for person_dir in sorted(self.watchlist_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            embeddings: list[np.ndarray] = []
            image_paths: list[Path] = []
            for pattern in image_patterns:
                image_paths.extend(person_dir.glob(pattern))
            for image_path in sorted(image_paths):
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning("Watchlist image unreadable: %s", image_path)
                    continue
                faces = self.detector.detect(image)
                if not faces:
                    provider = self.detector.status().provider
                    logger.warning(
                        "No face detected in watchlist image: %s (score_threshold=%.2f, provider=%s)",
                        image_path,
                        self.detector.score_thresh,
                        provider,
                    )
                    if self._failure_count < 3:
                        debug_faces = self.detector.detect(image)
                        logger.warning(
                            "SCRFD debug detect: faces=%d",
                            len(debug_faces),
                        )
                        if debug_faces:
                            top_face = max(debug_faces, key=lambda f: f.score)
                            logger.warning(
                                "SCRFD debug top score=%.4f box=%s",
                                top_face.score,
                                [top_face.x1, top_face.y1, top_face.x2, top_face.y2],
                            )
                        logger.warning("SCRFD IO report: %s", self.detector.io_report())
                    self._failure_count += 1
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

    def best_match(self, embedding: np.ndarray) -> tuple[str, float]:
        if not self.entries:
            return "Unknown", 0.0
        scores = [float(np.dot(entry.embedding, embedding)) for entry in self.entries]
        best_idx = int(np.argmax(scores))
        return self.entries[best_idx].name, scores[best_idx]

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        best_name, best_score = self.best_match(embedding)
        if best_score < self.similarity_threshold:
            return "Unknown", best_score
        return best_name, best_score

    @property
    def load_error(self) -> str | None:
        return self._load_error
