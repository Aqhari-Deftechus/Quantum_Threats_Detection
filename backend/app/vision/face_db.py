from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceDbBuildStats:
    people_processed: int
    images_scanned: int
    images_used: int
    identities: list[str]


class FaceDbStore:
    def __init__(self, dataset_dir: Path, cache_path: Path) -> None:
        self.dataset_dir = dataset_dir
        self.cache_path = cache_path
        self.embeddings: dict[str, np.ndarray] = {}

    def _iter_image_paths(self, person_dir: Path) -> list[Path]:
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(person_dir.glob(pattern))
        return sorted(paths)

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        vec = embedding.astype("float32")
        return vec / (np.linalg.norm(vec) + 1e-12)

    def load_cache(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            data = np.load(self.cache_path, allow_pickle=False)
            names = data["names"]
            vectors = data["embeddings"]
            if len(names) != len(vectors):
                logger.warning("Face DB cache mismatch; ignoring cache: %s", self.cache_path)
                return False
            loaded: dict[str, np.ndarray] = {}
            for idx, name in enumerate(names.tolist()):
                loaded[str(name)] = self._normalize(vectors[idx])
            self.embeddings = loaded
            logger.info("Face DB cache loaded: %d identities from %s", len(self.embeddings), self.cache_path)
            return bool(self.embeddings)
        except Exception:
            logger.exception("Failed loading face DB cache: %s", self.cache_path)
            return False

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        names = np.array(sorted(self.embeddings.keys()), dtype=object)
        vectors = np.stack([self.embeddings[name] for name in names.tolist()]).astype("float32")
        np.savez(self.cache_path, names=names, embeddings=vectors)
        logger.info("Face DB cache saved: %d identities -> %s", len(names), self.cache_path)

    def build_from_dataset(self, detector_app: object) -> FaceDbBuildStats:
        if not self.dataset_dir.exists() or not self.dataset_dir.is_dir():
            logger.warning("Face dataset folder not found: %s", self.dataset_dir)
            self.embeddings = {}
            return FaceDbBuildStats(0, 0, 0, [])

        db: dict[str, np.ndarray] = {}
        people_processed = 0
        images_scanned = 0
        images_used = 0

        for person_dir in sorted(self.dataset_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            people_processed += 1
            person_embeddings: list[np.ndarray] = []
            for image_path in self._iter_image_paths(person_dir):
                images_scanned += 1
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning("Unreadable face DB image: %s", image_path)
                    continue
                faces = detector_app.get(image)
                if len(faces) != 1:
                    logger.warning("Skip %s (faces=%d). Need exactly 1 face.", image_path, len(faces))
                    continue
                emb = getattr(faces[0], "embedding", None)
                if emb is None:
                    continue
                person_embeddings.append(self._normalize(np.asarray(emb, dtype="float32")))
                images_used += 1
            if person_embeddings:
                stacked = np.stack(person_embeddings).astype("float32")
                mean_emb = self._normalize(np.mean(stacked, axis=0))
                db[person_dir.name] = mean_emb
                logger.info(
                    "Face DB enrolled: %s (%d/%d images)",
                    person_dir.name,
                    len(person_embeddings),
                    len(self._iter_image_paths(person_dir)),
                )
            else:
                logger.warning("No valid enrollment images for %s", person_dir.name)

        self.embeddings = db
        if self.embeddings:
            self.save_cache()
        return FaceDbBuildStats(people_processed, images_scanned, images_used, sorted(self.embeddings.keys()))

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str, float]:
        if not self.embeddings:
            return "Unknown", 0.0
        probe = self._normalize(embedding)
        best_name = "Unknown"
        best_score = 0.0
        for name, db_emb in self.embeddings.items():
            score = float(np.dot(probe, db_emb))
            if score > best_score:
                best_score = score
                best_name = name
        if best_score < threshold:
            return "Unknown", best_score
        return best_name, best_score
