from __future__ import annotations

import logging
import shutil
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
    def __init__(
        self,
        dataset_dir: Path,
        cache_path: Path,
        enroll_det_size: tuple[int, int],
        enroll_det_conf_thresh: float,
        enroll_upscale_for_det: float,
        enroll_min_face_area: int,
        enroll_debug_dir: Path,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.cache_path = cache_path
        self.enroll_det_size = enroll_det_size
        self.enroll_det_conf_thresh = float(enroll_det_conf_thresh)
        self.enroll_upscale_for_det = float(enroll_upscale_for_det)
        self.enroll_min_face_area = int(enroll_min_face_area)
        self.enroll_debug_dir = enroll_debug_dir
        self.embeddings: dict[str, np.ndarray] = {}
        self.last_cache_rebuild_reason: str | None = None

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
        self.last_cache_rebuild_reason = None
        if not self.cache_path.exists():
            self.last_cache_rebuild_reason = "cache_file_missing"
            return False
        try:
            data = np.load(self.cache_path, allow_pickle=False)
            names = data["names"]
            vectors = data["embeddings"]
            if len(names) != len(vectors):
                reason = "cache_length_mismatch"
                logger.warning("Face DB cache mismatch; ignoring cache: %s", self.cache_path)
                logger.info("Face DB cache rebuilt: reason=%s", reason)
                self.last_cache_rebuild_reason = reason
                return False
            loaded: dict[str, np.ndarray] = {}
            for idx, name in enumerate(names.tolist()):
                loaded[str(name)] = self._normalize(vectors[idx])
            self.embeddings = loaded
            logger.info("Face DB cache loaded: %d identities", len(self.embeddings))
            return bool(self.embeddings)
        except Exception as exc:
            reason = f"cache_load_error:{exc.__class__.__name__}"
            logger.warning("Failed loading face DB cache: %s", self.cache_path)
            logger.info("Face DB cache rebuilt: reason=%s", reason)
            self.last_cache_rebuild_reason = reason
            return False

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_names = sorted(self.embeddings.keys())
        max_name_len = max((len(name) for name in sorted_names), default=1)
        names = np.array(sorted_names, dtype=f"<U{max_name_len}")
        vectors = np.stack([self.embeddings[name] for name in sorted_names]).astype("float32")
        np.savez(self.cache_path, names=names, embeddings=vectors)
        logger.info("Face DB cache saved: %d identities -> %s", len(names), self.cache_path)

    def _save_debug_failure(self, person_dir: Path, image_path: Path, reason: str) -> None:
        safe_person = person_dir.name or "unknown"
        target_dir = self.enroll_debug_dir / safe_person
        target_dir.mkdir(parents=True, exist_ok=True)
        copy_target = target_dir / image_path.name
        reason_target = target_dir / f"{image_path.stem}.txt"
        try:
            shutil.copy2(image_path, copy_target)
            reason_target.write_text(reason + "\n", encoding="utf-8")
        except Exception:
            logger.exception("Failed writing enrollment debug artifact for %s", image_path)

    def _detect_faces_for_enrollment(self, detector_app: object, image: np.ndarray):
        work_image = image
        if self.enroll_upscale_for_det > 1.0:
            work_image = cv2.resize(
                image,
                None,
                fx=self.enroll_upscale_for_det,
                fy=self.enroll_upscale_for_det,
                interpolation=cv2.INTER_LINEAR,
            )
        return detector_app.get(work_image), work_image

    def build_from_dataset(self, detector_app: object) -> FaceDbBuildStats:
        if not self.dataset_dir.exists() or not self.dataset_dir.is_dir():
            logger.warning("Face dataset folder not found: %s", self.dataset_dir)
            self.embeddings = {}
            return FaceDbBuildStats(0, 0, 0, [])

        db: dict[str, np.ndarray] = {}
        people_processed = 0
        images_scanned = 0
        images_used = 0

        if self.enroll_det_size and hasattr(detector_app, "prepare"):
            try:
                detector_app.prepare(ctx_id=0, det_size=self.enroll_det_size)
            except Exception:
                logger.debug("Detector prepare override failed; continuing with existing det_size.")

        for person_dir in sorted(self.dataset_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            people_processed += 1
            person_embeddings: list[np.ndarray] = []
            person_images = self._iter_image_paths(person_dir)
            for image_path in person_images:
                images_scanned += 1
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning("Unreadable face DB image: %s", image_path)
                    self._save_debug_failure(person_dir, image_path, "reason=unreadable_image")
                    continue
                faces, work_image = self._detect_faces_for_enrollment(detector_app, image)
                if len(faces) != 1:
                    logger.warning("Skip %s (faces=%d). Need exactly 1 face.", image_path, len(faces))
                    self._save_debug_failure(person_dir, image_path, f"reason=face_count_{len(faces)}")
                    continue
                emb = getattr(faces[0], "embedding", None)
                if emb is None:
                    self._save_debug_failure(person_dir, image_path, "reason=missing_embedding")
                    continue
                bbox = getattr(faces[0], "bbox", None)
                if bbox is not None and len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    if area < self.enroll_min_face_area:
                        self._save_debug_failure(
                            person_dir,
                            image_path,
                            f"reason=min_face_area area={area} threshold={self.enroll_min_face_area}",
                        )
                        continue
                score = float(getattr(faces[0], "det_score", 0.0))
                if score < self.enroll_det_conf_thresh:
                    self._save_debug_failure(
                        person_dir,
                        image_path,
                        f"reason=det_conf score={score:.4f} threshold={self.enroll_det_conf_thresh:.4f}",
                    )
                    continue
                _ = work_image  # keep for future debug extension without linter noise
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
                    len(person_images),
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
