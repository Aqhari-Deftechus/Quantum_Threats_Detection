from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import cv2
import numpy as np

from ..config import get_settings
from ..scrfd_detector import FaceBox
from .face_alignment import FaceAligner
from .face_recognizer import ArcFaceRecognizer
from .watchlist import WatchlistManager
from .detector_factory import DetectorProtocol, create_detector

logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    box: list[int]
    score: float
    label: str
    similarity: float
    quality: str
    landmarks: list[list[float]] | None


@dataclass
class VisionStatus:
    scrfd_ready: bool
    arcface_ready: bool
    provider: str
    last_error: str | None


class VisionService:
    def __init__(self) -> None:
        settings = get_settings()
        self.detector: DetectorProtocol = create_detector(settings)
        self.recognizer = ArcFaceRecognizer(settings.arcface_model_path)
        self.aligner = FaceAligner()
        self.watchlist = WatchlistManager(
            settings.watchlist_dir,
            self.detector,
            self.recognizer,
            self.aligner,
            settings.watchlist_match_threshold,
        )
        self._last_error: str | None = None
        self._frame_counter: dict[int, int] = {}
        self._last_faces: dict[int, list[FaceResult]] = {}

    def status(self) -> VisionStatus:
        scrfd_status = self.detector.status()
        arcface_status = self.recognizer.status()
        provider = scrfd_status.provider if scrfd_status.provider != "UNKNOWN" else arcface_status.provider
        last_error = self._last_error or scrfd_status.error or arcface_status.error or self.watchlist.load_error
        return VisionStatus(
            scrfd_ready=scrfd_status.ready,
            arcface_ready=arcface_status.ready,
            provider=provider,
            last_error=last_error,
        )

    def model_report(self) -> dict[str, Any]:
        scrfd_status = self.detector.status()
        arcface_status = self.recognizer.status()
        return {
            "provider": scrfd_status.provider if scrfd_status.provider != "UNKNOWN" else arcface_status.provider,
            "scrfd": {
                "ready": scrfd_status.ready,
                "error": scrfd_status.error,
                "path": str(scrfd_status.model.path),
                "size_bytes": scrfd_status.model.size_bytes,
                "is_lfs_pointer": scrfd_status.model.is_lfs_pointer,
            },
            "arcface": {
                "ready": arcface_status.ready,
                "error": arcface_status.error,
                "path": str(arcface_status.model.path),
                "size_bytes": arcface_status.model.size_bytes,
                "is_lfs_pointer": arcface_status.model.is_lfs_pointer,
            },
            "watchlist_error": self.watchlist.load_error,
        }

    def scrfd_io_report(self) -> dict[str, object]:
        return self.detector.io_report()

    def _face_to_result(self, face: FaceBox, name: str, similarity: float) -> FaceResult:
        landmarks = None
        if face.landmarks is not None:
            landmarks = face.landmarks.tolist()
        return FaceResult(
            box=[face.x1, face.y1, face.x2, face.y2],
            score=face.score,
            label=name,
            similarity=similarity,
            quality=face.quality,
            landmarks=landmarks,
        )

    def analyze_frame(self, camera_id: int, frame_bgr: np.ndarray, force_detect: bool = False) -> list[FaceResult]:
        settings = get_settings()
        counter = self._frame_counter.get(camera_id, 0) + 1
        self._frame_counter[camera_id] = counter
        if not force_detect and counter % max(settings.detect_every_n_frames, 1) != 0:
            return self._last_faces.get(camera_id, [])
        if not self.detector.status().ready:
            self._last_error = self.detector.status().error
            return []
        faces = self.detector.detect(frame_bgr)
        results: list[FaceResult] = []
        for face in faces:
            name = "Unknown"
            similarity = 0.0
            if face.landmarks is not None and self.recognizer.session is not None:
                aligned = self.aligner.align(frame_bgr, face.landmarks)
                if aligned is not None:
                    embedding = self.recognizer.embed(aligned.aligned)
                    if embedding is not None:
                        name, similarity = self.watchlist.match(embedding)
                        if name != "Unknown":
                            logger.info("Recognition: %s (%.2f)", name, similarity)
            results.append(self._face_to_result(face, name, similarity))
        self._last_faces[camera_id] = results
        return results

    def annotate(self, frame_bgr: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
        annotated = frame_bgr.copy()
        for face in faces:
            x1, y1, x2, y2 = face.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"[FACE] {face.label} | SIM {face.similarity:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated

    def placeholder_frame(self, message: str = "NO SIGNAL") -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            message,
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def reload_watchlist(self) -> None:
        self.watchlist.reload()
