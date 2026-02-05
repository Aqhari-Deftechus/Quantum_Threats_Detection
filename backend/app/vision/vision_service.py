from __future__ import annotations

from dataclasses import dataclass
import logging
import time
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


@dataclass
class CachedMatch:
    box: list[int]
    name: str
    similarity: float
    updated_at: float


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
        self._match_cache: dict[int, list["CachedMatch"]] = {}
        self._round_robin_index: dict[int, int] = {}

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

    def _resize_for_inference(
        self, frame_bgr: np.ndarray, short_side_target: int
    ) -> tuple[np.ndarray, float, float]:
        if short_side_target <= 0:
            return frame_bgr, 1.0, 1.0
        src_h, src_w = frame_bgr.shape[:2]
        short_side = min(src_h, src_w)
        if short_side <= short_side_target:
            return frame_bgr, 1.0, 1.0
        scale = short_side_target / float(short_side)
        resized_w = int(round(src_w * scale))
        resized_h = int(round(src_h * scale))
        resized = cv2.resize(frame_bgr, (resized_w, resized_h))
        scale_x = src_w / float(resized_w)
        scale_y = src_h / float(resized_h)
        return resized, scale_x, scale_y

    def _scale_face(self, face: FaceBox, scale_x: float, scale_y: float) -> FaceBox:
        if scale_x == 1.0 and scale_y == 1.0:
            return face
        landmarks = None
        if face.landmarks is not None:
            landmarks = face.landmarks.copy()
            landmarks[:, 0] = landmarks[:, 0] * scale_x
            landmarks[:, 1] = landmarks[:, 1] * scale_y
        return FaceBox(
            x1=int(round(face.x1 * scale_x)),
            y1=int(round(face.y1 * scale_y)),
            x2=int(round(face.x2 * scale_x)),
            y2=int(round(face.y2 * scale_y)),
            score=face.score,
            quality=face.quality,
            landmarks=landmarks,
        )

    def _iou(self, box_a: list[int], box_b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1 + 1)
        inter_h = max(0, inter_y2 - inter_y1 + 1)
        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
        area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom

    def _find_cached_match(
        self, cache: list[CachedMatch], box: list[int], iou_threshold: float
    ) -> tuple[int | None, CachedMatch | None]:
        best_idx = None
        best_match = None
        best_score = 0.0
        for idx, entry in enumerate(cache):
            score = self._iou(entry.box, box)
            if score >= iou_threshold and score > best_score:
                best_idx = idx
                best_match = entry
                best_score = score
        return best_idx, best_match

    def analyze_frame(self, camera_id: int, frame_bgr: np.ndarray, force_detect: bool = False) -> list[FaceResult]:
        settings = get_settings()
        counter = self._frame_counter.get(camera_id, 0) + 1
        self._frame_counter[camera_id] = counter
        if not force_detect and counter % max(settings.detect_every_n_frames, 1) != 0:
            return self._last_faces.get(camera_id, [])
        if not self.detector.status().ready:
            self._last_error = self.detector.status().error
            return []
        resized_frame, scale_x, scale_y = self._resize_for_inference(
            frame_bgr, settings.inference_short_side
        )
        faces = self.detector.detect(resized_frame)
        scaled_faces = [self._scale_face(face, scale_x, scale_y) for face in faces]
        results: list[FaceResult] = []
        now = time.time()
        cache = self._match_cache.get(camera_id, [])
        ttl = max(settings.embedding_cache_ttl_seconds, 0.1)
        cache = [entry for entry in cache if now - entry.updated_at <= ttl * 3]

        needs_match: list[int] = []
        cached_results: dict[int, tuple[str, float]] = {}
        cache_indices: dict[int, int | None] = {}

        for idx, face in enumerate(scaled_faces):
            cache_idx, entry = self._find_cached_match(cache, [face.x1, face.y1, face.x2, face.y2], 0.5)
            cache_indices[idx] = cache_idx
            if entry and (now - entry.updated_at) <= ttl:
                cached_results[idx] = (entry.name, entry.similarity)
            else:
                needs_match.append(idx)

        max_matches = max(settings.max_face_matches_per_cycle, 1)
        sorted_indices = sorted(needs_match, key=lambda i: scaled_faces[i].score, reverse=True)
        if sorted_indices and len(sorted_indices) > max_matches:
            start = self._round_robin_index.get(camera_id, 0) % len(sorted_indices)
            selected = [sorted_indices[(start + offset) % len(sorted_indices)] for offset in range(max_matches)]
            self._round_robin_index[camera_id] = (start + max_matches) % len(sorted_indices)
        else:
            selected = sorted_indices

        selected_set = set(selected)
        for idx, face in enumerate(scaled_faces):
            name = "Unknown"
            similarity = 0.0
            if idx in cached_results:
                name, similarity = cached_results[idx]
            elif idx in selected_set:
                detected_face = faces[idx]
                if detected_face.landmarks is not None and self.recognizer.session is not None:
                    aligned = self.aligner.align(resized_frame, detected_face.landmarks)
                    if aligned is not None:
                        embedding = self.recognizer.embed(aligned.aligned)
                        if embedding is not None:
                            name, similarity = self.watchlist.match(embedding)
                            if name != "Unknown":
                                logger.info("Recognition: %s (%.2f)", name, similarity)
                cache_entry = CachedMatch(
                    box=[face.x1, face.y1, face.x2, face.y2],
                    name=name,
                    similarity=similarity,
                    updated_at=now,
                )
                cache_idx = cache_indices.get(idx)
                if cache_idx is not None:
                    cache[cache_idx] = cache_entry
                else:
                    cache.append(cache_entry)
            else:
                name = "SCANNING"
                similarity = 0.0
            results.append(self._face_to_result(face, name, similarity))

        self._match_cache[camera_id] = cache
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
