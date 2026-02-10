from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import cv2
import numpy as np

from ..config import get_settings
from ..diagnostics import RollingMetric
from ..scrfd_detector import FaceBox
from .face_alignment import FaceAligner
from .face_recognizer import ArcFaceRecognizer
from .watchlist import WatchlistManager
from .detector_factory import DetectorProtocol, create_detector
from .face_engine import IntegratedFaceEngine, FaceRecord

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
    last_known_name: str
    last_known_similarity: float
    last_known_at: float


class VisionService:
    def __init__(self) -> None:
        settings = get_settings()
        self.engine_mode = settings.face_engine_mode
        self.integrated_engine = IntegratedFaceEngine(settings) if self.engine_mode == "integrated" else None

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

        self._diagnostics_enabled = settings.diagnostics_mode
        self._diagnostics_every_n = settings.diagnostics_log_every_n_frames
        self._detect_embed_metric = RollingMetric()
        self._match_metric = RollingMetric()
        self._draw_metric = RollingMetric()

        logger.info("Face engine mode: %s", self.engine_mode)

    def status(self) -> VisionStatus:
        if self.engine_mode == "integrated" and self.integrated_engine is not None:
            provider = self.integrated_engine.provider
            last_error = self.integrated_engine.error
            return VisionStatus(
                scrfd_ready=self.integrated_engine.ready,
                arcface_ready=self.integrated_engine.ready,
                provider=provider,
                last_error=last_error,
            )

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
        if self.engine_mode == "integrated" and self.integrated_engine is not None:
            return {
                "provider": self.integrated_engine.provider,
                "scrfd": {
                    "ready": self.integrated_engine.ready,
                    "error": self.integrated_engine.error,
                    "path": "insightface/buffalo_l",
                    "size_bytes": 0,
                    "is_lfs_pointer": False,
                },
                "arcface": {
                    "ready": self.integrated_engine.ready,
                    "error": self.integrated_engine.error,
                    "path": "insightface/buffalo_l",
                    "size_bytes": 0,
                    "is_lfs_pointer": False,
                },
                "watchlist_error": None,
            }

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
        if self.engine_mode == "integrated":
            settings = get_settings()
            return {
                "backend": "insightface-integrated",
                "model_name": settings.face_model_name,
                "det_size": list(settings.face_active_det_size),
                "provider": self.integrated_engine.provider if self.integrated_engine else "UNKNOWN",
                "ready": self.integrated_engine.ready if self.integrated_engine else False,
                "error": self.integrated_engine.error if self.integrated_engine else "Engine unavailable",
            }
        return self.detector.io_report()

    def _integrated_to_result(self, record: FaceRecord) -> FaceResult:
        return FaceResult(
            box=record.box,
            score=record.score,
            label=record.label,
            similarity=record.similarity,
            quality=record.quality,
            landmarks=record.landmarks,
        )

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
        self, frame_bgr: np.ndarray, short_side_target: int, resize_mode: str
    ) -> tuple[np.ndarray, float, float]:
        if resize_mode == "off":
            return frame_bgr, 1.0, 1.0
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

    def _analyze_legacy(self, camera_id: int, frame_bgr: np.ndarray, force_detect: bool = False) -> list[FaceResult]:
        settings = get_settings()
        detect_embed_start = time.perf_counter()
        counter = self._frame_counter.get(camera_id, 0) + 1
        self._frame_counter[camera_id] = counter
        if not force_detect and counter % settings.active_detect_every_n != 0:
            return self._last_faces.get(camera_id, [])
        if not self.detector.status().ready:
            self._last_error = self.detector.status().error
            return []
        resized_frame, scale_x, scale_y = self._resize_for_inference(
            frame_bgr, settings.inference_short_side, settings.active_resize_mode
        )
        faces = self.detector.detect(resized_frame)
        scaled_faces = [self._scale_face(face, scale_x, scale_y) for face in faces]
        min_area = settings.active_min_face_area
        filtered: list[tuple[FaceBox, FaceBox]] = []
        for face, scaled_face in zip(faces, scaled_faces):
            width = max(0, scaled_face.x2 - scaled_face.x1)
            height = max(0, scaled_face.y2 - scaled_face.y1)
            if width * height < min_area:
                continue
            filtered.append((face, scaled_face))
        results: list[FaceResult] = []
        now = time.time()
        cache = self._match_cache.get(camera_id, [])
        ttl = max(settings.embedding_cache_ttl_seconds, 0.1)
        cache = [entry for entry in cache if now - entry.updated_at <= ttl * 3]

        needs_match: list[int] = []
        cached_results: dict[int, tuple[str, float]] = {}
        cache_indices: dict[int, int | None] = {}

        for idx, (_, face) in enumerate(filtered):
            cache_idx, entry = self._find_cached_match(
                cache,
                [face.x1, face.y1, face.x2, face.y2],
                settings.match_iou_threshold,
            )
            cache_indices[idx] = cache_idx
            if entry and (now - entry.updated_at) <= ttl:
                cached_results[idx] = (entry.name, entry.similarity)
            else:
                needs_match.append(idx)

        max_matches = settings.active_max_face_matches
        sorted_indices = sorted(needs_match, key=lambda i: filtered[i][1].score, reverse=True)
        if sorted_indices and len(sorted_indices) > max_matches:
            start = self._round_robin_index.get(camera_id, 0) % len(sorted_indices)
            selected = [sorted_indices[(start + offset) % len(sorted_indices)] for offset in range(max_matches)]
            self._round_robin_index[camera_id] = (start + max_matches) % len(sorted_indices)
        else:
            selected = sorted_indices

        selected_set = set(selected)
        match_total = 0.0
        for idx, (_, face) in enumerate(filtered):
            name = "Unknown"
            similarity = 0.0
            if idx in cached_results:
                name, similarity = cached_results[idx]
                cache_idx = cache_indices.get(idx)
                if cache_idx is not None:
                    cache_entry = cache[cache_idx]
                    cache_entry.updated_at = now
                    if name != "Unknown":
                        cache_entry.last_known_at = now
            elif idx in selected_set:
                detected_face = filtered[idx][0]
                if detected_face.landmarks is not None and self.recognizer.session is not None:
                    aligned = self.aligner.align(resized_frame, detected_face.landmarks)
                    if aligned is not None:
                        embedding = self.recognizer.embed(aligned.aligned)
                        if embedding is not None:
                            match_start = time.perf_counter()
                            best_name, best_score = self.watchlist.best_match(embedding)
                            match_total += time.perf_counter() - match_start
                            if best_score >= settings.watchlist_match_threshold:
                                name, similarity = best_name, best_score
                            else:
                                name, similarity = "Unknown", best_score
                cache_entry = CachedMatch(
                    box=[face.x1, face.y1, face.x2, face.y2],
                    name=name,
                    similarity=similarity,
                    updated_at=now,
                    last_known_name="Unknown",
                    last_known_similarity=0.0,
                    last_known_at=0.0,
                )
                if name != "Unknown":
                    cache_entry.last_known_name = name
                    cache_entry.last_known_similarity = similarity
                    cache_entry.last_known_at = now
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
        detect_embed_total = time.perf_counter() - detect_embed_start
        detect_embed_total = max(detect_embed_total - match_total, 0.0)
        if self._diagnostics_enabled:
            self._detect_embed_metric.update(detect_embed_total)
            self._match_metric.update(match_total)
            if self._detect_embed_metric.should_log(self._diagnostics_every_n):
                logger.info(
                    "Diagnostics: detect_embed_avg_ms=%.2f match_avg_ms=%.2f camera_id=%s",
                    self._detect_embed_metric.average_ms(),
                    self._match_metric.average_ms(),
                    camera_id,
                )
        return results

    def analyze_frame(self, camera_id: int, frame_bgr: np.ndarray, force_detect: bool = False) -> list[FaceResult]:
        if self.engine_mode == "integrated" and self.integrated_engine is not None:
            records = self.integrated_engine.analyze_frame(camera_id, frame_bgr, force_detect=force_detect)
            results = [self._integrated_to_result(record) for record in records]
            self._last_faces[camera_id] = results
            return results
        return self._analyze_legacy(camera_id, frame_bgr, force_detect=force_detect)

    def annotate(self, frame_bgr: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
        draw_start = time.perf_counter()
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
        draw_total = time.perf_counter() - draw_start
        if self._diagnostics_enabled:
            self._draw_metric.update(draw_total)
            if self._draw_metric.should_log(self._diagnostics_every_n):
                logger.info("Diagnostics: draw_show_avg_ms=%.2f", self._draw_metric.average_ms())
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
        if self.engine_mode == "integrated" and self.integrated_engine is not None:
            self.integrated_engine.rebuild_face_db()
            return
        self.watchlist.reload()

    def rebuild_face_db(self) -> dict[str, Any]:
        if self.engine_mode == "integrated" and self.integrated_engine is not None:
            stats = self.integrated_engine.rebuild_face_db()
            return {
                "mode": "integrated",
                "people_processed": stats.people_processed,
                "images_scanned": stats.images_scanned,
                "images_used": stats.images_used,
                "identities": stats.identities,
            }
        self.watchlist.reload()
        return {
            "mode": "legacy",
            "people_processed": len(self.watchlist.entries),
            "images_scanned": 0,
            "images_used": 0,
            "identities": [entry.name for entry in self.watchlist.entries],
        }
