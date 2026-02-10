from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from ..config import Settings
from .face_db import FaceDbBuildStats, FaceDbStore

logger = logging.getLogger(__name__)


@dataclass
class FaceRecord:
    box: list[int]
    score: float
    label: str
    similarity: float
    quality: str
    landmarks: list[list[float]] | None


class IntegratedFaceEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._app = None
        self._ready = False
        self._provider = "UNKNOWN"
        self._error: str | None = None
        self._frame_counter: dict[int, int] = {}
        self._last_faces: dict[int, list[FaceRecord]] = {}
        self._db = FaceDbStore(
            settings.face_dataset_dir_resolved,
            settings.face_db_cache_path_resolved,
            settings.face_enroll_active_det_size,
            settings.face_enroll_det_conf_thresh,
            settings.face_enroll_upscale_for_det,
            settings.face_enroll_min_face_area,
            settings.face_enroll_debug_dir,
        )
        self._init_app()
        if self._ready:
            cache_loaded = self._db.load_cache()
            if not cache_loaded:
                logger.info(
                    "Face DB cache rebuilt: reason=%s",
                    self._db.last_cache_rebuild_reason or "cache_load_failed",
                )
                self.rebuild_face_db()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def identity_count(self) -> int:
        return len(self._db.embeddings)

    def _init_app(self) -> None:
        available = ort.get_available_providers()
        logger.info("ONNXRuntime providers at startup: %s", available)

        wants_cuda = self.settings.face_ctx_id == 0
        has_cuda = "CUDAExecutionProvider" in available
        if wants_cuda and not has_cuda:
            message = "GPU requested (QTD_FACE_CTX_ID=0) but CUDAExecutionProvider is unavailable."
            if self.settings.face_fail_if_no_cuda:
                self._error = message + " Failing because QTD_FACE_FAIL_IF_NO_CUDA=true."
                logger.error(self._error)
                self._ready = False
                return
            logger.warning("%s Falling back to CPU.", message)

        providers = [provider for provider in self.settings.face_active_providers if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")

        from insightface.app import FaceAnalysis

        try:
            self._app = FaceAnalysis(name=self.settings.face_model_name, providers=providers)
            ctx_id = self.settings.face_ctx_id
            if ctx_id == 0 and "CUDAExecutionProvider" not in providers:
                ctx_id = -1
            self._app.prepare(ctx_id=ctx_id, det_size=self.settings.face_active_det_size)
            self._provider = providers[0]
            self._ready = True
            self._error = None
            model_route = getattr(self._app, "models", {})
            provider_map: dict[str, object] = {}
            if isinstance(model_route, dict):
                for name, model in model_route.items():
                    session = getattr(model, "session", None)
                    if session is not None and hasattr(session, "get_providers"):
                        provider_map[name] = session.get_providers()
            logger.info("InsightFace session providers in use: %s", provider_map)
            logger.info(
                "Integrated face engine ready model=%s provider=%s det_size=%s",
                self.settings.face_model_name,
                self._provider,
                self.settings.face_active_det_size,
            )
        except Exception as exc:
            self._ready = False
            self._error = f"Integrated face engine init failed: {exc}"
            logger.exception("Integrated face engine initialization failed")

    def rebuild_face_db(self) -> FaceDbBuildStats:
        if not self._ready or self._app is None:
            return FaceDbBuildStats(0, 0, 0, [])
        stats = self._db.build_from_dataset(self._app)
        logger.info(
            "Face DB rebuild done people=%d images=%d used=%d enrolled=%s",
            stats.people_processed,
            stats.images_scanned,
            stats.images_used,
            stats.identities,
        )
        return stats

    def _iou(self, a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0:
            return 0.0
        return inter / denom

    def _to_record_dicts(self, faces: list[Any], scale: float, offset_x: int, offset_y: int) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for face in faces:
            bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32)
            if bbox.shape[0] < 4:
                continue
            if scale != 1.0:
                bbox = bbox / float(scale)
            bbox[0] += offset_x
            bbox[1] += offset_y
            bbox[2] += offset_x
            bbox[3] += offset_y

            landmarks = getattr(face, "kps", None)
            if landmarks is not None:
                landmarks = np.asarray(landmarks, dtype=np.float32)
                if scale != 1.0:
                    landmarks = landmarks / float(scale)
                landmarks[:, 0] += offset_x
                landmarks[:, 1] += offset_y

            records.append(
                {
                    "bbox": bbox,
                    "det_score": float(getattr(face, "det_score", 0.0)),
                    "embedding": np.asarray(getattr(face, "embedding", np.zeros((512,), dtype=np.float32))),
                    "landmarks": landmarks,
                }
            )
        return records

    def _detect_on_image(self, image_bgr: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> list[dict[str, Any]]:
        scale = self.settings.face_upscale_for_det if self.settings.face_upscale_for_det > 1.0 else 1.0
        if scale > 1.0:
            det_img = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            det_img = image_bgr
        faces = self._app.get(det_img) if self._app is not None else []
        faces = sorted(faces, key=lambda item: float(getattr(item, "det_score", 0.0)), reverse=True)
        return self._to_record_dicts(faces, scale=scale, offset_x=offset_x, offset_y=offset_y)

    def _detect_tile_scan(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        h, w = frame_bgr.shape[:2]
        cols, rows = self.settings.face_active_tile_grid
        tile_w = max(1, w // cols)
        tile_h = max(1, h // rows)
        overlap_x = int(tile_w * self.settings.face_tile_overlap)
        overlap_y = int(tile_h * self.settings.face_tile_overlap)
        records: list[dict[str, Any]] = []
        for row in range(rows):
            for col in range(cols):
                x1 = max(0, col * tile_w - overlap_x)
                y1 = max(0, row * tile_h - overlap_y)
                x2 = min(w, (col + 1) * tile_w + overlap_x)
                y2 = min(h, (row + 1) * tile_h + overlap_y)
                if x2 <= x1 or y2 <= y1:
                    continue
                tile = frame_bgr[y1:y2, x1:x2]
                records.extend(self._detect_on_image(tile, offset_x=x1, offset_y=y1))
        return records

    def _deduplicate(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sorted_records = sorted(records, key=lambda item: float(item.get("det_score", 0.0)), reverse=True)
        kept: list[dict[str, Any]] = []
        for record in sorted_records:
            box = record["bbox"]
            box_list = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            duplicate = False
            for existing in kept:
                e = existing["bbox"]
                e_list = [int(e[0]), int(e[1]), int(e[2]), int(e[3])]
                if self._iou(box_list, e_list) >= self.settings.face_tile_dedup_iou:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(record)
        return kept

    def _run_detection(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        tile_enabled = self.settings.face_tile_scan_enabled
        if not tile_enabled or self.settings.face_full_frame_det_when_tile:
            records.extend(self._detect_on_image(frame_bgr))
        if tile_enabled:
            records.extend(self._detect_tile_scan(frame_bgr))
        deduped = self._deduplicate(records)
        deduped = sorted(deduped, key=lambda item: float(item.get("det_score", 0.0)), reverse=True)
        return deduped[: max(1, self.settings.face_max_faces)]

    def analyze_frame(self, camera_id: int, frame_bgr: np.ndarray, force_detect: bool = False) -> list[FaceRecord]:
        if not self._ready:
            return self._last_faces.get(camera_id, [])

        counter = self._frame_counter.get(camera_id, 0) + 1
        self._frame_counter[camera_id] = counter
        if not force_detect and counter % self.settings.face_active_detect_every_n != 0:
            return self._last_faces.get(camera_id, [])

        h, w = frame_bgr.shape[:2]
        records = self._run_detection(frame_bgr)
        output: list[FaceRecord] = []

        for record in records:
            det_score = float(record.get("det_score", 0.0))
            if det_score < self.settings.face_det_conf_thresh:
                continue

            bbox = record["bbox"]
            x1 = int(np.clip(bbox[0], 0, w - 1))
            y1 = int(np.clip(bbox[1], 0, h - 1))
            x2 = int(np.clip(bbox[2], 0, w - 1))
            y2 = int(np.clip(bbox[3], 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < self.settings.face_min_face_area:
                continue

            emb = record.get("embedding")
            if emb is None:
                continue
            name, similarity = self._db.match(np.asarray(emb, dtype=np.float32), self.settings.face_recog_thresh)
            raw_landmarks = record.get("landmarks")
            landmarks = raw_landmarks.tolist() if raw_landmarks is not None else None
            output.append(
                FaceRecord(
                    box=[x1, y1, x2, y2],
                    score=det_score,
                    label=name,
                    similarity=float(similarity),
                    quality="MED",
                    landmarks=landmarks,
                )
            )

        self._last_faces[camera_id] = output
        return output
