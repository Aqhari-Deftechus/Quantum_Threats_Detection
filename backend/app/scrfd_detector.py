from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort

from .vision.onnx_utils import ModelFileStatus, create_session, inspect_model_file

logger = logging.getLogger(__name__)


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    quality: str
    landmarks: np.ndarray | None


@dataclass
class DetectorStatus:
    ready: bool
    error: str | None
    provider: str
    model: ModelFileStatus


class ScrfdDetector:
    def __init__(self, model_path: Path, score_thresh: float = 0.3, nms_thresh: float = 0.4) -> None:
        self.model_path = model_path
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.input_shape: tuple[int, int] = (640, 640)
        self._feat_strides = [8, 16, 32]
        self._load_error: str | None = None
        self._last_error: str | None = None
        self._output_names: list[str] = []
        self._output_shapes: list[tuple[int | str | None, ...]] = []
        self.provider = "UNKNOWN"
        self.model_status = inspect_model_file(model_path)
        self._load_session()

    def status(self) -> DetectorStatus:
        return DetectorStatus(
            ready=self.session is not None,
            error=self._last_error or self._load_error,
            provider=self.provider,
            model=self.model_status,
        )

    def _load_session(self) -> None:
        if not self.model_status.exists or self.model_status.is_lfs_pointer:
            self._load_error = self.model_status.error or "SCRFD model unavailable."
            logger.error(self._load_error)
            return
        try:
            self.session, self.provider = create_session(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            shape = self.session.get_inputs()[0].shape
            if isinstance(shape[2], int) and isinstance(shape[3], int):
                self.input_shape = (shape[3], shape[2])
            self._output_names = [output.name for output in self.session.get_outputs()]
            self._output_shapes = [tuple(output.shape) for output in self.session.get_outputs()]
            logger.info("SCRFD input: %s %s", self.input_name, shape)
            for output_name, output_shape in zip(self._output_names, self._output_shapes):
                logger.info("SCRFD output: %s %s", output_name, output_shape)
            logger.info("SCRFD loaded: %s", self.model_path)
        except Exception:
            self.session = None
            self._load_error = "SCRFD model failed to load. Check ONNX runtime logs for details."
            logger.exception(self._load_error)

    def _quality(self, face: np.ndarray) -> str:
        h, w = face.shape[:2]
        size_score = min(h, w)
        blur_score = cv2.Laplacian(face, cv2.CV_64F).var() if face.size else 0
        if size_score > 120 and blur_score > 150:
            return "HIGH"
        if size_score > 60 and blur_score > 60:
            return "MED"
        return "LOW"

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        if boxes.size == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def _prepare(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, float, int, int]:
        input_w, input_h = self.input_shape
        src_h, src_w = frame_bgr.shape[:2]
        scale = min(input_w / src_w, input_h / src_h)
        resized_w = int(src_w * scale)
        resized_h = int(src_h * scale)
        resized = cv2.resize(frame_bgr, (resized_w, resized_h))
        canvas = np.zeros((input_h, input_w, 3), dtype=np.float32)
        pad_x = (input_w - resized_w) // 2
        pad_y = (input_h - resized_h) // 2
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized.astype(np.float32)
        canvas = (canvas - 127.5) / 128.0
        blob = np.transpose(canvas, (2, 0, 1))[None, :, :, :]
        scale_x = 1.0 / scale
        scale_y = 1.0 / scale
        return blob, scale_x, scale_y, pad_x, pad_y

    def _infer_stride_from_shape(self, shape: tuple[int | None, ...]) -> int | None:
        input_w, input_h = self.input_shape
        if len(shape) == 4:
            feat_h = shape[2]
            feat_w = shape[3]
            if isinstance(feat_h, int) and isinstance(feat_w, int) and feat_h > 0 and feat_w > 0:
                return int(input_h / feat_h)
        if len(shape) >= 2 and isinstance(shape[-2], int):
            total = int(shape[-2])
            for stride in self._feat_strides:
                feat_h = int(input_h / stride)
                feat_w = int(input_w / stride)
                if feat_h > 0 and feat_w > 0 and total % (feat_h * feat_w) == 0:
                    return stride
        return None

    def _score_from_tensor(self, scores: np.ndarray, feat_h: int, feat_w: int) -> np.ndarray:
        if scores.ndim == 4:
            _, channels, _, _ = scores.shape
            scores = scores.reshape((channels, feat_h, feat_w))
            scores = scores.transpose((1, 2, 0)).reshape((-1,))
        elif scores.ndim in (2, 3):
            scores = scores.reshape((-1,))
        else:
            raise ValueError(f"SCRFD score tensor has invalid shape: {scores.shape}")
        if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
            scores = 1.0 / (1.0 + np.exp(-scores))
        return scores

    def _bbox_from_tensor(self, bboxes: np.ndarray, feat_h: int, feat_w: int) -> np.ndarray:
        if bboxes.ndim == 4:
            _, channels, _, _ = bboxes.shape
            anchors = max(int(channels // 4), 1)
            bboxes = bboxes.reshape((anchors, 4, feat_h, feat_w))
            return bboxes.transpose((2, 3, 0, 1)).reshape((-1, 4))
        if bboxes.ndim == 3:
            return bboxes.reshape((-1, 4))
        if bboxes.ndim == 2:
            return bboxes.reshape((-1, 4))
        raise ValueError(f"SCRFD bbox tensor has invalid shape: {bboxes.shape}")

    def _kps_from_tensor(self, kps: np.ndarray, feat_h: int, feat_w: int) -> np.ndarray:
        if kps.ndim == 4:
            _, channels, _, _ = kps.shape
            anchors = max(int(channels // 10), 1)
            kps = kps.reshape((anchors, 10, feat_h, feat_w))
            return kps.transpose((2, 3, 0, 1)).reshape((-1, 10))
        if kps.ndim == 3:
            return kps.reshape((-1, 10))
        if kps.ndim == 2:
            return kps.reshape((-1, 10))
        raise ValueError(f"SCRFD kps tensor has invalid shape: {kps.shape}")

    def _split_outputs(self, outputs: list[np.ndarray]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
        scores_map: dict[int, np.ndarray] = {}
        bboxes_map: dict[int, np.ndarray] = {}
        kps_map: dict[int, np.ndarray] = {}

        for idx, output in enumerate(outputs):
            name = self._output_names[idx].lower() if idx < len(self._output_names) else ""
            shape = self._output_shapes[idx] if idx < len(self._output_shapes) else output.shape
            stride = self._infer_stride_from_shape(tuple(shape))
            if stride is None:
                continue
            if any(token in name for token in ("score", "cls", "conf")):
                scores_map[stride] = output
                continue
            if any(token in name for token in ("bbox", "box", "loc")):
                bboxes_map[stride] = output
                continue
            if any(token in name for token in ("kps", "landmark")):
                kps_map[stride] = output
                continue
            if output.ndim >= 3:
                last_dim = output.shape[-1]
                channel_dim = output.shape[1] if output.ndim == 4 else last_dim
                if last_dim == 4 or channel_dim == 4:
                    bboxes_map[stride] = output
                elif last_dim == 10 or channel_dim == 10:
                    kps_map[stride] = output
                else:
                    scores_map[stride] = output
            else:
                scores_map[stride] = output
        return scores_map, bboxes_map, kps_map

    def _decode(
        self,
        outputs: Iterable[np.ndarray],
        scale_x: float,
        scale_y: float,
        pad_x: int,
        pad_y: int,
        frame: np.ndarray,
    ) -> list[FaceBox]:
        outputs_list = list(outputs)
        if not outputs_list:
            self._last_error = "SCRFD returned no outputs."
            logger.debug("SCRFD decode skipped: no outputs")
            return []
        scores_map, bboxes_map, kps_map = self._split_outputs(outputs_list)
        use_kps = bool(kps_map)

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_kps: list[np.ndarray] = []

        input_w, input_h = self.input_shape

        for stride in self._feat_strides:
            scores = scores_map.get(stride)
            bboxes = bboxes_map.get(stride)
            if scores is None or bboxes is None:
                logger.debug("SCRFD stride %s missing scores or bboxes", stride)
                continue
            feat_h = int(input_h / stride)
            feat_w = int(input_w / stride)
            scores = self._score_from_tensor(scores, feat_h, feat_w)
            bboxes = self._bbox_from_tensor(bboxes, feat_h, feat_w)
            kps = None
            if use_kps and stride in kps_map:
                kps = self._kps_from_tensor(kps_map[stride], feat_h, feat_w)

            anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w], axis=-1).astype(np.float32)
            anchor_centers = anchor_centers * stride
            anchor_centers = anchor_centers.reshape((-1, 2))
            if anchor_centers.shape[0] == 0:
                logger.debug("SCRFD stride %s has zero anchor centers", stride)
                continue
            if scores.size % anchor_centers.shape[0] != 0:
                logger.debug(
                    "SCRFD stride %s score count mismatch: scores=%s centers=%s",
                    stride,
                    scores.size,
                    anchor_centers.shape[0],
                )
                continue
            num_anchors = max(int(scores.size // anchor_centers.shape[0]), 1)
            if num_anchors > 1:
                anchor_centers = np.repeat(anchor_centers, num_anchors, axis=0)
            if bboxes.shape[0] != anchor_centers.shape[0]:
                logger.debug(
                    "SCRFD stride %s bbox count mismatch: bboxes=%s centers=%s",
                    stride,
                    bboxes.shape[0],
                    anchor_centers.shape[0],
                )
                continue
            if use_kps and kps is not None and kps.shape[0] != anchor_centers.shape[0]:
                logger.debug(
                    "SCRFD stride %s kps count mismatch: kps=%s centers=%s",
                    stride,
                    kps.shape[0],
                    anchor_centers.shape[0],
                )
                continue

            pos_inds = np.where(scores >= self.score_thresh)[0]
            if pos_inds.size == 0:
                continue
            scores = scores[pos_inds]
            bboxes = bboxes[pos_inds]
            centers = anchor_centers[pos_inds]
            if use_kps and kps is not None:
                kps = kps[pos_inds]

            bboxes = bboxes * stride
            x1 = centers[:, 0] - bboxes[:, 0]
            y1 = centers[:, 1] - bboxes[:, 1]
            x2 = centers[:, 0] + bboxes[:, 2]
            y2 = centers[:, 1] + bboxes[:, 3]
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            all_boxes.append(boxes)
            all_scores.append(scores)
            if use_kps and kps is not None:
                kps = kps.reshape((-1, 5, 2))
                kps = kps * stride
                kps = kps + centers[:, None, :]
                all_kps.append(kps)

        if not all_boxes:
            self._last_error = "SCRFD produced no detections after decoding."
            logger.debug("SCRFD decode completed with zero faces")
            return []

        all_boxes_np = np.concatenate(all_boxes, axis=0)
        all_scores_np = np.concatenate(all_scores, axis=0)
        all_kps_np = np.concatenate(all_kps, axis=0) if all_kps else None

        keep = self._nms(all_boxes_np, all_scores_np)
        boxes_kept = all_boxes_np[keep]
        scores_kept = all_scores_np[keep]
        kps_kept = all_kps_np[keep] if all_kps_np is not None else None

        faces: list[FaceBox] = []
        for idx, (box, score) in enumerate(zip(boxes_kept, scores_kept)):
            x1 = int(max(0, (box[0] - pad_x) * scale_x))
            y1 = int(max(0, (box[1] - pad_y) * scale_y))
            x2 = int(min(frame.shape[1], (box[2] - pad_x) * scale_x))
            y2 = int(min(frame.shape[0], (box[3] - pad_y) * scale_y))
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame[y1:y2, x1:x2]
            landmarks = None
            if kps_kept is not None:
                kps = kps_kept[idx].copy()
                kps[:, 0] = (kps[:, 0] - pad_x) * scale_x
                kps[:, 1] = (kps[:, 1] - pad_y) * scale_y
                landmarks = kps
            faces.append(
                FaceBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(score),
                    quality=self._quality(face_crop),
                    landmarks=landmarks,
                )
            )
        return faces

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        if self.session is None or self.input_name is None:
            return []
        try:
            blob, scale_x, scale_y, pad_x, pad_y = self._prepare(frame_bgr)
            outputs = self.session.run(None, {self.input_name: blob})
            faces = self._decode(outputs, scale_x, scale_y, pad_x, pad_y, frame_bgr)
            if faces:
                self._last_error = None
            return faces
        except Exception as exc:
            self._last_error = f"SCRFD inference failed: {exc}"
            logger.exception("SCRFD inference failed.")
            return []

    def io_report(self) -> dict[str, object]:
        return {
            "backend": "onnx",
            "input_name": self.input_name,
            "input_shape": list(self.session.get_inputs()[0].shape) if self.session else None,
            "output_names": self._output_names,
            "output_shapes": [list(shape) for shape in self._output_shapes],
            "provider": self.provider,
        }
