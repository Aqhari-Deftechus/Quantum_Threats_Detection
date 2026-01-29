from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    quality: str


class ScrfdDetector:
    def __init__(self, model_path: Path, score_thresh: float = 0.5, nms_thresh: float = 0.4) -> None:
        self.model_path = model_path
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.session = None
        self.input_name: str | None = None
        self.input_shape: tuple[int, int] = (640, 640)
        self._feat_strides = [8, 16, 32]
        self._load_error: str | None = None
        self._load_session()

    def _looks_like_lfs_pointer(self, path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                first_line = handle.readline().strip()
            return first_line == b"version https://git-lfs.github.com/spec/v1"
        except OSError:
            return False

    def _load_session(self) -> None:
        logger = logging.getLogger(__name__)
        try:
            import onnxruntime as ort

            if not self.model_path.exists():
                self._load_error = f"SCRFD model missing at {self.model_path}"
                logger.error(self._load_error)
                return
            if self._looks_like_lfs_pointer(self.model_path):
                self._load_error = (
                    "SCRFD model is a Git LFS pointer. Run `git lfs pull` to download the real model."
                )
                logger.error(self._load_error)
                return
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            shape = self.session.get_inputs()[0].shape
            if isinstance(shape[2], int) and isinstance(shape[3], int):
                self.input_shape = (shape[3], shape[2])
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

    def _prepare(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
        input_w, input_h = self.input_shape
        resized = cv2.resize(frame_bgr, (input_w, input_h))
        scale_x = frame_bgr.shape[1] / input_w
        scale_y = frame_bgr.shape[0] / input_h
        blob = cv2.dnn.blobFromImage(resized, 1.0 / 128.0, (input_w, input_h), (127.5, 127.5, 127.5))
        return blob, scale_x, scale_y

    def _infer_num_anchors(self, scores: np.ndarray, feat_h: int, feat_w: int) -> int:
        total = scores.shape[1] if scores.ndim >= 2 else scores.shape[0]
        anchors = max(int(total // (feat_h * feat_w)), 1)
        return anchors

    def _decode(self, outputs: Iterable[np.ndarray], scale_x: float, scale_y: float, frame: np.ndarray) -> list[FaceBox]:
        outputs_list = list(outputs)
        if not outputs_list:
            return []
        use_kps = len(outputs_list) % 3 == 0
        stride_count = len(outputs_list) // (3 if use_kps else 2)
        scores_list = outputs_list[:stride_count]
        bboxes_list = outputs_list[stride_count : stride_count * 2]

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        input_w, input_h = self.input_shape

        for idx, stride in enumerate(self._feat_strides[:stride_count]):
            scores = scores_list[idx]
            bboxes = bboxes_list[idx]
            feat_h = int(input_h / stride)
            feat_w = int(input_w / stride)
            num_anchors = self._infer_num_anchors(scores, feat_h, feat_w)

            scores = scores.reshape(-1)
            bboxes = bboxes.reshape((-1, 4))

            anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w], axis=-1).astype(np.float32)
            anchor_centers = anchor_centers * stride
            anchor_centers = anchor_centers.reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = np.repeat(anchor_centers, num_anchors, axis=0)

            pos_inds = np.where(scores >= self.score_thresh)[0]
            if pos_inds.size == 0:
                continue
            scores = scores[pos_inds]
            bboxes = bboxes[pos_inds]
            anchor_centers = anchor_centers[pos_inds]

            x1 = anchor_centers[:, 0] - bboxes[:, 0]
            y1 = anchor_centers[:, 1] - bboxes[:, 1]
            x2 = anchor_centers[:, 0] + bboxes[:, 2]
            y2 = anchor_centers[:, 1] + bboxes[:, 3]
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            all_boxes.append(boxes)
            all_scores.append(scores)

        if not all_boxes:
            return []

        all_boxes_np = np.concatenate(all_boxes, axis=0)
        all_scores_np = np.concatenate(all_scores, axis=0)

        keep = self._nms(all_boxes_np, all_scores_np)
        boxes_kept = all_boxes_np[keep]
        scores_kept = all_scores_np[keep]

        faces: list[FaceBox] = []
        for box, score in zip(boxes_kept, scores_kept):
            x1 = int(max(0, box[0] * scale_x))
            y1 = int(max(0, box[1] * scale_y))
            x2 = int(min(frame.shape[1], box[2] * scale_x))
            y2 = int(min(frame.shape[0], box[3] * scale_y))
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame[y1:y2, x1:x2]
            faces.append(
                FaceBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(score),
                    quality=self._quality(face_crop),
                )
            )
        return faces

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        if self.session is None or self.input_name is None:
            return []
        try:
            blob, scale_x, scale_y = self._prepare(frame_bgr)
            outputs = self.session.run(None, {self.input_name: blob})
            return self._decode(outputs, scale_x, scale_y, frame_bgr)
        except Exception:
            return []
