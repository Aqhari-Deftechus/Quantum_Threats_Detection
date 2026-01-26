from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.session = None
        self.input_name: str | None = None
        self.input_shape: tuple[int, int] = (640, 640)
        self._load_session()

    def _load_session(self) -> None:
        try:
            import onnxruntime as ort

            if not self.model_path.exists():
                return
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            shape = self.session.get_inputs()[0].shape
            if isinstance(shape[2], int) and isinstance(shape[3], int):
                self.input_shape = (shape[3], shape[2])
        except Exception:
            self.session = None

    def _quality(self, face: np.ndarray) -> str:
        h, w = face.shape[:2]
        size_score = min(h, w)
        blur_score = cv2.Laplacian(face, cv2.CV_64F).var()
        if size_score > 120 and blur_score > 150:
            return "HIGH"
        if size_score > 60 and blur_score > 60:
            return "MED"
        return "LOW"

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        if self.session is None or self.input_name is None:
            return self._demo_face(frame_bgr)
        try:
            resized = cv2.resize(frame_bgr, self.input_shape)
            blob = cv2.dnn.blobFromImage(resized, 1.0 / 128.0, self.input_shape, (127.5, 127.5, 127.5))
            outputs = self.session.run(None, {self.input_name: blob})
            if not outputs:
                return []
        except Exception:
            return self._demo_face(frame_bgr)
        return self._demo_face(frame_bgr)

    def _demo_face(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        h, w = frame_bgr.shape[:2]
        size = int(min(h, w) * 0.3)
        x1 = int((w - size) / 2)
        y1 = int((h - size) / 2)
        x2 = x1 + size
        y2 = y1 + size
        face = frame_bgr[y1:y2, x1:x2]
        quality = self._quality(face) if face.size else "LOW"
        return [FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=0.5, quality=quality)]

