from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from ..scrfd_detector import DetectorStatus, FaceBox
from .onnx_utils import ModelFileStatus, select_providers

logger = logging.getLogger(__name__)


@dataclass
class _InsightFaceState:
    ready: bool
    error: str | None
    provider: str


class InsightFaceDetector:
    def __init__(self, score_thresh: float, det_size: tuple[int, int]) -> None:
        self.score_thresh = score_thresh
        self.det_size = det_size
        self._state = _InsightFaceState(ready=False, error=None, provider="UNKNOWN")
        self._app = None
        self._init_app()

    def _init_app(self) -> None:
        providers = select_providers()
        try:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(name="buffalo_l", providers=providers)
            ctx_id = 0 if providers and providers[0] == "CUDAExecutionProvider" else -1
            self._app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            self._state = _InsightFaceState(ready=True, error=None, provider=providers[0] if providers else "CPU")
        except Exception as exc:
            logger.exception("InsightFace initialization failed.")
            self._state = _InsightFaceState(ready=False, error=f"InsightFace init failed: {exc}", provider="UNKNOWN")

    def status(self) -> DetectorStatus:
        model_status = ModelFileStatus(
            path=Path("insightface/buffalo_l"),
            exists=True,
            size_bytes=0,
            is_lfs_pointer=False,
            error=self._state.error,
        )
        return DetectorStatus(
            ready=self._state.ready,
            error=self._state.error,
            provider=self._state.provider,
            model=model_status,
        )

    def _quality(self, face: np.ndarray) -> str:
        h, w = face.shape[:2]
        size_score = min(h, w)
        blur_score = 0.0
        if size_score > 120 and blur_score > 150:
            return "HIGH"
        if size_score > 60 and blur_score > 60:
            return "MED"
        return "LOW"

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        if not self._state.ready or self._app is None:
            return []
        faces = self._app.get(frame_bgr)
        results: list[FaceBox] = []
        for face in faces:
            score = float(getattr(face, "det_score", 0.0))
            if score < self.score_thresh:
                continue
            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                continue
            landmarks = getattr(face, "kps", None)
            results.append(
                FaceBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    quality="MED",
                    landmarks=landmarks,
                )
            )
        return results

    def io_report(self) -> dict[str, object]:
        return {
            "backend": "insightface",
            "model_name": "buffalo_l",
            "det_size": list(self.det_size),
            "provider": self._state.provider,
            "ready": self._state.ready,
            "error": self._state.error,
        }
