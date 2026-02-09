from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import Settings
from ..scrfd_detector import FaceBox, ScrfdDetector, DetectorStatus
from .insightface_detector import InsightFaceDetector


class DetectorProtocol(Protocol):
    score_thresh: float

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        ...

    def status(self) -> DetectorStatus:
        ...

    def io_report(self) -> dict[str, object]:
        ...


def create_detector(settings: Settings) -> DetectorProtocol:
    backend = settings.scrfd_backend.lower()
    if backend == "insightface":
        return InsightFaceDetector(
            score_thresh=settings.scrfd_score_threshold,
            det_size=settings.active_det_size,
        )
    return ScrfdDetector(
        settings.scrfd_model_path,
        score_thresh=settings.scrfd_score_threshold,
        nms_thresh=settings.scrfd_nms_threshold,
    )
