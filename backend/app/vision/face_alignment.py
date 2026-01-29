from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceAlignmentResult:
    aligned: np.ndarray
    matrix: np.ndarray


class FaceAligner:
    def __init__(self, output_size: int = 112) -> None:
        self.output_size = output_size
        self._dst = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        if output_size != 112:
            scale = output_size / 112.0
            self._dst *= scale

    def align(self, frame_bgr: np.ndarray, landmarks: np.ndarray) -> FaceAlignmentResult | None:
        if landmarks.shape != (5, 2):
            return None
        src = landmarks.astype(np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(src, self._dst, method=cv2.LMEDS)
        if matrix is None:
            return None
        aligned = cv2.warpAffine(frame_bgr, matrix, (self.output_size, self.output_size))
        return FaceAlignmentResult(aligned=aligned, matrix=matrix)
