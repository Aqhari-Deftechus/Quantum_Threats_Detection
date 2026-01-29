from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .onnx_utils import ModelFileStatus, create_session, inspect_model_file

logger = logging.getLogger(__name__)


@dataclass
class RecognizerStatus:
    ready: bool
    error: str | None
    provider: str
    model: ModelFileStatus


class ArcFaceRecognizer:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.provider = "UNKNOWN"
        self.model_status = inspect_model_file(model_path)
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_status.exists or self.model_status.is_lfs_pointer:
            self._load_error = self.model_status.error or "ArcFace model unavailable."
            logger.error(self._load_error)
            return
        if self.model_status.size_bytes < 5_000_000:
            self._load_error = (
                "ArcFace model file is too small. Run: git lfs install; git lfs pull; verify file size."
            )
            logger.error(self._load_error)
            return
        try:
            self.session, self.provider = create_session(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info("ArcFace loaded: %s", self.model_path)
        except Exception:
            self.session = None
            self._load_error = "ArcFace model failed to load. Check ONNX runtime logs for details."
            logger.exception(self._load_error)

    def status(self) -> RecognizerStatus:
        return RecognizerStatus(
            ready=self.session is not None,
            error=self._load_error,
            provider=self.provider,
            model=self.model_status,
        )

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray | None:
        if self.session is None or self.input_name is None or self.output_name is None:
            return None
        try:
            rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
            blob = rgb.astype(np.float32) / 255.0
            blob = (blob - 0.5) / 0.5
            blob = np.transpose(blob, (2, 0, 1))[None, ...]
            output = self.session.run([self.output_name], {self.input_name: blob})[0]
            embedding = output.reshape(-1).astype(np.float32)
            norm = np.linalg.norm(embedding) + 1e-12
            return embedding / norm
        except Exception:
            logger.exception("ArcFace inference failed.")
            return None
