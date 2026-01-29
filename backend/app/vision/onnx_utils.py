from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import onnxruntime as ort

logger = logging.getLogger(__name__)


@dataclass
class ModelFileStatus:
    path: Path
    exists: bool
    size_bytes: int
    is_lfs_pointer: bool
    error: str | None = None


def inspect_model_file(path: Path) -> ModelFileStatus:
    exists = path.exists()
    size_bytes = 0
    is_lfs_pointer = False
    error = None
    if not exists:
        error = f"Missing model file: {path}"
        return ModelFileStatus(path=path, exists=False, size_bytes=0, is_lfs_pointer=False, error=error)
    try:
        size_bytes = path.stat().st_size
        with path.open("rb") as handle:
            first_line = handle.readline().strip()
        if first_line == b"version https://git-lfs.github.com/spec/v1" or size_bytes < 1_000_000:
            is_lfs_pointer = True
            error = (
                "Model file is a Git LFS pointer. Run: git lfs install; git lfs pull; verify file size."
            )
    except OSError as exc:
        error = f"Failed to read model file: {exc}"
    return ModelFileStatus(
        path=path,
        exists=exists,
        size_bytes=size_bytes,
        is_lfs_pointer=is_lfs_pointer,
        error=error,
    )


def select_providers() -> list[str]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        logger.info("ONNXRuntime provider: CUDAExecutionProvider")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.info("ONNXRuntime provider: CPUExecutionProvider")
    return ["CPUExecutionProvider"]


def create_session(model_path: Path) -> tuple[ort.InferenceSession, str]:
    providers = select_providers()
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
        return session, providers[0]
    except Exception:
        logger.warning("Falling back to CPUExecutionProvider for %s", model_path)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return session, "CPUExecutionProvider"
