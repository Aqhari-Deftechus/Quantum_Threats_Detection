from __future__ import annotations

import time
from threading import Lock
from typing import Generator, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

from ..config import get_settings
from ..scrfd_detector import ScrfdDetector
from ..state import camera_registry

router = APIRouter(prefix="/cameras")

_detector_lock = Lock()
_detector: Optional[ScrfdDetector] = None


def _get_detector() -> ScrfdDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            settings = get_settings()
            _detector = ScrfdDetector(settings.scrfd_model_path)
        return _detector


def _draw_faces(frame_bgr: np.ndarray, faces: list) -> np.ndarray:
    annotated = frame_bgr.copy()
    for face in faces:
        cv2.rectangle(annotated, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
        label = f"UNKNOWN {face.score:.2f}"
        cv2.putText(
            annotated,
            label,
            (face.x1, max(face.y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _mjpeg_generator(camera_id: int) -> Generator[bytes, None, None]:
    detector = _get_detector()
    while True:
        runtime = camera_registry.snapshot_metrics(camera_id)
        #runtime = camera_registry.get(camera_id)

        if not runtime or not runtime.worker:
            time.sleep(0.1)
            continue
        latest = runtime.worker.get_latest_bgr()
        if not latest:
            time.sleep(0.01)
            continue
        _, frame_bgr = latest
        faces = detector.detect(frame_bgr)
        if faces:
            print(f"[SCRFD] camera={camera_id} faces={len(faces)} score_top={faces[0].score:.3f}")
        annotated = _draw_faces(frame_bgr, faces)
        success, buffer = cv2.imencode(".jpg", annotated)
        if not success:
            time.sleep(0.01)
            continue
        frame = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.03)


@router.get("/{camera_id}/mjpeg")
async def camera_mjpeg(camera_id: int) -> StreamingResponse:
    runtime = camera_registry.get(camera_id)
    if not runtime:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(_mjpeg_generator(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")
