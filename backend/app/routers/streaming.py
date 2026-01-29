from __future__ import annotations

import logging
import time
from typing import Generator

import cv2
from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

from ..state import analysis_store, camera_registry, vision_service

router = APIRouter(prefix="/cameras")
logger = logging.getLogger(__name__)


def _mjpeg_generator(camera_id: int) -> Generator[bytes, None, None]:
    while True:
        try:
            runtime = camera_registry.snapshot_metrics(camera_id)
            if not runtime or not runtime.worker:
                frame_bgr = vision_service.placeholder_frame("NO SIGNAL")
                faces = []
            else:
                latest = runtime.worker.get_latest_bgr()
                if not latest:
                    frame_bgr = vision_service.placeholder_frame("NO SIGNAL")
                    faces = []
                else:
                    _, frame_bgr = latest
                    faces = vision_service.analyze_frame(camera_id, frame_bgr, force_detect=True)

            analysis_store.update(camera_id, [face.__dict__ for face in faces])
            annotated = vision_service.annotate(frame_bgr, faces)
            success, buffer = cv2.imencode(".jpg", annotated)
            if not success:
                time.sleep(0.01)
                continue
            frame = buffer.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.03)
        except GeneratorExit:
            return
        except Exception:
            logger.exception("MJPEG generator failed; keeping stream alive.")
            frame_bgr = vision_service.placeholder_frame("STREAM ERROR")
            success, buffer = cv2.imencode(".jpg", frame_bgr)
            if success:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(0.2)


@router.get("/{camera_id}/mjpeg")
async def camera_mjpeg(camera_id: int) -> StreamingResponse:
    runtime = camera_registry.get(camera_id)
    if not runtime:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(_mjpeg_generator(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")
