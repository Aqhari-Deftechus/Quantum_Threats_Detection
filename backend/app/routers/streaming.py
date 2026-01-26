from __future__ import annotations

import time
from typing import Generator

from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

from ..state import camera_registry

router = APIRouter(prefix="/cameras")


def _mjpeg_generator(camera_id: int) -> Generator[bytes, None, None]:
    while True:
        runtime = camera_registry.snapshot_metrics(camera_id)
        if not runtime or not runtime.worker:
            time.sleep(0.1)
            continue
        packet = runtime.worker.get_latest_frame()
        if not packet:
            time.sleep(0.01)
            continue
        frame = packet.frame
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
