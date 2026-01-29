from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile

from ..schemas import DetectionResponse, FaceDetectionOut
from ..state import analysis_store, vision_service

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)) -> DetectionResponse:
    content = await file.read()
    image_array = np.frombuffer(content, dtype=np.uint8)
    frame_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return DetectionResponse(faces=[], timestamp=datetime.now(timezone.utc))
    faces = vision_service.analyze_frame(camera_id=-1, frame_bgr=frame_bgr, force_detect=True)
    analysis_store.update(-1, [face.__dict__ for face in faces])
    output_faces: list[FaceDetectionOut] = [
        FaceDetectionOut(
            box=face.box,
            score=face.score,
            label=face.label,
            similarity=face.similarity,
            quality=face.quality,
            landmarks=face.landmarks,
        )
        for face in faces
    ]
    return DetectionResponse(faces=output_faces, timestamp=datetime.now(timezone.utc))


@router.post("/watchlist/reload")
async def reload_watchlist() -> dict[str, Any]:
    vision_service.reload_watchlist()
    return {"status": "reloaded"}
