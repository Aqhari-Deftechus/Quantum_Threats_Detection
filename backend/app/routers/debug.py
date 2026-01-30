from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..state import analysis_store, camera_registry, vision_service

router = APIRouter(prefix="/debug")


@router.get("/models")
def debug_models() -> dict[str, object]:
    return vision_service.model_report()


@router.get("/scrfd/io")
def debug_scrfd_io() -> dict[str, object]:
    return vision_service.scrfd_io_report()


@router.get("/cameras/{camera_id}/last_detection")
def debug_last_detection(camera_id: int) -> dict[str, object]:
    runtime = camera_registry.get(camera_id)
    if not runtime:
        raise HTTPException(status_code=404, detail="Camera not found")
    snapshot = analysis_store.get(camera_id)
    faces = snapshot.faces if snapshot else []
    top_score = max((face.get("score", 0.0) for face in faces), default=0.0)
    first_bbox = faces[0].get("box") if faces else None
    vision_status = vision_service.status()
    return {
        "camera_id": camera_id,
        "last_frame_ts": snapshot.timestamp.isoformat() if snapshot else None,
        "face_count": len(faces),
        "top_score": top_score,
        "first_bbox": first_bbox,
        "scrfd_model_ready": vision_status.scrfd_ready,
        "arcface_model_ready": vision_status.arcface_ready,
    }
