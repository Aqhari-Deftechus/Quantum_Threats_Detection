from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import create_audit_event
from ..camera_registry import CameraRuntime
from ..config import get_settings
from ..db import get_session
from ..models import Camera
from ..schemas import (
    CameraCreate,
    CameraHealth,
    CameraOut,
    CameraTestRequest,
    CameraUpdate,
    CameraAnalysisResponse,
    FaceDetectionOut,
    WebRTCPlaybackResponse,
)
from ..state import analysis_store, camera_registry
from ..utils import redact_rtsp

router = APIRouter(prefix="/cameras")


def _camera_status(runtime: CameraRuntime | None, enabled: bool) -> str:
    if not enabled:
        return "DISABLED"
    if runtime is None:
        return "DOWN"
    return runtime.status


def _build_whep_url(camera: Camera) -> str:
    settings = get_settings()
    base = settings.mediamtx_whep_base_url.rstrip("/")
    path = settings.mediamtx_whep_path_template.format(
        camera_id=camera.id,
        camera_name=camera.name,
    )
    path = path.strip("/")
    return f"{base}/{path}/whep"


@router.get("", response_model=list[CameraOut])
def list_cameras(session: Session = Depends(get_session)) -> list[CameraOut]:
    cameras = session.scalars(select(Camera)).all()
    output: list[CameraOut] = []
    for camera in cameras:
        runtime = camera_registry.snapshot_metrics(camera.id)
        output.append(
            CameraOut(
                id=camera.id,
                name=camera.name,
                source_type=camera.source_type,
                source_redacted=redact_rtsp(camera.source),
                enabled=camera.enabled,
                decoder_mode=camera.decoder_mode,
                created_at=camera.created_at,
                updated_at=camera.updated_at,
                camera_status=_camera_status(runtime, camera.enabled),
            )
        )
    return output


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: int, session: Session = Depends(get_session)) -> CameraOut:
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    runtime = camera_registry.snapshot_metrics(camera.id)
    return CameraOut(
        id=camera.id,
        name=camera.name,
        source_type=camera.source_type,
        source_redacted=redact_rtsp(camera.source),
        enabled=camera.enabled,
        decoder_mode=camera.decoder_mode,
        created_at=camera.created_at,
        updated_at=camera.updated_at,
        camera_status=_camera_status(runtime, camera.enabled),
    )


@router.post("", response_model=CameraOut)
def create_camera(payload: CameraCreate, session: Session = Depends(get_session)) -> CameraOut:
    camera = Camera(
        name=payload.name,
        source=payload.source,
        source_type=payload.source_type,
        enabled=payload.enabled,
        decoder_mode=payload.decoder_mode,
    )
    session.add(camera)
    session.commit()
    session.refresh(camera)
    camera_registry.load_from_db([camera])
    if camera.enabled:
        camera_registry.start_worker(camera)
    create_audit_event(session, "camera_created", {"camera_id": camera.id, "name": camera.name})
    return get_camera(camera.id, session)


@router.put("/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: int, payload: CameraUpdate, session: Session = Depends(get_session)) -> CameraOut:
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(camera, key, value)
    camera.updated_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(camera)
    create_audit_event(session, "camera_updated", {"camera_id": camera.id, "name": camera.name})
    return get_camera(camera.id, session)


@router.post("/{camera_id}/enable", response_model=CameraOut)
def enable_camera(camera_id: int, session: Session = Depends(get_session)) -> CameraOut:
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    camera.enabled = True
    session.commit()
    session.refresh(camera)
    camera_registry.load_from_db([camera])
    camera_registry.start_worker(camera)
    create_audit_event(session, "camera_enabled", {"camera_id": camera.id, "name": camera.name})
    return get_camera(camera.id, session)


@router.delete("/{camera_id}")
def delete_camera(camera_id: int, session: Session = Depends(get_session)) -> dict[str, str]:
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    session.delete(camera)
    session.commit()
    camera_registry.remove(camera_id)
    create_audit_event(session, "camera_deleted", {"camera_id": camera_id})
    return {"status": "deleted"}


@router.post("/test")
def test_camera(payload: CameraTestRequest) -> dict[str, str]:
    return {
        "status": "OK",
        "source_type": payload.source_type,
        "source_redacted": redact_rtsp(payload.source),
    }


@router.post("/{camera_id}/restart")
def restart_camera(camera_id: int, session: Session = Depends(get_session)) -> dict[str, str]:
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    camera_registry.restart(camera_id)
    camera_registry.start_worker(camera)
    create_audit_event(session, "camera_restarted", {"camera_id": camera.id})
    return {"status": "restarting"}


@router.get("/{camera_id}/health", response_model=CameraHealth)
def camera_health(camera_id: int, session: Session = Depends(get_session)) -> CameraHealth:
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    runtime = camera_registry.snapshot_metrics(camera_id)
    if not runtime:
        return CameraHealth(
            camera_id=camera_id,
            status="DOWN",
            last_seen=None,
            fps=0.0,
            dropped_frames=0,
            queue_depth=0,
            latency_ms=0.0,
        )
    return CameraHealth(
        camera_id=camera_id,
        status=runtime.status,
        last_seen=runtime.last_seen,
        fps=runtime.fps,
        dropped_frames=runtime.dropped_frames,
        queue_depth=runtime.queue_depth,
        latency_ms=runtime.latency_ms,
    )


@router.get("/{camera_id}/webrtc-playback", response_model=WebRTCPlaybackResponse)
def camera_webrtc_playback(camera_id: int, session: Session = Depends(get_session)) -> WebRTCPlaybackResponse:
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return WebRTCPlaybackResponse(protocol="whep", whep_url=_build_whep_url(camera))


@router.get("/{camera_id}/analysis", response_model=CameraAnalysisResponse)
def camera_analysis(camera_id: int, session: Session = Depends(get_session)) -> CameraAnalysisResponse:
    camera = session.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    snapshot = analysis_store.get(camera_id)
    faces = snapshot.faces if snapshot else []
    output_faces = [
        FaceDetectionOut(
            box=face.get("box", []),
            score=face.get("score", 0.0),
            label=face.get("label", "Unknown"),
            similarity=face.get("similarity", 0.0),
            quality=face.get("quality", "LOW"),
            landmarks=face.get("landmarks"),
        )
        for face in faces
    ]
    return CameraAnalysisResponse(
        camera_id=camera_id,
        timestamp=snapshot.timestamp if snapshot else datetime.now(timezone.utc),
        faces=output_faces,
    )
