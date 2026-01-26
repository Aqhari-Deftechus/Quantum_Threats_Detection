from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..audit import verify_chain
from ..db import get_session
from ..schemas import HealthResponse, StatusResponse, SystemHealthResponse
from ..state import camera_registry, matcher

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="OK", timestamp=datetime.now(timezone.utc))


@router.get("/status", response_model=StatusResponse)
async def status(session: Session = Depends(get_session)) -> StatusResponse:
    match_status = matcher.status()
    chain_ok, _ = verify_chain(session)
    cameras = camera_registry.list()
    live = sum(1 for camera in cameras if camera.status == "LIVE")
    down = sum(1 for camera in cameras if camera.status in {"DOWN", "DEGRADED"})
    total_fps = sum(camera.fps for camera in cameras)
    total_latency = sum(camera.latency_ms for camera in cameras)
    total_queue = sum(camera.queue_depth for camera in cameras)
    total_dropped = sum(camera.dropped_frames for camera in cameras)
    count = len(cameras) or 1
    return StatusResponse(
        system="OK",
        inference="CPU",
        matcher=match_status.mode,
        matcher_index_status=match_status.index_status,
        evidence="INTEGRITY OK" if chain_ok else "VERIFY FAIL",
        cameras_live=live,
        cameras_total=len(cameras),
        cameras_down=down,
        cameras_avg_fps=total_fps / count,
        cameras_avg_latency_ms=total_latency / count,
        cameras_queue_depth=total_queue,
        cameras_dropped_frames=total_dropped,
        ws_status="CONNECTED" if cameras else "RECONNECTING",
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/system/health", response_model=SystemHealthResponse)
async def system_health(session: Session = Depends(get_session)) -> SystemHealthResponse:
    chain_ok, _ = verify_chain(session)
    metrics = {
        "fps": 0.0,
        "latency_ms": 0.0,
        "dropped_frames": 0.0,
    }
    return SystemHealthResponse(
        status="OK" if chain_ok else "DEGRADED",
        metrics=metrics,
        timestamp=datetime.now(timezone.utc),
    )
