from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..schemas import WebRTCOfferRequest, WebRTCAnswerResponse
from ..state import camera_registry
from ..webrtc import WebRTCManager

router = APIRouter(prefix="/webrtc")
logger = logging.getLogger(__name__)

webrtc_manager = WebRTCManager(camera_registry=camera_registry)


@router.post("/offer", response_model=WebRTCAnswerResponse)
async def webrtc_offer(payload: WebRTCOfferRequest) -> WebRTCAnswerResponse:
    settings = get_settings()
    if not settings.webrtc_enabled:
        raise HTTPException(status_code=503, detail="WebRTC disabled")

    runtime = camera_registry.get(payload.camera_id)
    if not runtime:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not runtime.worker:
        raise HTTPException(status_code=409, detail="Camera not ready")

    try:
        answer = await webrtc_manager.handle_offer(payload.camera_id, payload.sdp, payload.type)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    return WebRTCAnswerResponse(sdp=answer.sdp, type=answer.type)
