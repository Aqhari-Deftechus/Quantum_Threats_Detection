from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter

from ..config import get_settings

router = APIRouter(prefix="/diagnostics")


@router.get("/ffmpeg")
async def ffmpeg_diagnostics() -> dict[str, str]:
    settings = get_settings()
    resolved = shutil.which(settings.ffmpeg_path)
    return {
        "configured": settings.ffmpeg_path,
        "resolved": resolved or "not_found",
        "status": "OK" if resolved else "NOT_FOUND",
    }
