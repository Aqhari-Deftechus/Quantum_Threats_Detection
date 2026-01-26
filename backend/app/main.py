from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import Base, engine
from .logging import setup_logging
from .models import Camera
from .routers import cameras_router, diagnostics_router, health_router, identities_router, streaming_router
from .detection_service import run_detection_loop
from .state import camera_registry, ws_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    Base.metadata.create_all(bind=engine)
    clip_dir = settings.clip_storage_dir_resolved
    clip_dir.mkdir(parents=True, exist_ok=True)

    with engine.begin() as connection:
        session = Session(bind=connection)
        cameras = session.scalars(select(Camera)).all()
        camera_registry.load_from_db(cameras)
        for camera in cameras:
            if camera.enabled:
                camera_registry.start_worker(camera)

    asyncio.create_task(run_detection_loop(camera_registry, ws_manager))
    yield


settings = get_settings()
static_dir = settings.clip_storage_dir_resolved.parent
static_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(cameras_router, prefix="/api")
app.include_router(diagnostics_router, prefix="/api")
app.include_router(identities_router, prefix="/api")
app.include_router(streaming_router, prefix="/api")

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/api/webrtc/offer")
async def webrtc_offer() -> dict[str, str]:
    return {"status": "scaffold", "detail": "WebRTC offer endpoint placeholder"}


@app.post("/api/webrtc/answer")
async def webrtc_answer() -> dict[str, str]:
    return {"status": "scaffold", "detail": "WebRTC answer endpoint placeholder"}
