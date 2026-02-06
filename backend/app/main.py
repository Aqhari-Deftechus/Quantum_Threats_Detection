from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import Base, engine
from .logging import setup_logging
from .models import Camera, Identity, IdentityEmbedding
from .routers import cameras_router, diagnostics_router, health_router, identities_router, streaming_router, webrtc_router
from .routers.debug import router as debug_router
from .routers.vision import router as vision_router
from .detection_service import run_detection_loop
from .state import camera_registry, ws_manager, matcher

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
        embeddings = session.scalars(select(IdentityEmbedding)).all()
        name_map = {identity.id: identity.name for identity in session.scalars(select(Identity)).all()}
        if embeddings:
            vectors = [np.frombuffer(embedding.embedding, dtype="float32") for embedding in embeddings]
            identity_ids = [embedding.identity_id for embedding in embeddings]
            matcher.rebuild(np.vstack(vectors), identity_ids, name_map)
        else:
            matcher.rebuild(np.empty((0, matcher.dimension), dtype="float32"), [], name_map)

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
app.include_router(webrtc_router, prefix="/api")
app.include_router(vision_router, prefix="/api")
app.include_router(debug_router, prefix="/api")

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
