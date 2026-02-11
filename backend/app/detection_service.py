from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from .camera_registry import CameraRegistry
from .config import get_settings
from .ws import WebSocketManager
from .state import analysis_store, vision_service


def _overlay_payload(camera_id: int, faces: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "1.0",
        "type": "overlay",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "appearance_mode": "default",
        "threshold_profile": "standard",
        "policy": {
            "decision": "ALLOW",
            "reason_code": "DEMO_RULE",
        },
        "data": {
            "camera_id": camera_id,
            "faces": faces,
        },
    }


async def run_detection_loop(camera_registry: CameraRegistry, ws_manager: WebSocketManager) -> None:
    settings = get_settings()
    interval = max(float(settings.ws_event_interval_seconds), 0.02)
    next_due_by_camera: dict[int, float] = {}

    while True:
        await asyncio.sleep(0.01)
        cameras = camera_registry.list()
        if not cameras:
            continue

        now = time.monotonic()
        camera_count = max(len(cameras), 1)

        for index, runtime in enumerate(cameras):
            if not runtime.worker:
                continue

            base_due = next_due_by_camera.get(runtime.camera_id)
            if base_due is None:
                stagger = (interval / camera_count) * index
                base_due = now + stagger
                next_due_by_camera[runtime.camera_id] = base_due

            if now < base_due:
                continue

            next_due = base_due + interval
            if now - base_due > interval * 3:
                next_due = now + interval
            next_due_by_camera[runtime.camera_id] = next_due

            latest = runtime.worker.get_latest_bgr()
            if not latest:
                continue
            _, frame = latest
            if settings.detection_mode == "MANUAL":
                continue

            faces = []
            analyzed = vision_service.analyze_frame(runtime.camera_id, frame, force_detect=False)
            for face in analyzed:
                faces.append(
                    {
                        "box": face.box,
                        "score": face.score,
                        "quality": face.quality,
                        "label": face.label,
                        "similarity": face.similarity,
                        "landmarks": face.landmarks,
                    }
                )

            analysis_store.update(runtime.camera_id, faces)

            payload = _overlay_payload(runtime.camera_id, faces)
            await ws_manager.broadcast(payload)
