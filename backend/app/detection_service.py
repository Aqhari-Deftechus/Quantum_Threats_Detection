from __future__ import annotations

import asyncio
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
    frame_counters: dict[int, int] = {}

    while True:
        await asyncio.sleep(0.05)
        cameras = camera_registry.list()
        for runtime in cameras:
            if not runtime.worker:
                continue
            latest = runtime.worker.get_latest_bgr()
            if not latest:
                continue
            timestamp, frame = latest
            frame_counters[runtime.camera_id] = frame_counters.get(runtime.camera_id, 0) + 1
            if settings.detection_mode == "MANUAL":
                continue
            if frame_counters[runtime.camera_id] % max(settings.detect_every_n_frames, 1) != 0:
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
