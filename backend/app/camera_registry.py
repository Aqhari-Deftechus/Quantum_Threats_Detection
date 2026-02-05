from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
import time
from typing import Dict, Optional

from .camera_worker import CameraWorker
from .config import get_settings
from .models import Camera


@dataclass
class CameraRuntime:
    camera_id: int
    status: str
    last_seen: Optional[datetime]
    fps: float
    dropped_frames: int
    queue_depth: int
    latency_ms: float
    state: str
    queue_maxsize: int
    queue_size: int
    worker: Optional[CameraWorker] = None


class CameraRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._registry: Dict[int, CameraRuntime] = {}

    def load_from_db(self, cameras: list[Camera]) -> None:
        with self._lock:
            for camera in cameras:
                if camera.enabled:
                    self._registry[camera.id] = CameraRuntime(
                        camera_id=camera.id,
                        status="CONNECTING",
                        last_seen=None,
                        fps=0.0,
                        dropped_frames=0,
                        queue_depth=0,
                        latency_ms=0.0,
                        state="UP",
                        queue_maxsize=get_settings().camera_queue_size,
                        queue_size=0,
                        worker=None,
                    )
                else:
                    self._registry[camera.id] = CameraRuntime(
                        camera_id=camera.id,
                        status="DISABLED",
                        last_seen=None,
                        fps=0.0,
                        dropped_frames=0,
                        queue_depth=0,
                        latency_ms=0.0,
                        state="DOWN",
                        queue_maxsize=get_settings().camera_queue_size,
                        queue_size=0,
                        worker=None,
                    )

    def get(self, camera_id: int) -> Optional[CameraRuntime]:
        with self._lock:
            return self._registry.get(camera_id)

    def update_status(self, camera_id: int, status: str) -> None:
        with self._lock:
            runtime = self._registry.get(camera_id)
            if runtime:
                runtime.status = status
                runtime.last_seen = datetime.now(timezone.utc)

    def list(self) -> list[CameraRuntime]:
        with self._lock:
            camera_ids = list(self._registry.keys())
        snapshots: list[CameraRuntime] = []
        for camera_id in camera_ids:
            runtime = self.snapshot_metrics(camera_id)
            if runtime:
                snapshots.append(runtime)
        return snapshots

    def restart(self, camera_id: int) -> None:
        with self._lock:
            runtime = self._registry.get(camera_id)
            if runtime:
                if runtime.worker:
                    runtime.worker.stop()
                runtime.status = "CONNECTING"
                runtime.last_seen = datetime.now(timezone.utc)

    def remove(self, camera_id: int) -> None:
        with self._lock:
            runtime = self._registry.pop(camera_id, None)
            if runtime and runtime.worker:
                runtime.worker.stop()

    def start_worker(self, camera: Camera) -> None:
        with self._lock:
            runtime = self._registry.get(camera.id)
            if not runtime:
                runtime = CameraRuntime(
                    camera_id=camera.id,
                    status="CONNECTING",
                    last_seen=None,
                    fps=0.0,
                    dropped_frames=0,
                    queue_depth=0,
                    latency_ms=0.0,
                    state="UP",
                    queue_maxsize=get_settings().camera_queue_size,
                    queue_size=0,
                    worker=None,
                )
                self._registry[camera.id] = runtime
            if runtime.worker:
                runtime.worker.stop()
            runtime.worker = CameraWorker(
                camera_id=camera.id,
                source=camera.source,
                source_type=camera.source_type,
                decoder_mode=camera.decoder_mode,
                queue_maxsize=runtime.queue_maxsize,
                capture_fps=get_settings().capture_fps,
            )
            runtime.worker.start()
            runtime.status = "CONNECTING"

    def snapshot_metrics(self, camera_id: int) -> Optional[CameraRuntime]:
        with self._lock:
            runtime = self._registry.get(camera_id)
            if not runtime or not runtime.worker:
                return runtime
            worker = runtime.worker
            runtime.status = worker.status
            runtime.fps = worker.fps
            runtime.dropped_frames = worker.dropped_frames
            runtime.queue_depth = worker.get_queue_depth()
            runtime.queue_size = runtime.queue_depth
            if worker.last_frame_time:
                runtime.latency_ms = max(0.0, (time.time() - worker.last_frame_time) * 1000)
                runtime.last_seen = datetime.now(timezone.utc)
            return runtime
