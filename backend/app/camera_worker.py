from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import cv2
import numpy as np

from .config import get_settings
from .diagnostics import RollingMetric

logger = logging.getLogger(__name__)

@dataclass
class FramePacket:
    frame: bytes
    timestamp: float


class CameraWorker:
    def __init__(
        self,
        camera_id: int,
        source: str,
        source_type: str,
        queue_maxsize: int,
        capture_fps: int,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.source_type = source_type
        self.queue_maxsize = queue_maxsize
        self.capture_fps = capture_fps
        self.queue: Deque[FramePacket] = deque(maxlen=queue_maxsize)
        self.dropped_frames = 0
        self.fps = 0.0
        self.last_frame_time: Optional[float] = None
        self.last_frame_bgr: Optional[tuple[float, "np.ndarray"]] = None
        self.status = "CONNECTING"
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        settings = get_settings()
        self._diagnostics_enabled = settings.diagnostics_mode
        self._diagnostics_every_n = settings.diagnostics_log_every_n_frames
        self._camera_read_metric = RollingMetric()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _open_capture(self) -> cv2.VideoCapture:
        if self.source_type == "WEBCAM":
            return cv2.VideoCapture(int(self.source))
        return cv2.VideoCapture(self.source)

    def _run(self) -> None:
        capture = self._open_capture()
        if not capture.isOpened():
            self.status = "DOWN"
            return
        self.status = "LIVE"
        last_tick = time.time()
        frame_count = 0
        frame_interval = 1.0 / max(self.capture_fps, 1)
        while not self._stop_event.is_set():
            read_start = time.perf_counter()
            ok, frame = capture.read()
            read_end = time.perf_counter()
            if self._diagnostics_enabled:
                self._camera_read_metric.update(read_end - read_start)
                if self._camera_read_metric.should_log(self._diagnostics_every_n):
                    logger.info(
                        "Diagnostics: camera_read_avg_ms=%.2f camera_id=%s",
                        self._camera_read_metric.average_ms(),
                        self.camera_id,
                    )
            if not ok:
                self.status = "DEGRADED"
                time.sleep(0.1)
                continue

            now = time.time()
            if self.last_frame_time and (now - self.last_frame_time) < frame_interval:
                time.sleep(frame_interval - (now - self.last_frame_time))
                continue

            frame_count += 1
            now = time.time()
            if now - last_tick >= 1.0:
                self.fps = frame_count / (now - last_tick)
                frame_count = 0
                last_tick = now

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue

            packet = FramePacket(frame=buffer.tobytes(), timestamp=now)
            with self._lock:
                if len(self.queue) >= self.queue_maxsize:
                    self.queue.popleft()
                    self.dropped_frames += 1
                self.queue.append(packet)
                self.last_frame_time = now
                self.last_frame_bgr = (now, frame)

        capture.release()
        self.status = "DOWN"

    def get_latest_frame(self) -> Optional[FramePacket]:
        with self._lock:
            if not self.queue:
                return None
            return self.queue[-1]

    def get_queue_depth(self) -> int:
        with self._lock:
            return len(self.queue)

    def get_latest_bgr(self) -> Optional[tuple[float, "np.ndarray"]]:
        with self._lock:
            return self.last_frame_bgr
