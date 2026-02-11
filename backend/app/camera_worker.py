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
        decoder_mode: str,
        queue_maxsize: int,
        capture_fps: int,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.source_type = source_type
        self.decoder_mode = decoder_mode
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
        settings = get_settings()
        if self.source_type == "WEBCAM":
            return cv2.VideoCapture(int(self.source))

        use_ffmpeg = self.decoder_mode == "ffmpeg"
        if self.source_type == "RTSP" and settings.rtsp_prefer_ffmpeg:
            use_ffmpeg = True

        if use_ffmpeg:
            ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
            if ffmpeg_backend is None:
                logger.warning("FFmpeg backend requested but not available; using default backend.")
                return cv2.VideoCapture(self.source)
            return cv2.VideoCapture(self.source, ffmpeg_backend)
        return cv2.VideoCapture(self.source)

    def _configure_rtsp_timeouts(self, capture: cv2.VideoCapture) -> None:
        settings = get_settings()
        open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if open_timeout is None:
            logger.warning("OpenCV RTSP open timeout property not available; continuing without it.")
        else:
            if not capture.set(open_timeout, 5000):
                logger.warning("Failed to set RTSP open timeout; continuing without it.")
        if read_timeout is None:
            logger.warning("OpenCV RTSP read timeout property not available; continuing without it.")
        else:
            if not capture.set(read_timeout, 5000):
                logger.warning("Failed to set RTSP read timeout; continuing without it.")

        buffer_size_prop = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
        if buffer_size_prop is not None:
            buffer_size = max(1, int(settings.rtsp_buffer_size))
            if not capture.set(buffer_size_prop, buffer_size):
                logger.warning("Failed to set RTSP buffer size; continuing with backend default.")

    def _drain_latest_frame(self, capture: cv2.VideoCapture, frame: np.ndarray) -> np.ndarray:
        settings = get_settings()
        if not settings.rtsp_read_latest_frame:
            return frame
        latest = frame
        for _ in range(max(0, int(settings.rtsp_drain_frames))):
            grabbed = capture.grab()
            if not grabbed:
                break
            ok, refreshed = capture.retrieve()
            if not ok:
                break
            latest = refreshed
        return latest

    def _run(self) -> None:
        settings = get_settings()
        reconnect_threshold = 10
        reconnect_delay_s = max(0.1, float(settings.face_reconnect_wait_sec))
        consecutive_failures = 0
        logger.info("Camera connect starting: camera_id=%s source_type=%s", self.camera_id, self.source_type)
        capture = self._open_capture()
        if self.source_type == "RTSP":
            self._configure_rtsp_timeouts(capture)
        if not capture.isOpened():
            logger.warning("Camera connect failed: camera_id=%s", self.camera_id)
            self.status = "DOWN"
            return
        logger.info("Camera connect success: camera_id=%s", self.camera_id)
        flush_frames = max(0, int(settings.face_flush_frames_on_connect))
        for _ in range(flush_frames):
            capture.read()
        self.status = "LIVE"
        last_tick = time.time()
        frame_count = 0
        frame_interval = 1.0 / max(self.capture_fps, 1)
        next_capture_deadline = time.perf_counter()
        while not self._stop_event.is_set():
            now_tick = time.perf_counter()
            if now_tick < next_capture_deadline:
                time.sleep(min(next_capture_deadline - now_tick, frame_interval))
                continue

            read_start = time.perf_counter()
            ok, frame = capture.read()
            read_end = time.perf_counter()
            next_capture_deadline += frame_interval
            if read_end - next_capture_deadline > frame_interval * 3:
                next_capture_deadline = read_end
            timeout_s = max(1, int(settings.face_read_timeout_sec))
            if (read_end - read_start) > float(timeout_s):
                logger.warning("Camera read timeout exceeded: camera_id=%s elapsed=%.2fs", self.camera_id, read_end - read_start)
                ok = False
            if self._diagnostics_enabled:
                self._camera_read_metric.update(read_end - read_start)
                if self._camera_read_metric.should_log(self._diagnostics_every_n):
                    logger.info(
                        "Diagnostics: camera_read_avg_ms=%.2f camera_id=%s",
                        self._camera_read_metric.average_ms(),
                        self.camera_id,
                    )
            if not ok:
                consecutive_failures += 1
                logger.warning(
                    "Camera read failed: camera_id=%s failures=%s",
                    self.camera_id,
                    consecutive_failures,
                )
                self.status = "DEGRADED"
                if consecutive_failures >= reconnect_threshold:
                    logger.warning("Camera reconnecting: camera_id=%s", self.camera_id)
                    capture.release()
                    time.sleep(reconnect_delay_s)
                    capture = self._open_capture()
                    if self.source_type == "RTSP":
                        self._configure_rtsp_timeouts(capture)
                    if capture.isOpened():
                        logger.info("Camera reconnect success: camera_id=%s", self.camera_id)
                        flush_frames = max(0, int(settings.face_flush_frames_on_connect))
                        for _ in range(flush_frames):
                            capture.read()
                        self.status = "LIVE"
                        consecutive_failures = 0
                    else:
                        logger.warning("Camera reconnect failed: camera_id=%s", self.camera_id)
                        self.status = "DOWN"
                time.sleep(0.1)
                continue
            consecutive_failures = 0

            if self.source_type == "RTSP":
                frame = self._drain_latest_frame(capture, frame)

            now = time.time()
            frame_count += 1
            now = time.time()
            if now - last_tick >= 1.0:
                self.fps = frame_count / (now - last_tick)
                frame_count = 0
                last_tick = now

            packet = FramePacket(frame=b"", timestamp=now)
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
