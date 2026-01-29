from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any


@dataclass
class AnalysisSnapshot:
    camera_id: int
    timestamp: datetime
    faces: list[dict[str, Any]]


class AnalysisStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._store: dict[int, AnalysisSnapshot] = {}

    def update(self, camera_id: int, faces: list[dict[str, Any]]) -> AnalysisSnapshot:
        snapshot = AnalysisSnapshot(camera_id=camera_id, timestamp=datetime.now(timezone.utc), faces=faces)
        with self._lock:
            self._store[camera_id] = snapshot
        return snapshot

    def get(self, camera_id: int) -> AnalysisSnapshot | None:
        with self._lock:
            return self._store.get(camera_id)
