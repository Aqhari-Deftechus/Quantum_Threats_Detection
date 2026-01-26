from __future__ import annotations

from .camera_registry import CameraRegistry
from .matcher import Matcher
from .ws import WebSocketManager


camera_registry = CameraRegistry()
matcher = Matcher()
ws_manager = WebSocketManager()
