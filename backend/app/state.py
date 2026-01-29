from __future__ import annotations

from .analysis_store import AnalysisStore
from .camera_registry import CameraRegistry
from .matcher import Matcher
from .vision.vision_service import VisionService
from .ws import WebSocketManager


camera_registry = CameraRegistry()
matcher = Matcher()
ws_manager = WebSocketManager()
vision_service = VisionService()
analysis_store = AnalysisStore()
