from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QTD_", env_file=".env", extra="ignore")

    app_name: str = "Quantum Threats Detection"
    environment: str = "development"
    db_url: str = "sqlite:///./qtd.db"
    log_level: str = "INFO"
    evidence_worm_mode: bool = True
    ffmpeg_path: str = "ffmpeg"
    audit_chain_id: str = "default"
    clip_storage_dir: Path = Path("backend/static/event_clips")
    ws_event_interval_seconds: float = 1.0
    camera_queue_size: int = 2
    capture_fps: int = 12

    scrfd_model_path: Path = Path("backend/models/scrfd_10g_bnkps.onnx")
    arcface_model_path: Path = Path("backend/models/glintr100.onnx")
    watchlist_dir: Path = Path("backend/watchlist")
    watchlist_match_threshold: float = 0.35
    scrfd_score_threshold: float = 0.3
    scrfd_nms_threshold: float = 0.4
    scrfd_backend: str = "insightface"
    insightface_det_size: tuple[int, int] = (640, 640)
    detection_mode: str = "AUTO"
    detect_every_n_frames: int = 5
    inference_short_side: int = 720
    max_face_matches_per_cycle: int = 5
    embedding_cache_ttl_seconds: float = 1.0
    match_iou_threshold: float = 0.4
    min_face_area: int = 80 * 80
    unknown_grace_seconds: float = 1.0
    unknown_grace_margin: float = 0.03
    diagnostics_mode: bool = False
    diagnostics_log_every_n_frames: int = 30

    matcher_dimension: int = 128
    matcher_faiss_enabled: bool = True

    retention_days: int = 30

    webrtc_enabled: bool = True
    webrtc_stun_urls: str = "stun:stun.l.google.com:19302"
    webrtc_turn_url: str = ""
    webrtc_turn_user: str = ""
    webrtc_turn_pass: str = ""
    webrtc_signalling_path: str = "/api/webrtc/offer"
    webrtc_max_peers: int = 2
    webrtc_video_codec: str = "H264"
    webrtc_fps: int = 12
    webrtc_resolution: str = "1280x720"
    mediamtx_whep_base_url: str = "http://127.0.0.1:8889"
    mediamtx_whep_path_template: str = "camera-{camera_id}"

    @property
    def clip_storage_dir_resolved(self) -> Path:
        return self.clip_storage_dir.resolve()


@lru_cache

def get_settings() -> Settings:
    return Settings()
