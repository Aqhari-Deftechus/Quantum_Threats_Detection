from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
ROOT_ENV_FILE = REPO_ROOT / ".env"
BACKEND_ENV_FILE = BACKEND_DIR / ".env"


def _clean_env_path_value(raw: str) -> str:
    value = raw.strip()
    if value.lower().startswith('r"') and value.endswith('"'):
        return value[2:-1]
    if value.lower().startswith("r'") and value.endswith("'"):
        return value[2:-1]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QTD_",
        env_file=(str(ROOT_ENV_FILE), str(BACKEND_ENV_FILE)),
        extra="ignore",
    )

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

    long_distance_mode: bool = False
    long_distance_det_size: str = "960x960"

    detection_mode: str = "AUTO"
    detect_every_n_frames: int = 5
    inference_short_side: int = 720
    inference_resize_mode: str = Field(default="auto", pattern="^(auto|off)$")

    max_face_matches_per_cycle: int = 5
    long_distance_max_face_matches_per_cycle: int = 12
    max_matches_hard_cap: int = 64

    embedding_cache_ttl_seconds: float = 1.0
    match_iou_threshold: float = 0.4
    min_face_area: int = 80 * 80
    min_face_area_warn_threshold: int = 2500
    unknown_grace_seconds: float = 1.0
    unknown_grace_margin: float = 0.03
    diagnostics_mode: bool = False
    diagnostics_log_every_n_frames: int = 30

    rtsp_prefer_ffmpeg: bool = True
    rtsp_buffer_size: int = 1
    rtsp_read_latest_frame: bool = True

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

    # Integrated face engine kill-switch
    face_engine_mode: str = Field(default="integrated", pattern="^(integrated|legacy)$")

    # Face source / runtime
    face_dataset_dir: Path = Path("faces_db")
    face_source: str = "0"
    face_model_name: str = "buffalo_l"

    # GPU
    face_ctx_id: int = 0
    face_onnx_providers: str = "CUDAExecutionProvider,CPUExecutionProvider"
    face_fail_if_no_cuda: bool = False

    # Detection
    face_det_size: str = "1600,1600"
    face_det_conf_thresh: float = 0.05
    face_upscale_for_det: float = 2.0
    face_min_face_area: int = 30 * 30
    face_max_faces: int = 100

    # Tile scan
    face_tile_scan_enabled: bool = True
    face_tile_grid: str = "2,2"
    face_tile_overlap: float = 0.05
    face_tile_dedup_iou: float = 0.50
    face_full_frame_det_when_tile: bool = False

    # Speed
    face_detect_every_n_frames: int = 5

    # Recognition
    face_recog_thresh: float = 0.20

    # RTSP robustness
    face_read_timeout_sec: int = 10
    face_reconnect_wait_sec: float = 2.0
    face_flush_frames_on_connect: int = 8

    # Face DB cache
    face_db_cache_path: Path = Path("backend/models/face_db.npz")

    @property
    def clip_storage_dir_resolved(self) -> Path:
        return self.clip_storage_dir.resolve()

    @property
    def face_db_cache_path_resolved(self) -> Path:
        return self.face_db_cache_path.resolve()

    @property
    def face_dataset_dir_raw(self) -> str:
        return str(self.face_dataset_dir)

    @property
    def face_dataset_dir_resolved(self) -> Path:
        raw_text = _clean_env_path_value(str(self.face_dataset_dir))
        candidate = Path(raw_text)
        if candidate.is_absolute():
            return candidate

        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

        root_candidate = (REPO_ROOT / candidate).resolve()
        if root_candidate.exists():
            return root_candidate

        backend_candidate = (BACKEND_DIR / candidate).resolve()
        if backend_candidate.exists():
            return backend_candidate

        return root_candidate

    @property
    def active_face_env_file(self) -> str:
        if ROOT_ENV_FILE.exists() and BACKEND_ENV_FILE.exists():
            return f"{ROOT_ENV_FILE} (base), {BACKEND_ENV_FILE} (override)"
        if ROOT_ENV_FILE.exists():
            return str(ROOT_ENV_FILE)
        if BACKEND_ENV_FILE.exists():
            return str(BACKEND_ENV_FILE)
        return "none"

    def _parse_det_size(self, value: str) -> tuple[int, int]:
        try:
            clean = value.lower().replace(" ", "")
            parts = clean.split("x")
            if len(parts) != 2:
                raise ValueError("det_size must be in WxH format")
            width = max(64, int(parts[0]))
            height = max(64, int(parts[1]))
            return width, height
        except Exception:
            return self.insightface_det_size

    def _parse_csv_pair(self, value: str, default: tuple[int, int], minimum: int = 1) -> tuple[int, int]:
        try:
            parts = [chunk.strip() for chunk in value.split(",")]
            if len(parts) != 2:
                return default
            first = max(minimum, int(parts[0]))
            second = max(minimum, int(parts[1]))
            return first, second
        except Exception:
            return default

    def _parse_provider_csv(self, value: str) -> list[str]:
        chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
        if not chunks:
            return ["CPUExecutionProvider"]
        return chunks

    @property
    def active_det_size(self) -> tuple[int, int]:
        if self.long_distance_mode:
            return self._parse_det_size(self.long_distance_det_size)
        return self.insightface_det_size

    @property
    def face_active_det_size(self) -> tuple[int, int]:
        return self._parse_csv_pair(self.face_det_size, default=(1600, 1600), minimum=64)

    @property
    def face_active_tile_grid(self) -> tuple[int, int]:
        return self._parse_csv_pair(self.face_tile_grid, default=(2, 2), minimum=1)

    @property
    def face_active_providers(self) -> list[str]:
        return self._parse_provider_csv(self.face_onnx_providers)

    @property
    def face_active_source(self) -> int | str:
        raw = str(self.face_source).strip()
        if raw == "0":
            return 0
        return raw

    @property
    def active_resize_mode(self) -> str:
        if self.long_distance_mode and self.inference_resize_mode == "auto" and self.inference_short_side >= 1080:
            return "auto"
        return self.inference_resize_mode

    @property
    def active_detect_every_n(self) -> int:
        return max(int(self.detect_every_n_frames), 1)

    @property
    def face_active_detect_every_n(self) -> int:
        return max(int(self.face_detect_every_n_frames), 1)

    @property
    def active_min_face_area(self) -> int:
        return max(int(self.min_face_area), 0)

    @property
    def active_max_face_matches(self) -> int:
        base = self.long_distance_max_face_matches_per_cycle if self.long_distance_mode else self.max_face_matches_per_cycle
        capped = max(1, int(base))
        return min(capped, max(1, int(self.max_matches_hard_cap)))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    logger.info(
        "Active Face Settings | QTD_FACE_DATASET_DIR=%s | resolved_dataset_dir=%s | cwd=%s | env_file=%s",
        settings.face_dataset_dir_raw,
        settings.face_dataset_dir_resolved,
        Path.cwd(),
        settings.active_face_env_file,
    )
    return settings
