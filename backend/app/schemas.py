from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CameraBase(BaseModel):
    name: str
    source: str
    source_type: str = Field(pattern="^(WEBCAM|RTSP|VIDEO_FILE)$")
    decoder_mode: str = Field(default="opencv", pattern="^(ffmpeg|opencv|none)$")


class CameraCreate(CameraBase):
    enabled: bool = True


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = Field(default=None, pattern="^(WEBCAM|RTSP|VIDEO_FILE)$")
    enabled: Optional[bool] = None
    decoder_mode: Optional[str] = Field(default=None, pattern="^(ffmpeg|opencv|none)$")


class CameraOut(BaseModel):
    id: int
    name: str
    source_type: str
    source_redacted: str
    enabled: bool
    decoder_mode: str
    created_at: datetime
    updated_at: datetime
    camera_status: str

    class Config:
        from_attributes = True


class CameraTestRequest(BaseModel):
    source: str
    source_type: str = Field(pattern="^(WEBCAM|RTSP|VIDEO_FILE)$")


class CameraHealth(BaseModel):
    camera_id: int
    status: str
    last_seen: Optional[datetime]
    fps: float
    dropped_frames: int
    queue_depth: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class StatusResponse(BaseModel):
    system: str
    inference: str
    matcher: str
    matcher_index_status: str
    evidence: str
    cameras_live: int
    cameras_total: int
    cameras_down: int
    cameras_avg_fps: float
    cameras_avg_latency_ms: float
    cameras_queue_depth: int
    cameras_dropped_frames: int
    ws_status: str
    timestamp: datetime


class SystemHealthResponse(BaseModel):
    status: str
    metrics: dict[str, float]
    scrfd_model_ready: bool | None = None
    arcface_model_ready: bool | None = None
    onnxruntime_provider: str | None = None
    last_error: Optional[str] = None
    timestamp: datetime


class FaceDetectionOut(BaseModel):
    box: list[int]
    score: float
    label: str
    similarity: float
    quality: str
    landmarks: list[list[float]] | None = None


class DetectionResponse(BaseModel):
    faces: list[FaceDetectionOut]
    timestamp: datetime


class CameraAnalysisResponse(BaseModel):
    camera_id: int
    timestamp: datetime
    faces: list[FaceDetectionOut]


class IdentityCreate(BaseModel):
    name: str
    notes: str = ""


class IdentityUpdate(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None


class IdentityOut(BaseModel):
    id: int
    name: str
    notes: str
    created_at: datetime
    updated_at: datetime
    embedding_count: int

    class Config:
        from_attributes = True


class EmbeddingEnrollRequest(BaseModel):
    embeddings: list[list[float]] | None = None


class WebRTCOfferRequest(BaseModel):
    camera_id: int
    sdp: str
    type: str = Field(pattern="^(offer)$")


class WebRTCAnswerResponse(BaseModel):
    sdp: str
    type: str
