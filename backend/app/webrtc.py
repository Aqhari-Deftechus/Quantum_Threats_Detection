from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender
from aiortc.mediastreams import VideoStreamTrack
from av import VideoFrame

from .camera_registry import CameraRegistry
from .config import get_settings
from .state import vision_service

logger = logging.getLogger(__name__)


@dataclass
class WebRTCSession:
    peer: RTCPeerConnection
    camera_id: int


def _parse_resolution(value: str) -> Optional[tuple[int, int]]:
    if not value:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        return None
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


class CameraVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, camera_id: int, camera_registry: CameraRegistry) -> None:
        super().__init__()
        settings = get_settings()
        self.camera_id = camera_id
        self.camera_registry = camera_registry
        self.fps = max(settings.webrtc_fps, 1)
        self.resolution = _parse_resolution(settings.webrtc_resolution)

    async def recv(self) -> VideoFrame:
        await asyncio.sleep(1 / self.fps)
        runtime = self.camera_registry.snapshot_metrics(self.camera_id)
        frame_bgr = None
        if runtime and runtime.worker:
            latest = runtime.worker.get_latest_bgr()
            if latest:
                _, frame_bgr = latest
        if frame_bgr is None:
            frame_bgr = vision_service.placeholder_frame("NO SIGNAL")
        if self.resolution:
            frame_bgr = cv2.resize(frame_bgr, self.resolution)
        frame = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        frame.pts, frame.time_base = await self.next_timestamp()
        return frame


class WebRTCManager:
    def __init__(self, camera_registry: CameraRegistry) -> None:
        self.camera_registry = camera_registry
        self.active: dict[str, WebRTCSession] = {}

    def _build_configuration(self) -> RTCConfiguration:
        settings = get_settings()
        ice_servers: list[RTCIceServer] = []
        if settings.webrtc_stun_urls:
            stun_urls = [url.strip() for url in settings.webrtc_stun_urls.split(",") if url.strip()]
            if stun_urls:
                ice_servers.append(RTCIceServer(urls=stun_urls))
        if settings.webrtc_turn_url:
            ice_servers.append(
                RTCIceServer(
                    urls=[settings.webrtc_turn_url],
                    username=settings.webrtc_turn_user or None,
                    credential=settings.webrtc_turn_pass or None,
                )
            )
        return RTCConfiguration(iceServers=ice_servers)

    async def create_peer(self, camera_id: int) -> RTCPeerConnection:
        peer = RTCPeerConnection(configuration=self._build_configuration())
        track = CameraVideoTrack(camera_id=camera_id, camera_registry=self.camera_registry)
        sender = peer.addTrack(track)
        settings = get_settings()
        desired_codec = settings.webrtc_video_codec.upper()
        capabilities = RTCRtpSender.getCapabilities("video")
        if capabilities and capabilities.codecs:
            preferred = [codec for codec in capabilities.codecs if codec.name.upper() == desired_codec]
            if preferred:
                sender.setCodecPreferences(preferred)

        @peer.on("connectionstatechange")
        async def on_state_change() -> None:
            if peer.connectionState in {"failed", "closed", "disconnected"}:
                await self.close_peer(peer)

        return peer

    async def close_peer(self, peer: RTCPeerConnection) -> None:
        for key, session in list(self.active.items()):
            if session.peer == peer:
                self.active.pop(key, None)
        await peer.close()

    async def handle_offer(self, camera_id: int, sdp: str, sdp_type: str) -> RTCSessionDescription:
        settings = get_settings()
        if len(self.active) >= settings.webrtc_max_peers:
            raise RuntimeError("WebRTC peer limit reached.")

        peer = await self.create_peer(camera_id)
        await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))

        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)

        session_id = f"{camera_id}:{id(peer)}"
        self.active[session_id] = WebRTCSession(peer=peer, camera_id=camera_id)
        logger.info("WebRTC peer created: session_id=%s camera_id=%s", session_id, camera_id)

        return peer.localDescription
