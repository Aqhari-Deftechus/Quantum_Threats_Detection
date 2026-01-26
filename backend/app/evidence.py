from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from .config import get_settings
from .models import EvidenceEvent


@dataclass
class ClipResult:
    clip_path: Optional[str]
    clip_sha256: Optional[str]


def _hash_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_event_clip(session: Session, camera_id: int, event_type: str) -> EvidenceEvent:
    settings = get_settings()
    clip_dir = settings.clip_storage_dir_resolved
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_path = clip_dir / f"event_{camera_id}_stub.mp4"
    if not clip_path.exists():
        clip_path.write_bytes(b"")

    clip_sha256 = _hash_file(clip_path)
    event = EvidenceEvent(
        camera_id=camera_id,
        event_type=event_type,
        clip_path=str(clip_path),
        clip_sha256=clip_sha256,
        integrity_status="OK",
        confidence=0.5,
    )
    session.add(event)
    session.commit()
    session.refresh(event)
    return event
