from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import AuditEvent


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _hash_event(prev_hash: str, payload: dict[str, Any], timestamp: datetime) -> str:
    data = f"{prev_hash}{_canonical_json(payload)}{timestamp.isoformat()}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def create_audit_event(session: Session, event_type: str, payload: dict[str, Any], actor: str = "system") -> AuditEvent:
    settings = get_settings()
    latest = session.scalars(
        select(AuditEvent).where(AuditEvent.hash_chain_id == settings.audit_chain_id).order_by(AuditEvent.id.desc())
    ).first()
    prev_hash = latest.event_hash if latest else "GENESIS"
    created_at = datetime.now(timezone.utc)
    event_hash = _hash_event(prev_hash, payload, created_at)
    event = AuditEvent(
        event_type=event_type,
        actor=actor,
        payload=_canonical_json(payload),
        created_at=created_at,
        hash_chain_id=settings.audit_chain_id,
        prev_event_hash=prev_hash,
        event_hash=event_hash,
        hash_algo="SHA256",
    )
    session.add(event)
    session.commit()
    session.refresh(event)
    return event


def verify_chain(session: Session) -> tuple[bool, Optional[str]]:
    settings = get_settings()
    events = session.scalars(
        select(AuditEvent).where(AuditEvent.hash_chain_id == settings.audit_chain_id).order_by(AuditEvent.id)
    ).all()
    prev_hash = "GENESIS"
    for event in events:
        payload = json.loads(event.payload)
        expected = _hash_event(prev_hash, payload, event.created_at)
        if expected != event.event_hash:
            return False, event.event_hash
        prev_hash = event.event_hash
    return True, None
