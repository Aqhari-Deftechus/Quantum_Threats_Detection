from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..audit import create_audit_event
from ..db import get_session
from ..models import Identity, IdentityEmbedding
from ..schemas import EmbeddingEnrollRequest, IdentityCreate, IdentityOut, IdentityUpdate
from ..state import matcher

router = APIRouter(prefix="/identities")


def _to_blob(embedding: np.ndarray) -> bytes:
    return embedding.astype("float32").tobytes()


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms


def _update_centroid(session: Session, identity: Identity) -> None:
    embeddings = session.scalars(select(IdentityEmbedding).where(IdentityEmbedding.identity_id == identity.id)).all()
    if not embeddings:
        identity.centroid = None
        session.commit()
        return
    vectors = [np.frombuffer(embedding.embedding, dtype="float32") for embedding in embeddings]
    centroid = np.mean(np.vstack(vectors), axis=0).astype("float32")
    centroid = _normalize_embeddings(centroid.reshape(1, -1))[0]
    identity.centroid = _to_blob(centroid)
    identity.updated_at = datetime.now(timezone.utc)
    session.commit()


def _add_embeddings_background(embeddings: np.ndarray, identity_ids: list[int], identity_names: dict[int, str]) -> None:
    matcher.add_embeddings(embeddings, identity_ids, identity_names)


def _identity_out(session: Session, identity: Identity) -> IdentityOut:
    count = session.scalar(
        select(func.count()).select_from(IdentityEmbedding).where(IdentityEmbedding.identity_id == identity.id)
    )
    return IdentityOut(
        id=identity.id,
        name=identity.name,
        notes=identity.notes,
        created_at=identity.created_at,
        updated_at=identity.updated_at,
        embedding_count=int(count or 0),
    )


@router.get("", response_model=list[IdentityOut])
def list_identities(session: Session = Depends(get_session)) -> list[IdentityOut]:
    identities = session.scalars(select(Identity)).all()
    return [_identity_out(session, identity) for identity in identities]


@router.post("", response_model=IdentityOut)
def create_identity(payload: IdentityCreate, session: Session = Depends(get_session)) -> IdentityOut:
    identity = Identity(name=payload.name, notes=payload.notes)
    session.add(identity)
    session.commit()
    session.refresh(identity)
    create_audit_event(session, "identity_created", {"identity_id": identity.id, "name": identity.name})
    return _identity_out(session, identity)


@router.put("/{identity_id}", response_model=IdentityOut)
def update_identity(identity_id: int, payload: IdentityUpdate, session: Session = Depends(get_session)) -> IdentityOut:
    identity = session.get(Identity, identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(identity, key, value)
    identity.updated_at = datetime.now(timezone.utc)
    session.commit()
    create_audit_event(session, "identity_updated", {"identity_id": identity.id, "name": identity.name})
    return _identity_out(session, identity)


@router.post("/{identity_id}/embeddings", response_model=IdentityOut)
def enroll_embedding(
    identity_id: int,
    payload: EmbeddingEnrollRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> IdentityOut:
    identity = session.get(Identity, identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found")

    if payload.embeddings:
        embedding_array = np.array(payload.embeddings, dtype="float32")
    else:
        embedding_array = np.random.random((1, matcher.dimension)).astype("float32")

    if embedding_array.ndim != 2 or embedding_array.shape[1] != matcher.dimension:
        raise HTTPException(
            status_code=400,
            detail=f"Embeddings must be shape (n, {matcher.dimension})",
        )

    normalized = _normalize_embeddings(embedding_array)
    for vector in normalized:
        session.add(IdentityEmbedding(identity_id=identity.id, embedding=_to_blob(vector)))
    session.commit()

    _update_centroid(session, identity)
    create_audit_event(session, "identity_embedding_enrolled", {"identity_id": identity.id})

    background_tasks.add_task(
        _add_embeddings_background,
        normalized,
        [identity.id] * normalized.shape[0],
        {identity.id: identity.name},
    )

    return _identity_out(session, identity)
