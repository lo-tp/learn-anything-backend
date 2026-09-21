"""Session status routes: state & progress read-back."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session as DBSession

from core.security import require_auth
from db import Phase, Session, get_db

router = APIRouter(tags=["sessions"], dependencies=[Depends(require_auth)])


# --- Schemas ---


class Progress(BaseModel):
    current_step_id: str | None
    completed_steps: list[str]
    total_steps: int
    step_scores: dict[str, float]


class SessionState(BaseModel):
    session_id: str
    phase: Phase
    narrowed_goal: str | None
    progress: Progress


class SessionListItem(BaseModel):
    session_id: str
    phase: Phase
    goal: str
    narrowed_goal: str | None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        # The OpenAPI contract declares created_at as RFC 3339 date-time,
        # which requires an explicit offset. Normalize to UTC so clients
        # never parse a bare wall-clock string as local time (this also
        # covers legacy rows written before the timestamptz migration).
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class SessionList(BaseModel):
    sessions: list[SessionListItem]


# --- Routes ---


@router.get("/sessions", response_model=SessionList)
def list_sessions(
    phase: list[Phase] | None = Query(default=None),
    db: DBSession = Depends(get_db),
) -> SessionList:
    """List all sessions, newest first, optionally filtered by phase(s)."""
    query = db.query(Session).order_by(Session.created_at.desc())
    if phase is not None:
        query = query.filter(Session.phase.in_([p.value for p in phase]))
    return SessionList(
        sessions=[
            SessionListItem(
                session_id=s.session_id,
                phase=Phase(s.phase),
                goal=s.goal,
                narrowed_goal=s.narrowed_goal,
                created_at=s.created_at,
            )
            for s in query.all()
        ]
    )


@router.get("/sessions/{session_id}", response_model=SessionState)
def get_session(session_id: str, db: DBSession = Depends(get_db)) -> SessionState:
    """Get current session state & progress."""
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    plan = session.plan
    total_steps = len(plan.steps) if plan else 0
    completed_steps = [p.step_id for p in session.step_progress if p.complete]
    step_scores: dict[str, float] = {}
    for p in session.step_progress:
        if p.score is not None:
            step_scores[p.step_id] = p.score
    current_step_id: str | None = None
    if plan:
        for step in plan.steps:
            if step["id"] not in completed_steps:
                current_step_id = step["id"]
                break
    return SessionState(
        session_id=session_id,
        phase=Phase(session.phase),
        narrowed_goal=session.narrowed_goal,
        progress=Progress(
            current_step_id=current_step_id,
            completed_steps=completed_steps,
            total_steps=total_steps,
            step_scores=step_scores,
        ),
    )
