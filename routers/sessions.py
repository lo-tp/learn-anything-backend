"""Session status routes: state & progress read-back."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session as DBSession

from core.security import require_auth, require_sign_in
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


def session_list_item(session: Session) -> SessionListItem:
    """Build one list item from a ``Session`` row.

    The only place this shape is built: ``routers.explore`` renders the same
    item for the public Explore feed, so a field added here is added there too
    — and the generated client types stay authoritative for both surfaces.
    """
    return SessionListItem(
        session_id=session.session_id,
        phase=Phase(session.phase),
        goal=session.goal,
        narrowed_goal=session.narrowed_goal,
        created_at=session.created_at,
    )


# --- Routes ---


@router.get(
    "/sessions",
    response_model=SessionList,
    # History is the surface a User works in, so it requires a sign-in cookie
    # whatever DEV_MODE says (#145): an anonymous caller gets 401, not an empty
    # list. GET /sessions/{session_id} keeps the bounded gate — opening one
    # Session's deck is browsing, and a Visitor is expected to do that.
    dependencies=[Depends(require_sign_in)],
)
def list_sessions(
    phase: list[Phase] | None = Query(default=None),
    db: DBSession = Depends(get_db),
) -> SessionList:
    """List History: every Session record, newest first, optionally filtered by
    phase(s).

    Requires a sign-in cookie (#145). Records are not scoped per User yet: every
    signed-in caller is shown the same list (see
    ``tests/routers/test_auth_gate.TestSessionsSharedAcrossUsers``).
    """
    query = db.query(Session).order_by(Session.created_at.desc())
    if phase is not None:
        query = query.filter(Session.phase.in_([p.value for p in phase]))
    return SessionList(sessions=[session_list_item(s) for s in query.all()])


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
