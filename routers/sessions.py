"""Session read routes: the list and one Session's state & progress.

Both are public reads (#178): a Visitor browses the same list and the same
deck as a User, so this router carries no sign-in dependency and the flag
never opens or closes it. Ownership begins at the write — see
``routers.clarify`` (creating a Session) and ``routers.plan`` (its plan).
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session as DBSession

from db import Phase, Session, get_db

router = APIRouter(tags=["sessions"])


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
    item for the Explore feed, so a field added here is added there too — and
    the generated client types stay authoritative for both surfaces.

    Both surfaces are public reads, so this shape deliberately carries no owner
    identity — no email, display name, or user id (#144). A field added here is
    a field every Visitor can read.
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
)
def list_sessions(
    phase: list[Phase] | None = Query(default=None),
    db: DBSession = Depends(get_db),
) -> SessionList:
    """List Sessions: every Session record, newest first, optionally filtered
    by phase(s).

    A public read (#178): a Visitor is answered the same list as a User, and no
    cap is applied — the surfaces that show this list have no pager (#143), so
    everything listed has to be in the answer. Which phases to show is the
    caller's question, not a server-side policy: the app asks for the Sessions
    that reached materials, and nothing hides the rest from a Visitor.

    The Explore feed (``routers.explore``) is the other list surface, and is
    uncapped for the same reason; it stays a separate route because the two are
    expected to diverge.

    Records are not scoped per User yet: every caller is shown the same list
    (see ``tests/routers/test_auth_gate.TestSessionsSharedAcrossUsers``), and
    the payload carries no owner identity.
    """
    query = db.query(Session).order_by(Session.created_at.desc())
    if phase is not None:
        query = query.filter(Session.phase.in_([p.value for p in phase]))
    return SessionList(sessions=[session_list_item(s) for s in query.all()])


@router.get("/sessions/{session_id}", response_model=SessionState)
def get_session(session_id: str, db: DBSession = Depends(get_db)) -> SessionState:
    """Get current session state & progress.

    A public read (#178): opening one Session's deck is browsing, which is what
    a Visitor does (ADR-0004).
    """
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
