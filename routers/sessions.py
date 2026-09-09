"""Session lifecycle routes: creation, clarification loop, status read-back."""

import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from db import Phase, Session, get_db
from graphs import clarify_graph, graph_config
from graphs.clarify import ClarifyState

router = APIRouter(tags=["sessions"])


# --- Schemas ---


class GoalIn(BaseModel):
    goal: str


class ClarifyIn(BaseModel):
    answer: str


class ClarifyResult(BaseModel):
    """Outcome of a Clarify graph call: either questions or a narrowed goal."""

    session_id: str
    phase: Literal["clarifying", "probing"]
    narrowed_goal: str | None = None
    clarifying_questions: list[str] | None = None


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


# --- Helpers ---


def initial_clarify_state(goal: str) -> ClarifyState:
    return {
        "goal": goal,
        "working_goal": goal,
        "open_dimensions": [],
        "round_count": 0,
        "clarifying_questions": [],
        "last_answer": None,
        "narrowed_goal": None,
    }


def interpret_clarify(
    result: dict[str, Any], session: Session, db: DBSession
) -> ClarifyResult:
    """Map a Clarify graph run result onto the session row + response.

    The graph either ended with a ``narrowed_goal`` (advance to probing) or
    paused at the interrupt with ``clarifying_questions`` (stay clarifying).
    """
    if "__interrupt__" in result:
        return ClarifyResult(
            session_id=session.session_id,
            phase=Phase.CLARIFYING.value,
            clarifying_questions=result.get("clarifying_questions", []),
        )
    session.narrowed_goal = result["narrowed_goal"]
    session.phase = Phase.PROBING.value
    db.commit()
    return ClarifyResult(
        session_id=session.session_id,
        phase=Phase.PROBING.value,
        narrowed_goal=session.narrowed_goal,
    )


# --- Routes ---


@router.post(
    "/sessions",
    response_model=ClarifyResult,
    response_model_exclude_none=True,
)
def create_session(body: GoalIn, db: DBSession = Depends(get_db)) -> ClarifyResult:
    """Create a session and make the first Clarify graph call."""
    session = Session(
        session_id=uuid.uuid4().hex, goal=body.goal, phase=Phase.CLARIFYING.value
    )
    db.add(session)
    db.commit()
    result = clarify_graph.invoke(
        initial_clarify_state(body.goal),
        graph_config(session.session_id, "clarify"),
    )
    return interpret_clarify(result, session, db)


@router.post(
    "/sessions/{session_id}/clarify",
    response_model=ClarifyResult,
    response_model_exclude_none=True,
)
def clarify_session(
    session_id: str, body: ClarifyIn, db: DBSession = Depends(get_db)
) -> ClarifyResult:
    """Resume the Clarify graph loop with the learner's answer."""
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.phase != Phase.CLARIFYING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Session is in phase '{session.phase}', not 'clarifying'",
        )
    result = clarify_graph.invoke(
        Command(resume={"answer": body.answer}),
        graph_config(session_id, "clarify"),
    )
    return interpret_clarify(result, session, db)


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
