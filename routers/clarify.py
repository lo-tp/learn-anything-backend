"""Clarify phase: session creation (first Clarify call) and the clarify loop."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from db import Phase, Session, get_db
from graphs import clarify_graph, graph_config
from graphs.clarify import ClarifyState

router = APIRouter(tags=["clarify"])


# --- Schemas ---


class GoalIn(BaseModel):
    goal: str


class ClarifyIn(BaseModel):
    answer: str


class ClarifyResult(BaseModel):
    """Outcome of a Clarify graph call: either questions or a narrowed goal."""

    session_id: str
    phase: Phase
    narrowed_goal: str | None = None
    clarifying_questions: list[str] | None = None


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
            phase=Phase.CLARIFYING,
            clarifying_questions=result.get("clarifying_questions", []),
        )
    session.narrowed_goal = result["narrowed_goal"]
    session.phase = Phase.PROBING.value
    db.commit()
    return ClarifyResult(
        session_id=session.session_id,
        phase=Phase.PROBING,
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
