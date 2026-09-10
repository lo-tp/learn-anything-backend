"""Probing phase: single combined endpoint driving the Probe graph loop."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from db import Phase, ProbeQuestion, Session, get_db
from graphs import graph_config, probe_graph

router = APIRouter(tags=["probe"])


# --- Schemas ---


class ProbeIn(BaseModel):
    question_id: str | None = None
    selected_index: int | None = None


class ProbeQuestionOut(BaseModel):
    id: str
    text: str
    options: list[str]
    correct_index: int
    explanation: str
    strand: str
    difficulty: int


class ProbeOut(BaseModel):
    phase: Phase
    question: ProbeQuestionOut | None = None
    boundary_map: dict[str, dict] | None = None


# --- Helpers ---


def _get_session_or_404(db: DBSession, session_id: str) -> Session:
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _persist_question(
    db: DBSession, session_id: str, q: dict
) -> ProbeQuestion:
    """Insert a new ProbeQuestion row from the graph's question dict."""
    row = ProbeQuestion(
        id=q["id"],
        session_id=session_id,
        question_id=q["id"],
        text=q["text"],
        options=q["options"],
        correct_index=q["correct_index"],
        explanation=q["explanation"],
        strand=q["strand"],
        difficulty=q["difficulty"],
    )
    db.add(row)
    db.commit()
    return row


def _update_answer(
    db: DBSession, question: ProbeQuestion, selected_index: int
) -> None:
    """Mark a ProbeQuestion row as answered."""
    question.selected_index = selected_index
    question.is_correct = selected_index == question.correct_index
    question.answered_at = datetime.now(UTC)
    db.commit()


# --- Route ---


@router.post(
    "/sessions/{session_id}/probe",
    response_model=ProbeOut,
    response_model_exclude_none=True,
)
def probe_session(
    session_id: str, body: ProbeIn, db: DBSession = Depends(get_db)
) -> ProbeOut:
    """Start the probe or submit the next answer (combined endpoint)."""
    session = _get_session_or_404(db, session_id)
    if session.phase != Phase.PROBING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Session is in phase '{session.phase}', not 'probing'",
        )

    config = graph_config(session_id, "probe")

    if body.question_id is None:
        # --- First call: start the probe graph ---
        result = probe_graph.invoke(
            {
                "goal": session.narrowed_goal or session.goal,
                "history": [],
                "boundary_map": {},
                "question_count": 0,
            },
            config,
        )
        question = result.get("next_question")
        if question is None:
            raise HTTPException(status_code=500, detail="Graph did not produce a question")

        _persist_question(db, session_id, question)
        return ProbeOut(phase=Phase.PROBING, question=ProbeQuestionOut(**question))

    # --- Subsequent call: resume with the learner's answer ---
    if body.selected_index is None:
        raise HTTPException(status_code=422, detail="selected_index is required")

    # Look up the question to validate
    question = next(
        (q for q in session.probe_questions if q.question_id == body.question_id),
        None,
    )
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    if question.answered_at is not None:
        raise HTTPException(status_code=409, detail="Question already answered")
    if not 0 <= body.selected_index < len(question.options):
        raise HTTPException(
            status_code=422,
            detail=f"selected_index must be 0-{len(question.options) - 1}",
        )

    result = probe_graph.invoke(
        Command(resume={"question_id": body.question_id, "selected_index": body.selected_index}),
        config,
    )

    if "__interrupt__" in result:
        # Next question — update previous row, persist new one
        _update_answer(db, question, body.selected_index)
        next_q = result.get("next_question")
        if next_q is None:
            raise HTTPException(status_code=500, detail="Graph did not produce a question")
        _persist_question(db, session_id, next_q)
        return ProbeOut(phase=Phase.PROBING, question=ProbeQuestionOut(**next_q))

    # Probe complete — boundary_map is the exit output
    _update_answer(db, question, body.selected_index)
    boundary_map = result.get("boundary_map", {})
    session.boundary_map = boundary_map
    session.phase = Phase.PLANNING.value
    db.commit()
    return ProbeOut(phase=Phase.PLANNING, boundary_map=boundary_map)
