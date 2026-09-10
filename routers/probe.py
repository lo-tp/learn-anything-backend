"""Probing phase routes: retrieve questions, submit answers."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from db import Phase, Session, get_db

router = APIRouter(tags=["probe"])


# --- Schemas ---


class ProbeQuestionOut(BaseModel):
    question_id: str
    text: str
    options: list[str]
    strand: str
    difficulty: int
    selected_index: int | None = None
    is_correct: bool | None = None


class ProbeListOut(BaseModel):
    session_id: str
    total: int
    answered: int
    questions: list[ProbeQuestionOut]


class ProbeAnswerIn(BaseModel):
    selected_index: int


class ProbeAnswerOut(BaseModel):
    question_id: str
    selected_index: int
    is_correct: bool
    correct_index: int


# --- Routes ---


def _get_session_or_404(db: DBSession, session_id: str) -> Session:
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions/{session_id}/probe", response_model=ProbeListOut)
def get_probe_questions(
    session_id: str, db: DBSession = Depends(get_db)
) -> ProbeListOut:
    """Get all probe questions for the session."""
    session = _get_session_or_404(db, session_id)
    if session.phase != Phase.PROBING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Session is in phase '{session.phase}', not 'probing'",
        )
    questions = session.probe_questions
    return ProbeListOut(
        session_id=session_id,
        total=len(questions),
        answered=sum(1 for q in questions if q.answered_at is not None),
        questions=[
            ProbeQuestionOut(
                question_id=q.question_id,
                text=q.text,
                options=q.options,
                strand=q.strand,
                difficulty=q.difficulty,
                selected_index=q.selected_index,
                is_correct=q.is_correct,
            )
            for q in questions
        ],
    )


@router.post(
    "/sessions/{session_id}/probe/{question_id}/answer",
    response_model=ProbeAnswerOut,
)
def answer_probe_question(
    session_id: str,
    question_id: str,
    body: ProbeAnswerIn,
    db: DBSession = Depends(get_db),
) -> ProbeAnswerOut:
    """Submit an answer to a probe question."""
    session = _get_session_or_404(db, session_id)
    if session.phase != Phase.PROBING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Session is in phase '{session.phase}', not 'probing'",
        )

    question = next(
        (q for q in session.probe_questions if q.question_id == question_id),
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

    is_correct = body.selected_index == question.correct_index
    question.selected_index = body.selected_index
    question.is_correct = is_correct
    question.answered_at = datetime.now(UTC)
    db.commit()

    return ProbeAnswerOut(
        question_id=question_id,
        selected_index=body.selected_index,
        is_correct=is_correct,
        correct_index=question.correct_index,
    )
