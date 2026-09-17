"""Probing phase: single combined endpoint driving the Probe graph loop.

The first call starts the probe (no answers); subsequent calls submit the
learner's answers for the whole current batch at once.
"""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from core.language import DEFAULT_LANGUAGE
from db import Phase, ProbeQuestion, Session, get_db
from graphs import graph_config, probe_graph

router = APIRouter(tags=["probe"])


# --- Schemas ---


class AnswerIn(BaseModel):
    question_id: uuid.UUID
    selected_index: int


class ProbeIn(BaseModel):
    answers: list[AnswerIn] | None = None


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
    questions: list[ProbeQuestionOut] | None = None
    boundary_map: dict[str, dict] | None = None


# --- Helpers ---


def _get_session_or_404(db: DBSession, session_id: str) -> Session:
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _persist_batch(
    db: DBSession, session_id: str, questions: list[dict]
) -> list[ProbeQuestion]:
    """Insert new ProbeQuestion rows from the graph's batch of question dicts."""
    rows = [
        ProbeQuestion(
            id=q["id"],
            session_id=session_id,
            question_id=uuid.UUID(q["id"]),
            text=q["text"],
            options=q["options"],
            correct_index=q["correct_index"],
            explanation=q["explanation"],
            strand=q["strand"],
            difficulty=q["difficulty"],
        )
        for q in questions
    ]
    db.add_all(rows)
    db.commit()
    return rows


def _update_answers(
    db: DBSession,
    questions: list[ProbeQuestion],
    answers: list[tuple[uuid.UUID, int]],
) -> None:
    """Mark ProbeQuestion rows as answered."""
    by_id = {q.question_id: q for q in questions}
    answered_at = datetime.now(UTC)
    for question_id, selected_index in answers:
        question = by_id[question_id]
        question.selected_index = selected_index
        question.is_correct = selected_index == question.correct_index
        question.answered_at = answered_at
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
    """Start the probe or submit the answers for the current batch (combined endpoint)."""
    session = _get_session_or_404(db, session_id)
    if session.phase != Phase.PROBING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Session is in phase '{session.phase}', not 'probing'",
        )

    config = graph_config(session_id, "probe")

    if body.answers is None:
        # --- First call: start the probe graph ---
        result = probe_graph.invoke(
            {
                "goal": session.narrowed_goal or session.goal,
                "language": session.language or DEFAULT_LANGUAGE,
                "strands": [],
                "history": [],
                "boundary_map": {},
                "question_count": 0,
            },
            config,
        )
        questions = result.get("next_batch")
        if not questions:
            raise HTTPException(status_code=500, detail="Graph did not produce questions")

        _persist_batch(db, session_id, questions)
        return ProbeOut(
            phase=Phase.PROBING,
            questions=[ProbeQuestionOut(**q) for q in questions],
        )

    # --- Subsequent call: resume with the learner's batch of answers ---
    by_id = {q.question_id: q for q in session.probe_questions}

    # Validate: every submitted question exists in this session.
    if any(a.question_id not in by_id for a in body.answers):
        raise HTTPException(status_code=404, detail="Question not found")
    # Validate: no submitted question has already been answered.
    if any(by_id[a.question_id].answered_at is not None for a in body.answers):
        raise HTTPException(status_code=409, detail="Question already answered")
    # Validate: the answers cover the whole current batch — no partial submission.
    unanswered_ids = {
        q.question_id for q in session.probe_questions if q.answered_at is None
    }
    if {a.question_id for a in body.answers} != unanswered_ids:
        raise HTTPException(
            status_code=422,
            detail="All questions in the current batch must be answered.",
        )
    # Validate: each selected_index is in range for its question's option count.
    for a in body.answers:
        question = by_id[a.question_id]
        if not 0 <= a.selected_index < len(question.options):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"selected_index for question {a.question_id} "
                    f"must be 0-{len(question.options) - 1}"
                ),
            )

    result = probe_graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": str(a.question_id), "selected_index": a.selected_index}
                    for a in body.answers
                ]
            }
        ),
        config,
    )

    answered_questions = [by_id[a.question_id] for a in body.answers]
    answer_pairs = [(a.question_id, a.selected_index) for a in body.answers]

    if "__interrupt__" in result:
        # Next batch — update the answered rows, persist the new batch
        _update_answers(db, answered_questions, answer_pairs)
        next_batch = result.get("next_batch")
        if not next_batch:
            raise HTTPException(status_code=500, detail="Graph did not produce questions")
        _persist_batch(db, session_id, next_batch)
        return ProbeOut(
            phase=Phase.PROBING,
            questions=[ProbeQuestionOut(**q) for q in next_batch],
        )

    # Probe complete — boundary_map is the exit output
    _update_answers(db, answered_questions, answer_pairs)
    boundary_map = result.get("boundary_map", {})
    session.boundary_map = boundary_map
    session.phase = Phase.PLANNING.value
    db.commit()
    return ProbeOut(phase=Phase.PLANNING, boundary_map=boundary_map)
