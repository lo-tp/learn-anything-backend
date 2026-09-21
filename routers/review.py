"""Review API: the learner's personal spaced-repetition deck.

Per-user (unlike the shared-session routers): every endpoint resolves the
current user via ``security.get_current_user`` and scopes by ``user_id``.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from core import security
from core.srs import next_due_at, sm2_apply
from db import ReviewCard, User, get_db
from services.review import count_due, get_due_cards, record_missed_question

router = APIRouter(tags=["review"], dependencies=[Depends(security.get_current_user)])


# --- Schemas ---


class ReviewQuestionIn(BaseModel):
    text: str
    options: list[str] = Field(min_length=2)
    correct_index: int = Field(ge=0)
    explanation: str


class ReviewCardIn(BaseModel):
    source: str = Field(pattern="^(probe|material)$")
    session_id: str
    question_id: str
    question: ReviewQuestionIn
    step_id: str | None = None
    selected_index: int | None = None  # provenance (the wrong pick)


class ReviewQuestionOut(BaseModel):
    text: str
    options: list[str]
    correct_index: int
    explanation: str


class ReviewCardOut(BaseModel):
    id: int
    source: str
    question: ReviewQuestionOut
    session_id: str
    step_id: str | None = None
    due_at: datetime
    interval_days: int
    ease: float
    lapses: int


class ReviewAnswerIn(BaseModel):
    selected_index: int


class ReviewAnswerOut(BaseModel):
    was_correct: bool
    due_at: datetime
    interval_days: int
    ease: float
    lapses: int
    is_retired: bool


class ReviewSummaryOut(BaseModel):
    due_count: int
    total_active: int
    total_retired: int


def _card_out(card: ReviewCard) -> ReviewCardOut:
    return ReviewCardOut(
        id=card.id,
        source=card.source,
        question=ReviewQuestionOut(**card.question),
        session_id=card.session_id,
        step_id=card.step_id,
        due_at=card.due_at,
        interval_days=card.interval_days,
        ease=card.ease,
        lapses=card.lapses,
    )


# --- Routes ---


@router.post("/review/cards", response_model=ReviewCardOut, status_code=201)
def record_card(
    body: ReviewCardIn,
    user: User = Depends(security.get_current_user),
    db: DBSession = Depends(get_db),
) -> ReviewCardOut:
    """Capture a missed question (material misses from the client; probe
    misses are recorded server-side). Re-miss resets the card's SRS."""
    if not (0 <= body.question.correct_index < len(body.question.options)):
        raise HTTPException(status_code=422, detail="correct_index out of range")
    if body.selected_index is not None and not (
        0 <= body.selected_index < len(body.question.options)
    ):
        raise HTTPException(status_code=422, detail="selected_index out of range")
    card = record_missed_question(
        db,
        user_id=user.id,
        source=body.source,
        session_id=body.session_id,
        source_question_id=body.question_id,
        question=body.question.model_dump(),
        step_id=body.step_id,
    )
    return _card_out(card)


@router.get("/review/due", response_model=list[ReviewCardOut])
def due_cards(
    user: User = Depends(security.get_current_user),
    db: DBSession = Depends(get_db),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[ReviewCardOut]:
    """The learner's due, non-retired cards ordered by ``due_at``."""
    return [_card_out(c) for c in get_due_cards(db, user.id, limit)]


@router.post("/review/cards/{card_id}/answer", response_model=ReviewAnswerOut)
def answer_card(
    card_id: int,
    body: ReviewAnswerIn,
    user: User = Depends(security.get_current_user),
    db: DBSession = Depends(get_db),
) -> ReviewAnswerOut:
    """Submit an answer: update the card's SRS state and return the result."""
    card = db.get(ReviewCard, card_id)
    if card is None or card.user_id != user.id:
        raise HTTPException(status_code=404, detail="Card not found")
    if not (0 <= body.selected_index < len(card.question["options"])):
        raise HTTPException(status_code=422, detail="selected_index out of range")
    was_correct = body.selected_index == card.question["correct_index"]
    interval, ease, lapses, retired = sm2_apply(
        card.interval_days, card.ease, card.lapses, was_correct
    )
    card.interval_days = interval
    card.ease = ease
    card.lapses = lapses
    card.is_retired = retired
    card.due_at = next_due_at()
    db.commit()
    return ReviewAnswerOut(
        was_correct=was_correct,
        due_at=card.due_at,
        interval_days=interval,
        ease=ease,
        lapses=lapses,
        is_retired=retired,
    )


@router.get("/review/summary", response_model=ReviewSummaryOut)
def summary(
    user: User = Depends(security.get_current_user),
    db: DBSession = Depends(get_db),
) -> ReviewSummaryOut:
    """Lightweight due / active / retired counts for the top-bar badge."""
    due = count_due(db, user.id)
    active = (
        db.query(ReviewCard)
        .filter(ReviewCard.user_id == user.id, ReviewCard.is_retired.is_(False))
        .count()
    )
    retired = (
        db.query(ReviewCard)
        .filter(ReviewCard.user_id == user.id, ReviewCard.is_retired.is_(True))
        .count()
    )
    return ReviewSummaryOut(due_count=due, total_active=active, total_retired=retired)
