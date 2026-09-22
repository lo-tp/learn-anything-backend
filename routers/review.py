"""Review API: the learner's personal spaced-repetition deck.

Per-user (unlike the shared-session routers): every endpoint resolves the
current user via ``security.get_current_user`` and scopes by ``user_id``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from fsrs import Card, Rating
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from core import security
from core.srs import fsrs_apply
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
    lapses: int


class ReviewAnswerIn(BaseModel):
    rating: Literal["again", "hard", "good", "easy"]


class ReviewAnswerOut(BaseModel):
    due_at: datetime
    interval_days: float
    lapses: int


class ReviewSummaryOut(BaseModel):
    due_count: int
    total_cards: int


_RATING_MAP: dict[str, Rating] = {
    "again": Rating.Again,
    "hard": Rating.Hard,
    "good": Rating.Good,
    "easy": Rating.Easy,
}


def _card_out(card: ReviewCard) -> ReviewCardOut:
    return ReviewCardOut(
        id=card.id,
        source=card.source,
        question=ReviewQuestionOut(**card.question),
        session_id=card.session_id,
        step_id=card.step_id,
        due_at=card.due_at,
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
    """The learner's due cards ordered by ``due_at``."""
    return [_card_out(c) for c in get_due_cards(db, user.id, limit)]


@router.post("/review/cards/{card_id}/answer", response_model=ReviewAnswerOut)
def answer_card(
    card_id: int,
    body: ReviewAnswerIn,
    user: User = Depends(security.get_current_user),
    db: DBSession = Depends(get_db),
) -> ReviewAnswerOut:
    """Submit a confidence rating: update the card's FSRS state and return
    the new due date, derived interval, and lapse count."""
    card = db.get(ReviewCard, card_id)
    if card is None or card.user_id != user.id:
        raise HTTPException(status_code=404, detail="Card not found")

    rating = _RATING_MAP[body.rating]
    now = datetime.now(UTC)
    fsrs_card = Card.from_dict(card.fsrs_state)  # type: ignore[arg-type]
    interval_days, new_due, new_lapses, new_fsrs_card = fsrs_apply(
        fsrs_card, rating, now=now, lapses=card.lapses
    )

    card.fsrs_state = cast(dict[str, Any], new_fsrs_card.to_dict())
    card.lapses = new_lapses
    card.due_at = new_due
    db.commit()

    return ReviewAnswerOut(
        due_at=card.due_at,
        interval_days=interval_days,
        lapses=new_lapses,
    )


@router.get("/review/summary", response_model=ReviewSummaryOut)
def summary(
    user: User = Depends(security.get_current_user),
    db: DBSession = Depends(get_db),
) -> ReviewSummaryOut:
    """Lightweight due / total counts for the top-bar badge."""
    due = count_due(db, user.id)
    total = db.query(ReviewCard).filter(ReviewCard.user_id == user.id).count()
    return ReviewSummaryOut(due_count=due, total_cards=total)
