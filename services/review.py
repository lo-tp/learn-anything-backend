"""Review-card persistence shared by the review router and the probe hook.

Both capture paths (the ``POST /review/cards`` material capture and the
server-side probe miss) call ``record_missed_question`` so a card is recorded
identically no matter where the miss happened.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fsrs import Card, State
from sqlalchemy.orm import Session as DBSession

from db import ReviewCard


def _fresh_fsrs_state(now: datetime) -> dict[str, Any]:
    """Serialize a fresh FSRS Card (first-learning-step, due now)."""
    return dict(
        Card(
            card_id=0,
            state=State.Learning,
            due=now,
            last_review=now,
            stability=None,
            difficulty=None,
            step=0,
        ).to_dict()
    )


def record_missed_question(
    db: DBSession,
    *,
    user_id: int,
    source: str,
    session_id: str,
    source_question_id: str,
    question: dict,
    step_id: str | None = None,
) -> ReviewCard:
    """Upsert a review card for a missed question; a re-miss resets its FSRS.

    One card per unique ``(user_id, source, session_id, source_question_id)``.
    A re-miss resets the card to a fresh FSRS schedule (New state, due now)
    and counts the lapse.
    """
    card = (
        db.query(ReviewCard)
        .filter(
            ReviewCard.user_id == user_id,
            ReviewCard.source == source,
            ReviewCard.session_id == session_id,
            ReviewCard.source_question_id == source_question_id,
        )
        .first()
    )
    now = datetime.now(UTC)
    if card is None:
        card = ReviewCard(
            user_id=user_id,
            source=source,
            session_id=session_id,
            source_question_id=source_question_id,
            question=question,
            step_id=step_id,
            fsrs_state=_fresh_fsrs_state(now),
            lapses=0,
            due_at=now,
        )
        db.add(card)
    else:
        # Re-miss: reset to a fresh FSRS card and count the lapse.
        card.fsrs_state = _fresh_fsrs_state(now)
        card.lapses += 1
        card.due_at = now
    db.commit()
    return card


def get_due_cards(
    db: DBSession, user_id: int, limit: int = 20
) -> list[ReviewCard]:
    """The learner's due cards ordered by ``due_at`` (then id)."""
    now = datetime.now(UTC)
    return (
        db.query(ReviewCard)
        .filter(
            ReviewCard.user_id == user_id,
            ReviewCard.due_at <= now,
        )
        .order_by(ReviewCard.due_at, ReviewCard.id)
        .limit(limit)
        .all()
    )


def count_due(db: DBSession, user_id: int) -> int:
    """Count the learner's due cards."""
    now = datetime.now(UTC)
    return (
        db.query(ReviewCard)
        .filter(
            ReviewCard.user_id == user_id,
            ReviewCard.due_at <= now,
        )
        .count()
    )
