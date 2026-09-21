"""Review-card persistence shared by the review router and the probe hook.

Both capture paths (the ``POST /review/cards`` material capture and the
server-side probe miss) call ``record_missed_question`` so a card is recorded
identically no matter where the miss happened.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session as DBSession

from db import ReviewCard


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
    """Upsert a review card for a missed question; a re-miss resets its SRS.

    One card per unique ``(user_id, source, session_id, source_question_id)``.
    A re-miss resets the card to a fresh schedule (interval 0, ease 2.5,
    due now, not retired) and counts the lapse.
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
            interval_days=0,
            ease=2.5,
            lapses=0,
            due_at=now,
            is_retired=False,
        )
        db.add(card)
    else:
        # Re-miss: reset to a fresh card and count the lapse.
        card.interval_days = 0
        card.ease = 2.5
        card.lapses += 1
        card.due_at = now
        card.is_retired = False
    db.commit()
    return card


def get_due_cards(
    db: DBSession, user_id: int, limit: int = 20
) -> list[ReviewCard]:
    """The learner's due, non-retired cards ordered by ``due_at`` (then id)."""
    now = datetime.now(UTC)
    return (
        db.query(ReviewCard)
        .filter(
            ReviewCard.user_id == user_id,
            ReviewCard.is_retired.is_(False),
            ReviewCard.due_at <= now,
        )
        .order_by(ReviewCard.due_at, ReviewCard.id)
        .limit(limit)
        .all()
    )


def count_due(db: DBSession, user_id: int) -> int:
    """Count the learner's due, non-retired cards."""
    now = datetime.now(UTC)
    return (
        db.query(ReviewCard)
        .filter(
            ReviewCard.user_id == user_id,
            ReviewCard.is_retired.is_(False),
            ReviewCard.due_at <= now,
        )
        .count()
    )
