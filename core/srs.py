"""SRS (spaced repetition) FSRS wrapper.

Pure, no I/O — unit-testable in isolation (see ``tests/core/test_srs_fsrs.py``).

The FSRS wrapper (``fsrs_apply``) tracks lapses locally — a lapse is counted
when a card in Review state is rated Again.
"""

from __future__ import annotations

from datetime import datetime

from fsrs import Card, Rating, Scheduler, State

_fsrs_scheduler = Scheduler(enable_fuzzing=False)


def fsrs_apply(
    card: Card,
    rating: Rating,
    now: datetime,
    lapses: int = 0,
) -> tuple[float, datetime, int, Card]:
    """Apply one FSRS review and return ``(new_interval_days, new_due, new_lapses, new_card)``.

    - ``new_interval_days`` is the fractional days until the card is due.
    - ``new_due`` is the absolute due datetime.
    - ``new_lapses`` is the local lapse count (incremented when a card in
      Review state is rated Again).
    - ``new_card`` is the updated FSRS card state for the next review.
    """
    is_lapse = card.state == State.Review and rating == Rating.Again
    new_lapses = lapses + (1 if is_lapse else 0)
    new_card, _log = _fsrs_scheduler.review_card(card, rating, review_datetime=now)
    interval_days = (new_card.due - now).total_seconds() / 86_400
    return interval_days, new_card.due, new_lapses, new_card
