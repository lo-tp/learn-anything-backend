"""Unit tests for the ReviewCard model (issue #116).

Exercises the ORM model against a real in-memory SQLite database via
``Base.metadata.create_all`` — this is the seam the model ticket owns; the
migration (Alembic) is validated separately with ``alembic upgrade head``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fsrs import Card, State
from sqlalchemy.exc import IntegrityError

from db import ReviewCard, ReviewSource, User


def _user_id(db) -> int:
    """Insert a real ``User`` row and return its id (the card's FK target)."""
    user = User(email="review@example.com", display_name="R", password_hash="x")
    db.add(user)
    db.commit()
    return user.id


def _question() -> dict:
    return {
        "text": "2 + 2 = ?",
        "options": ["3", "4"],
        "correct_index": 1,
        "explanation": "2 + 2 = 4",
    }


def _fresh_fsrs_state(now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    return dict(Card(due=now, last_review=now, state=State.Learning, step=0,
                     stability=None, difficulty=None, card_id=0).to_dict())


def test_review_source_values():
    assert ReviewSource.PROBE.value == "probe"
    assert ReviewSource.MATERIAL.value == "material"


def test_create_and_read_card(db):
    user_id = _user_id(db)
    card = ReviewCard(
        user_id=user_id,
        source=ReviewSource.PROBE.value,
        source_question_id="q1",
        session_id="s1",
        step_id=None,
        question=_question(),
        fsrs_state=_fresh_fsrs_state(),
        lapses=0,
        due_at=datetime.now(UTC),
    )
    db.add(card)
    db.commit()

    got = db.get(ReviewCard, card.id)
    assert got is not None
    assert got.user_id == user_id
    assert got.source == "probe"
    assert got.source_question_id == "q1"
    assert got.session_id == "s1"
    assert got.step_id is None
    assert got.question == _question()
    assert got.lapses == 0
    assert got.fsrs_state is not None
    assert got.created_at is not None
    assert got.updated_at is not None


def test_defaults_applied(db):
    """Omitted SRS/state fields fall back to the model defaults."""
    user_id = _user_id(db)
    card = ReviewCard(
        user_id=user_id,
        source=ReviewSource.MATERIAL.value,
        source_question_id="q2",
        session_id="s2",
        step_id="step_1",
        question=_question(),
        fsrs_state=_fresh_fsrs_state(),
    )
    db.add(card)
    db.commit()

    assert card.lapses == 0
    assert card.due_at is not None
    assert card.source == "material"
    assert card.step_id == "step_1"


def test_unique_identity_constraint(db):
    """One card per unique (user_id, source, session_id, source_question_id)."""
    user_id = _user_id(db)
    base = {
        "user_id": user_id,
        "source": "probe",
        "session_id": "s1",
        "source_question_id": "q1",
        "question": _question(),
        "fsrs_state": _fresh_fsrs_state(),
    }
    db.add(ReviewCard(**base))
    db.commit()

    db.add(ReviewCard(**base))
    with pytest.raises(IntegrityError):
        db.commit()
