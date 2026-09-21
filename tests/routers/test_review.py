"""Integration tests for the Review API (``routers/review.py``) — #107.

Against the real DB + a real ``User`` row (JWT cookie). Covers CRUD,
per-user scoping, 404s, 422s, and retirement.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from core.security import create_token
from db import ReviewCard, User


def _cookie(user: User) -> dict:
    return {"access_token": create_token(user.email)}


def _question(**overrides) -> dict:
    base = {
        "text": "What is 2+2?",
        "options": ["3", "4", "5"],
        "correct_index": 1,
        "explanation": "4 is correct",
    }
    base.update(overrides)
    return base


def _card_body(qid: str = "q-1", **overrides) -> dict:
    base = {
        "source": "material",
        "session_id": "sess-1",
        "question_id": qid,
        "question": _question(),
    }
    base.update(overrides)
    return base


def _seed_card(db, user_id, *, qid="q-1", due_at=None, is_retired=False, **overrides):
    card = ReviewCard(
        user_id=user_id,
        source="material",
        session_id="sess-1",
        source_question_id=qid,
        question=_question(),
        interval_days=overrides.pop("interval_days", 0),
        ease=overrides.pop("ease", 2.5),
        lapses=overrides.pop("lapses", 0),
        due_at=due_at or datetime.now(UTC),
        is_retired=is_retired,
        **overrides,
    )
    db.add(card)
    db.commit()
    return card


# --- POST /review/cards ---


class TestRecordCard:
    def test_creates_card_201_with_snapshot(self, client, db, user):
        resp = client.post("/review/cards", json=_card_body(), cookies=_cookie(user))
        assert resp.status_code == 201
        body = resp.json()
        assert body["source"] == "material"
        assert body["session_id"] == "sess-1"
        assert body["question"] == _question()
        assert body["interval_days"] == 0
        assert body["ease"] == 2.5
        assert body["lapses"] == 0

        row = db.get(ReviewCard, body["id"])
        assert row.question == _question()
        assert row.user_id == user.id

    def test_remiss_resets_srs_and_increments_lapses(self, client, db, user):
        cookie = _cookie(user)
        body = _card_body()
        first = client.post("/review/cards", json=body, cookies=cookie)
        assert first.status_code == 201
        card_id = first.json()["id"]

        # Advance the card once (correct answer -> interval grows to 1).
        answer = client.post(
            f"/review/cards/{card_id}/answer",
            json={"selected_index": 1},
            cookies=cookie,
        )
        assert answer.status_code == 200
        assert answer.json()["interval_days"] == 1

        # Re-miss the same identity key: SRS resets, lapses increments.
        again = client.post("/review/cards", json=body, cookies=cookie)
        assert again.status_code == 201
        again_body = again.json()
        assert again_body["id"] == card_id
        assert again_body["interval_days"] == 0
        assert again_body["ease"] == 2.5
        assert again_body["lapses"] == 1

        row = db.get(ReviewCard, card_id)
        assert row.lapses == 1
        assert row.interval_days == 0

    def test_no_cookie_is_401(self, client, user):
        resp = client.post("/review/cards", json=_card_body())
        assert resp.status_code == 401

    def test_correct_index_out_of_range_is_422(self, client, user):
        body = _card_body(question=_question(correct_index=9))
        resp = client.post("/review/cards", json=body, cookies=_cookie(user))
        assert resp.status_code == 422

    def test_selected_index_out_of_range_is_422(self, client, user):
        body = _card_body(selected_index=7)
        resp = client.post("/review/cards", json=body, cookies=_cookie(user))
        assert resp.status_code == 422

    def test_single_option_is_422(self, client, user):
        body = _card_body(
            question=_question(options=["only"], correct_index=0)
        )
        resp = client.post("/review/cards", json=body, cookies=_cookie(user))
        assert resp.status_code == 422


# --- GET /review/due ---


class TestDueCards:
    def test_returns_only_current_users_due_nonretired_ordered(self, client, db, user):
        now = datetime.now(UTC)
        other = User(email="other@example.com", display_name="Other",
                     password_hash="x")
        db.add(other)
        db.commit()

        _seed_card(db, user.id, qid="future", due_at=now + timedelta(days=1))
        late = _seed_card(db, user.id, qid="late", due_at=now - timedelta(hours=1))
        early = _seed_card(db, user.id, qid="early",
                           due_at=now - timedelta(days=1))
        _seed_card(db, user.id, qid="retired", is_retired=True,
                   due_at=now - timedelta(days=2))
        _seed_card(db, other.id, qid="other", due_at=now - timedelta(days=1))

        resp = client.get("/review/due", cookies=_cookie(user))
        assert resp.status_code == 200
        ids = [c["id"] for c in resp.json()]
        # Ordered by due_at: earliest first; only the current user's due,
        # non-retired cards.
        assert ids == [early.id, late.id]

    def test_limit_bounds_the_deck(self, client, db, user):
        now = datetime.now(UTC)
        for i in range(3):
            _seed_card(db, user.id, qid=f"q{i}",
                       due_at=now - timedelta(minutes=i))
        resp = client.get(
            "/review/due", params={"limit": 2}, cookies=_cookie(user)
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_limit_out_of_range_is_422(self, client, user):
        resp = client.get(
            "/review/due", params={"limit": 101}, cookies=_cookie(user)
        )
        assert resp.status_code == 422


# --- POST /review/cards/{id}/answer ---


class TestAnswerCard:
    def test_correct_grows_interval(self, client, db, user):
        card = _seed_card(db, user.id, qid="c1")
        resp = client.post(
            f"/review/cards/{card.id}/answer",
            json={"selected_index": 1},
            cookies=_cookie(user),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["was_correct"] is True
        assert body["interval_days"] == 1  # fresh card -> due in 1 day
        assert body["lapses"] == 0

    def test_wrong_resets_interval_and_increments_lapse(self, client, db, user):
        card = _seed_card(db, user.id, qid="c2", interval_days=8, ease=2.6)
        resp = client.post(
            f"/review/cards/{card.id}/answer",
            json={"selected_index": 0},
            cookies=_cookie(user),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["was_correct"] is False
        assert body["interval_days"] == 1
        assert body["lapses"] == 1

    def test_unknown_id_is_404(self, client, user):
        resp = client.post(
            "/review/cards/999999/answer",
            json={"selected_index": 0},
            cookies=_cookie(user),
        )
        assert resp.status_code == 404

    def test_another_users_card_is_404(self, client, db, user):
        other = User(email="other@example.com", display_name="Other",
                     password_hash="x")
        db.add(other)
        db.commit()
        card = _seed_card(db, other.id, qid="theirs")
        resp = client.post(
            f"/review/cards/{card.id}/answer",
            json={"selected_index": 0},
            cookies=_cookie(user),
        )
        assert resp.status_code == 404

    def test_selected_index_out_of_range_is_422(self, client, db, user):
        card = _seed_card(db, user.id, qid="c3")
        resp = client.post(
            f"/review/cards/{card.id}/answer",
            json={"selected_index": 5},
            cookies=_cookie(user),
        )
        assert resp.status_code == 422


# --- Retirement ---


class TestRetirement:
    def test_retired_card_leaves_due_and_counts_retired(self, client, db, user):
        # A card close to retirement: one correct answer pushes it past 21.
        card = _seed_card(db, user.id, qid="retire", interval_days=10, ease=2.5)
        resp = client.post(
            f"/review/cards/{card.id}/answer",
            json={"selected_index": 1},
            cookies=_cookie(user),
        )
        assert resp.status_code == 200
        assert resp.json()["is_retired"] is True

        db.expire_all()
        row = db.get(ReviewCard, card.id)
        assert row.is_retired is True

        # The retired card no longer appears on the due deck.
        deck = client.get("/review/due", cookies=_cookie(user)).json()
        assert [c["id"] for c in deck] == []

        # The summary counts it retired.
        summary = client.get("/review/summary", cookies=_cookie(user)).json()
        assert summary["total_retired"] == 1
        assert summary["total_active"] == 0
        assert summary["due_count"] == 0


# --- GET /review/summary ---


class TestSummary:
    def test_counts_due_active_retired(self, client, db, user):
        now = datetime.now(UTC)
        _seed_card(db, user.id, qid="due", due_at=now - timedelta(days=1))
        _seed_card(db, user.id, qid="pending",
                   due_at=now + timedelta(days=1))
        _seed_card(db, user.id, qid="retired", is_retired=True,
                   due_at=now - timedelta(days=1))

        resp = client.get("/review/summary", cookies=_cookie(user))
        assert resp.status_code == 200
        body = resp.json()
        assert body["due_count"] == 1
        assert body["total_active"] == 2  # due + pending (both non-retired)
        assert body["total_retired"] == 1

    def test_empty_deck_is_zero(self, client, db, user):
        resp = client.get("/review/summary", cookies=_cookie(user))
        assert resp.json() == {
            "due_count": 0, "total_active": 0, "total_retired": 0
        }
