"""Integration tests for the Review API (``routers/review.py``) — #116.

Against the real DB + a real ``User`` row (JWT cookie). Covers the
confidence contract: capture, answer (again/hard/good/easy), due deck,
summary, and re-miss reset.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fsrs import Card, State

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


def _seed_card(db, user_id, *, qid="q-1", due_at=None, lapses=0, **overrides):
    """Seed a ReviewCard with a fresh FSRS state."""
    now = due_at or datetime.now(UTC)
    fsrs_state = dict(Card(due=now, last_review=now, state=State.Learning, step=0,
                           stability=None, difficulty=None, card_id=0).to_dict())
    card = ReviewCard(
        user_id=user_id,
        source="material",
        session_id="sess-1",
        source_question_id=qid,
        question=_question(),
        fsrs_state=fsrs_state,
        lapses=lapses,
        due_at=now,
        **overrides,
    )
    db.add(card)
    db.commit()
    return card


# --- POST /review/cards ---


class TestRecordCard:
    def test_creates_card_201_with_fresh_fsrs(self, client, db, user):
        resp = client.post("/review/cards", json=_card_body(), cookies=_cookie(user))
        assert resp.status_code == 201
        body = resp.json()
        assert body["source"] == "material"
        assert body["session_id"] == "sess-1"
        assert body["question"] == _question()
        assert body["lapses"] == 0

        row = db.get(ReviewCard, body["id"])
        assert row.question == _question()
        assert row.user_id == user.id
        assert row.lapses == 0

    def test_remiss_resets_fsrs_and_increments_lapses(self, client, db, user):
        cookie = _cookie(user)
        body = _card_body()
        first = client.post("/review/cards", json=body, cookies=cookie)
        assert first.status_code == 201
        card_id = first.json()["id"]

        # Advance the card once (good answer).
        answer = client.post(
            f"/review/cards/{card_id}/answer",
            json={"rating": "good"},
            cookies=cookie,
        )
        assert answer.status_code == 200
        assert answer.json()["lapses"] == 0

        # Re-miss the same identity key: FSRS resets, lapses increments.
        again = client.post("/review/cards", json=body, cookies=cookie)
        assert again.status_code == 201
        again_body = again.json()
        assert again_body["id"] == card_id
        assert again_body["lapses"] == 1

        row = db.get(ReviewCard, card_id)
        assert row.lapses == 1

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
    def test_returns_only_current_users_due_cards_ordered(self, client, db, user):
        now = datetime.now(UTC)
        other = User(email="other@example.com", display_name="Other",
                     password_hash="x")
        db.add(other)
        db.commit()

        _seed_card(db, user.id, qid="future", due_at=now + timedelta(days=1))
        late = _seed_card(db, user.id, qid="late", due_at=now - timedelta(hours=1))
        early = _seed_card(db, user.id, qid="early",
                           due_at=now - timedelta(days=1))
        _seed_card(db, other.id, qid="other", due_at=now - timedelta(days=1))

        resp = client.get("/review/due", cookies=_cookie(user))
        assert resp.status_code == 200
        ids = [c["id"] for c in resp.json()]
        # Ordered by due_at: earliest first; only the current user's due cards.
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
    def test_good_on_fresh_card_gives_learning_step(self, client, db, user):
        card = _seed_card(db, user.id, qid="c1")
        resp = client.post(
            f"/review/cards/{card.id}/answer",
            json={"rating": "good"},
            cookies=_cookie(user),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "due_at" in body
        assert "interval_days" in body
        assert body["lapses"] == 0
        # Fresh card rated Good: sub-day learning step
        assert 0 < body["interval_days"] < 1

    def test_again_on_fresh_card_gives_shorter_step(self, client, db, user):
        card = _seed_card(db, user.id, qid="c2")
        resp = client.post(
            f"/review/cards/{card.id}/answer",
            json={"rating": "again"},
            cookies=_cookie(user),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert 0 < body["interval_days"] < 1

    def test_easy_on_fresh_card_gives_longer_step(self, client, db, user):
        card = _seed_card(db, user.id, qid="c3")
        resp = client.post(
            f"/review/cards/{card.id}/answer",
            json={"rating": "easy"},
            cookies=_cookie(user),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["interval_days"] > 0

    def test_hard_on_fresh_card_between_again_and_easy(self, client, db, user):
        # Three fresh cards, one per rating
        cards = {}
        for r in ("again", "hard", "easy"):
            c = _seed_card(db, user.id, qid=f"fresh-{r}")
            resp = client.post(
                f"/review/cards/{c.id}/answer",
                json={"rating": r},
                cookies=_cookie(user),
            )
            cards[r] = resp.json()["interval_days"]

        assert cards["again"] < cards["hard"] < cards["easy"]

    def test_lapse_count_increments_on_review_again(self, client, db, user):
        """Rating Again on a card in Review state increments lapses."""
        card = _seed_card(db, user.id, qid="lapse")
        cookie = _cookie(user)

        # Two Good answers graduate the card to Review state.
        client.post(
            f"/review/cards/{card.id}/answer",
            json={"rating": "good"},
            cookies=cookie,
        )
        client.post(
            f"/review/cards/{card.id}/answer",
            json={"rating": "good"},
            cookies=cookie,
        )

        # The card is now due in the future; force it due now for the test.
        db.expire_all()
        row = db.get(ReviewCard, card.id)
        row.due_at = datetime.now(UTC)
        db.commit()

        # Third answer (Again): should lapse since state was Review.
        resp = client.post(
            f"/review/cards/{card.id}/answer",
            json={"rating": "again"},
            cookies=cookie,
        )
        assert resp.status_code == 200
        assert resp.json()["lapses"] == 1

    def test_invalid_rating_is_422(self, client, db, user):
        card = _seed_card(db, user.id, qid="c4")
        resp = client.post(
            f"/review/cards/{card.id}/answer",
            json={"rating": "meh"},
            cookies=_cookie(user),
        )
        assert resp.status_code == 422

    def test_missing_rating_is_422(self, client, db, user):
        card = _seed_card(db, user.id, qid="c5")
        resp = client.post(
            f"/review/cards/{card.id}/answer",
            json={},
            cookies=_cookie(user),
        )
        assert resp.status_code == 422

    def test_unknown_id_is_404(self, client, user):
        resp = client.post(
            "/review/cards/999999/answer",
            json={"rating": "good"},
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
            json={"rating": "good"},
            cookies=_cookie(user),
        )
        assert resp.status_code == 404


# --- GET /review/summary ---


class TestSummary:
    def test_counts_due_and_total(self, client, db, user):
        now = datetime.now(UTC)
        _seed_card(db, user.id, qid="due", due_at=now - timedelta(days=1))
        _seed_card(db, user.id, qid="pending",
                   due_at=now + timedelta(days=1))

        resp = client.get("/review/summary", cookies=_cookie(user))
        assert resp.status_code == 200
        body = resp.json()
        assert body["due_count"] == 1
        assert body["total_cards"] == 2
        # No retired count.
        assert "total_retired" not in body

    def test_empty_deck_is_zero(self, client, db, user):
        resp = client.get("/review/summary", cookies=_cookie(user))
        assert resp.json() == {"due_count": 0, "total_cards": 0}
