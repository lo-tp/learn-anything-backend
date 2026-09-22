"""Unit tests for the FSRS scheduler wrapper (core/srs.py).

Covers issue #115 acceptance criteria:
fresh card, growth across grades, again lapses, due date.
"""

from datetime import UTC, datetime, timedelta

from fsrs import Card, Rating, State

from core.srs import fsrs_apply

NOW = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


class TestFreshCard:
    def test_fresh_good_gives_learning_step_due(self):
        """A fresh card reviewed with Good should have a sub-day due date (learning step)."""
        card = Card()
        interval_days, new_due, lapses, new_card = fsrs_apply(
            card, Rating.Good, now=NOW, lapses=0
        )
        # Learning step is 10 minutes -> sub-day interval
        assert 0 < interval_days < 1
        assert new_due > NOW
        assert lapses == 0
        assert new_card.state == State.Learning

    def test_fresh_again_gives_shorter_step(self):
        """Fresh card with Again should give a shorter step than Good."""
        card = Card()
        interval_again, _, _, _ = fsrs_apply(
            card, Rating.Again, now=NOW, lapses=0
        )
        interval_good, _, _, _ = fsrs_apply(
            Card(), Rating.Good, now=NOW, lapses=0
        )
        assert interval_again < interval_good

    def test_fresh_easy_gives_longest_step(self):
        """Fresh card with Easy should give the longest learning step."""
        card = Card()
        interval_easy, _, _, _ = fsrs_apply(
            card, Rating.Easy, now=NOW, lapses=0
        )
        interval_good, _, _, _ = fsrs_apply(
            Card(), Rating.Good, now=NOW, lapses=0
        )
        assert interval_easy > interval_good


class TestGrowthAcrossGrades:
    """A card in Review: Again < Hard < Good < Easy in interval."""

    def _review_card(self) -> Card:
        return Card(
            state=State.Review,
            due=NOW - timedelta(days=1),
            stability=10.0,
            difficulty=5.0,
            last_review=NOW - timedelta(days=1),
            step=None,
        )

    def test_grade_ordering(self):
        """Again < Hard < Good < Easy for a card in Review state."""
        interval_again, _, _, _ = fsrs_apply(self._review_card(), Rating.Again, now=NOW, lapses=0)
        interval_hard, _, _, _ = fsrs_apply(self._review_card(), Rating.Hard, now=NOW, lapses=0)
        interval_good, _, _, _ = fsrs_apply(self._review_card(), Rating.Good, now=NOW, lapses=0)
        interval_easy, _, _, _ = fsrs_apply(self._review_card(), Rating.Easy, now=NOW, lapses=0)
        assert interval_again < interval_hard < interval_good < interval_easy

    def test_good_grows_beyond_one_day(self):
        """Good on a card with stability 10d should give a multi-day interval."""
        interval, _, _, new_card = fsrs_apply(self._review_card(), Rating.Good, now=NOW, lapses=0)
        assert interval > 1
        assert new_card.state == State.Review

    def test_hard_grows_less_than_good(self):
        """Hard grows the interval less than Good."""
        interval_hard, _, _, _ = fsrs_apply(self._review_card(), Rating.Hard, now=NOW, lapses=0)
        interval_good, _, _, _ = fsrs_apply(self._review_card(), Rating.Good, now=NOW, lapses=0)
        assert interval_hard < interval_good


class TestAgainLapses:
    """Rating Again on a card in Review state should increment the lapse count."""

    def test_review_again_increments_lapses(self):
        card = Card(
            state=State.Review,
            due=NOW - timedelta(days=1),
            stability=10.0,
            difficulty=5.0,
            last_review=NOW - timedelta(days=1),
            step=None,
        )
        _, _, lapses, new_card = fsrs_apply(card, Rating.Again, now=NOW, lapses=3)
        assert lapses == 4
        # Card should transition to Relearning
        assert new_card.state == State.Relearning

    def test_review_good_does_not_lapse(self):
        card = Card(
            state=State.Review,
            due=NOW - timedelta(days=1),
            stability=10.0,
            difficulty=5.0,
            last_review=NOW - timedelta(days=1),
            step=None,
        )
        _, _, lapses, _ = fsrs_apply(card, Rating.Good, now=NOW, lapses=2)
        assert lapses == 2

    def test_learning_again_does_not_lapse(self):
        """A card in Learning state doesn't lapse (it hasn't graduated yet)."""
        card = Card(
            state=State.Learning,
            due=NOW,
            stability=0.5,
            difficulty=5.0,
            last_review=NOW,
            step=0,
        )
        _, _, lapses, _ = fsrs_apply(card, Rating.Again, now=NOW, lapses=0)
        assert lapses == 0


class TestDueDate:
    def test_due_date_is_after_now(self):
        card = Card()
        _, new_due, _, _ = fsrs_apply(card, Rating.Good, now=NOW, lapses=0)
        assert new_due > NOW

    def test_due_date_reflects_interval(self):
        """The due date minus now should equal the interval."""
        card = Card(
            state=State.Review,
            due=NOW - timedelta(days=1),
            stability=10.0,
            difficulty=5.0,
            last_review=NOW - timedelta(days=1),
            step=None,
        )
        interval_days, new_due, _, _ = fsrs_apply(card, Rating.Good, now=NOW, lapses=0)
        expected = NOW + timedelta(days=interval_days)
        assert new_due == expected

    def test_review_good_due_is_far_future(self):
        card = Card(
            state=State.Review,
            due=NOW - timedelta(days=1),
            stability=10.0,
            difficulty=5.0,
            last_review=NOW - timedelta(days=1),
            step=None,
        )
        interval_days, _, _, _ = fsrs_apply(card, Rating.Good, now=NOW, lapses=0)
        # stability 10, desired retention 0.9 -> interval ~21 days
        assert interval_days > 10
