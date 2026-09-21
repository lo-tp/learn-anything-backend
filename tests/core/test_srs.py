"""Unit tests for the SRS wrapper (core/srs.py).

Covers every transition from issue #106 acceptance criteria:
fresh->1d, growth by ease, wrong->reset+lapse, ease floor 1.3,
retirement at >= 21 days, and re-activation of retired cards.
"""

from datetime import UTC, datetime, timedelta

import pytest

from core.srs import RETIRE_DAYS, next_due_at, sm2_apply


class TestFreshCard:
    def test_fresh_correct_gives_one_day(self):
        interval, ease, lapses, is_retired = sm2_apply(0, 2.5, 0, was_correct=True)
        assert interval == 1
        assert ease == pytest.approx(2.6)
        assert lapses == 0
        assert is_retired is False

    def test_fresh_wrong_resets_to_one_day_and_lapses(self):
        interval, ease, lapses, is_retired = sm2_apply(0, 2.5, 0, was_correct=False)
        assert interval == 1
        assert ease == pytest.approx(1.7)
        assert lapses == 1
        assert is_retired is False

    def test_zero_interval_with_lapses_correct_is_treated_as_fresh(self):
        # A re-miss resets the card to interval 0 while keeping lapses; a
        # subsequent correct answer must not stall the card at interval 0.
        interval, _ease, lapses, is_retired = sm2_apply(0, 2.5, 2, was_correct=True)
        assert interval == 1
        assert lapses == 2
        assert is_retired is False


class TestGrowth:
    def test_correct_grows_by_round_interval_times_ease(self):
        interval, ease, lapses, is_retired = sm2_apply(7, 2.5, 0, was_correct=True)
        assert interval == round(7 * 2.5)  # 18
        assert interval == 18
        assert ease == pytest.approx(2.6)
        assert lapses == 0
        assert is_retired is False

    def test_growth_does_not_inflate_lapses(self):
        interval, _, lapses, _ = sm2_apply(20, 2.5, 3, was_correct=True)
        assert interval == 50
        assert lapses == 3


class TestWrongAnswer:
    def test_wrong_resets_interval_to_one_and_increments_lapses(self):
        interval, ease, lapses, is_retired = sm2_apply(10, 2.5, 0, was_correct=False)
        assert interval == 1
        assert ease == pytest.approx(1.7)
        assert lapses == 1
        assert is_retired is False

    def test_wrong_on_lapsed_card_keeps_counting(self):
        interval, _, lapses, _ = sm2_apply(5, 2.5, 4, was_correct=False)
        assert interval == 1
        assert lapses == 5


class TestEaseFloor:
    def test_ease_floored_at_1_3(self):
        interval, ease, lapses, is_retired = sm2_apply(10, 1.3, 1, was_correct=False)
        assert interval == 1
        assert ease == pytest.approx(1.3)
        assert lapses == 2
        assert is_retired is False

    def test_ease_never_dips_below_floor_from_higher_value(self):
        _, ease, _, _ = sm2_apply(10, 1.5, 1, was_correct=False)
        assert ease == pytest.approx(1.3)


class TestRetirement:
    def test_retired_when_interval_at_threshold(self):
        # Boundary: growth lands exactly on RETIRE_DAYS.
        interval, ease, lapses, is_retired = sm2_apply(1, 21.0, 0, was_correct=True)
        assert interval == RETIRE_DAYS
        assert ease == pytest.approx(21.1)
        assert lapses == 0
        assert is_retired is True

    def test_retired_when_interval_above_threshold(self):
        interval, ease, lapses, is_retired = sm2_apply(21, 2.5, 0, was_correct=True)
        assert interval == 52
        assert ease == pytest.approx(2.6)
        assert lapses == 0
        assert is_retired is True

    def test_not_retired_below_threshold(self):
        _, _, _, is_retired = sm2_apply(2, 10.0, 0, was_correct=True)
        assert is_retired is False


class TestReactivation:
    def test_wrong_on_retired_card_reactivates_it(self):
        interval, ease, lapses, is_retired = sm2_apply(50, 2.5, 3, was_correct=False)
        assert interval == 1
        assert interval < RETIRE_DAYS
        assert ease == pytest.approx(1.7)
        assert lapses == 4
        assert is_retired is False


class TestNextDueAt:
    def test_next_due_at_is_one_day_after_now(self):
        now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
        assert next_due_at(now) == now + timedelta(days=1)

    def test_next_due_at_defaults_to_current_time(self):
        before = datetime.now(UTC)
        result = next_due_at()
        after = datetime.now(UTC)
        assert before + timedelta(days=1) <= result <= after + timedelta(days=1)
