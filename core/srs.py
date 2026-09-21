"""SRS (spaced repetition) wrapper over the ``supermemo2`` package.

Pure, no I/O — unit-testable in isolation (see ``tests/core/test_srs.py``).

``supermemo2`` tracks ``repetitions`` (consecutive correct reviews, reset on
a wrong answer). This module tracks ``lapses`` (total wrong answers, never
reset) instead, so lapses are computed locally, not from the package.
Retirement (``interval >= RETIRE_DAYS``) is also our extension, not in the
package.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from supermemo2 import first_review, review

RETIRE_DAYS = 21


def sm2_apply(
    interval_days: int, ease: float, lapses: int, was_correct: bool
) -> tuple[int, float, int, bool]:
    """Apply one SM2 review and return ``(new_interval, new_ease, new_lapses, is_retired)``.

    - Fresh card (interval 0) + correct -> interval 1.
    - interval >= 1 + correct -> ``round(interval * ease)``.
    - Wrong -> ``lapses + 1``, interval resets to 1.
    - Ease: correct +0.1, wrong -0.8, floored at 1.3.
    - ``is_retired`` is True when the new interval >= RETIRE_DAYS; a wrong
      answer on a retired card re-activates it (interval < RETIRE_DAYS).
    """
    quality = 5 if was_correct else 0
    if interval_days == 0:
        r = first_review(quality)
    else:
        # repetitions >= 2 selects SM2's growth branch; the package's
        # repetitions value is discarded because we track lapses locally.
        r = review(quality, ease, interval_days, 2)
    new_lapses = lapses + (0 if was_correct else 1)
    new_interval = r["interval"]
    if was_correct and interval_days >= 1:
        # The package ceils the growth interval; the SM2 spec rounds.
        new_interval = round(interval_days * ease)
    return new_interval, r["easiness"], new_lapses, new_interval >= RETIRE_DAYS


def next_due_at(now: datetime | None = None) -> datetime:
    """One day after ``now`` (defaults to the current time)."""
    return (now or datetime.now(UTC)) + timedelta(days=1)
