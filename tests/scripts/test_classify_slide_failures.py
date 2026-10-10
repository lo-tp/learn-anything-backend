"""Tests for ``scripts/classify_slide_failures.py`` (issue #164, C1.2).

Two sources of truth:

* ``tests/fixtures/slide_failure_attempts.json`` — the real output of the script
  over every persisted failed attempt in the development database, checked in by
  ``--dump-fixture``. It pins what the classifier does to real errors, so
  changing a token list shows up here as a changed distribution.
* A seeded in-memory database — the script's real read path (``fetch_attempts``
  over ``failed_slides``), exercised without a Postgres.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from db.models import FailedSlide
from graphs.material.sandbox import MAX_MATERIAL_ATTEMPTS, SLIDE_FAILURE_CLASSES
from scripts.classify_slide_failures import (
    FailedAttempt,
    build_report,
    classify_attempts,
    classify_error,
    fetch_attempts,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "slide_failure_attempts.json"
)


@pytest.fixture(scope="module")
def real_sample() -> dict:
    """The checked-in real sample (the script's own output on real data)."""
    return json.loads(FIXTURE_PATH.read_text())


def _attempts_of(sample: dict) -> list[FailedAttempt]:
    """Rebuild persisted attempts from the fixture, honouring each entry's count."""
    attempts: list[FailedAttempt] = []
    for entry in sample["attempts"]:
        for _ in range(entry["count"]):
            attempts.append(
                FailedAttempt(
                    id=entry["id"],
                    session_id=entry["session_id"],
                    step_id=entry["step_id"],
                    slide_index=entry["slide_index"],
                    created_at=datetime.fromisoformat(entry["created_at"]),
                    error=entry["error"],
                    jsx=entry["jsx_excerpt"],
                )
            )
    return attempts


class TestRealSample:
    """The classifier over real persisted errors."""

    def test_every_attempt_maps_to_exactly_one_declared_class(self, real_sample):
        classes = Counter()
        for entry in real_sample["attempts"]:
            failure_class, _reduced = classify_error(entry["error"])
            assert failure_class in SLIDE_FAILURE_CLASSES
            # The class the script recorded is the class it still produces.
            assert failure_class == entry["failure_class"]
            classes[failure_class] += entry["count"]
        assert sum(classes.values()) == real_sample["attempt_count"]

    def test_the_recorded_distribution_is_pinned(self, real_sample):
        """Re-classifying the real errors reproduces the checked-in counts."""
        classes = Counter()
        for entry in real_sample["attempts"]:
            failure_class, _ = classify_error(entry["error"])
            classes[failure_class] += entry["count"]
        assert dict(classes) == {
            failure_class: count
            for failure_class, count in real_sample["distribution"].items()
            if count
        }

    def test_the_sample_is_a_real_one(self, real_sample):
        """A real sample, not a toy: it carries the whole failed-attempt table."""
        assert real_sample["attempt_count"] >= 100
        assert real_sample["slide_count"] >= 50
        assert len(real_sample["attempts"]) >= 25
        # Every class the real data produced is represented by a real error.
        produced = {c for c, n in real_sample["distribution"].items() if n}
        assert produced <= set(SLIDE_FAILURE_CLASSES)
        assert {"syntax_error", "content_spec_invalid", "truncated_output"} <= produced
        # ``sandbox_timeout`` has no real rows yet (the compile always answered),
        # which is why it is covered by the unit tests in test_sandbox.py.
        assert "sandbox_timeout" not in produced

    def test_the_sample_keeps_the_raw_stored_error(self, real_sample):
        """The reduce step is exercised too: rows are stored transport-wrapped."""
        wrapped = [
            e for e in real_sample["attempts"] if e["error"].startswith("Transport/HTTP")
        ]
        assert wrapped, "expected the real rows to be stored transport-wrapped"
        for entry in wrapped:
            _failure_class, reduced = classify_error(entry["error"])
            assert not reduced.startswith("Transport/HTTP")
            assert "for url" not in reduced


class TestClassificationIsDeterministicAndTotal:
    def test_opaque_errors_are_other_never_a_raise(self):
        assert classify_error("500 Internal Server Error")[0] == "other"
        assert classify_error("")[0] == "other"

    def test_the_transport_wrapper_does_not_change_the_class(self):
        raw = (
            "Transport/HTTP error: Client error '400 code must declare "
            "`export default`' for url 'http://localhost:3001/api/compile' "
            "For more information check: https://example.com"
        )
        assert classify_error(raw) == (
            "content_spec_invalid",
            "400 code must declare `export default`",
        )


class TestReport:
    def test_report_lists_every_class_including_the_ones_that_never_happened(
        self, real_sample
    ):
        rows = classify_attempts(_attempts_of(real_sample))
        report = build_report(rows, session_id=None, since=None)
        for failure_class in SLIDE_FAILURE_CLASSES:
            assert failure_class in report
        assert f"{len(rows)} failed attempt(s)" in report
        assert "Every attempt mapped to exactly one class: yes" in report
        # The distribution section sums back to the attempts it was given.
        total_line = next(line for line in report.splitlines() if "TOTAL" in line)
        assert str(len(rows)) in total_line

    def test_report_groups_the_errors_behind_each_class(self, real_sample):
        rows = classify_attempts(_attempts_of(real_sample))
        report = build_report(rows, session_id=None, since=None)
        assert "2. The errors behind each class" in report
        # The single most common real failure is named with its count.
        assert "code must be at least 40 characters" in report

    def test_report_says_so_when_there_is_nothing_to_classify(self):
        report = build_report([], session_id=None, since=None)
        assert "No failed attempts found." in report

    def test_a_slide_that_failed_every_attempt_is_reported_as_exhausted(self):
        """The skip-rate side of the report: class of the last failed attempt."""
        jsx = "export default function S() { return <div>x</div>; }"
        # One slide fails every outer attempt (the last one a truncation), and
        # one slide fails once and is produced on a later attempt.
        cycle = [
            "500 Build failed with 1 error: ERROR: Unexpected \">\"",
            "400 code must be at least 40 characters",
            "500 Build failed with 1 error: ERROR: Unexpected end of file",
        ]
        attempts = [
            FailedAttempt(
                id=n + 1,
                session_id="sess-1",
                step_id="s1",
                slide_index=3,
                created_at=datetime(2026, 10, 1, tzinfo=UTC),
                error=cycle[n % len(cycle)],
                jsx=jsx,
            )
            for n in range(MAX_MATERIAL_ATTEMPTS)
        ]
        attempts.append(
            FailedAttempt(
                id=99,
                session_id="sess-1",
                step_id="s1",
                slide_index=4,
                created_at=datetime(2026, 10, 1, tzinfo=UTC),
                error="500 Internal Server Error",
                jsx=jsx,
            )
        )
        report = build_report(classify_attempts(attempts), session_id=None, since=None)
        outcomes = report.split("3. Slides")[1]
        assert "slides with failed attempts: 2" in outcomes
        assert (
            f"exhausted all {MAX_MATERIAL_ATTEMPTS} attempts "
            "(placeholder/skipped): 1"
        ) in outcomes
        assert "fewer failures than attempts" in outcomes
        # The exhausted slide is attributed to its LAST failed attempt, and the
        # recovered slide to its only one; neither earlier class is attributed.
        assert "truncated_output" in outcomes
        assert "other" in outcomes
        assert "syntax_error" not in outcomes
        assert "content_spec_invalid" not in outcomes


class TestReadsPersistedAttempts:
    """AC: the script runs against persisted failed attempts (seeded database)."""

    def _seed(self, factory: sessionmaker) -> None:
        with factory() as db:
            for n, (session_id, when) in enumerate(
                [
                    ("sess-old", datetime(2026, 8, 1, tzinfo=UTC)),
                    ("sess-new", datetime(2026, 9, 20, tzinfo=UTC)),
                    ("sess-new", datetime(2026, 9, 21, tzinfo=UTC)),
                ]
            ):
                db.add(
                    FailedSlide(
                        id=100 + n,
                        session_id=session_id,
                        step_id="s1",
                        slide_index=n + 1,
                        prompt="[]",
                        jsx="x",
                        error=(
                            "Transport/HTTP error: Client error '400 code must "
                            "declare `export default`' for url 'http://x/api/compile'"
                        ),
                        created_at=when,
                    )
                )
            db.commit()

    def test_reads_rows_in_write_order(self, db_session_factory):
        self._seed(db_session_factory)
        attempts = fetch_attempts(db_session_factory)
        assert [a.id for a in attempts] == [100, 101, 102]
        assert [a.session_id for a in attempts] == [
            "sess-old",
            "sess-new",
            "sess-new",
        ]

    def test_filters_by_session_and_since(self, db_session_factory):
        self._seed(db_session_factory)
        assert [a.id for a in fetch_attempts(db_session_factory, session_id="sess-new")] == [
            101,
            102,
        ]
        assert [
            a.id
            for a in fetch_attempts(
                db_session_factory, since=datetime(2026, 9, 21, tzinfo=UTC)
            )
        ] == [102]

    def test_a_seeded_database_produces_a_distribution_report(self, db_session_factory):
        self._seed(db_session_factory)
        rows = classify_attempts(fetch_attempts(db_session_factory))
        report = build_report(rows, session_id=None, since=None)
        assert "3 failed attempt(s) across 3 slide(s) in 2 session(s)" in report
        assert "content_spec_invalid" in report
