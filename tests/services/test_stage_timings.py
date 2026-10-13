"""Tests for services/stage_timings.py — recording what a stage cost (#194).

Seams: :func:`stage_timing_rows` (the shape of a row, shared by every driver of
a graph — including the material one, which needs the rows inside its own
slide-commit transaction) and :func:`record_stage_timings` (the whole write).
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")  # before `db` is imported

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from db import Base, GraphStageTiming
from services.stage_timings import (
    empty_stage_timings,
    record_stage_timings,
    stage_timing_rows,
)


@pytest.fixture()
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


class TestStageTimingRows:
    def test_the_graph_is_a_column_not_part_of_the_stage_name(self):
        """The same stage name the material graph writes, so one query reads
        both: `scripts/slide_analysis.py --graph clarify|probe|plan`."""
        (row,) = stage_timing_rows(
            "session-1",
            "probe",
            [{"stage": "evaluate_batch", "context": {"question_count": 3}, "duration_seconds": 1.5}],
        )
        assert (row.session_id, row.graph, row.stage) == ("session-1", "probe", "evaluate_batch")
        assert row.context == {"question_count": 3}
        assert row.duration_seconds == 1.5

    def test_a_stage_with_no_attribution_is_still_recorded(self):
        (row,) = stage_timing_rows("session-1", "plan", [{"stage": "research_topic", "duration_seconds": 2.0}])
        assert row.context is None

    def test_no_timings_is_no_rows(self):
        assert stage_timing_rows("session-1", "clarify", []) == []


class TestRecordStageTimings:
    def test_every_stage_this_request_ran_becomes_a_row(self, db):
        written = record_stage_timings(
            db,
            "session-1",
            "clarify",
            {
                "stage_timings": [
                    {"stage": "assess_goal", "context": {"round_count": 0}, "duration_seconds": 1.0},
                    {"stage": "generate_questions", "duration_seconds": 2.0},
                ]
            },
        )
        assert written == 2
        rows = db.scalars(select(GraphStageTiming).order_by(GraphStageTiming.stage)).all()
        assert [r.stage for r in rows] == ["assess_goal", "generate_questions"]

    def test_a_request_that_ran_no_model_stage_writes_nothing(self, db):
        """The rows describe work this service did, not time the learner spent."""
        assert record_stage_timings(db, "session-1", "plan", {}) == 0
        assert db.scalars(select(GraphStageTiming)).all() == []

    def test_the_channel_is_emptied_for_the_next_request(self):
        """A checkpoint carries the previous request's timings; its rows are
        already written, so the resume must start from an empty channel — which
        is what a channel's reducer would not do on its own."""
        overwrite = empty_stage_timings()
        assert overwrite.value == []
        assert overwrite.type == "__overwrite__"
