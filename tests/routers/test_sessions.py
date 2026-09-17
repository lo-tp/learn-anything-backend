"""Unit tests for routers/sessions.py."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from db import Phase, Plan, StepProgress

# RFC 3339 date-time with an explicit UTC designator (pydantic emits "Z"
# for UTC; "+00:00" is also acceptable).
UTC_OFFSET_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|\+00:00)$")


class TestListSessions:
    def test_empty(self, client, db):
        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert resp.json() == {"sessions": []}

    def test_returns_all_newest_first(self, client, db, make_session):
        make_session(
            session_id="a",
            phase=Phase.CLARIFYING,
            goal="goal A",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        make_session(
            session_id="b",
            phase=Phase.PROBING,
            goal="goal B",
            created_at=datetime(2024, 2, 1, tzinfo=UTC),
        )

        resp = client.get("/sessions")
        assert resp.status_code == 200
        ids = [s["session_id"] for s in resp.json()["sessions"]]
        assert ids == ["b", "a"]
        # Spot-check a serialized field.
        first = resp.json()["sessions"][0]
        assert first["goal"] == "goal B"
        assert first["phase"] == "probing"
        # created_at must carry an explicit UTC offset (RFC 3339), otherwise
        # clients parse it as local time.
        assert UTC_OFFSET_RE.match(first["created_at"]), first["created_at"]

    def test_naive_legacy_created_at_serialized_as_utc(self, client, db, make_session):
        # Rows written before the timestamptz migration (or via a driver that
        # strips the offset) come back naive; the response must still carry
        # an explicit UTC offset so the instant is unambiguous.
        make_session(
            session_id="a",
            # Naive UTC wall-clock: what legacy rows look like when read back.
            created_at=datetime(2026, 9, 17, 1, 2, 49, tzinfo=UTC).replace(
                tzinfo=None
            ),
        )
        resp = client.get("/sessions")
        created_at = resp.json()["sessions"][0]["created_at"]
        assert UTC_OFFSET_RE.match(created_at), created_at
        assert created_at.startswith("2026-09-17T01:02:49")

    def test_filter_by_single_phase(self, client, db, make_session):
        make_session(session_id="a", phase=Phase.CLARIFYING)
        make_session(session_id="b", phase=Phase.PROBING)
        make_session(session_id="c", phase=Phase.PROBING)

        resp = client.get("/sessions", params={"phase": ["probing"]})
        ids = sorted(s["session_id"] for s in resp.json()["sessions"])
        assert ids == ["b", "c"]

    def test_filter_by_multiple_phases(self, client, db, make_session):
        make_session(session_id="a", phase=Phase.CLARIFYING)
        make_session(session_id="b", phase=Phase.PROBING)
        make_session(session_id="c", phase=Phase.PLANNING)

        resp = client.get(
            "/sessions", params={"phase": ["clarifying", "planning"]}
        )
        ids = sorted(s["session_id"] for s in resp.json()["sessions"])
        assert ids == ["a", "c"]

    def test_filter_matches_nothing(self, client, db, make_session):
        make_session(session_id="a", phase=Phase.CLARIFYING)

        resp = client.get("/sessions", params={"phase": ["complete"]})
        assert resp.json() == {"sessions": []}


class TestGetSession:
    def test_not_found(self, client, db):
        resp = client.get("/sessions/nope")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Session not found"

    def test_no_plan(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.CLARIFYING).session_id
        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "a"
        assert body["phase"] == "clarifying"
        assert body["narrowed_goal"] is None
        assert body["progress"] == {
            "current_step_id": None,
            "completed_steps": [],
            "total_steps": 0,
            "step_scores": {},
        }

    def test_with_plan_and_progress(self, client, db, make_session):
        sid = make_session(
            session_id="a", phase=Phase.EXECUTING, narrowed_goal="ng"
        ).session_id
        db.add(
            Plan(
                session_id=sid,
                version=1,
                prose_summary="ps",
                dependency_dag="dag",
                steps=[
                    {"id": "s1", "title": "T1", "description": "d",
                     "depends_on": [], "depth": 0},
                    {"id": "s2", "title": "T2", "description": "d",
                     "depends_on": ["s1"], "depth": 1},
                    {"id": "s3", "title": "T3", "description": "d",
                     "depends_on": ["s2"], "depth": 2},
                ],
                adjustments=[],
            )
        )
        db.add(StepProgress(session_id=sid, step_id="s1", complete=True, score=0.9))
        db.add(StepProgress(session_id=sid, step_id="s2", complete=True, score=1.0))
        db.add(StepProgress(session_id=sid, step_id="s3", complete=False, score=None))
        db.commit()

        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["narrowed_goal"] == "ng"
        assert body["progress"] == {
            "current_step_id": "s3",
            "completed_steps": ["s1", "s2"],
            "total_steps": 3,
            "step_scores": {"s1": 0.9, "s2": 1.0},
        }

    def test_all_steps_complete(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.COMPLETE).session_id
        db.add(
            Plan(
                session_id=sid,
                version=1,
                prose_summary="ps",
                dependency_dag="dag",
                steps=[{"id": "s1", "title": "T", "description": "d",
                        "depends_on": [], "depth": 0}],
                adjustments=[],
            )
        )
        db.add(StepProgress(session_id=sid, step_id="s1", complete=True, score=1.0))
        db.commit()

        resp = client.get(f"/sessions/{sid}")
        body = resp.json()
        assert body["progress"]["current_step_id"] is None
        assert body["progress"]["completed_steps"] == ["s1"]
