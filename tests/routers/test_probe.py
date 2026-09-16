"""Unit tests for routers/probe.py."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import patch

from db import Phase, ProbeQuestion, Session


def _qid() -> str:
    return str(uuid.uuid4())


def _seed_probe(db, sid: str, qids: list[str], answered: set[str] | None = None):
    """Insert ProbeQuestion rows for ``sid``; mark those in ``answered`` done."""
    answered = answered or set()
    for i, qid in enumerate(qids):
        db.add(
            ProbeQuestion(
                id=qid,
                session_id=sid,
                question_id=uuid.UUID(qid),
                text=f"Q{i}",
                options=["A", "B"],
                correct_index=0,
                explanation="e",
                strand="algebra",
                difficulty=1,
                answered_at=datetime.now(UTC) if qid in answered else None,
            )
        )
    db.commit()


def _question_dict(i: int = 0, **overrides) -> dict:
    base = {
        "id": _qid(),
        "text": f"Q{i}?",
        "options": ["A", "B"],
        "correct_index": 0,
        "explanation": "e",
        "strand": "algebra",
        "difficulty": 1,
    }
    base.update(overrides)
    return base


# --- shared error branches ---


class TestProbeErrors:
    def test_not_found(self, client, db):
        resp = client.post("/sessions/nope/probe", json={})
        assert resp.status_code == 404

    def test_wrong_phase(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.CLARIFYING).session_id
        resp = client.post(f"/sessions/{sid}/probe", json={})
        assert resp.status_code == 409
        assert "not 'probing'" in resp.json()["detail"]


# --- first call: start the probe (answers is None) ---


class TestProbeStart:
    def test_returns_first_batch_and_persists(self, client, db, make_session):
        sid = make_session(
            session_id="a", phase=Phase.PROBING, narrowed_goal="ng"
        ).session_id
        batch = [_question_dict(0), _question_dict(1, options=["A", "B", "C"])]
        with patch("routers.probe.probe_graph") as graph:
            graph.invoke.return_value = {"next_batch": batch}
            resp = client.post(f"/sessions/{sid}/probe", json={})

        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "probing"
        assert len(body["questions"]) == 2
        assert body["questions"][0]["id"] == batch[0]["id"]

        db.expire_all()
        rows = db.query(ProbeQuestion).filter_by(session_id=sid).all()
        assert len(rows) == 2

    def test_start_uses_goal_when_no_narrowed_goal(self, client, db, make_session):
        sid = make_session(
            session_id="a", phase=Phase.PROBING, narrowed_goal=None,
            goal="Learn calculus",
        ).session_id
        with patch("routers.probe.probe_graph") as graph:
            graph.invoke.return_value = {"next_batch": [_question_dict()]}
            client.post(f"/sessions/{sid}/probe", json={})

        args, _ = graph.invoke.call_args
        assert args[0]["goal"] == "Learn calculus"

    def test_start_no_questions_is_500(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PROBING).session_id
        with patch("routers.probe.probe_graph") as graph:
            graph.invoke.return_value = {"next_batch": None}
            resp = client.post(f"/sessions/{sid}/probe", json={})
        assert resp.status_code == 500


# --- subsequent call: resume with a batch of answers ---


class TestProbeResume:
    def test_unknown_question_is_404(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PROBING).session_id
        _seed_probe(db, sid, [_qid()])
        unknown = str(uuid.uuid4())
        resp = client.post(
            f"/sessions/{sid}/probe",
            json={"answers": [{"question_id": unknown, "selected_index": 0}]},
        )
        assert resp.status_code == 404

    def test_already_answered_is_409(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PROBING).session_id
        q1 = _qid()
        _seed_probe(db, sid, [q1], answered={q1})
        resp = client.post(
            f"/sessions/{sid}/probe",
            json={"answers": [{"question_id": q1, "selected_index": 0}]},
        )
        assert resp.status_code == 409

    def test_partial_submission_is_422(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PROBING).session_id
        q1, q2 = _qid(), _qid()
        _seed_probe(db, sid, [q1, q2])
        resp = client.post(
            f"/sessions/{sid}/probe",
            json={"answers": [{"question_id": q1, "selected_index": 0}]},
        )
        assert resp.status_code == 422

    def test_selected_index_out_of_range_is_422(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PROBING).session_id
        q1 = _qid()
        _seed_probe(db, sid, [q1])  # options == ["A", "B"]
        resp = client.post(
            f"/sessions/{sid}/probe",
            json={"answers": [{"question_id": q1, "selected_index": 5}]},
        )
        assert resp.status_code == 422

    def test_resume_interrupt_no_batch_is_500(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PROBING).session_id
        q1 = _qid()
        _seed_probe(db, sid, [q1])
        with patch("routers.probe.probe_graph") as graph:
            graph.invoke.return_value = {"__interrupt__": object(),
                                         "next_batch": None}
            resp = client.post(
                f"/sessions/{sid}/probe",
                json={"answers": [{"question_id": q1, "selected_index": 0}]},
            )
        assert resp.status_code == 500

    def test_resume_next_batch(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PROBING).session_id
        q1 = _qid()
        _seed_probe(db, sid, [q1])
        next_batch = [_question_dict(9)]
        with patch("routers.probe.probe_graph") as graph:
            graph.invoke.return_value = {
                "__interrupt__": object(),
                "next_batch": next_batch,
            }
            resp = client.post(
                f"/sessions/{sid}/probe",
                json={"answers": [{"question_id": q1, "selected_index": 1}]},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "probing"
        assert len(body["questions"]) == 1

        # The answered row was updated with the choice and correctness.
        db.expire_all()
        row = db.get(ProbeQuestion, q1)
        assert row.selected_index == 1
        assert row.is_correct is False  # correct_index is 0
        assert row.answered_at is not None

    def test_resume_complete_returns_boundary_map(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PROBING).session_id
        q1 = _qid()
        _seed_probe(db, sid, [q1])
        boundary_map = {"algebra": {"floor": 1, "ceiling": 5, "gap_type": "unknown"}}
        with patch("routers.probe.probe_graph") as graph:
            graph.invoke.return_value = {"boundary_map": boundary_map}
            resp = client.post(
                f"/sessions/{sid}/probe",
                json={"answers": [{"question_id": q1, "selected_index": 0}]},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "planning"
        assert body["boundary_map"] == boundary_map
        assert "questions" not in body  # excluded when None

        db.expire_all()
        session = db.get(Session, sid)
        assert session.phase == Phase.PLANNING.value
        assert session.boundary_map == boundary_map
        row = db.get(ProbeQuestion, q1)
        assert row.selected_index == 0
        assert row.is_correct is True  # correct_index is 0
