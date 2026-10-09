"""Unit tests for routers/clarify.py."""

from __future__ import annotations

from unittest.mock import patch

from core.llm import llm
from db import Phase, Session


def _signed_in(auth_cookie: str) -> dict[str, str]:
    """A sign-in cookie: creating a Session requires one from every caller (#145).

    A Visitor's refusal — and the fact that the refused request writes nothing —
    is asserted in ``test_auth_gate.TestDataOwningEndpointsAlwaysRefuseAVisitor``.
    """
    return {"access_token": auth_cookie}


# --- POST /sessions (create + first Clarify call) ---


class TestCreateSession:
    """A signed-in caller starts a Session and gets the first Clarify call."""

    def test_interrupt_returns_clarifying_questions(
        self, client, db, make_session, auth_cookie
    ):
        with (
            patch("routers.clarify.clarify_graph") as graph,
            patch("routers.clarify.detect_language", return_value="English") as dl,
        ):
            graph.invoke.return_value = {
                "__interrupt__": object(),
                "clarifying_questions": ["What area?", "How deep?"],
            }
            resp = client.post(
                "/sessions",
                json={"goal": "Learn calculus"},
                cookies=_signed_in(auth_cookie),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "clarifying"
        assert body["clarifying_questions"] == ["What area?", "How deep?"]
        assert "narrowed_goal" not in body  # excluded when None

        # Session row was persisted with the detected language.
        session = db.get(Session, body["session_id"])
        assert session is not None
        assert session.phase == Phase.CLARIFYING.value
        assert session.language == "English"
        assert session.goal == "Learn calculus"
        dl.assert_called_once_with("Learn calculus", llm)

    def test_complete_returns_narrowed_goal(
        self, client, db, make_session, auth_cookie
    ):
        with (
            patch("routers.clarify.clarify_graph") as graph,
            patch("routers.clarify.detect_language", return_value="English"),
        ):
            graph.invoke.return_value = {"narrowed_goal": "Master derivatives"}
            resp = client.post(
                "/sessions",
                json={"goal": "Learn calculus"},
                cookies=_signed_in(auth_cookie),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "probing"
        assert body["narrowed_goal"] == "Master derivatives"
        assert "clarifying_questions" not in body  # excluded when None

        session = db.get(Session, body["session_id"])
        assert session.phase == Phase.PROBING.value
        assert session.narrowed_goal == "Master derivatives"

    def test_initial_state_and_config_passed_to_graph(
        self, client, db, make_session, auth_cookie
    ):
        with (
            patch("routers.clarify.clarify_graph") as graph,
            patch("routers.clarify.detect_language", return_value="Spanish"),
        ):
            graph.invoke.return_value = {"narrowed_goal": "ng"}
            client.post(
                "/sessions",
                json={"goal": "Aprender cálculo"},
                cookies=_signed_in(auth_cookie),
            )

        args, _kwargs = graph.invoke.call_args
        initial_state = args[0]
        assert initial_state["goal"] == "Aprender cálculo"
        assert initial_state["language"] == "Spanish"
        assert initial_state["round_count"] == 0
        config = args[1]
        # Thread is namespaced per (session, graph).
        assert config["configurable"]["thread_id"].endswith(":clarify")


# --- POST /sessions/{session_id}/clarify (resume the loop) ---


class TestClarifySession:
    def test_not_found(self, client, db):
        resp = client.post("/sessions/nope/clarify", json={"answer": "x"})
        assert resp.status_code == 404

    def test_wrong_phase(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PROBING).session_id
        resp = client.post(f"/sessions/{sid}/clarify", json={"answer": "x"})
        assert resp.status_code == 409
        assert "not 'clarifying'" in resp.json()["detail"]

    def test_resume_interrupt(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.CLARIFYING).session_id
        with (
            patch("routers.clarify.clarify_graph") as graph,
            patch("routers.clarify.has_meaningful_signal", return_value=False),
            patch("routers.clarify.detect_language", return_value="English"),
        ):
            graph.invoke.return_value = {
                "__interrupt__": object(),
                "clarifying_questions": ["More detail?"],
            }
            resp = client.post(
                f"/sessions/{sid}/clarify", json={"answer": "derivatives"}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "clarifying"
        assert body["clarifying_questions"] == ["More detail?"]

    def test_resume_complete(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.CLARIFYING).session_id
        with (
            patch("routers.clarify.clarify_graph") as graph,
            patch("routers.clarify.has_meaningful_signal", return_value=False),
            patch("routers.clarify.detect_language", return_value="English"),
        ):
            graph.invoke.return_value = {"narrowed_goal": "Master limits"}
            resp = client.post(
                f"/sessions/{sid}/clarify", json={"answer": "limits please"}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "probing"
        assert body["narrowed_goal"] == "Master limits"

        db.expire_all()
        session = db.get(Session, sid)
        assert session.phase == Phase.PROBING.value
        assert session.narrowed_goal == "Master limits"

    def test_re_detects_language_when_meaningful(self, client, db, make_session):
        sid = make_session(
            session_id="a", phase=Phase.CLARIFYING, language="English"
        ).session_id
        with (
            patch("routers.clarify.clarify_graph") as graph,
            patch("routers.clarify.has_meaningful_signal", return_value=True),
            patch("routers.clarify.detect_language", return_value="French") as dl,
        ):
            graph.invoke.return_value = {"narrowed_goal": "ng"}
            resp = client.post(
                f"/sessions/{sid}/clarify",
                json={"answer": "Je veux apprendre les dérivées"},
            )

        assert resp.status_code == 200
        dl.assert_called_once_with("Je veux apprendre les dérivées", llm)
        db.expire_all()
        assert db.get(Session, sid).language == "French"

    def test_keeps_language_for_low_signal_answer(self, client, db, make_session):
        sid = make_session(
            session_id="a", phase=Phase.CLARIFYING, language="English"
        ).session_id
        with (
            patch("routers.clarify.clarify_graph") as graph,
            patch("routers.clarify.has_meaningful_signal", return_value=False),
            patch("routers.clarify.detect_language", return_value="English") as dl,
        ):
            graph.invoke.return_value = {"narrowed_goal": "ng"}
            client.post(f"/sessions/{sid}/clarify", json={"answer": "yes"})

        # No re-detection for a one-word answer.
        dl.assert_not_called()
        db.expire_all()
        assert db.get(Session, sid).language == "English"
