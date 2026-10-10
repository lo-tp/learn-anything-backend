"""Tests for the sign-in gates on Session endpoints (#89, #145, #178).

Acceptance criteria:
- The Session **reads** — the list, one Session's state, one Session's
  materials — answer 200 to an anonymous caller whatever ``DEV_MODE`` says
  (#178). Browsing is public (ADR-0004); an unreadable cookie does not turn a
  public read into a refusal.
- The Session **writes** — the intake phases, plan generate/adjust/approve,
  dev regeneration — answer 401 to a Visitor when ``DEV_MODE`` is off, and
  are open when it is on, so a local run can walk the phases without a cookie.
  A cookie that cannot be decoded is a Visitor on a write too: the same 401.
- Creating a Session still answers 401 to a Visitor whatever ``DEV_MODE`` says
  (#145): the refusal lands before the write, so no ownerless Session exists.
- ``DEV_MODE`` is read per request so tests can toggle it; it defaults to false.
- Sessions remain shared across users (no per-user scoping).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import jwt

from db import Session

COOKIE_NAME = "access_token"

# A Session read, seeded so an anonymous request is a real 200 rather than a
# 404 (#178: the reads carry no owner, so no cookie is asked for).
PUBLIC_READ = "/sessions/public-1"


def _make_token(email: str = "ada@example.com") -> str:
    """Create a valid sign-in JWT (same shape as what login sets)."""
    from datetime import UTC, datetime, timedelta

    secret = os.environ["JWT_SECRET"]
    now = datetime.now(UTC)
    payload = {"sub": email, "iat": now, "exp": now + timedelta(days=30)}
    return jwt.encode(payload, secret, algorithm="HS256")


def _expired_token() -> str:
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    return jwt.encode(
        {"sub": "a@b.com", "iat": now - timedelta(days=31), "exp": now - timedelta(days=1)},
        os.environ["JWT_SECRET"],
        algorithm="HS256",
    )


class TestSessionReadsArePublic:
    """#178: a Visitor reads a Session list, deck and materials unasked.

    Every test here deletes ``DEV_MODE`` on purpose: that is how production is
    configured (``.env.prod`` sets ``DEV_MODE=0``), and it is the configuration
    in which a Session deck used to answer 401 to a Visitor.
    """

    def test_the_list_answers_a_visitor_with_content(
        self, client, monkeypatch, make_session
    ):
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(session_id="public-1", goal="Learn reservoir computing")

        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert [s["session_id"] for s in resp.json()["sessions"]] == ["public-1"]

    def test_one_sessions_state_answers_a_visitor(self, client, monkeypatch, make_session):
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(session_id="public-1")

        resp = client.get(PUBLIC_READ)
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "public-1"

    def test_materials_answer_a_visitor(self, client, monkeypatch, make_session):
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(session_id="public-1")

        resp = client.get(f"{PUBLIC_READ}/materials")
        assert resp.status_code == 200
        assert resp.json()["generated_steps"] == []

    def test_an_unreadable_cookie_does_not_refuse_a_read(
        self, client, monkeypatch, make_session
    ):
        # The reads stopped asking, so a stale or invalid token is not their
        # business: a Visitor and a User with an expired token see the same deck.
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(session_id="public-1")

        for cookie in ("garbage", _expired_token()):
            resp = client.get(PUBLIC_READ, cookies={COOKIE_NAME: cookie})
            assert resp.status_code == 200, cookie

    def test_reads_are_public_whatever_dev_mode_says(
        self, client, monkeypatch, make_session
    ):
        # One answer for every value of the flag: the reads no longer key off
        # it, so dev and production serve the same thing.
        make_session(session_id="public-1")

        for dev_mode in ("true", "1", "false", "0"):
            monkeypatch.setenv("DEV_MODE", dev_mode)
            for path in ("/sessions", PUBLIC_READ, f"{PUBLIC_READ}/materials"):
                assert client.get(path).status_code == 200, path

    def test_a_signed_in_read_is_answered_as_before(
        self, client, monkeypatch, make_session
    ):
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(session_id="public-1")

        resp = client.get(PUBLIC_READ, cookies={COOKIE_NAME: _make_token()})
        assert resp.status_code == 200


class TestSessionWritesRefuseAVisitor:
    """Ownership begins at the write: an intake or plan write needs a User."""

    def test_clarifying_a_session_refuses_a_visitor(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.post("/sessions/any-1/clarify", json={"answer": "x"})
        assert resp.status_code == 401

    def test_probing_refuses_a_visitor(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.post("/sessions/any-1/probe", json={})
        assert resp.status_code == 401

    def test_plan_writes_refuse_a_visitor(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        for path, body in (
            ("/sessions/any-1/plan/generate", None),
            ("/sessions/any-1/plan/adjust", {"adjustment": "add a quiz"}),
            ("/sessions/any-1/plan/approve", None),
        ):
            resp = client.post(path, json=body) if body else client.post(path)
            assert resp.status_code == 401, path

    def test_dev_regeneration_refuses_a_visitor(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.post("/dev/sessions/any-1/regenerate")
        assert resp.status_code == 401

    def test_a_gated_write_refuses_an_unreadable_cookie(self, client, monkeypatch):
        # A cookie that does not decode is a Visitor, not a User (#89): on a
        # public read it changes nothing (above); on a gated write it is still
        # the refusal.
        monkeypatch.delenv("DEV_MODE", raising=False)
        for cookie in ("garbage", _expired_token()):
            resp = client.post(
                "/sessions/any-1/plan/generate", cookies={COOKIE_NAME: cookie}
            )
            assert resp.status_code == 401, cookie

    def test_the_same_writes_are_open_in_dev(self, client, monkeypatch, make_session):
        # The bounded gate is kept on the writes so a local run can walk the
        # phases without a cookie (#89). One request, then the flag flipped, then
        # the same request: DEV_MODE is read per request, not at import — and the
        # refusal is gone, replaced by the endpoint's own answer (409, wrong
        # phase).
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(session_id="any-1")

        refused = client.post("/sessions/any-1/plan/generate")
        assert refused.status_code == 401

        monkeypatch.setenv("DEV_MODE", "true")
        resp = client.post("/sessions/any-1/plan/generate")
        assert resp.status_code == 409


class TestDataOwningWritesAlwaysRefuseAVisitor:
    """#145: the write that creates data requires a sign-in cookie always.

    Every test here sets ``DEV_MODE=true`` on purpose: that is the configuration
    in which the bug existed (the dev backend answered ``200 {"sessions": []}``
    and quietly created a Session that belongs to nobody). The refusal is the
    same 401 the gate already returns, and it lands before any write.
    """

    def test_creating_a_session_refuses_a_visitor_and_writes_nothing(
        self, client, db, monkeypatch
    ):
        monkeypatch.setenv("DEV_MODE", "true")

        with (
            patch("routers.clarify.clarify_graph") as graph,
            patch("routers.clarify.detect_language", return_value="English"),
        ):
            resp = client.post("/sessions", json={"goal": "Learn calculus"})

        assert resp.status_code == 401
        assert resp.json() == {"detail": "Not authenticated"}
        # The refusal happens before the endpoint runs, so there is no ownerless
        # Session to surface in the list and no graph work was started.
        assert db.query(Session).all() == []
        graph.invoke.assert_not_called()

    def test_a_signed_in_call_is_answered_as_before(
        self, client, db, monkeypatch, auth_cookie
    ):
        monkeypatch.setenv("DEV_MODE", "true")

        with (
            patch("routers.clarify.clarify_graph") as graph,
            patch("routers.clarify.detect_language", return_value="English"),
        ):
            graph.invoke.return_value = {"narrowed_goal": "Master derivatives"}
            resp = client.post(
                "/sessions",
                json={"goal": "Learn calculus"},
                cookies={COOKIE_NAME: auth_cookie},
            )

        assert resp.status_code == 200
        assert resp.json()["phase"] == "probing"
        assert db.query(Session).count() == 1


class TestAuthEndpointsNotGated:
    """Register and login are public (you can't sign in if auth is gated)."""

    def test_register_is_public(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.post(
            "/auth/register",
            json={"email": "x@y.com", "password": "supersecret"},
        )
        assert resp.status_code == 201

    def test_login_is_public(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        client.post(
            "/auth/register",
            json={"email": "x@y.com", "password": "supersecret"},
        )
        resp = client.post(
            "/auth/login",
            json={"email": "x@y.com", "password": "supersecret"},
        )
        assert resp.status_code == 200


class TestSessionsSharedAcrossUsers:
    """Sessions remain shared — no per-user scoping."""

    def test_user_a_can_see_user_b_session(
        self, client, monkeypatch, make_session, auth_cookie
    ):
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(session_id="shared-1", goal="Learn X")

        asks = [
            {},
            {COOKIE_NAME: auth_cookie},
            {COOKIE_NAME: _make_token("a@example.com")},
            {COOKIE_NAME: _make_token("b@example.com")},
        ]
        answers = [
            client.get("/sessions/shared-1", cookies=cookies).json()["session_id"]
            for cookies in asks
        ]
        assert answers == ["shared-1"] * 4
