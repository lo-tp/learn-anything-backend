"""Tests for the sign-in gate on session endpoints (#89, #145).

Acceptance criteria:
- With DEV_MODE=false, the session-work endpoints answer 401 without a valid
  token and with an invalid token, and 200 with a valid token.
- With DEV_MODE=true, those endpoints are public (no token required).
- With DEV_MODE=true, the two data-owning endpoints — the caller's own Session
  list and creating a Session — still answer 401 (#145). A Visitor is refused
  whatever the dev-mode flag says, and the refusal writes no record.
- DEV_MODE is read per request so tests can toggle it; it defaults to false.
- Sessions remain shared across users (no per-user scoping).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import jwt

from db import Session

COOKIE_NAME = "access_token"

# An endpoint that keeps the DEV_MODE-bounded gate (#151's Visitor deck view and
# #146 depend on the session-work surface keeping it), seeded so a signed-in
# request is a real 200 rather than a 404.
GATED_BY_DEV_MODE = "/sessions/gated-1"


def _make_token(email: str = "ada@example.com") -> str:
    """Create a valid sign-in JWT (same shape as what login sets)."""
    from datetime import UTC, datetime, timedelta

    secret = os.environ["JWT_SECRET"]
    now = datetime.now(UTC)
    payload = {"sub": email, "iat": now, "exp": now + timedelta(days=30)}
    return jwt.encode(payload, secret, algorithm="HS256")


class TestGateBlocked:
    """DEV_MODE unset/false: token is required."""

    def test_no_cookie_returns_401(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.get(GATED_BY_DEV_MODE)
        assert resp.status_code == 401

    def test_invalid_cookie_returns_401(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.get(GATED_BY_DEV_MODE, cookies={COOKIE_NAME: "garbage"})
        assert resp.status_code == 401

    def test_expired_cookie_returns_401(self, client, monkeypatch):
        from datetime import UTC, datetime, timedelta

        monkeypatch.delenv("DEV_MODE", raising=False)
        secret = os.environ["JWT_SECRET"]
        now = datetime.now(UTC)
        expired = jwt.encode(
            {"sub": "a@b.com", "iat": now - timedelta(days=31),
             "exp": now - timedelta(days=1)},
            secret,
            algorithm="HS256",
        )
        resp = client.get(GATED_BY_DEV_MODE, cookies={COOKIE_NAME: expired})
        assert resp.status_code == 401

    def test_valid_cookie_returns_200(self, client, monkeypatch, make_session):
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(session_id="gated-1")
        token = _make_token()
        resp = client.get(GATED_BY_DEV_MODE, cookies={COOKIE_NAME: token})
        assert resp.status_code == 200


class TestGateOpen:
    """DEV_MODE=true: the session-work endpoints are public."""

    def test_no_cookie_returns_200(self, client, monkeypatch, make_session):
        monkeypatch.setenv("DEV_MODE", "true")
        make_session(session_id="gated-1")
        resp = client.get(GATED_BY_DEV_MODE)
        assert resp.status_code == 200

    def test_valid_cookie_also_works(self, client, monkeypatch, make_session):
        monkeypatch.setenv("DEV_MODE", "true")
        make_session(session_id="gated-1")
        token = _make_token()
        resp = client.get(GATED_BY_DEV_MODE, cookies={COOKIE_NAME: token})
        assert resp.status_code == 200


class TestDevModeReadPerRequest:
    """DEV_MODE is read per request, not cached at startup."""

    def test_toggle_between_requests(self, client, monkeypatch, make_session):
        make_session(session_id="gated-1")

        # First request: gate on
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.get(GATED_BY_DEV_MODE)
        assert resp.status_code == 401

        # Second request: gate off (no restart needed)
        monkeypatch.setenv("DEV_MODE", "true")
        resp = client.get(GATED_BY_DEV_MODE)
        assert resp.status_code == 200

        # Third request: gate on again
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.get(GATED_BY_DEV_MODE)
        assert resp.status_code == 401


class TestDataOwningEndpointsAlwaysRefuseAVisitor:
    """#145: the endpoints that own data require a sign-in cookie always.

    Every test here sets ``DEV_MODE=true`` on purpose: that is the configuration
    in which the bug existed (the dev backend answered ``200 {"sessions": []}``
    and quietly created a Session that belongs to nobody). The refusal is the
    same 401 the gate already returns, and it lands before any write.
    """

    def test_history_list_refuses_a_visitor(self, client, monkeypatch, make_session):
        monkeypatch.setenv("DEV_MODE", "true")
        make_session(session_id="owned-1", goal="Learn X")

        resp = client.get("/sessions")
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Not authenticated"}

    def test_history_list_refuses_an_unreadable_cookie(self, client, monkeypatch):
        monkeypatch.setenv("DEV_MODE", "true")

        resp = client.get("/sessions", cookies={COOKIE_NAME: "garbage"})
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Invalid token"}

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
        # Session to surface in Explore and no graph work was started.
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

    def test_user_a_can_see_user_b_session(self, client, monkeypatch, make_session):
        monkeypatch.delenv("DEV_MODE", raising=False)

        # Create sessions
        make_session(session_id="shared-1", goal="Learn X")

        # Token for user A
        token_a = _make_token("a@example.com")
        resp = client.get(
            "/sessions/shared-1", cookies={COOKIE_NAME: token_a}
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "shared-1"

        # Token for user B sees the same session
        token_b = _make_token("b@example.com")
        resp = client.get(
            "/sessions/shared-1", cookies={COOKIE_NAME: token_b}
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "shared-1"
