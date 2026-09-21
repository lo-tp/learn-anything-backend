"""Tests for the sign-in gate on session endpoints (#89).

Acceptance criteria:
- With DEV_MODE=false, session endpoints return 401 without a valid token
  and with an invalid token.
- With DEV_MODE=false, session endpoints return 200 with a valid token.
- With DEV_MODE=true, session endpoints are public (no token required).
- DEV_MODE is read per request so tests can toggle it; it defaults to false.
- Sessions remain shared across users (no per-user scoping).
"""

from __future__ import annotations

import os

import jwt

COOKIE_NAME = "access_token"


def _make_token(email: str = "ada@example.com") -> str:
    """Create a valid sign-in JWT (same shape as what login sets)."""
    from datetime import UTC, datetime, timedelta

    secret = os.environ["JWT_SECRET"]
    now = datetime.now(UTC)
    payload = {"sub": email, "iat": now, "exp": now + timedelta(days=30)}
    return jwt.encode(payload, secret, algorithm="HS256")


# A sample endpoint from each gated router.
SAMPLE_ENDPOINTS = [
    ("GET", "/sessions"),
    ("GET", "/sessions/some-id"),
    ("POST", "/sessions", {"goal": "test"}),
]


class TestGateBlocked:
    """DEV_MODE unset/false: token is required."""

    def test_no_cookie_returns_401(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.get("/sessions")
        assert resp.status_code == 401

    def test_invalid_cookie_returns_401(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.get("/sessions", cookies={COOKIE_NAME: "garbage"})
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
        resp = client.get("/sessions", cookies={COOKIE_NAME: expired})
        assert resp.status_code == 401

    def test_valid_cookie_returns_200(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        token = _make_token()
        resp = client.get("/sessions", cookies={COOKIE_NAME: token})
        assert resp.status_code == 200


class TestGateOpen:
    """DEV_MODE=true: endpoints are public."""

    def test_no_cookie_returns_200(self, client, monkeypatch):
        monkeypatch.setenv("DEV_MODE", "true")
        resp = client.get("/sessions")
        assert resp.status_code == 200

    def test_valid_cookie_also_works(self, client, monkeypatch):
        monkeypatch.setenv("DEV_MODE", "true")
        token = _make_token()
        resp = client.get("/sessions", cookies={COOKIE_NAME: token})
        assert resp.status_code == 200


class TestDevModeReadPerRequest:
    """DEV_MODE is read per request, not cached at startup."""

    def test_toggle_between_requests(self, client, monkeypatch):
        # First request: gate on
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.get("/sessions")
        assert resp.status_code == 401

        # Second request: gate off (no restart needed)
        monkeypatch.setenv("DEV_MODE", "true")
        resp = client.get("/sessions")
        assert resp.status_code == 200

        # Third request: gate on again
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.get("/sessions")
        assert resp.status_code == 401


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
