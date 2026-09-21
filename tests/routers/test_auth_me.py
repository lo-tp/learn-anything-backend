"""Unit tests for me (read + update) and logout endpoints (#88)."""

from __future__ import annotations

import os

import jwt
import pytest

COOKIE_NAME = "access_token"


def _login(client, email: str = "ada@example.com") -> str:
    """Register + login, return the cookie value."""
    client.post("/auth/register", json={
        "email": email, "password": "supersecret",
    })
    resp = client.post("/auth/login", json={
        "email": email, "password": "supersecret",
    })
    assert resp.status_code == 200
    return resp.cookies.get(COOKIE_NAME)


class TestMeRead:
    def test_returns_current_user(self, client, monkeypatch):
        monkeypatch.setenv("DEV_MODE", "true")
        client.post("/auth/register", json={
            "email": "ada@example.com",
            "password": "supersecret",
            "display_name": "Ada",
        })
        cookie = _login(client)

        resp = client.get("/auth/me", cookies={COOKIE_NAME: cookie})
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "ada@example.com"
        assert body["display_name"] == "Ada"

    def test_no_cookie_returns_401(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_invalid_cookie_returns_401(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.get("/auth/me", cookies={COOKIE_NAME: "bad"})
        assert resp.status_code == 401

    def test_token_for_unknown_user_returns_401(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        # A valid JWT but for a user that doesn't exist in the DB.
        secret = os.environ["JWT_SECRET"]
        from datetime import UTC, datetime, timedelta
        now = datetime.now(UTC)
        token = jwt.encode(
            {"sub": "ghost@example.com", "iat": now, "exp": now + timedelta(days=30)},
            secret, algorithm="HS256",
        )
        resp = client.get("/auth/me", cookies={COOKIE_NAME: token})
        assert resp.status_code == 401


class TestMeUpdate:
    def test_updates_display_name(self, client, monkeypatch):
        monkeypatch.setenv("DEV_MODE", "true")
        client.post("/auth/register", json={
            "email": "ada@example.com",
            "password": "supersecret",
        })
        cookie = _login(client)

        resp = client.patch(
            "/auth/me",
            json={"display_name": "Countess"},
            cookies={COOKIE_NAME: cookie},
        )
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "Countess"

    def test_persists_display_name(self, client, monkeypatch, db):
        from db import User
        monkeypatch.setenv("DEV_MODE", "true")
        client.post("/auth/register", json={
            "email": "ada@example.com",
            "password": "supersecret",
        })
        cookie = _login(client)

        client.patch(
            "/auth/me",
            json={"display_name": "Countess"},
            cookies={COOKIE_NAME: cookie},
        )

        user = db.query(User).filter(User.email == "ada@example.com").first()
        assert user.display_name == "Countess"

    def test_no_cookie_returns_401(self, client, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.patch("/auth/me", json={"display_name": "X"})
        assert resp.status_code == 401

    def test_empty_display_name_rejected(self, client, monkeypatch):
        monkeypatch.setenv("DEV_MODE", "true")
        client.post("/auth/register", json={
            "email": "ada@example.com",
            "password": "supersecret",
        })
        cookie = _login(client)

        resp = client.patch(
            "/auth/me",
            json={"display_name": ""},
            cookies={COOKIE_NAME: cookie},
        )
        assert resp.status_code == 422


class TestLogout:
    def test_clears_cookie(self, client, monkeypatch):
        monkeypatch.setenv("DEV_MODE", "true")
        client.post("/auth/register", json={
            "email": "ada@example.com",
            "password": "supersecret",
        })
        cookie = _login(client)

        resp = client.post(
            "/auth/logout", cookies={COOKIE_NAME: cookie}
        )
        assert resp.status_code == 200
        # Cookie is cleared: expires in the past or max-age=0.
        set_cookie = resp.headers.get("set-cookie", "")
        assert "access_token" in set_cookie
        # The cleared cookie has max-age=0 (or expires in the past).
        assert "max-age=0" in set_cookie.lower() or "expires=" in set_cookie.lower()

    def test_no_cookie_still_returns_200(self, client, monkeypatch):
        """Logout is idempotent: no cookie → still 200 (no-op)."""
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.post("/auth/logout")
        assert resp.status_code == 200
