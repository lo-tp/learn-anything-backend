"""Unit tests for routers/auth.py (register + login)."""

from __future__ import annotations

import os

import jwt

COOKIE_NAME = "access_token"
THIRTY_DAYS = 30 * 24 * 60 * 60


def _decode(cookie_value: str) -> dict:
    # Read the same runtime secret ``core.security`` signs with, so the
    # decode always matches regardless of .env loading during the suite.
    return jwt.decode(
        cookie_value, os.environ["JWT_SECRET"], algorithms=["HS256"]
    )


class TestRegister:
    def test_success(self, client):
        resp = client.post(
            "/auth/register",
            json={
                "email": "ada@example.com",
                "password": "supersecret",
                "display_name": "Ada",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["email"] == "ada@example.com"
        assert body["display_name"] == "Ada"
        assert isinstance(body["id"], int)

    def test_display_name_defaults_to_local_part(self, client):
        resp = client.post(
            "/auth/register",
            json={"email": "ada@example.com", "password": "supersecret"},
        )
        assert resp.status_code == 201
        assert resp.json()["display_name"] == "ada"

    def test_blank_display_name_defaults_to_local_part(self, client):
        resp = client.post(
            "/auth/register",
            json={
                "email": "Ada@Example.com",
                "password": "supersecret",
                "display_name": "",
            },
        )
        assert resp.status_code == 201
        assert resp.json()["display_name"] == "ada"
        # email is normalized to lowercase on store.
        assert resp.json()["email"] == "ada@example.com"

    def test_duplicate_email_returns_409(self, client):
        client.post(
            "/auth/register",
            json={"email": "ada@example.com", "password": "supersecret"},
        )
        resp = client.post(
            "/auth/register",
            json={"email": "ada@example.com", "password": "othersecret"},
        )
        assert resp.status_code == 409

    def test_duplicate_email_is_case_insensitive(self, client):
        client.post(
            "/auth/register",
            json={"email": "ada@example.com", "password": "supersecret"},
        )
        resp = client.post(
            "/auth/register",
            json={"email": "ADA@Example.COM", "password": "othersecret"},
        )
        assert resp.status_code == 409

    def test_weak_password_returns_422(self, client):
        resp = client.post(
            "/auth/register",
            json={"email": "ada@example.com", "password": "short"},
        )
        assert resp.status_code == 422

    def test_invalid_email_returns_422(self, client):
        resp = client.post(
            "/auth/register",
            json={"email": "not-an-email", "password": "supersecret"},
        )
        assert resp.status_code == 422


class TestLogin:
    def test_valid_credentials_set_cookie(self, client):
        client.post(
            "/auth/register",
            json={"email": "ada@example.com", "password": "supersecret"},
        )
        resp = client.post(
            "/auth/login",
            json={"email": "ada@example.com", "password": "supersecret"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"message": "Signed in"}

        set_cookie = resp.headers["set-cookie"].lower()
        cookie_value = resp.cookies.get(COOKIE_NAME)
        assert cookie_value is not None

        # Cookie attributes: httpOnly, SameSite=Lax, 30-day max-age.
        assert "httponly" in set_cookie
        assert "samesite=lax" in set_cookie
        assert f"max-age={THIRTY_DAYS}" in set_cookie

        # JWT claims.
        payload = _decode(cookie_value)
        assert payload["sub"] == "ada@example.com"
        assert "iat" in payload
        assert "exp" in payload
        assert payload["exp"] - payload["iat"] == THIRTY_DAYS

    def test_login_normalizes_email_case(self, client):
        client.post(
            "/auth/register",
            json={"email": "Ada@Example.com", "password": "supersecret"},
        )
        resp = client.post(
            "/auth/login",
            json={"email": "ada@example.com", "password": "supersecret"},
        )
        assert resp.status_code == 200

    def test_wrong_password_returns_401(self, client):
        client.post(
            "/auth/register",
            json={"email": "ada@example.com", "password": "supersecret"},
        )
        resp = client.post(
            "/auth/login",
            json={"email": "ada@example.com", "password": "wrongpassword"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid credentials"
        assert COOKIE_NAME not in resp.cookies

    def test_unknown_email_returns_401(self, client):
        resp = client.post(
            "/auth/login",
            json={"email": "ghost@example.com", "password": "supersecret"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid credentials"
        assert COOKIE_NAME not in resp.cookies


class TestCookieScope:
    """Where the sign-in cookie is allowed to travel (infra M6).

    The app gate (`learn.` verifying this JWT with the same secret) and the token
    issuer (`api.`) are two hosts on one domain. Without a shared parent domain the
    cookie never reaches the host that reads it, and a signed-in browser is bounced
    back to the login page by its own frontend.
    """

    def _login(self, client, email: str) -> str:
        client.post(
            "/auth/register", json={"email": email, "password": "supersecret"}
        )
        resp = client.post(
            "/auth/login", json={"email": email, "password": "supersecret"}
        )
        assert resp.status_code == 200
        return resp.headers["set-cookie"].lower()

    def test_unset_cookie_domain_keeps_it_host_only(
        self, client, monkeypatch
    ):
        # Local development: the API and the app are separate origins on
        # localhost, and a parent-domain cookie would leak across them.
        monkeypatch.delenv("COOKIE_DOMAIN", raising=False)
        monkeypatch.setenv("FRONTEND_DOMAIN", "http://localhost:3000")
        set_cookie = self._login(client, "host-only@example.com")
        assert "domain=" not in set_cookie
        assert "secure" not in set_cookie

    def test_cookie_domain_shares_the_session_across_surfaces(
        self, client, monkeypatch
    ):
        monkeypatch.setenv("COOKIE_DOMAIN", ".lotp.xyz")
        monkeypatch.setenv("FRONTEND_DOMAIN", "https://learn.lotp.xyz")
        set_cookie = self._login(client, "shared@example.com")
        assert "domain=.lotp.xyz" in set_cookie
        # A production surface is https-only, so the cookie should never travel
        # over a plaintext connection.
        assert "secure" in set_cookie

    def test_logout_clears_the_cookie_it_set(self, client, monkeypatch):
        """A domain-scoped cookie is not removed by a host-only delete."""
        monkeypatch.setenv("COOKIE_DOMAIN", ".lotp.xyz")
        monkeypatch.setenv("FRONTEND_DOMAIN", "https://learn.lotp.xyz")
        self._login(client, "logout@example.com")
        resp = client.post("/auth/logout")
        assert resp.status_code == 200
        set_cookie = resp.headers["set-cookie"].lower()
        assert "domain=.lotp.xyz" in set_cookie
