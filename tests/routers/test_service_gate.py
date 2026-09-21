"""Tests for the service-identity gate on the internal slides endpoint (#103).

Acceptance criteria:
- GET /slides/{id} → 200 with a valid X-Service-Token header, under both
  DEV_MODE=0 and DEV_MODE=1.
- GET /slides/{id} → 401 with no header.
- GET /slides/{id} → 401 with a wrong header.
- GET /slides/{id} → 401 with a valid user cookie and no service header
  (mutual exclusion: the human cookie is never accepted on this route).
- SANDBOX_SERVICE_TOKEN unset in the environment → 401 (fail-secure,
  no dev bypass).
"""

from __future__ import annotations

from db import SlideContent

SERVICE_HEADER = "X-Service-Token"
TOKEN = "s3rvice-t0k3n"


def _seed_slide(db, slide_id: str = "s1_stepA_slide_1") -> None:
    db.add(
        SlideContent(
            slide_id=slide_id,
            session_id="s1",
            step_id="stepA",
            content="<div>hi</div>",
        )
    )
    db.commit()


class TestValidToken:
    """A valid X-Service-Token is accepted under any DEV_MODE."""

    def test_valid_token_dev_mode_off(self, client, db, make_session,
                                       monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        monkeypatch.setenv("SANDBOX_SERVICE_TOKEN", TOKEN)
        make_session(session_id="s1")
        _seed_slide(db)

        resp = client.get(
            "/slides/s1_stepA_slide_1", headers={SERVICE_HEADER: TOKEN}
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "slide_id": "s1_stepA_slide_1",
            "content": "<div>hi</div>",
        }

    def test_valid_token_dev_mode_on(self, client, db, make_session,
                                     monkeypatch):
        monkeypatch.setenv("DEV_MODE", "1")
        monkeypatch.setenv("SANDBOX_SERVICE_TOKEN", TOKEN)
        make_session(session_id="s1")
        _seed_slide(db)

        resp = client.get(
            "/slides/s1_stepA_slide_1", headers={SERVICE_HEADER: TOKEN}
        )
        assert resp.status_code == 200


class TestMutualExclusion:
    """The human cookie is never accepted on this route."""

    def test_valid_cookie_without_header_returns_401(self, client, db,
                                                     make_session,
                                                     monkeypatch,
                                                     auth_cookie):
        monkeypatch.delenv("DEV_MODE", raising=False)
        monkeypatch.setenv("SANDBOX_SERVICE_TOKEN", TOKEN)
        make_session(session_id="s1")
        _seed_slide(db)

        resp = client.get(
            "/slides/s1_stepA_slide_1",
            cookies={"access_token": auth_cookie},
        )
        assert resp.status_code == 401

    def test_valid_cookie_and_wrong_header_returns_401(self, client, db,
                                                       make_session,
                                                       monkeypatch,
                                                       auth_cookie):
        monkeypatch.delenv("DEV_MODE", raising=False)
        monkeypatch.setenv("SANDBOX_SERVICE_TOKEN", TOKEN)
        make_session(session_id="s1")
        _seed_slide(db)

        resp = client.get(
            "/slides/s1_stepA_slide_1",
            cookies={"access_token": auth_cookie},
            headers={SERVICE_HEADER: "not-the-token"},
        )
        assert resp.status_code == 401


class TestFailSecure:
    """No header, wrong header, and unset secret all → 401."""

    def test_no_header_returns_401(self, client, db, make_session,
                                   monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        monkeypatch.setenv("SANDBOX_SERVICE_TOKEN", TOKEN)
        make_session(session_id="s1")
        _seed_slide(db)

        resp = client.get("/slides/s1_stepA_slide_1")
        assert resp.status_code == 401

    def test_wrong_header_returns_401(self, client, db, make_session,
                                       monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        monkeypatch.setenv("SANDBOX_SERVICE_TOKEN", TOKEN)
        make_session(session_id="s1")
        _seed_slide(db)

        resp = client.get(
            "/slides/s1_stepA_slide_1", headers={SERVICE_HEADER: "not-the-token"}
        )
        assert resp.status_code == 401

    def test_env_unset_returns_401(self, client, db, make_session,
                                   monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        monkeypatch.delenv("SANDBOX_SERVICE_TOKEN", raising=False)
        make_session(session_id="s1")
        _seed_slide(db)

        resp = client.get(
            "/slides/s1_stepA_slide_1", headers={SERVICE_HEADER: TOKEN}
        )
        assert resp.status_code == 401

    def test_env_unset_no_dev_bypass(self, client, db, make_session,
                                     monkeypatch):
        """DEV_MODE=1 does not open the service gate."""
        monkeypatch.setenv("DEV_MODE", "1")
        monkeypatch.delenv("SANDBOX_SERVICE_TOKEN", raising=False)
        make_session(session_id="s1")
        _seed_slide(db)

        resp = client.get(
            "/slides/s1_stepA_slide_1", headers={SERVICE_HEADER: TOKEN}
        )
        assert resp.status_code == 401
