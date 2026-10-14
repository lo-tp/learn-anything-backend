"""Unit tests for routers/slides.py."""

from __future__ import annotations

from db import SlideContent

SERVICE_HEADER = "X-Service-Token"


class TestGetSlide:
    def test_found(self, client, db, make_session, service_token):
        sid = make_session(session_id="s1").session_id
        db.add(
            SlideContent(
                slide_id="s1_stepA_slide_1",
                session_id=sid,
                step_id="stepA",
                content="<div>hi</div>",
            )
        )
        db.commit()

        resp = client.get(
            "/slides/s1_stepA_slide_1", headers={SERVICE_HEADER: service_token}
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "slide_id": "s1_stepA_slide_1",
            "content": "<div>hi</div>",
        }

    def test_not_found(self, client, db, service_token):
        resp = client.get("/slides/does-not-exist",
                          headers={SERVICE_HEADER: service_token})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Slide not found"

    def test_placeholder_hidden_outside_dev(self, client, db, make_session,
                                            monkeypatch, service_token):
        monkeypatch.delenv("DEV_MODE", raising=False)
        sid = make_session(session_id="s1").session_id
        db.add(
            SlideContent(
                slide_id="s1_stepA_slide_1",
                session_id=sid,
                step_id="stepA",
                content="<div>placeholder</div>",
                is_placeholder=True,
            )
        )
        db.commit()

        resp = client.get("/slides/s1_stepA_slide_1",
                          headers={SERVICE_HEADER: service_token})
        assert resp.status_code == 404

    def test_placeholder_visible_in_dev(self, client, db, make_session,
                                        monkeypatch, service_token):
        monkeypatch.setenv("DEV_MODE", "1")
        sid = make_session(session_id="s1").session_id
        db.add(
            SlideContent(
                slide_id="s1_stepA_slide_1",
                session_id=sid,
                step_id="stepA",
                content="<div>placeholder</div>",
                is_placeholder=True,
            )
        )
        db.commit()

        resp = client.get("/slides/s1_stepA_slide_1",
                          headers={SERVICE_HEADER: service_token})
        assert resp.status_code == 200
        assert resp.json() == {
            "slide_id": "s1_stepA_slide_1",
            "content": "<div>placeholder</div>",
        }

    def test_placeholder_visible_when_dev_mode_is_spelled_true(
        self, client, db, make_session, monkeypatch, service_token
    ):
        """DEV_MODE means one thing whichever way it is spelled.

        ``core.security.is_dev_mode`` — the predicate the write gates and the
        regenerate endpoint use — accepts "true" as well as "1". A placeholder
        the deck read will show has to be one this endpoint will serve too, or
        the deck has a slot it cannot fill.
        """
        monkeypatch.setenv("DEV_MODE", "true")
        sid = make_session(session_id="s1").session_id
        db.add(
            SlideContent(
                slide_id="s1_stepA_slide_1",
                session_id=sid,
                step_id="stepA",
                content="<div>placeholder</div>",
                is_placeholder=True,
            )
        )
        db.commit()

        resp = client.get("/slides/s1_stepA_slide_1",
                          headers={SERVICE_HEADER: service_token})
        assert resp.status_code == 200
