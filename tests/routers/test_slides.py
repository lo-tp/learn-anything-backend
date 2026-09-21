"""Unit tests for routers/slides.py."""

from __future__ import annotations

from db import SlideContent


class TestGetSlide:
    def test_found(self, client, db, make_session):
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

        resp = client.get("/slides/s1_stepA_slide_1")
        assert resp.status_code == 200
        assert resp.json() == {
            "slide_id": "s1_stepA_slide_1",
            "content": "<div>hi</div>",
        }

    def test_not_found(self, client, db):
        resp = client.get("/slides/does-not-exist")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Slide not found"

    def test_placeholder_hidden_outside_dev(self, client, db, make_session,
                                            monkeypatch, auth_cookie):
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
                          cookies={"access_token": auth_cookie})
        assert resp.status_code == 404

    def test_placeholder_visible_in_dev(self, client, db, make_session,
                                        monkeypatch):
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

        resp = client.get("/slides/s1_stepA_slide_1")
        assert resp.status_code == 200
        assert resp.json() == {
            "slide_id": "s1_stepA_slide_1",
            "content": "<div>placeholder</div>",
        }
