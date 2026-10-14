"""Tests for MOCK_LLM mode end-to-end behavior (#118).

Acceptance criteria:
- Approving the plan in mock mode does not advance the phase to
  GENERATING and does not schedule generate_materials; it returns a clear
  message that slide generation is disabled.
- generate_materials is a defensive no-op in mock mode.
- dev_regenerate is refused in mock mode (it's dev-only).
- regenerating a single slide is refused too (#197): mock mode has no slides.
"""

from __future__ import annotations

from unittest.mock import patch

from db import Phase, Plan, Session, SlideContent
from routers.plan import generate_materials, regenerate_slide_material


def _valid_plan() -> dict:
    return {
        "prose_summary": "ps",
        "dependency_dag": "dag",
        "steps": [
            {
                "id": "s1",
                "letter": "A",
                "title": "T1",
                "description": "d",
                "depends_on": [],
                "depth": 1,
            },
        ],
    }


# --- Approving the plan never enters slide generation ---


class TestMockApprovePlan:
    def test_approve_stays_reviewing_and_does_not_schedule(
        self, client, db, make_session, monkeypatch
    ):
        sid = make_session(session_id="a", phase=Phase.REVIEWING).session_id
        db.add(
            Plan(
                session_id=sid,
                version=1,
                prose_summary="ps",
                dependency_dag="dag",
                steps=_valid_plan()["steps"],
                adjustments=[],
            )
        )
        db.commit()

        monkeypatch.setenv("MOCK_LLM", "1")
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
            patch("routers.plan.generate_materials") as gen,
        ):
            graph.invoke.return_value = {"current_plan": _valid_plan()}
            resp = client.post(f"/sessions/{sid}/plan/approve")

        assert resp.status_code == 202
        body = resp.json()
        assert body["phase"] == "reviewing"  # NOT "generating"
        assert "mock" in body["message"].lower()
        gen.assert_not_called()

        db.expire_all()
        assert db.get(Session, sid).phase == Phase.REVIEWING.value


class TestMockGenerateMaterialsGuard:
    def test_noop_in_mock_mode(self, db, db_engine, make_session, monkeypatch):
        monkeypatch.setenv("MOCK_LLM", "1")
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(
            Plan(
                session_id=sid,
                version=1,
                prose_summary="ps",
                dependency_dag="dag",
                steps=[{"id": "s1"}],
                adjustments=[],
            )
        )
        db.commit()

        with patch("routers.plan.material_graph") as graph:
            generate_materials(sid)  # defensive no-op even if scheduled

        assert graph.invoke.call_count == 0
        db.expire_all()
        # Phase is left untouched by the no-op.
        assert db.get(Session, sid).phase == Phase.GENERATING.value


class TestMockDevRegenerateRefused:
    def test_refused_even_in_dev_mode(self, client, db, make_session, monkeypatch):
        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.setenv("DEV_MODE", "1")
        sid = make_session(session_id="a", phase=Phase.GENERATING).session_id

        resp = client.post(f"/dev/sessions/{sid}/regenerate")
        assert resp.status_code == 400
        assert "mock" in resp.json()["detail"].lower()


class TestMockSlideRegenerationRefused:
    def test_noop_even_if_the_task_is_somehow_scheduled(
        self, db, db_engine, make_session, monkeypatch
    ):
        """The same guard generate_materials carries: mock mode has no slides."""
        monkeypatch.setenv("MOCK_LLM", "1")
        sid = make_session(session_id="a", phase=Phase.EXECUTING).session_id
        db.add(
            SlideContent(
                slide_id="a_s1_slide_1", session_id=sid, step_id="s1",
                content="x",
                content_spec={"title": "T", "key_points": ["k"], "visual_hint": "v"},
            )
        )
        db.commit()

        with patch("routers.plan.regenerate_slide") as regen:
            regenerate_slide_material(sid, "a_s1_slide_1", None)

        assert regen.call_count == 0

    def test_refused_even_for_a_slide_that_has_a_stored_spec(
        self, client, db, make_session, monkeypatch
    ):
        """Mock mode never writes a slide, so there is none to rewrite."""
        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.setenv("DEV_MODE", "1")
        sid = make_session(session_id="a", phase=Phase.EXECUTING).session_id
        db.add(
            SlideContent(
                slide_id="a_s1_slide_1", session_id=sid, step_id="s1",
                content="x",
                content_spec={"title": "T", "key_points": ["k"], "visual_hint": "v"},
            )
        )
        db.commit()

        resp = client.post(f"/sessions/{sid}/slides/a_s1_slide_1/regenerate")
        assert resp.status_code == 400
        assert "mock" in resp.json()["detail"].lower()
