"""Unit tests for routers/plan.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import sessionmaker

from db import (
    FailedSlide,
    GraphStageTiming,
    Phase,
    Plan,
    PlanPreference,
    Session,
    SlideContent,
    SlidePreference,
    StepMaterial,
)
from routers.plan import (
    StepOut,
    _ordered_materials,
    generate_materials,
    validate_plan,
)


def _valid_plan(**overrides) -> dict:
    plan = {
        "prose_summary": "ps",
        "dependency_dag": "dag",
        "steps": [
            {"id": "s1", "letter": "A", "title": "T1", "description": "d",
             "depends_on": [], "depth": 0},
            {"id": "s2", "letter": "B", "title": "T2", "description": "d",
             "depends_on": ["s1"], "depth": 1},
        ],
    }
    plan.update(overrides)
    return plan


def _seed_plan_session(db, make_session, sid, phase, steps=None, **extra):
    make_session(session_id=sid, phase=phase, **extra)
    db.add(
        Plan(
            session_id=sid,
            version=1,
            prose_summary="ps",
            dependency_dag="dag",
            steps=steps or _valid_plan()["steps"],
            adjustments=[],
        )
    )
    db.commit()
    return sid


def _adjusted_plan_approve_result() -> dict:
    """The approve resume state for a plan that went through one adjustment.

    Exactly what the plan graph carries when a learner adjusts and then
    approves: the version they rejected, the one they chose, their words, and
    the prompt of the design pass that produced the chosen version (#167).
    """
    return {
        "current_plan": _valid_plan(prose_summary="ps v2"),
        "prior_plan": _valid_plan(prose_summary="ps v1"),
        "adjustment": "add a quiz step",
        "prompt_context": [
            {"role": "system", "content": "Refine the existing plan"},
            {
                "role": "human",
                "content": "Goal: ng\nUser adjustment request: add a quiz step",
            },
        ],
    }


def _advance_plan_row_to_version(db, sid, version, adjustments):
    """Make the Plan row look like it has been adjusted ``version - 1`` times."""
    row = db.query(Plan).filter_by(session_id=sid).one()
    row.version = version
    row.adjustments = adjustments
    db.commit()


# --- validate_plan (pure helper) ---


class TestStepOut:
    def test_letter_passes_through(self):
        s = StepOut(id="s1", letter="A", title="T1", description="d",
                    depends_on=[], depth=1)
        assert s.letter == "A"

    def test_legacy_step_without_letter_defaults_empty(self):
        s = StepOut(id="s1", title="T1", description="d",
                    depends_on=[], depth=1)
        assert s.letter == ""


class TestValidatePlan:
    def test_accepts_valid_plan(self):
        validate_plan(_valid_plan())

    def test_rejects_empty_steps(self):
        with pytest.raises(HTTPException) as exc:
            validate_plan({"steps": []})
        assert exc.value.status_code == 500

    def test_rejects_unknown_dependency(self):
        with pytest.raises(HTTPException) as exc:
            validate_plan(
                {"steps": [{"id": "s1", "depends_on": ["nope"]},
                           {"id": "s2", "depends_on": []}]}
            )
        assert exc.value.status_code == 500

    def test_rejects_cycle(self):
        with pytest.raises(HTTPException) as exc:
            validate_plan(
                {"steps": [
                    {"id": "s1", "depends_on": ["s2"]},
                    {"id": "s2", "depends_on": ["s1"]},
                ]}
            )
        assert exc.value.status_code == 500


# --- _ordered_materials (pure helper) ---


class TestOrderedMaterials:
    def test_orders_by_plan_then_id(self, db, make_session):
        sid = make_session(session_id="a", phase=Phase.EXECUTING).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1"}, {"id": "s2"}], adjustments=[]))
        # Insert s2 first (lower row id) to prove ordering is by plan, not id.
        db.add(StepMaterial(session_id=sid, step_id="s2", slides=[],
                            questions=[], summary={"step_id": "s2", "title": "T2",
                                                   "key_points": []}))
        db.add(StepMaterial(session_id=sid, step_id="s1", slides=[],
                            questions=[], summary={"step_id": "s1", "title": "T1",
                                                   "key_points": []}))
        db.commit()
        session = db.get(Session, sid)

        ordered = _ordered_materials(session)
        assert [m.step_id for m in ordered] == ["s1", "s2"]

    def test_rows_not_in_plan_sort_after_plan_rows(self, db, make_session):
        sid = make_session(session_id="a", phase=Phase.EXECUTING).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag", steps=[{"id": "s1"}], adjustments=[]))
        db.add(StepMaterial(session_id=sid, step_id="zzz", slides=[],
                            questions=[], summary={"step_id": "zzz", "title": "T",
                                                   "key_points": []}))
        db.add(StepMaterial(session_id=sid, step_id="s1", slides=[],
                            questions=[], summary={"step_id": "s1", "title": "T",
                                                   "key_points": []}))
        db.commit()
        session = db.get(Session, sid)

        ordered = _ordered_materials(session)
        assert ordered[0].step_id == "s1"
        assert ordered[1].step_id == "zzz"

    def test_no_plan_keeps_row_id_order(self, db, make_session):
        sid = make_session(session_id="a", phase=Phase.EXECUTING).session_id
        db.add(StepMaterial(session_id=sid, step_id="b", slides=[],
                            questions=[], summary={"step_id": "b", "title": "T",
                                                   "key_points": []}))
        db.add(StepMaterial(session_id=sid, step_id="a", slides=[],
                            questions=[], summary={"step_id": "a", "title": "T",
                                                   "key_points": []}))
        db.commit()
        session = db.get(Session, sid)

        ordered = _ordered_materials(session)
        # No plan -> insertion (row-id) order: b (inserted first) then a.
        assert [m.step_id for m in ordered] == ["b", "a"]


# --- POST /sessions/{session_id}/plan/generate ---


class TestGeneratePlan:
    def test_not_found(self, client, db):
        resp = client.post("/sessions/nope/plan/generate")
        assert resp.status_code == 404

    def test_wrong_phase_reviewing(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.REVIEWING).session_id
        resp = client.post(f"/sessions/{sid}/plan/generate")
        assert resp.status_code == 409
        assert "adjust" in resp.json()["detail"]

    def test_wrong_phase_other(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.CLARIFYING).session_id
        resp = client.post(f"/sessions/{sid}/plan/generate")
        assert resp.status_code == 409
        assert "not 'planning'" in resp.json()["detail"]

    def test_success_persists_plan_and_advances_phase(self, client, db, make_session):
        sid = make_session(
            session_id="a", phase=Phase.PLANNING, narrowed_goal="ng"
        ).session_id
        plan = _valid_plan()
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
        ):
            graph.invoke.return_value = {"__interrupt__": object(),
                                         "current_plan": plan}
            resp = client.post(f"/sessions/{sid}/plan/generate")

        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "reviewing"
        assert body["plan"]["prose_summary"] == "ps"
        assert len(body["plan"]["steps"]) == 2
        assert [s["letter"] for s in body["plan"]["steps"]] == ["A", "B"]

        db.expire_all()
        session = db.get(Session, sid)
        assert session.phase == Phase.REVIEWING.value
        assert session.plan.version == 1
        assert session.plan.steps == plan["steps"]

    def test_graph_exception_is_500_and_cleans_thread(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PLANNING).session_id
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer") as cp,
        ):
            graph.invoke.side_effect = Exception("boom")
            resp = client.post(f"/sessions/{sid}/plan/generate")
        assert resp.status_code == 500
        cp.delete_thread.assert_called_once_with(f"{sid}:plan")

    def test_cleanup_thread_failure_still_500(self, client, db, make_session):
        """A failing checkpoint cleanup must not mask the 500 from the graph."""
        sid = make_session(session_id="a", phase=Phase.PLANNING).session_id
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer") as cp,
        ):
            graph.invoke.side_effect = Exception("boom")
            cp.delete_thread.side_effect = Exception("checkpoint down")
            resp = client.post(f"/sessions/{sid}/plan/generate")
        assert resp.status_code == 500

    def test_no_interrupt_is_500(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PLANNING).session_id
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
        ):
            graph.invoke.return_value = {"current_plan": _valid_plan()}
            resp = client.post(f"/sessions/{sid}/plan/generate")
        assert resp.status_code == 500

    def test_no_current_plan_is_500(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PLANNING).session_id
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
        ):
            graph.invoke.return_value = {"__interrupt__": object(),
                                         "current_plan": None}
            resp = client.post(f"/sessions/{sid}/plan/generate")
        assert resp.status_code == 500

    def test_invalid_plan_is_500(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PLANNING).session_id
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
        ):
            graph.invoke.return_value = {
                "__interrupt__": object(),
                "current_plan": _valid_plan(steps=[]),
            }
            resp = client.post(f"/sessions/{sid}/plan/generate")
        assert resp.status_code == 500


# --- POST /sessions/{session_id}/plan/adjust ---


class TestAdjustPlan:
    def test_not_found(self, client, db):
        resp = client.post("/sessions/nope/plan/adjust",
                           json={"adjustment": "x"})
        assert resp.status_code == 404

    def test_wrong_phase(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PLANNING).session_id
        resp = client.post(f"/sessions/{sid}/plan/adjust",
                           json={"adjustment": "x"})
        assert resp.status_code == 409

    def test_success_increments_version_and_records_adjustment(
        self, client, db, make_session
    ):
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING,
            steps=_valid_plan()["steps"],
        )
        # Existing plan carries a prior adjustment.
        existing = db.query(Plan).filter_by(session_id=sid).one()
        existing.adjustments = ["earlier"]
        db.commit()

        new_plan = _valid_plan(prose_summary="ps v2")
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.detect_language", return_value="English"),
            patch("routers.plan.has_meaningful_signal", return_value=False),
        ):
            graph.invoke.return_value = {"__interrupt__": object(),
                                         "current_plan": new_plan}
            resp = client.post(
                f"/sessions/{sid}/plan/adjust", json={"adjustment": "add a quiz"}
            )

        assert resp.status_code == 200
        assert resp.json()["phase"] == "reviewing"

        db.expire_all()
        plan_row = db.query(Plan).filter_by(session_id=sid).one()
        assert plan_row.version == 2
        assert plan_row.adjustments == ["earlier", "add a quiz"]
        assert plan_row.prose_summary == "ps v2"

    def test_re_detects_language_when_meaningful(self, client, db, make_session):
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING,
            steps=_valid_plan()["steps"],
            language="English",
        )
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.detect_language", return_value="German") as dl,
            patch("routers.plan.has_meaningful_signal", return_value=True),
        ):
            graph.invoke.return_value = {"__interrupt__": object(),
                                         "current_plan": _valid_plan()}
            resp = client.post(
                f"/sessions/{sid}/plan/adjust",
                json={"adjustment": "Bitte füge mehr Beispiele hinzu"},
            )
        assert resp.status_code == 200
        dl.assert_called_once()
        db.expire_all()
        assert db.get(Session, sid).language == "German"

    def test_graph_exception_is_500_and_cleans_thread(self, client, db, make_session):
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer") as cp,
        ):
            graph.invoke.side_effect = Exception("boom")
            resp = client.post(
                f"/sessions/{sid}/plan/adjust", json={"adjustment": "x"}
            )
        assert resp.status_code == 500
        cp.delete_thread.assert_called_once_with(f"{sid}:plan")

    def test_no_interrupt_is_500(self, client, db, make_session):
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
        ):
            graph.invoke.return_value = {"current_plan": _valid_plan()}
            resp = client.post(
                f"/sessions/{sid}/plan/adjust", json={"adjustment": "x"}
            )
        assert resp.status_code == 500

    def test_no_current_plan_is_500(self, client, db, make_session):
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
        ):
            graph.invoke.return_value = {"__interrupt__": object(),
                                         "current_plan": None}
            resp = client.post(
                f"/sessions/{sid}/plan/adjust", json={"adjustment": "x"}
            )
        assert resp.status_code == 500

    def test_invalid_plan_is_500(self, client, db, make_session):
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
        ):
            graph.invoke.return_value = {"__interrupt__": object(),
                                         "current_plan": _valid_plan(steps=[])}
            resp = client.post(
                f"/sessions/{sid}/plan/adjust", json={"adjustment": "x"}
            )
        assert resp.status_code == 500


# --- POST /sessions/{session_id}/plan/approve ---


class TestApprovePlan:
    def test_not_found(self, client, db):
        resp = client.post("/sessions/nope/plan/approve")
        assert resp.status_code == 404

    def test_wrong_phase(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.PLANNING).session_id
        resp = client.post(f"/sessions/{sid}/plan/approve")
        assert resp.status_code == 409

    def test_success_202_and_schedules_generation(self, client, db, make_session):
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
            patch("routers.plan.localize_status", return_value="Plan aprobado.") as ls,
            patch("routers.plan.generate_materials") as gen,
        ):
            graph.invoke.return_value = {"current_plan": _valid_plan()}
            resp = client.post(f"/sessions/{sid}/plan/approve")

        assert resp.status_code == 202
        body = resp.json()
        assert body["phase"] == "generating"
        assert body["message"] == "Plan aprobado."
        ls.assert_called_once()
        gen.assert_called_once_with(sid)

        db.expire_all()
        assert db.get(Session, sid).phase == Phase.GENERATING.value

    def test_graph_exception_is_500(self, client, db, make_session):
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
            patch("routers.plan.generate_materials"),
        ):
            graph.invoke.side_effect = Exception("boom")
            resp = client.post(f"/sessions/{sid}/plan/approve")
        assert resp.status_code == 500

    def test_no_plan_is_500(self, client, db, make_session):
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
            patch("routers.plan.generate_materials"),
        ):
            graph.invoke.return_value = {"current_plan": None}
            resp = client.post(f"/sessions/{sid}/plan/approve")
        assert resp.status_code == 500


# --- Preference pair captured at approve (#167) ---


class TestApproveCapturesPreference:
    """An approve after the adjust loop stores one instruction-conditioned
    pair. The learner-facing response is the same either way."""

    def _approve(self, client, graph_result, sid):
        with (
            patch("routers.plan.plan_graph") as graph,
            patch("routers.plan.checkpointer"),
            patch("routers.plan.localize_status", return_value="ok"),
            patch("routers.plan.generate_materials") as gen,
        ):
            graph.invoke.return_value = graph_result
            resp = client.post(f"/sessions/{sid}/plan/approve")
        return resp, gen

    def test_adjusted_plan_writes_one_row_with_the_pair(
        self, client, db, make_session
    ):
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        _advance_plan_row_to_version(db, sid, 2, ["add a quiz step"])

        resp, gen = self._approve(client, _adjusted_plan_approve_result(), sid)
        assert resp.status_code == 202
        gen.assert_called_once_with(sid)

        db.expire_all()
        rows = db.query(PlanPreference).filter_by(session_id=sid).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.rejected_version == 1
        assert row.chosen_version == 2
        assert row.adjustment == "add a quiz step"
        # Both versions as the learner saw them, not just their version numbers.
        assert row.rejected_plan["prose_summary"] == "ps v1"
        assert row.chosen_plan["prose_summary"] == "ps v2"
        assert [m["role"] for m in row.prompt_context] == ["system", "human"]
        assert "add a quiz step" in row.prompt_context[1]["content"]

    def test_unmodified_plan_writes_no_row(self, client, db, make_session):
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )

        resp, _ = self._approve(
            client, {"current_plan": _valid_plan()}, sid
        )
        assert resp.status_code == 202

        db.expire_all()
        assert db.query(PlanPreference).filter_by(session_id=sid).count() == 0
        # The approve itself behaved exactly as before.
        assert db.get(Session, sid).phase == Phase.GENERATING.value

    @pytest.mark.parametrize("adjustment", ["", "   ", None])
    def test_adjustment_that_says_nothing_writes_no_row(
        self, client, db, make_session, adjustment
    ):
        """A pair is instruction-conditioned, so a blank adjustment (which does
        run a real refinement pass) leaves nothing to pair on (#167)."""
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        _advance_plan_row_to_version(db, sid, 2, [adjustment or ""])

        pair = _adjusted_plan_approve_result()
        if adjustment is None:
            pair.pop("adjustment")
        else:
            pair["adjustment"] = adjustment
        resp, gen = self._approve(client, pair, sid)

        assert resp.status_code == 202
        gen.assert_called_once_with(sid)

        db.expire_all()
        assert db.query(PlanPreference).filter_by(session_id=sid).count() == 0
        assert db.get(Session, sid).phase == Phase.GENERATING.value

    def test_user_visible_behaviour_is_unchanged(self, client, db, make_session):
        """Capturing the pair changes nothing the learner sees (#167)."""
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        _advance_plan_row_to_version(db, sid, 2, ["add a quiz step"])

        resp, _ = self._approve(client, _adjusted_plan_approve_result(), sid)

        assert resp.status_code == 202
        assert resp.json() == {"phase": "generating", "message": "ok"}
        db.expire_all()
        row = db.query(Plan).filter_by(session_id=sid).one()
        assert row.version == 2
        assert row.prose_summary == "ps v2"

    def test_pair_is_written_once_even_if_a_second_approve_gets_through(
        self, client, db, make_session, monkeypatch
    ):
        """Mock mode leaves the session in reviewing, so a repeat approve must
        not store the same pair twice."""
        monkeypatch.setenv("MOCK_LLM", "1")
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        _advance_plan_row_to_version(db, sid, 2, ["add a quiz step"])

        self._approve(client, _adjusted_plan_approve_result(), sid)
        self._approve(client, _adjusted_plan_approve_result(), sid)

        db.expire_all()
        assert db.query(PlanPreference).filter_by(session_id=sid).count() == 1

    def test_write_failure_does_not_fail_the_approve(self, client, db, make_session):
        """Training data is best-effort: it must never break an approve."""
        sid = _seed_plan_session(
            db, make_session, "a", Phase.REVIEWING, steps=_valid_plan()["steps"]
        )
        _advance_plan_row_to_version(db, sid, 2, ["add a quiz step"])

        with patch(
            "routers.plan.PlanPreference", side_effect=RuntimeError("db down")
        ):
            resp, gen = self._approve(client, _adjusted_plan_approve_result(), sid)

        assert resp.status_code == 202
        assert resp.json()["phase"] == "generating"
        gen.assert_called_once_with(sid)
        db.expire_all()
        assert db.query(PlanPreference).filter_by(session_id=sid).count() == 0


# --- GET /sessions/{session_id}/materials ---


class TestGetMaterials:
    def test_not_found(self, client, db):
        resp = client.get("/sessions/nope/materials")
        assert resp.status_code == 404

    def test_empty(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.GENERATING).session_id
        resp = client.get(f"/sessions/{sid}/materials")
        assert resp.status_code == 200
        assert resp.json() == {"phase": "generating", "generated_steps": []}

    def test_unknown_phase_is_500(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.EXECUTING).session_id
        session = db.get(Session, sid)
        session.phase = "bogus"
        db.commit()
        resp = client.get(f"/sessions/{sid}/materials")
        assert resp.status_code == 500

    def test_materials_with_slides_and_questions_in_plan_order(
        self, client, db, make_session
    ):
        sid = make_session(session_id="a", phase=Phase.EXECUTING).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1"}, {"id": "s2"}], adjustments=[]))
        # s2 inserted first (lower row id) to prove plan ordering.
        db.add(StepMaterial(
            session_id=sid, step_id="s2",
            slides=["a_s2_slide_1"],
            questions=[{"id": "q1", "text": "?", "options": ["A", "B"],
                        "correct_index": 0, "explanation": "e"}],
            summary={"step_id": "s2", "title": "T2", "key_points": ["k"]},
        ))
        db.add(StepMaterial(
            session_id=sid, step_id="s1", slides=[], questions=[],
            summary={"step_id": "s1", "title": "T1", "key_points": []},
        ))
        db.commit()

        resp = client.get(f"/sessions/{sid}/materials")
        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "executing"
        assert [g["step_id"] for g in body["generated_steps"]] == ["s1", "s2"]

        s2 = body["generated_steps"][1]
        assert s2["summary"] == {"step_id": "s2", "title": "T2", "key_points": ["k"]}
        # Slides come before questions in the item list.
        assert [i["type"] for i in s2["items"]] == ["slide", "question"]
        assert s2["items"][0] == {"type": "slide",
                                  "slide_id": "a_s2_slide_1"}
        assert s2["items"][1] == {
            "type": "question", "id": "q1", "text": "?",
            "options": ["A", "B"], "correct_index": 0, "explanation": "e",
        }

    def test_placeholder_slides_hidden_outside_dev(self, client, db,
                                                    make_session, monkeypatch,
                                                    auth_cookie):
        monkeypatch.delenv("DEV_MODE", raising=False)
        sid = make_session(session_id="a", phase=Phase.EXECUTING).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag", steps=[{"id": "s1"}], adjustments=[]))
        db.add(StepMaterial(
            session_id=sid, step_id="s1",
            slides=["a_s1_slide_1", "a_s1_slide_2"], questions=[],
            summary={"step_id": "s1", "title": "T1", "key_points": []},
        ))
        db.add(SlideContent(slide_id="a_s1_slide_1", session_id=sid,
                            step_id="s1", content="<div/>"))
        db.add(SlideContent(slide_id="a_s1_slide_2", session_id=sid,
                            step_id="s1", content="<div>ph</div>",
                            is_placeholder=True))
        db.commit()

        resp = client.get(f"/sessions/{sid}/materials",
                          cookies={"access_token": auth_cookie})
        assert resp.status_code == 200
        items = resp.json()["generated_steps"][0]["items"]
        # The placeholder slot is omitted from the deck in production.
        assert items == [{"type": "slide", "slide_id": "a_s1_slide_1"}]

    def test_placeholder_slides_visible_in_dev(self, client, db, make_session,
                                               monkeypatch):
        monkeypatch.setenv("DEV_MODE", "1")
        sid = make_session(session_id="a", phase=Phase.EXECUTING).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag", steps=[{"id": "s1"}], adjustments=[]))
        db.add(StepMaterial(
            session_id=sid, step_id="s1",
            slides=["a_s1_slide_1", "a_s1_slide_2"], questions=[],
            summary={"step_id": "s1", "title": "T1", "key_points": []},
        ))
        db.add(SlideContent(slide_id="a_s1_slide_1", session_id=sid,
                            step_id="s1", content="<div/>"))
        db.add(SlideContent(slide_id="a_s1_slide_2", session_id=sid,
                            step_id="s1", content="<div>ph</div>",
                            is_placeholder=True))
        db.commit()

        resp = client.get(f"/sessions/{sid}/materials")
        assert resp.status_code == 200
        items = resp.json()["generated_steps"][0]["items"]
        # In dev mode the placeholder keeps its visible slot.
        assert items == [
            {"type": "slide", "slide_id": "a_s1_slide_1"},
            {"type": "slide", "slide_id": "a_s1_slide_2"},
        ]


# --- POST /dev/sessions/{session_id}/regenerate (dev-only) ---


class TestDevRegenerate:
    def test_disabled_by_default(self, client, db, make_session, monkeypatch,
                                  auth_cookie):
        monkeypatch.delenv("DEV_MODE", raising=False)
        sid = make_session(session_id="a", phase=Phase.GENERATING).session_id
        resp = client.post(f"/dev/sessions/{sid}/regenerate",
                           cookies={"access_token": auth_cookie})
        assert resp.status_code == 404

    def test_no_plan_is_409(self, client, db, make_session, monkeypatch):
        monkeypatch.setenv("DEV_MODE", "1")
        sid = make_session(session_id="a", phase=Phase.GENERATING).session_id
        resp = client.post(f"/dev/sessions/{sid}/regenerate")
        assert resp.status_code == 409

    def test_reachable_without_auth_when_dev_mode_is_true(
        self, client, db, make_session, monkeypatch
    ):
        """DEV_MODE=true (the auth gate's dev-mode value) opens the endpoint:

        no auth cookie is sent and the endpoint is not hidden behind a 404.
        """
        monkeypatch.setenv("DEV_MODE", "true")
        sid = make_session(session_id="a", phase=Phase.GENERATING).session_id
        resp = client.post(f"/dev/sessions/{sid}/regenerate")
        assert resp.status_code == 409  # reached, not 401/404

    def test_success_wipes_materials_and_restarts(self, client, db, make_session,
                                                  monkeypatch):
        monkeypatch.setenv("DEV_MODE", "1")
        sid = make_session(
            session_id="a", phase=Phase.EXECUTING, boundary_map={"algebra": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag", steps=[{"id": "s1"}], adjustments=[]))
        db.add(StepMaterial(session_id=sid, step_id="s1",
                            slides=["a_s1_slide_1"], questions=[],
                            summary={"step_id": "s1", "title": "T",
                                     "key_points": []}))
        db.add(SlideContent(slide_id="a_s1_slide_1", session_id=sid,
                            step_id="s1", content="<div/>"))
        db.commit()

        with (
            patch("routers.plan.checkpointer") as cp,
            patch("routers.plan.generate_materials") as gen,
        ):
            resp = client.post(f"/dev/sessions/{sid}/regenerate")

        assert resp.status_code == 202
        assert resp.json()["phase"] == "generating"
        gen.assert_called_once_with(sid)
        cp.delete_thread.assert_called_with(f"{sid}:material:s1")

        db.expire_all()
        assert db.query(StepMaterial).filter_by(session_id=sid).count() == 0
        assert db.query(SlideContent).filter_by(session_id=sid).count() == 0
        assert db.get(Session, sid).phase == Phase.GENERATING.value

    def test_checkpoint_cleanup_failure_is_best_effort(
        self, client, db, make_session, monkeypatch
    ):
        monkeypatch.setenv("DEV_MODE", "1")
        sid = make_session(
            session_id="a", phase=Phase.EXECUTING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag", steps=[{"id": "s1"}], adjustments=[]))
        db.commit()

        with (
            patch("routers.plan.checkpointer") as cp,
            patch("routers.plan.generate_materials") as gen,
        ):
            cp.delete_thread.side_effect = Exception("boom")
            resp = client.post(f"/dev/sessions/{sid}/regenerate")

        # Best-effort cleanup failure must not break the endpoint.
        assert resp.status_code == 202
        gen.assert_called_once_with(sid)


# --- generate_materials background driver (called directly) ---


def _timing(
    stage: str,
    slide_index: int | None = None,
    attempt: int | None = None,
    seconds: float = 0.1,
) -> dict:
    """One stage_timings entry as the material graph nodes would produce it."""
    return {
        "stage": stage,
        "context": {
            "step_id": "s1", "slide_index": slide_index, "attempt": attempt,
        },
        "duration_seconds": seconds,
    }


def _two_slide_results(pairs: list[dict] | None = None) -> list[dict]:
    """A 3-call interrupt/resume sequence: two slides, then the final state.

    Each state carries the ACCUMULATING ``stage_timings`` list (3 / 5 / 7
    entries), exactly as the real material graph would. ``pairs`` is the
    preference pairs the graph's ``compile_slide`` node would have left behind
    (#168) — present at the pause of the slide that produced them, and carried
    in every later state — empty by default, because both slides here compile
    first try."""
    question = {"id": "q1", "text": "?", "options": ["A", "B"],
                "correct_index": 0, "explanation": "e"}
    t_plan = _timing("plan_slide_contents", seconds=0.5)
    t_w1 = _timing("write_slide", 1, 1, 0.2)
    t_c1 = _timing("compile_slide", 1, 1, 0.3)
    t_w2 = _timing("write_slide", 2, 1, 0.2)
    t_c2 = _timing("compile_slide", 2, 1, 0.3)
    t_q = _timing("write_questions", seconds=0.4)
    t_s = _timing("summarize_step", seconds=0.1)
    return [
        {"__interrupt__": object(), "slides": ["<div>1</div>"],
         # The pair belongs to the compile that produced it, so it is already
         # in state at this pause (#168) — and accumulates from here on.
         "slide_preferences": list(pairs or []),
         "stage_timings": [t_plan, t_w1, t_c1]},
        {"__interrupt__": object(), "slides": ["<div>1</div>", "<div>2</div>"],
         "slide_preferences": list(pairs or []),
         "stage_timings": [t_plan, t_w1, t_c1, t_w2, t_c2]},
        {"slides": ["<div>1</div>", "<div>2</div>"],
         "questions": [question],
         "summary": {"step_id": "s1", "title": "T1", "key_points": ["k"]},
         "failed_attempts": [{"index": 0, "attempt": 1, "prompt": "p", "jsx": "j",
                              "error": "e"}],
         "slide_preferences": list(pairs or []),
         "stage_timings": [t_plan, t_w1, t_c1, t_w2, t_c2, t_q, t_s]},
    ]


# One pair as graphs.material.nodes._slide_preference would produce it, for
# slide 1 of this step: attempt 1 failed, attempt 2 compiled.
_PREFERENCE_PAIR = {
    "index": 0,
    "rejected_attempt": 1,
    "chosen_attempt": 2,
    "prompt_context": '[{"role": "system", "content": "s"}]',
    "chosen_prompt": '[{"role": "system", "content": "s + the retry note"}]',
    "rejected_jsx": "<div>broken</div>",
    "rejected_error": "syntax error",
    "chosen_jsx": "<div>fixed</div>",
}


@pytest.fixture()
def fake_session_factory(db_engine):
    """Point ``routers.plan.SessionFactory`` at the test engine so the
    driver's own DB session sees the seeded rows."""
    TestingSession = sessionmaker(bind=db_engine)
    factory = MagicMock()
    factory.side_effect = lambda: TestingSession()
    with patch("routers.plan.SessionFactory", factory):
        yield factory


class TestGenerateMaterials:
    def test_session_not_found_returns_early(
        self, db, db_engine, fake_session_factory
    ):
        with patch("routers.plan.material_graph") as graph:
            generate_materials("missing")

        # No session -> no graph run and nothing persisted.
        assert graph.invoke.call_count == 0
        assert db.query(StepMaterial).count() == 0
        assert db.query(SlideContent).count() == 0

    def test_no_plan_returns_early(self, db, db_engine, make_session,
                                    fake_session_factory):
        sid = make_session(session_id="a", phase=Phase.GENERATING).session_id
        with patch("routers.plan.material_graph") as graph:
            generate_materials(sid)
        assert graph.invoke.call_count == 0
        db.expire_all()
        assert db.get(Session, sid).phase == Phase.GENERATING.value  # unchanged

    def test_happy_path_commits_slides_questions_summary_and_failed(
        self, db, db_engine, make_session, fake_session_factory
    ):
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        db.commit()

        with patch("routers.plan.material_graph") as graph:
            graph.invoke.side_effect = _two_slide_results()
            generate_materials(sid)

        assert graph.invoke.call_count == 3
        db.expire_all()

        session = db.get(Session, sid)
        assert session.phase == Phase.EXECUTING.value

        step_row = db.query(StepMaterial).filter_by(
            session_id=sid, step_id="s1").one()
        assert step_row.is_complete is True
        assert step_row.slides == ["a_s1_slide_1", "a_s1_slide_2"]
        assert step_row.questions[0]["id"] == "q1"
        assert step_row.summary == {"step_id": "s1", "title": "T1",
                                    "key_points": ["k"]}

        # One SlideContent per compiled slide, with the last spec as content.
        s1 = db.get(SlideContent, "a_s1_slide_1")
        s2 = db.get(SlideContent, "a_s1_slide_2")
        assert s1.content == "<div>1</div>"
        assert s2.content == "<div>2</div>"

        # Failed attempt persisted (index is 1-based in the DB).
        failed = db.query(FailedSlide).filter_by(session_id=sid, step_id="s1").one()
        assert failed.slide_index == 1
        assert failed.error == "e"

        # One GraphStageTiming row per stage execution (7 for this 2-slide
        # run) plus a final generate_step row with the step's total time.
        # All attributed to the material graph, in execution order.
        timings = (
            db.query(GraphStageTiming)
            .filter_by(session_id=sid)
            .order_by(GraphStageTiming.id)
            .all()
        )
        assert len(timings) == 8
        assert all(t.graph == "material" for t in timings)
        assert [t.stage for t in timings] == [
            "plan_slide_contents", "write_slide", "compile_slide",
            "write_slide", "compile_slide", "write_questions",
            "summarize_step", "generate_step",
        ]
        # Context carries the step and 1-based slide position/attempt.
        assert timings[0].context == {
            "step_id": "s1", "slide_index": None, "attempt": None,
        }
        assert timings[1].context == {
            "step_id": "s1", "slide_index": 1, "attempt": 1,
        }
        assert timings[3].context == {
            "step_id": "s1", "slide_index": 2, "attempt": 1,
        }
        # The generate_step row carries the total material time for the step.
        assert timings[-1].context == {
            "step_id": "s1", "slide_index": None, "attempt": None,
        }
        assert timings[-1].duration_seconds > 0
        assert all(t.duration_seconds >= 0 for t in timings)

    # --- preference pairs (#168): the compiler annotated them, the driver stores them ---

    def test_a_fixed_slide_stores_exactly_one_preference_row(
        self, db, db_engine, make_session, fake_session_factory
    ):
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        db.commit()

        with patch("routers.plan.material_graph") as graph:
            graph.invoke.side_effect = _two_slide_results(
                pairs=[dict(_PREFERENCE_PAIR)]
            )
            generate_materials(sid)

        db.expire_all()
        (row,) = db.query(SlidePreference).filter_by(session_id=sid).all()
        assert (row.step_id, row.slide_index) == ("s1", 1)  # 1-based, as failed_slides
        assert (row.rejected_attempt, row.chosen_attempt) == (1, 2)
        assert row.prompt_context == _PREFERENCE_PAIR["prompt_context"]
        assert row.chosen_prompt == _PREFERENCE_PAIR["chosen_prompt"]
        assert row.rejected_jsx == "<div>broken</div>"
        assert row.rejected_error == "syntax error"
        assert row.chosen_jsx == "<div>fixed</div>"
        assert row.created_at is not None

        # Capturing a pair changes nothing the learner gets: the step is still
        # complete, the slides are still there, the phase still advances.
        step_row = db.query(StepMaterial).filter_by(
            session_id=sid, step_id="s1").one()
        assert step_row.is_complete is True
        assert step_row.slides == ["a_s1_slide_1", "a_s1_slide_2"]
        assert db.get(Session, sid).phase == Phase.EXECUTING.value

    def test_a_pair_is_stored_at_its_compile_not_at_the_end_of_the_step(
        self, db, db_engine, make_session, fake_session_factory
    ):
        """The pair belongs to the compile that produced it: a step that dies
        after that slide has still left its training data behind."""
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        db.commit()

        paused = {"__interrupt__": object(), "slides": ["<div>1</div>"],
                  "slide_preferences": [dict(_PREFERENCE_PAIR)]}
        with patch("routers.plan.material_graph") as graph:
            # The step gets one slide, then the graph run dies.
            graph.invoke.side_effect = [paused, RuntimeError("generation died")]
            generate_materials(sid)

        db.expire_all()
        (row,) = db.query(SlidePreference).filter_by(session_id=sid).all()
        assert (row.step_id, row.slide_index) == ("s1", 1)
        assert row.chosen_jsx == "<div>fixed</div>"
        # The step itself did not survive: session lands in error, as before.
        assert db.get(Session, sid).phase == Phase.ERROR.value
        assert db.query(StepMaterial).filter_by(
            session_id=sid, step_id="s1").one().is_complete is False

    def test_the_driver_stores_no_row_when_the_graph_leaves_no_pair(
        self, db, db_engine, make_session, fake_session_factory
    ):
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        db.commit()

        with patch("routers.plan.material_graph") as graph:
            # A first-attempt success leaves no pair in the graph state (see
            # tests/graphs/material), so the driver has nothing to store.
            graph.invoke.side_effect = _two_slide_results()
            generate_materials(sid)

        db.expire_all()
        assert db.query(SlidePreference).filter_by(session_id=sid).count() == 0

    def test_a_failed_preference_write_does_not_cost_the_step(
        self, db, db_engine, make_session, fake_session_factory
    ):
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        db.commit()

        with (
            patch("routers.plan.material_graph") as graph,
            patch("routers.plan.SlidePreference") as pref,
        ):
            graph.invoke.side_effect = _two_slide_results(
                pairs=[dict(_PREFERENCE_PAIR)]
            )
            pref.side_effect = Exception("insert failed")
            generate_materials(sid)

        db.expire_all()
        # Offline training data failed; the generated step did not.
        assert db.query(SlidePreference).filter_by(session_id=sid).count() == 0
        step_row = db.query(StepMaterial).filter_by(
            session_id=sid, step_id="s1").one()
        assert step_row.is_complete is True
        assert db.get(SlideContent, "a_s1_slide_1").content == "<div>1</div>"
        assert db.get(Session, sid).phase == Phase.EXECUTING.value

    def test_placeholder_slide_persists_is_placeholder_flag(
        self, db, db_engine, make_session, fake_session_factory
    ):
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        db.commit()

        # The first paused slide is a placeholder; the second is a real slide.
        question = {"id": "q1", "text": "?", "options": ["A", "B"],
                    "correct_index": 0, "explanation": "e"}
        sequence = [
            {"__interrupt__": object(), "slides": ["<div>ph</div>"],
             "current_slide_is_placeholder": True},
            {"__interrupt__": object(), "slides": ["<div>ph</div>", "<div>2</div>"],
             "current_slide_is_placeholder": False},
            {"slides": ["<div>ph</div>", "<div>2</div>"],
             "questions": [question],
             "summary": {"step_id": "s1", "title": "T1", "key_points": ["k"]},
             "failed_attempts": [],
             "stage_timings": []},
        ]
        with patch("routers.plan.material_graph") as graph:
            graph.invoke.side_effect = sequence
            generate_materials(sid)

        db.expire_all()
        s1 = db.get(SlideContent, "a_s1_slide_1")
        s2 = db.get(SlideContent, "a_s1_slide_2")
        assert s1.is_placeholder is True
        assert s2.is_placeholder is False

    def test_skips_already_complete_step(self, db, db_engine, make_session,
                                         fake_session_factory):
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        db.add(StepMaterial(
            session_id=sid, step_id="s1", slides=[], questions=[],
            summary={"step_id": "s1", "title": "T1", "key_points": ["k"]},
            is_complete=True,
        ))
        db.commit()

        with patch("routers.plan.material_graph") as graph:
            generate_materials(sid)

        # A complete step is skipped — no graph run.
        assert graph.invoke.call_count == 0
        db.expire_all()
        assert db.get(Session, sid).phase == Phase.EXECUTING.value

    def test_re_runs_provisional_step_and_cleans_orphans(
        self, db, db_engine, make_session, fake_session_factory
    ):
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        # A provisional (incomplete) row plus orphan rows from a crashed run.
        db.add(StepMaterial(
            session_id=sid, step_id="s1", slides=["a_s1_slide_1"],
            questions=[],
            summary={"step_id": "s1", "title": "T1", "key_points": []},
            is_complete=False,
        ))
        db.add(SlideContent(slide_id="a_s1_slide_99", session_id=sid,
                            step_id="s1", content="orphan"))
        db.add(FailedSlide(session_id=sid, step_id="s1", slide_index=5,
                           prompt="p", jsx="j", error="e"))
        db.commit()

        with patch("routers.plan.material_graph") as graph:
            graph.invoke.side_effect = _two_slide_results()
            generate_materials(sid)

        db.expire_all()
        # Orphan slide row is gone; exactly the two freshly-compiled slides remain.
        assert db.get(SlideContent, "a_s1_slide_99") is None
        assert db.query(SlideContent).filter_by(session_id=sid,
                                                step_id="s1").count() == 2
        step_row = db.query(StepMaterial).filter_by(
            session_id=sid, step_id="s1").one()
        assert step_row.is_complete is True
        assert db.get(Session, sid).phase == Phase.EXECUTING.value

    def test_partial_timings_survive_mid_step_failure(
        self, db, db_engine, make_session, fake_session_factory
    ):
        """Stage timings committed at each slide interrupt survive even when
        the step later fails — the failure is recorded as the error phase."""
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        db.commit()

        first_state = {
            "__interrupt__": object(), "slides": ["<div>1</div>"],
            "stage_timings": _two_slide_results()[0]["stage_timings"],
        }
        with patch("routers.plan.material_graph") as graph:
            graph.invoke.side_effect = [first_state,
                                        Exception("sandbox down")]
            generate_materials(sid)

        db.expire_all()
        session = db.get(Session, sid)
        assert session.phase == Phase.ERROR.value
        # The 3 stage timings committed at the first interrupt are kept.
        timings = (
            db.query(GraphStageTiming)
            .filter_by(session_id=sid)
            .order_by(GraphStageTiming.id)
            .all()
        )
        assert len(timings) == 3
        assert [t.stage for t in timings] == [
            "plan_slide_contents", "write_slide", "compile_slide",
        ]

    def test_rerun_appends_timing_history(
        self, db, db_engine, make_session, fake_session_factory
    ):
        """A re-run of a step keeps the prior attempt's timing rows and
        appends a fresh set (append-only history, like failed_slides)."""
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        # A provisional (incomplete) row from a crashed run plus one timing
        # row from that prior attempt.
        db.add(StepMaterial(
            session_id=sid, step_id="s1", slides=[], questions=[],
            summary={"step_id": "s1", "title": "T1", "key_points": []},
            is_complete=False,
        ))
        db.add(GraphStageTiming(
            session_id=sid, graph="material",
            stage="plan_slide_contents",
            context={"step_id": "s1", "slide_index": None, "attempt": None},
            duration_seconds=0.9,
        ))
        db.commit()

        with patch("routers.plan.material_graph") as graph:
            graph.invoke.side_effect = _two_slide_results()
            generate_materials(sid)

        db.expire_all()
        timings = (
            db.query(GraphStageTiming)
            .filter_by(session_id=sid)
            .order_by(GraphStageTiming.id)
            .all()
        )
        # The old row (0.9s) survives; 8 fresh rows are appended after it
        # (7 stage rows + the generate_step total).
        assert len(timings) == 9
        assert timings[0].stage == "plan_slide_contents"
        assert timings[0].duration_seconds == 0.9
        assert [t.stage for t in timings[1:]] == [
            "plan_slide_contents", "write_slide", "compile_slide",
            "write_slide", "compile_slide", "write_questions",
            "summarize_step", "generate_step",
        ]

    def test_failure_sets_error_phase(self, db, db_engine, make_session,
                                      fake_session_factory):
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        db.commit()

        with patch("routers.plan.material_graph") as graph:
            graph.invoke.side_effect = Exception("sandbox down")
            generate_materials(sid)

        db.expire_all()
        session = db.get(Session, sid)
        assert session.phase == Phase.ERROR.value

    def test_error_recording_failure_is_swallowed(self, db, db_engine, make_session):
        """If persisting the error state also fails, the inner best-effort
        except swallows it and the driver returns without raising."""
        sid = make_session(
            session_id="a", phase=Phase.GENERATING, boundary_map={"a": {}}
        ).session_id
        db.add(Plan(session_id=sid, version=1, prose_summary="ps",
                    dependency_dag="dag",
                    steps=[{"id": "s1", "title": "T1", "description": "d",
                            "depends_on": [], "depth": 0}],
                    adjustments=[]))
        db.commit()

        # A session whose commit always fails: the first commit in the loop
        # raises, and the recovery commit inside the error handler does too.
        TestingSession = sessionmaker(bind=db_engine)
        boom_session = TestingSession()

        def _boom():
            raise RuntimeError("db down")

        boom_session.commit = _boom
        with patch("routers.plan.SessionFactory", return_value=boom_session):
            generate_materials(sid)  # must not raise

        db.expire_all()
        # Recovery commit failed, so the phase was never persisted as ERROR.
        assert db.get(Session, sid).phase == Phase.GENERATING.value
