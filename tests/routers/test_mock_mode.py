"""Tests for MOCK_LLM mode end-to-end behavior (#118).

Acceptance criteria:
- Approving the plan in mock mode does not advance the phase to
  GENERATING and does not schedule generate_materials; it returns a clear
  message that slide generation is disabled.
- generate_materials is a defensive no-op in mock mode.
- dev_regenerate is refused in mock mode (it's dev-only).
- regenerating a single slide is refused too (#197): mock mode has no slides.
- The Probing step of the new-session form is answered with a full batch of
  questions, the way the real flow answers it.
- The Clarify step is walked too: two rounds, five clarifying questions, then
  the Session moves on to probing.
"""

from __future__ import annotations

from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver

from core.mock_llm import MockChatModel
from db import GraphStageTiming, Phase, Plan, ProbeQuestion, Session, SlideContent
from graphs.clarify import build_clarify_graph
from graphs.probe import build_probe_graph
from routers.plan import generate_materials, regenerate_slide_material


def _mock_probe_graph():
    """The probe graph mock mode builds at startup, with its own checkpointer.

    Same construction as ``graphs/__init__.py`` makes when ``MOCK_LLM`` is on;
    the app's module-level graph cannot be reused here because each test needs
    its own thread state.
    """
    return build_probe_graph(MockChatModel(), checkpointer=MemorySaver())


def _mock_clarify_graph():
    """The clarify graph mock mode builds at startup, with its own checkpointer."""
    return build_clarify_graph(MockChatModel(), checkpointer=MemorySaver())


def _signed_in(auth_cookie: str) -> dict[str, str]:
    """A sign-in cookie: creating a Session requires one from every caller (#145),
    mock mode included — the RAM DB is seeded with a known account for this."""
    return {"access_token": auth_cookie}


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


class TestMockClarifyScript:
    """A new Session's first three turns, walked through the clarify router on
    the mock graph.

    This is the first screen of the new-session form. The mock used to answer
    "specific" on the first call, so a mock run skipped the Clarify step
    entirely and the form's question UI had nothing to walk; now it gets the
    same shape a real run gets — two rounds, 3 questions then 2, then a
    narrowed goal that moves the Session on to probing.
    """

    def test_the_intake_asks_five_questions_over_two_rounds(
        self, client, db, auth_cookie
    ):
        graph = _mock_clarify_graph()
        asked: list[str] = []
        with (
            patch("routers.clarify.clarify_graph", graph),
            patch("routers.clarify.detect_language", return_value="English"),
        ):
            first = client.post(
                "/sessions",
                json={"goal": "Learn calculus"},
                cookies=_signed_in(auth_cookie),
            )
            assert first.status_code == 200
            body = first.json()
            sid = body["session_id"]
            assert body["phase"] == "clarifying"
            assert "narrowed_goal" not in body
            assert len(body["clarifying_questions"]) == 3
            asked += body["clarifying_questions"]

            second = client.post(
                f"/sessions/{sid}/clarify",
                json={"answer": "Mock answer 1"},
                cookies=_signed_in(auth_cookie),
            )
            assert second.status_code == 200
            body = second.json()
            assert body["phase"] == "clarifying"
            assert len(body["clarifying_questions"]) == 2
            asked += body["clarifying_questions"]

            third = client.post(
                f"/sessions/{sid}/clarify",
                json={"answer": "Mock answer 2"},
                cookies=_signed_in(auth_cookie),
            )
            assert third.status_code == 200
            body = third.json()
            assert body["phase"] == "probing"
            assert body["narrowed_goal"]
            assert "clarifying_questions" not in body

        assert len(asked) == 5
        assert len(set(asked)) == 5  # five questions, not one round repeated

        db.expire_all()
        session = db.get(Session, sid)
        assert session is not None
        assert session.phase == Phase.PROBING.value
        assert session.narrowed_goal == body["narrowed_goal"]

    def test_an_answer_to_a_finished_clarify_is_refused(self, client, auth_cookie):
        """The phase has moved on; a late answer is a 409, not another round."""
        graph = _mock_clarify_graph()
        with (
            patch("routers.clarify.clarify_graph", graph),
            patch("routers.clarify.detect_language", return_value="English"),
        ):
            sid = client.post(
                "/sessions",
                json={"goal": "Learn calculus"},
                cookies=_signed_in(auth_cookie),
            ).json()["session_id"]
            client.post(
                f"/sessions/{sid}/clarify",
                json={"answer": "Mock answer 1"},
                cookies=_signed_in(auth_cookie),
            )
            client.post(
                f"/sessions/{sid}/clarify",
                json={"answer": "Mock answer 2"},
                cookies=_signed_in(auth_cookie),
            )
            late = client.post(
                f"/sessions/{sid}/clarify",
                json={"answer": "Mock answer 3"},
                cookies=_signed_in(auth_cookie),
            )

        assert late.status_code == 409
        assert "not 'clarifying'" in late.json()["detail"]


class TestMockProbeBatch:
    """A new Session walked through the probe router on the mock graph.

    Mock mode exists so the new-session form can be walked without a model, so
    what it hands that form has to be the shape the form is built for: one
    batch of ``PROBE_BATCH_SIZE`` questions, answered in one submission, that
    ends the phase. A single canned question was a different flow — one card,
    a one-answer payload, an evaluate step that never saw a batch.
    """

    def test_first_call_returns_a_batch_of_questions(self, client, db, make_session):
        sid = make_session(
            session_id="a", phase=Phase.PROBING, narrowed_goal="ng"
        ).session_id
        with patch("routers.probe.probe_graph", _mock_probe_graph()):
            resp = client.post(f"/sessions/{sid}/probe", json={})

        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "probing"
        questions = body["questions"]
        assert len(questions) == 3
        # Distinct questions across distinct strands: a batch spans the
        # unbracketed strands, it does not ask the same thing three times.
        assert len({q["id"] for q in questions}) == 3
        assert len({q["strand"] for q in questions}) == 3
        for q in questions:
            assert q["text"]
            assert len(q["options"]) == 5  # 4 options + "I don't know"
            assert 0 <= q["correct_index"] < 4
            assert 1 <= q["difficulty"] <= 5

        db.expire_all()
        rows = db.query(ProbeQuestion).filter_by(session_id=sid).all()
        assert len(rows) == 3
        assert {str(r.question_id) for r in rows} == {q["id"] for q in questions}

    def test_answering_the_batch_ends_probing(self, client, db, make_session):
        sid = make_session(
            session_id="a", phase=Phase.PROBING, narrowed_goal="ng"
        ).session_id
        graph = _mock_probe_graph()
        with patch("routers.probe.probe_graph", graph):
            first = client.post(f"/sessions/{sid}/probe", json={}).json()
            resp = client.post(
                f"/sessions/{sid}/probe",
                json={
                    "answers": [
                        {"question_id": q["id"], "selected_index": 0}
                        for q in first["questions"]
                    ]
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "planning"
        assert body.get("questions") is None

        db.expire_all()
        session = db.get(Session, sid)
        assert session.phase == Phase.PLANNING.value
        # Every strand the batch asked about is bracketed, so decide_next is
        # done and the Session moves on to Plan without a second round.
        assert len(session.boundary_map) == 3
        assert set(session.boundary_map) == {q["strand"] for q in first["questions"]}
        assert db.query(ProbeQuestion).filter_by(session_id=sid).count() == 3
        # The three stages the two requests ran are recorded, as for any other
        # phase (#194) — a mock run is measurable too.
        stages = [
            t.stage
            for t in db.query(GraphStageTiming)
            .filter_by(session_id=sid, graph="probe")
            .order_by(GraphStageTiming.id)
            .all()
        ]
        assert stages == ["decompose_strands", "generate_batch", "evaluate_batch"]

    def test_a_partial_answer_is_refused_before_the_graph_runs(
        self, client, db, make_session
    ):
        """The batch is the unit: answering part of it is not half a round."""
        sid = make_session(
            session_id="a", phase=Phase.PROBING, narrowed_goal="ng"
        ).session_id
        graph = _mock_probe_graph()
        with patch("routers.probe.probe_graph", graph):
            first = client.post(f"/sessions/{sid}/probe", json={}).json()
            resp = client.post(
                f"/sessions/{sid}/probe",
                json={
                    "answers": [
                        {
                            "question_id": first["questions"][0]["id"],
                            "selected_index": 0,
                        }
                    ]
                },
            )

        assert resp.status_code == 422
        assert "current batch" in resp.json()["detail"]


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
