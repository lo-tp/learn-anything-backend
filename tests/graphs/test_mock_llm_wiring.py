"""Tests for the graph/LLM seam in MOCK_LLM mode (#118).

Acceptance criteria:
- The pre-material graphs (clarify/probe/plan) are wired to MockChatModel
  in mock mode and the real llm otherwise.
- The material graph is always wired to the real llm (mock mode simply
  never invokes it).
- A session can go Clarify -> Probe -> Plan in mock mode with canned
  output (zero real LLM calls).
- The intake shape is the shape a real run produces: Clarify asks two rounds
  (3 clarifying questions, then 2), Probe asks a batch of three.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import graphs
from core.external_tools import external_search_tools
from core.llm import llm
from core.mock_llm import MockChatModel
from graphs import graph_config
from graphs.clarify import ClarifyState, build_clarify_graph
from graphs.plan import build_plan_graph
from graphs.probe import ProbeState, build_probe_graph
from services.stage_timings import empty_stage_timings


class TestWiringSeam:
    """The module-level seam in graphs/__init__.py."""

    def test_real_llm_by_default(self):
        # The main test process imports graphs with MOCK_LLM unset.
        assert graphs.pre_material_llm is llm

    def test_mock_llm_when_flag_set(self, monkeypatch):
        monkeypatch.setenv("MOCK_LLM", "1")
        reloaded = importlib.reload(graphs)
        try:
            assert isinstance(reloaded.pre_material_llm, MockChatModel)
            assert reloaded.pre_material_llm is not llm
        finally:
            monkeypatch.undo()
            importlib.reload(graphs)  # restore the original wiring

    def test_material_graph_always_real_llm(self, monkeypatch):
        """The material graph is built with the real llm even in mock mode —
        mock mode simply never invokes it."""
        import graphs.material as material_mod

        monkeypatch.setenv("MOCK_LLM", "1")
        captured: dict = {}
        real_build = material_mod.build_material_graph

        def spy(bound_llm, *args, **kwargs):
            captured["llm"] = bound_llm
            return real_build(bound_llm, *args, **kwargs)

        monkeypatch.setattr(material_mod, "build_material_graph", spy)
        reloaded = importlib.reload(graphs)
        try:
            assert reloaded.material_graph is not None
            current_real = importlib.import_module("core.llm").llm
            assert captured["llm"] is current_real  # real llm, not the mock
            assert not isinstance(captured["llm"], MockChatModel)
        finally:
            monkeypatch.undo()
            importlib.reload(graphs)  # restore the original wiring


# --- External search wiring (#166) ---


def _spy_plan_graph_build(monkeypatch) -> dict:
    """Capture what graphs/__init__.py hands build_plan_graph, then reload."""
    import graphs.plan as plan_mod

    captured: dict = {}
    real_build = plan_mod.build_plan_graph

    def spy(bound_llm, *args, **kwargs):
        captured["kwargs"] = kwargs
        return real_build(bound_llm, *args, **kwargs)

    monkeypatch.setattr(plan_mod, "build_plan_graph", spy)
    return captured


class TestExternalSearchWiring:
    """The graphs/__init__.py seam: which graph may reach external search."""

    def test_plan_graph_is_wired_to_the_external_search_boundary(self, monkeypatch):
        captured = _spy_plan_graph_build(monkeypatch)
        importlib.reload(graphs)
        try:
            assert captured["kwargs"]["search_tools"] is external_search_tools
        finally:
            monkeypatch.undo()
            importlib.reload(graphs)

    def test_mock_mode_builds_the_plan_graph_with_no_external_search(
        self, monkeypatch
    ):
        """#166: a mock run has no external-tool path to reach at all."""
        captured = _spy_plan_graph_build(monkeypatch)
        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-wired-key")
        with patch("core.external_tools.MultiServerMCPClient") as client_cls:
            importlib.reload(graphs)
            search_tools = captured["kwargs"]["search_tools"]
        try:
            assert search_tools is None
            client_cls.assert_not_called()
        finally:
            monkeypatch.undo()
            importlib.reload(graphs)


def _answer_batch(batch: list[dict]) -> dict:
    """A resume payload shaped exactly as routers/probe.py sends one: the whole
    batch, one answer per question."""
    return {"answers": [{"question_id": q["id"], "selected_index": 0} for q in batch]}


class TestCannedPipeline:
    """Clarify -> Probe -> Plan end-to-end on MockChatModel: zero real LLM."""

    def test_clarify_asks_two_rounds_then_narrows(self):
        """The mock's Clarify script walked through the real graph.

        Three questions, then two, then the narrowed goal: five clarifying
        questions in two learner turns, with the real 1-3-per-round limit never
        exceeded and the round cap never reached.
        """
        graph = build_clarify_graph(MockChatModel(), checkpointer=MemorySaver())
        config = graph_config("t:clarify", "clarify")
        state: ClarifyState = {
            "goal": "Learn calculus",
            "language": "English",
            "working_goal": "Learn calculus",
            "open_dimensions": [],
            "round_count": 0,
            "clarifying_questions": [],
            "last_answer": None,
            "narrowed_goal": None,
        }
        result = graph.invoke(state, config)
        assert "__interrupt__" in result
        first_round = result["clarifying_questions"]
        assert len(first_round) == 3

        result = graph.invoke(Command(resume={"answer": "Mock answer 1"}), config)
        assert "__interrupt__" in result
        second_round = result["clarifying_questions"]
        assert len(second_round) == 2

        result = graph.invoke(Command(resume={"answer": "Mock answer 2"}), config)
        assert "__interrupt__" not in result
        assert result["narrowed_goal"]
        # Ended by the script, not by MAX_CLARIFY_ROUNDS.
        assert result["round_count"] == 2
        asked = first_round + second_round
        assert len(asked) == 5
        assert len(set(asked)) == 5

    def test_probe_brackets_after_one_answered_batch(self):
        graph = build_probe_graph(MockChatModel(), checkpointer=MemorySaver())
        config = graph_config("t:probe", "probe")
        state: ProbeState = {
            "goal": "Learn calculus",
            "language": "English",
            "strands": [],
            "strand_descriptions": {},
            "boundary_map": {},
            "next_batch": [],
            "question_count": 0,
            "history": [],
        }
        result = graph.invoke(state, config)
        assert "__interrupt__" in result
        # A full batch: what the new-session form's Probing step shows.
        assert len(result["next_batch"]) == 3
        assert {q["strand"] for q in result["next_batch"]} == set(result["strands"])
        # Resume with the learner's (mock) answers for the whole batch.
        result = graph.invoke(
            Command(resume=_answer_batch(result["next_batch"])),
            config,
        )
        # evaluate_batch brackets every strand -> done, no second batch.
        assert "__interrupt__" not in result
        assert result["question_count"] == 3
        assert len(result["history"]) == 3
        for strand in result["strands"]:
            assert result["boundary_map"][strand]["floor"] is not None
            assert result["boundary_map"][strand]["ceiling"] is not None

    def test_plan_produces_canned_plan(self):
        graph = build_plan_graph(MockChatModel(), checkpointer=MemorySaver())
        result = graph.invoke(
            {
                "goal": "Learn calculus",
                "language": "English",
                "boundary_map": {},
                "research": None,
                "current_plan": None,
                "adjustment": None,
                "pass_count": 0,
            },
            {"configurable": {"thread_id": "t:plan"}},
        )
        assert "__interrupt__" in result
        plan = result["current_plan"]
        assert plan["prose_summary"]
        assert plan["dependency_dag"]
        assert plan["steps"]


class TestPreMaterialStageTimings:
    """Clarify, Probe and Plan record a timing per stage, the way Material already
    does (#194). The cost of these phases is invisible without it — Material's
    durations were the only measurable ones, which is how #194's fix was justified
    for slides and only for slides.

    The mock is used because these are the same graphs the app runs; the point is
    which stages get an entry, not what the model said.
    """

    @staticmethod
    def _stages(result: dict) -> list[str]:
        return [t["stage"] for t in result["stage_timings"]]

    def test_clarify_records_each_stage_and_not_the_pause(self):
        graph = build_clarify_graph(MockChatModel(), checkpointer=MemorySaver())
        result = graph.invoke(
            {
                "goal": "Learn calculus",
                "language": "English",
                "working_goal": "Learn calculus",
                "open_dimensions": [],
                "round_count": 0,
                "clarifying_questions": [],
                "last_answer": None,
                "narrowed_goal": None,
            },
            {"configurable": {"thread_id": "timings:clarify"}},
        )
        # A request that ends in questions records both model calls it made.
        # The pause itself is the learner's time, not this service's.
        assert self._stages(result) == ["assess_goal", "generate_questions"]
        assert all(t["duration_seconds"] >= 0 for t in result["stage_timings"])

    def test_probe_records_each_stage_and_not_the_pause(self):
        """`wait_for_answers` is the learner's time, not this service's."""
        graph = build_probe_graph(MockChatModel(), checkpointer=MemorySaver())
        config = graph_config("timings:probe", "probe")
        result = graph.invoke(
            {
                "goal": "Learn calculus",
                "language": "English",
                "strands": [],
                "strand_descriptions": {},
                "boundary_map": {},
                "next_batch": [],
                "question_count": 0,
                "history": [],
                "stage_timings": empty_stage_timings(),
            },
            config,
        )
        assert self._stages(result) == ["decompose_strands", "generate_batch"]

        result = graph.invoke(
            Command(resume=_answer_batch(result["next_batch"])),
            config,
        )
        assert self._stages(result) == [
            "decompose_strands",
            "generate_batch",
            "evaluate_batch",
        ]

    def test_the_batch_a_timing_belongs_to_is_recorded(self):
        """Attribution is what makes the rows answer "which batch was slow"."""
        graph = build_probe_graph(MockChatModel(), checkpointer=MemorySaver())
        result = graph.invoke(
            {
                "goal": "Learn calculus",
                "language": "English",
                "strands": [],
                "strand_descriptions": {},
                "boundary_map": {},
                "next_batch": [],
                "question_count": 0,
                "history": [],
            },
            graph_config("timings:probe:ctx", "probe"),
        )
        by_stage = {t["stage"]: t for t in result["stage_timings"]}
        assert by_stage["generate_batch"]["context"] == {"question_count": 0}

    def test_plan_records_each_stage_and_not_the_pause(self):
        graph = build_plan_graph(MockChatModel(), checkpointer=MemorySaver())
        result = graph.invoke(
            {
                "goal": "Learn calculus",
                "language": "English",
                "boundary_map": {},
                "research": None,
                "current_plan": None,
                "design_steps": [],
                "pass_count": 0,
            },
            {"configurable": {"thread_id": "timings:plan"}},
        )
        assert self._stages(result) == ["research_topic", "design_plan", "render_plan"]


class TestEachRequestRecordsOnlyItsOwnStages:
    """The driver contract the routers depend on (#194).

    A checkpoint keeps the timings of the stages already run, so a driver that
    persisted whatever the graph returned would write yesterday's stages again on
    every request. Each request therefore starts with the channel emptied, and
    records exactly what it ran.

    These invokes are the inputs the routers actually build — first request and
    resume alike, with :func:`services.stage_timings.empty_stage_timings` in
    them — so the emptying is tested against a real graph, not against a
    stand-in that could not tell an emptied channel from an ordinary one.
    """

    def test_a_resume_records_only_the_stages_it_ran(self):
        graph = build_probe_graph(MockChatModel(), checkpointer=MemorySaver())
        config = graph_config("clear:probe", "probe")
        result = graph.invoke(
            {
                "goal": "Learn calculus",
                "language": "English",
                "strands": [],
                "strand_descriptions": {},
                "boundary_map": {},
                "next_batch": [],
                "question_count": 0,
                "history": [],
            },
            config,
        )
        assert [t["stage"] for t in result["stage_timings"]] == [
            "decompose_strands",
            "generate_batch",
        ]

        result = graph.invoke(
            Command(
                resume=_answer_batch(result["next_batch"]),
                update={"stage_timings": empty_stage_timings()},
            ),
            config,
        )
        assert [t["stage"] for t in result["stage_timings"]] == ["evaluate_batch"]

    def test_a_clarify_resume_records_only_the_stages_it_ran(self):
        """A clarify resume re-enters the loop rather than continuing a stage.

        Folding the answer in, judging the goal again, and — because the mock's
        script has a second round left — asking that round: exactly what the
        second request cost, and nothing from the first one.
        """
        graph = build_clarify_graph(MockChatModel(), checkpointer=MemorySaver())
        config = graph_config("clear:clarify", "clarify")
        result = graph.invoke(
            {
                "goal": "Learn calculus",
                "language": "English",
                "working_goal": "Learn calculus",
                "open_dimensions": [],
                "round_count": 0,
                "clarifying_questions": [],
                "last_answer": None,
                "narrowed_goal": None,
                "stage_timings": empty_stage_timings(),
            },
            config,
        )
        assert [t["stage"] for t in result["stage_timings"]] == [
            "assess_goal",
            "generate_questions",
        ]

        result = graph.invoke(
            Command(
                resume={"answer": "Mock answer 1"},
                update={"language": "English", "stage_timings": empty_stage_timings()},
            ),
            config,
        )
        assert [t["stage"] for t in result["stage_timings"]] == [
            "refine_goal",
            "assess_goal",
            "generate_questions",
        ]
