"""Tests for the graph/LLM seam in MOCK_LLM mode (#118).

Acceptance criteria:
- The pre-material graphs (clarify/probe/plan) are wired to MockChatModel
  in mock mode and the real llm otherwise.
- The material graph is always wired to the real llm (mock mode simply
  never invokes it).
- A session can go Clarify -> Probe -> Plan in mock mode with canned
  output (zero real LLM calls).
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
from graphs.clarify import build_clarify_graph
from graphs.plan import build_plan_graph
from graphs.probe import ProbeState, build_probe_graph


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


class TestCannedPipeline:
    """Clarify -> Probe -> Plan end-to-end on MockChatModel: zero real LLM."""

    def test_clarify_returns_narrowed_goal_immediately(self):
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
            {"configurable": {"thread_id": "t:clarify"}},
        )
        # One canned narrowed goal, zero questions (no __interrupt__).
        assert "__interrupt__" not in result
        assert result["narrowed_goal"]

    def test_probe_brackets_after_one_answer(self):
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
        # One canned MCQ.
        assert len(result["next_batch"]) == 1
        question = result["next_batch"][0]
        # Resume with the learner's (mock) answer.
        result = graph.invoke(
            Command(
                resume={
                    "answers": [{"question_id": question["id"], "selected_index": 0}]
                }
            ),
            config,
        )
        # evaluate_batch brackets the strand -> done, no second question.
        assert "__interrupt__" not in result
        strand = result["strands"][0]
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
