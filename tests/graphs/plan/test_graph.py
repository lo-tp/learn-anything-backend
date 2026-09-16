"""Tests for graphs/plan/graph.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from graphs.plan.graph import build_plan_graph
from graphs.plan.state import PlanState


def _make_llm() -> MagicMock:
    return MagicMock()


# --- Graph construction ---


class TestBuildPlanGraph:
    def test_returns_compiled_graph(self):
        from langgraph.graph.state import CompiledStateGraph

        graph = build_plan_graph(_make_llm())
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_has_expected_nodes(self):
        graph = build_plan_graph(_make_llm())
        node_names = set(graph.nodes.keys())
        assert "research_topic" in node_names
        assert "design_plan" in node_names
        assert "render_plan" in node_names
        assert "wait_for_response" in node_names

    def test_with_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph.state import CompiledStateGraph

        checkpointer = MemorySaver()
        graph = build_plan_graph(_make_llm(), checkpointer=checkpointer)
        assert isinstance(graph, CompiledStateGraph)

    def test_different_llms_produce_independent_graphs(self):
        llm1 = MagicMock()
        llm2 = MagicMock()
        g1 = build_plan_graph(llm1)
        g2 = build_plan_graph(llm2)
        assert g1 is not g2


# --- State definition ---


class TestPlanState:
    def test_all_fields_optional(self):
        state: PlanState = {}
        assert state.get("goal") is None
        assert state.get("research") is None
        assert state.get("current_plan") is None
        assert state.get("approved") is None

    def test_full_state(self):
        state: PlanState = {
            "goal": "Learn calculus",
            "language": "English",
            "boundary_map": {"algebra": {"floor": None, "ceiling": None, "gap_type": "unknown"}},
            "research": {"unconditional_truths": ["U"], "core_concepts": ["C"]},
            "current_plan": {"prose_summary": "S", "steps": []},
            "design_steps": [{"title": "A", "description": "d", "depends_on": [], "depth": 1}],
            "adjustment": None,
            "pass_count": 1,
            "response": {"action": "adjust"},
            "approved": False,
        }
        assert state["pass_count"] == 1
        assert state["approved"] is False
