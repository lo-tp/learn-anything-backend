"""Tests for graphs/clarify/graph.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from graphs.clarify.graph import build_clarify_graph
from graphs.clarify.state import ClarifyState


def _make_llm() -> MagicMock:
    return MagicMock()


# --- Graph construction ---


class TestBuildClarifyGraph:
    def test_returns_compiled_graph(self):
        from langgraph.graph.state import CompiledStateGraph

        graph = build_clarify_graph(_make_llm())
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_has_expected_nodes(self):
        graph = build_clarify_graph(_make_llm())
        node_names = set(graph.nodes.keys())
        assert "assess_goal" in node_names
        assert "generate_questions" in node_names
        assert "wait_for_answer" in node_names
        assert "refine_goal" in node_names

    def test_with_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph.state import CompiledStateGraph

        checkpointer = MemorySaver()
        graph = build_clarify_graph(_make_llm(), checkpointer=checkpointer)
        assert isinstance(graph, CompiledStateGraph)

    def test_different_llms_produce_independent_graphs(self):
        llm1 = MagicMock()
        llm2 = MagicMock()
        g1 = build_clarify_graph(llm1)
        g2 = build_clarify_graph(llm2)
        assert g1 is not g2


# --- State definition ---


class TestClarifyState:
    def test_all_fields_optional(self):
        state: ClarifyState = {}
        assert state.get("goal") is None
        assert state.get("narrowed_goal") is None
        assert state.get("round_count") is None

    def test_full_state(self):
        state: ClarifyState = {
            "goal": "Learn physics",
            "language": "English",
            "working_goal": "Learn mechanics",
            "open_dimensions": ["depth"],
            "round_count": 1,
            "clarifying_questions": ["What level?"],
            "last_answer": "High school",
            "narrowed_goal": None,
        }
        assert state["goal"] == "Learn physics"
        assert state["round_count"] == 1
        assert state["narrowed_goal"] is None
