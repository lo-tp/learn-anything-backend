"""Tests for graphs/probe/graph.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from graphs.probe.graph import build_probe_graph
from graphs.probe.state import ProbeState


def _make_llm() -> MagicMock:
    return MagicMock()


# --- Graph construction ---


class TestBuildProbeGraph:
    def test_returns_compiled_graph(self):
        from langgraph.graph.state import CompiledStateGraph

        graph = build_probe_graph(_make_llm())
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_has_expected_nodes(self):
        graph = build_probe_graph(_make_llm())
        node_names = set(graph.nodes.keys())
        assert "decompose_strands" in node_names
        assert "generate_batch" in node_names
        assert "wait_for_answers" in node_names
        assert "evaluate_batch" in node_names
        assert "decide_next" in node_names

    def test_with_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph.state import CompiledStateGraph

        checkpointer = MemorySaver()
        graph = build_probe_graph(_make_llm(), checkpointer=checkpointer)
        assert isinstance(graph, CompiledStateGraph)

    def test_different_llms_produce_independent_graphs(self):
        llm1 = MagicMock()
        llm2 = MagicMock()
        g1 = build_probe_graph(llm1)
        g2 = build_probe_graph(llm2)
        assert g1 is not g2


# --- State definition ---


class TestProbeState:
    def test_all_fields_optional(self):
        state: ProbeState = {}
        assert state.get("goal") is None
        assert state.get("strands") is None
        assert state.get("boundary_map") is None

    def test_full_state(self):
        state: ProbeState = {
            "goal": "Learn calculus",
            "language": "English",
            "strands": ["algebra", "geometry"],
            "history": [],
            "boundary_map": {
                "algebra": {"floor": None, "ceiling": None, "gap_type": "unknown"}
            },
            "question_count": 0,
            "strand_descriptions": {"algebra": "Basic algebra."},
            "next_batch": None,
            "_resume": None,
            "_decision": None,
        }
        assert state["goal"] == "Learn calculus"
        assert state["question_count"] == 0
