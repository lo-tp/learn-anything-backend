"""Tests for graphs/material/graph.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from graphs.material.graph import build_material_graph
from graphs.material.schemas import (
    QuestionDraft,
    QuestionsOut,
    SlideContentsOut,
    SlideContentSpec,
    SlideOut,
    SummaryOut,
)
from graphs.material.state import MaterialState


def _make_llm() -> MagicMock:
    return MagicMock()


# --- Routing functions (tested via graph construction) ---


class TestRouting:
    """Test route_after_compile and route_after_pause logic.

    These are closures inside build_material_graph, so we test them by
    inspecting the compiled graph's behavior or by calling build_material_graph
    and checking the conditional edge mappings.
    """

    def test_graph_has_expected_nodes(self):
        graph = build_material_graph(_make_llm())
        # LangGraph stores nodes in .nodes dict
        node_names = set(graph.nodes.keys())
        assert "plan_slide_contents" in node_names
        assert "write_slide" in node_names
        assert "compile_slide" in node_names
        assert "pause_after_compile" in node_names
        assert "write_questions" in node_names
        assert "summarize_step" in node_names

    def test_graph_has_expected_edges(self):
        graph = build_material_graph(_make_llm())
        # Check that the graph is a valid compiled state graph
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "get_state_history")


# --- Graph construction ---


class TestBuildMaterialGraph:
    def test_returns_compiled_graph(self):
        from langgraph.graph.state import CompiledStateGraph

        graph = build_material_graph(_make_llm())
        assert isinstance(graph, CompiledStateGraph)

    def test_with_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph.state import CompiledStateGraph

        checkpointer = MemorySaver()
        graph = build_material_graph(_make_llm(), checkpointer=checkpointer)
        assert isinstance(graph, CompiledStateGraph)

    def test_different_llms_produce_independent_graphs(self):
        llm1 = MagicMock()
        llm2 = MagicMock()
        g1 = build_material_graph(llm1)
        g2 = build_material_graph(llm2)
        assert g1 is not g2


# --- State definition ---


class TestMaterialState:
    def test_all_fields_optional(self):
        """MaterialState is total=False, so any subset of keys is valid."""
        state: MaterialState = {}
        assert state.get("step") is None
        assert state.get("slides") is None

    def test_full_state(self):
        state: MaterialState = {
            "step": {"id": "s1"},
            "established_concepts": [],
            "learner_context": {},
            "language": "English",
            "slide_contents": [{"title": "A", "key_points": [], "visual_hint": ""}],
            "slides": ["code"],
            "slide_index": 1,
            "attempts_by_slide": [1],
            "current_slide_jsx": "jsx",
            "current_slide_prompt": "[]",
            "last_compile_error": None,
            "failed_attempts": [],
            "stage_timings": [],
            "compile_result": "success",
            "questions": [],
            "summary": {},
        }
        assert state["compile_result"] == "success"
        assert state["slide_index"] == 1


# --- End-to-end preference-pair capture (#168) ---


class TestPreferencePairAccumulation:
    def test_only_the_failed_then_fixed_slide_produces_a_pair(self):
        """Of a 3-slide step, only slide 1 needed a retry: the finished state
        carries exactly one pair for it, and none for the others."""
        from langgraph.checkpoint.memory import MemorySaver

        llm = _make_llm()
        graph = build_material_graph(llm, checkpointer=MemorySaver())

        with (
            patch(
                "graphs.material.nodes.structured_invoke",
                side_effect=[
                    SlideContentsOut(
                        slide_contents=[
                            SlideContentSpec(
                                title=t, key_points=["k"], visual_hint="v"
                            )
                            for t in ("A", "B", "C")
                        ]
                    ),
                    QuestionsOut(
                        questions=[
                            QuestionDraft(
                                text=f"Q{i}?",
                                options=["A", "B"],
                                correct_index=0,
                                explanation="e",
                            )
                            for i in range(3)
                        ]
                    ),
                    SummaryOut(key_points=["kp1"]),
                ],
            ),
            patch(
                "graphs.material.nodes.structured_invoke_messages",
                side_effect=[
                    SlideOut(code=jsx)
                    for jsx in (
                        "slide1_v1",  # rejected by the compiler
                        "slide1_v2",  # the fix
                        "slide2_v1",
                        "slide3_v1",
                    )
                ],
            ),
            patch(
                "graphs.material.nodes.with_unknown_option",
                side_effect=lambda llm, lang, opts: opts,
            ),
            patch(
                "graphs.material.nodes._compile_slide",
                side_effect=[
                    (None, "syntax error"),
                    ("compiled_1", ""),
                    ("compiled_2", ""),
                    ("compiled_3", ""),
                ],
            ),
        ):
            config: RunnableConfig = {
                "configurable": {"thread_id": "preference-e2e"},
            }
            state = graph.invoke(
                {
                    "step": {"id": "s1", "title": "T1", "description": "d"},
                    "established_concepts": [],
                    "learner_context": {},
                    "language": "English",
                },
                config,
            )
            while state.get("__interrupt__"):
                state = graph.invoke(Command(resume=True), config)

        assert state["compile_result"] == "success"
        (pair,) = state["slide_preferences"]
        assert pair["index"] == 0
        assert pair["rejected_attempt"] == 1
        assert pair["chosen_attempt"] == 2
        assert pair["rejected_jsx"] == "slide1_v1"
        assert pair["chosen_jsx"] == "slide1_v2"
        assert pair["rejected_error"] == "syntax error"
        # Both halves keep the prompt they were really written from: today they
        # differ by the one retry instruction, which is exactly what the pair
        # is honest about rather than claiming away.
        assert json.loads(pair["prompt_context"]) == json.loads(
            state["failed_attempts"][0]["prompt"]
        )
        assert "failed to compile" not in pair["prompt_context"]
        assert "failed to compile" in pair["chosen_prompt"]
        assert json.loads(pair["chosen_prompt"])[0]["role"] == "system"
        # The slide that failed carries the pairing, for the debug record too.
        assert state["failed_attempts"][0]["attempt"] == 1


# --- End-to-end stage timing accumulation ---


class TestStageTimingAccumulation:
    def test_full_run_accumulates_one_timing_per_stage(self):
        """A 3-slide step run (schema requires >= 3 slides) produces 9
        stage_timings entries in execution order: plan, then
        write/compile per slide, then questions, summary."""
        from langgraph.checkpoint.memory import MemorySaver

        llm = _make_llm()
        graph = build_material_graph(llm, checkpointer=MemorySaver())

        with (
            patch(
                "graphs.material.nodes.structured_invoke",
                side_effect=[
                    SlideContentsOut(
                        slide_contents=[
                            SlideContentSpec(
                                title=t, key_points=["k"], visual_hint="v"
                            )
                            for t in ("A", "B", "C")
                        ]
                    ),
                    QuestionsOut(
                        questions=[
                            QuestionDraft(
                                text=f"Q{i}?",
                                options=["A", "B"],
                                correct_index=0,
                                explanation="e",
                            )
                            for i in range(3)
                        ]
                    ),
                    SummaryOut(key_points=["kp1"]),
                ],
            ),
            patch(
                "graphs.material.nodes.structured_invoke_messages",
                side_effect=[
                    SlideOut(code=f"export default function {t}() {{}}")
                    for t in ("A", "B", "C")
                ],
            ),
            patch(
                "graphs.material.nodes.with_unknown_option",
                side_effect=lambda llm, lang, opts: opts,
            ),
            patch("graphs.material.nodes._compile_slide", return_value=("code", "")),
        ):
            config: RunnableConfig = {
                "configurable": {"thread_id": "timing-e2e"},
            }
            state = graph.invoke(
                {
                    "step": {"id": "s1", "title": "T1", "description": "d"},
                    "established_concepts": [],
                    "learner_context": {},
                    "language": "English",
                },
                config,
            )
            while state.get("__interrupt__"):
                state = graph.invoke(Command(resume=True), config)

        timings = state["stage_timings"]
        assert [t["stage"] for t in timings] == [
            "plan_slide_contents",
            "write_slide",
            "compile_slide",
            "write_slide",
            "compile_slide",
            "write_slide",
            "compile_slide",
            "write_questions",
            "summarize_step",
        ]
        assert all(t["duration_seconds"] >= 0 for t in timings)
        # Context carries the step, 1-based slide position, and attempt.
        assert timings[0]["context"] == {
            "step_id": "s1", "slide_index": None, "attempt": None,
        }
        assert timings[1]["context"] == {
            "step_id": "s1", "slide_index": 1, "attempt": 1,
        }
        assert timings[5]["context"] == {
            "step_id": "s1", "slide_index": 3, "attempt": 1,
        }
