"""Tests for graphs/plan/graph.py."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graphs.plan.graph import build_plan_graph
from graphs.plan.schemas import (
    DesignOut,
    RenderedStep,
    RenderOut,
    ResearchOut,
    StepDraft,
)
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


# --- External search scope (#166) ---

_RESEARCH = ResearchOut(
    unconditional_truths=["U"],
    core_concepts=["C"],
    standard_framing="SF",
    common_gotchas=["G"],
)


def _plan_outputs(_llm, schema, _third, _fourth=None, **_bind_kwargs):
    """The un-tooled path: design_plan (message list), then render_plan."""
    if schema is DesignOut:
        return DesignOut(
            steps=[StepDraft(title="A", description="d", depends_on=[], depth=1)]
        )
    return RenderOut(
        prose_summary="S",
        dependency_dag="graph LR\n  s0[A]",
        steps=[
            RenderedStep(
                id="s0", letter="A", title="A", description="d", depends_on=[], depth=1
            )
        ],
    )


def _run_plan(graph) -> dict:
    return graph.invoke(
        {
            "goal": "Learn calculus",
            "language": "English",
            "boundary_map": {},
            "research": None,
            "current_plan": None,
            "adjustment": None,
            "pass_count": 0,
        },
        {"configurable": {"thread_id": "scope-test:plan"}},
    )


class TestSearchToolScope:
    """#166: the external search tool reaches research_topic and nothing else."""

    def test_search_tools_reach_only_research_topic(self):
        tools = [MagicMock(), MagicMock()]
        consulted: list[int] = []

        def loader():
            consulted.append(1)
            return tools

        graph = build_plan_graph(
            _make_llm(), checkpointer=MemorySaver(), search_tools=loader
        )

        grounded: list = []
        with (
            patch(
                "graphs.plan.nodes.structured_invoke_with_tools",
                side_effect=lambda llm, schema, system, human, offered: grounded.append(
                    offered
                )
                or _RESEARCH,
            ),
            patch(
                "graphs.plan.nodes.structured_invoke", side_effect=_plan_outputs
            ) as render_spy,
            patch(
                "graphs.plan.nodes.structured_invoke_messages",
                side_effect=_plan_outputs,
            ) as design_spy,
        ):
            result = _run_plan(graph)

        assert result["current_plan"]["steps"][0]["id"] == "s0"  # the graph ran
        assert grounded == [tools]  # one grounded call: research_topic's
        assert consulted == [1]  # and only it consulted the boundary
        # design_plan and render_plan went through the un-tooled paths.
        assert [c.args[1] for c in design_spy.call_args_list] == [DesignOut]
        assert [c.args[1] for c in render_spy.call_args_list] == [RenderOut]


# --- Preference pair left behind by the adjust loop (#167) ---


def _per_pass_outputs():
    """A side_effect whose plan grows one step per pass (v1, then v2, …)."""
    passes = {DesignOut: 0, RenderOut: 0}

    def outputs(_llm, schema, _third, _fourth=None, **_bind_kwargs):
        passes[schema] += 1
        n = passes[schema]
        if schema is DesignOut:
            steps = [StepDraft(title="A", description="d", depends_on=[], depth=1)]
            if n > 1:
                steps.append(
                    StepDraft(title="B", description="d", depends_on=["A"], depth=2)
                )
            return DesignOut(steps=steps)
        steps = [
            RenderedStep(
                id="s0", letter="A", title="A", description="d", depends_on=[], depth=1
            )
        ]
        if n > 1:
            steps.append(
                RenderedStep(
                    id="s1",
                    letter="B",
                    title="B",
                    description="d",
                    depends_on=["s0"],
                    depth=2,
                )
            )
        return RenderOut(
            prose_summary=f"prose v{n}",
            dependency_dag=f"dag v{n}",
            steps=steps,
        )

    return outputs


def _config(thread: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread}}


def _initial_state() -> PlanState:
    """The first-pass input the plan endpoint sends."""
    return {
        "goal": "Learn calculus",
        "language": "English",
        "boundary_map": {},
        "research": None,
        "current_plan": None,
        "adjustment": None,
        "pass_count": 0,
    }


@contextmanager
def _plan_llm_outputs() -> Iterator[None]:
    """Run a plan graph with no network: research is canned, and every pass
    renders a plan one step bigger than the last (v1, v2, …)."""
    outputs = _per_pass_outputs()
    with (
        patch(
            "graphs.plan.nodes.structured_invoke_with_tools", return_value=_RESEARCH
        ),
        patch("graphs.plan.nodes.structured_invoke", side_effect=outputs),
        patch("graphs.plan.nodes.structured_invoke_messages", side_effect=outputs),
    ):
        yield


class TestPreferencePairCapture:
    """An adjusted plan leaves its rejected version in the state, so the
    approve handler can record one preference pair (#167)."""

    def test_adjust_then_approve_keeps_both_versions_and_the_instruction(self):
        graph = build_plan_graph(_make_llm(), checkpointer=MemorySaver())
        config = _config("pair-test:plan")

        with _plan_llm_outputs():
            graph.invoke(_initial_state(), config)
            graph.invoke(
                Command(resume={"action": "adjust", "text": "Add a step on limits"}),
                config,
            )
            result = graph.invoke(Command(resume={"action": "approve"}), config)

        assert result["approved"] is True
        # The version the learner chose, and the one they rejected for it.
        assert [s["id"] for s in result["current_plan"]["steps"]] == ["s0", "s1"]
        assert result["prior_plan"]["prose_summary"] == "prose v1"
        assert [s["id"] for s in result["prior_plan"]["steps"]] == ["s0"]
        # The instruction the chosen version was conditioned on.
        assert result["adjustment"] == "Add a step on limits"
        assert "Add a step on limits" in result["prompt_context"][1]["content"]

    def test_approving_without_an_adjustment_leaves_nothing_to_pair(self):
        graph = build_plan_graph(_make_llm(), checkpointer=MemorySaver())
        config = _config("unadjusted-test:plan")

        with _plan_llm_outputs():
            graph.invoke(_initial_state(), config)
            result = graph.invoke(Command(resume={"action": "approve"}), config)

        assert result["approved"] is True
        assert "prior_plan" not in result
