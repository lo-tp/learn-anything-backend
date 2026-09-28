"""Tests for graphs/plan/nodes.py."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from graphs.plan.nodes import (
    make_design_plan,
    make_render_plan,
    make_research_topic,
    make_wait_for_response,
)
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


# --- make_research_topic ---


class TestResearchTopic:
    def test_returns_research_dict(self):
        llm = _make_llm()
        mock_out = ResearchOut(
            unconditional_truths=["U1"],
            core_concepts=["C1", "C2"],
            standard_framing="SF",
            common_gotchas=["G1"],
        )
        with patch(
            "graphs.plan.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_research_topic(llm)
            result = node({"goal": "Learn calculus"})

        assert result["research"]["unconditional_truths"] == ["U1"]
        assert result["research"]["core_concepts"] == ["C1", "C2"]
        assert result["research"]["standard_framing"] == "SF"
        assert result["research"]["common_gotchas"] == ["G1"]

    def test_includes_goal_in_prompt(self):
        llm = _make_llm()
        mock_out = ResearchOut(
            unconditional_truths=[], core_concepts=[], standard_framing="", common_gotchas=[]
        )
        with patch(
            "graphs.plan.nodes.structured_invoke", return_value=mock_out
        ) as mock_invoke:
            node = make_research_topic(llm)
            node({"goal": "Quantum mechanics"})
        assert "Quantum mechanics" in str(mock_invoke.call_args)


# --- make_design_plan ---


class TestDesignPlan:
    def _state(self, **overrides) -> PlanState:
        base: PlanState = {
            "goal": "Learn calculus",
            "boundary_map": {"algebra": {"floor": "x", "ceiling": None, "gap_type": "narrow"}},
            "research": {
                "unconditional_truths": ["U"],
                "core_concepts": ["C"],
                "standard_framing": "SF",
                "common_gotchas": ["G"],
            },
            "language": "English",
        }
        base.update(overrides)  # type: ignore[arg-type]
        return base

    def _mock_design_out(self) -> DesignOut:
        return DesignOut(
            steps=[
                StepDraft(title="A", description="d", depth=1),
                StepDraft(title="B", description="d", depends_on=["A"], depth=2),
            ]
        )

    def test_initial_pass_returns_design_steps(self):
        llm = _make_llm()
        with patch(
            "graphs.plan.nodes.structured_invoke", return_value=self._mock_design_out()
        ):
            node = make_design_plan(llm)
            result = node(self._state())

        assert len(result["design_steps"]) == 2
        assert result["design_steps"][0]["title"] == "A"
        assert result["design_steps"][1]["depends_on"] == ["A"]

    @pytest.mark.real_prompts
    def test_refinement_pass_uses_refine_system(self):
        llm = _make_llm()
        state = self._state(
            current_plan={"steps": [{"id": "s0", "title": "Old"}]},
            adjustment="Add a step on limits",
        )
        with patch(
            "graphs.plan.nodes.structured_invoke", return_value=self._mock_design_out()
        ) as mock_invoke:
            node = make_design_plan(llm)
            node(state)
        # Refine system should be used
        assert "Refine the existing plan" in str(mock_invoke.call_args)

    @pytest.mark.real_prompts
    def test_initial_pass_uses_initial_system(self):
        llm = _make_llm()
        with patch(
            "graphs.plan.nodes.structured_invoke", return_value=self._mock_design_out()
        ) as mock_invoke:
            node = make_design_plan(llm)
            node(self._state())
        assert "Design a dependency-ordered" in str(mock_invoke.call_args)

    def test_includes_boundary_map_in_context(self):
        llm = _make_llm()
        with patch(
            "graphs.plan.nodes.structured_invoke", return_value=self._mock_design_out()
        ) as mock_invoke:
            node = make_design_plan(llm)
            node(self._state())
        assert "algebra" in str(mock_invoke.call_args)


# --- make_render_plan ---


class TestRenderPlan:
    def _state(self, **overrides) -> PlanState:
        base: PlanState = {
            "goal": "Learn calculus",
            "boundary_map": {},
            "design_steps": [
                {"title": "A", "description": "d", "depends_on": [], "depth": 1},
            ],
            "language": "English",
        }
        base.update(overrides)  # type: ignore[arg-type]
        return base

    def test_returns_current_plan(self):
        llm = _make_llm()
        mock_out = RenderOut(
            prose_summary="Summary.",
            dependency_dag="graph LR",
            steps=[RenderedStep(id="s0", letter="A", title="A", description="d", depth=1)],
        )
        with patch(
            "graphs.plan.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_render_plan(llm)
            result = node(self._state())

        assert result["current_plan"]["prose_summary"] == "Summary."
        assert result["current_plan"]["dependency_dag"] == "graph LR"
        assert result["current_plan"]["steps"][0]["id"] == "s0"

    def test_includes_baseline_ids_when_current_plan_exists(self):
        llm = _make_llm()
        mock_out = RenderOut(
            prose_summary="S",
            dependency_dag="D",
            steps=[RenderedStep(id="s0", letter="A", title="A", description="d", depth=1)],
        )
        with patch(
            "graphs.plan.nodes.structured_invoke", return_value=mock_out
        ) as mock_invoke:
            node = make_render_plan(llm)
            node(self._state(current_plan={"steps": [{"id": "s7"}, {"id": "s8"}]}))
        assert "s7" in str(mock_invoke.call_args)
        assert "s8" in str(mock_invoke.call_args)


# --- make_wait_for_response ---


class TestWaitForResponse:
    def test_approve(self):
        llm = _make_llm()
        node = make_wait_for_response(llm)
        with patch(
            "graphs.plan.nodes.interrupt", return_value={"action": "approve"}
        ):
            result = node({})
        assert result["approved"] is True
        assert result["response"]["action"] == "approve"

    def test_adjust(self):
        llm = _make_llm()
        node = make_wait_for_response(llm)
        with patch(
            "graphs.plan.nodes.interrupt",
            return_value={"action": "adjust", "text": "Add more depth"},
        ):
            result = node({"pass_count": 0})
        assert "approved" not in result
        assert result["adjustment"] == "Add more depth"
        assert result["pass_count"] == 1

    def test_increments_pass_count(self):
        llm = _make_llm()
        node = make_wait_for_response(llm)
        with patch(
            "graphs.plan.nodes.interrupt",
            return_value={"action": "adjust", "text": "x"},
        ):
            result = node({"pass_count": 2})
        assert result["pass_count"] == 3
