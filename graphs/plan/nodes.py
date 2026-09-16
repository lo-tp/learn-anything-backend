"""Node factories for the plan graph."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.types import interrupt

from language import DEFAULT_LANGUAGE, language_instruction

from ..common import structured_invoke
from .prompts import (
    DESIGN_PLAN_INITIAL_SYSTEM,
    DESIGN_PLAN_REFINE_SYSTEM,
    RENDER_PLAN_SYSTEM,
    RESEARCH_TOPIC_SYSTEM,
)
from .schemas import DesignOut, RenderOut, ResearchOut
from .state import PlanState


def make_research_topic(llm: BaseChatModel):
    """Scope the field: core concepts, first principles, framing, gotchas."""

    def research_topic(state: PlanState) -> dict:
        out = structured_invoke(
            llm,
            ResearchOut,
            RESEARCH_TOPIC_SYSTEM,
            f"Goal: {state.get('goal', '')}",
        )
        return {
            "research": {
                "unconditional_truths": out.unconditional_truths,
                "core_concepts": out.core_concepts,
                "standard_framing": out.standard_framing,
                "common_gotchas": out.common_gotchas,
            }
        }

    return research_topic


def make_design_plan(llm: BaseChatModel):
    """Build/refine the dependency-ordered step list."""

    def design_plan(state: PlanState) -> dict:
        research = state.get("research") or {}
        boundary_map = state.get("boundary_map") or {}
        current_plan = state.get("current_plan")
        adjustment = state.get("adjustment")

        # Build the input context
        parts: list[str] = []
        parts.append(f"Goal: {state.get('goal', '')}")
        parts.append(f"Boundary map (learner's current level): {boundary_map}")
        if research:
            parts.append(
                f"Topic research:\n"
                f"  Unconditional truths: {research.get('unconditional_truths', [])}\n"
                f"  Core concepts: {research.get('core_concepts', [])}\n"
                f"  Standard framing: {research.get('standard_framing', '')}\n"
                f"  Common gotchas: {research.get('common_gotchas', [])}"
            )
        if current_plan:
            parts.append(
                f"Current plan (baseline to refine):\n"
                f"  Steps: {current_plan.get('steps', [])}"
            )
        if adjustment:
            parts.append(f"User adjustment request: {adjustment}")

        is_refinement = current_plan is not None
        system = (
            DESIGN_PLAN_REFINE_SYSTEM if is_refinement else DESIGN_PLAN_INITIAL_SYSTEM
        )
        # Step titles and descriptions are shown to the learner.
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)
        out = structured_invoke(
            llm,
            DesignOut,
            system,
            "\n".join(parts),
        )
        return {"design_steps": [s.model_dump() for s in out.steps]}

    return design_plan


def make_render_plan(llm: BaseChatModel):
    """Format: prose summary, mermaid DAG, assign step IDs, convert deps to IDs."""

    def render_plan(state: PlanState) -> dict:
        design_steps = state.get("design_steps") or []
        boundary_map = state.get("boundary_map") or {}
        current_plan = state.get("current_plan")

        # Build the input context
        parts: list[str] = []
        parts.append(f"Goal: {state.get('goal', '')}")
        parts.append(f"Boundary map: {boundary_map}")
        parts.append(f"Design steps (dependencies by title): {design_steps}")
        if current_plan:
            parts.append(
                f"Baseline plan step IDs (keep stable for retained steps): "
                f"{[s['id'] for s in current_plan.get('steps', [])]}"
            )

        system = RENDER_PLAN_SYSTEM
        # prose_summary, step titles/descriptions, and DAG labels are user-facing.
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)

        out = structured_invoke(
            llm,
            RenderOut,
            system,
            "\n".join(parts),
        )
        return {
            "current_plan": {
                "prose_summary": out.prose_summary,
                "dependency_dag": out.dependency_dag,
                "steps": [s.model_dump() for s in out.steps],
            }
        }

    return render_plan


def make_wait_for_response(llm: BaseChatModel):
    """Pause (interrupt); resumes with approve or adjust."""

    def wait_for_response(state: PlanState) -> dict:
        payload = interrupt(None)
        action = payload.get("action")
        if action == "approve":
            return {"response": payload, "approved": True}
        return {
            "response": payload,
            "adjustment": payload.get("text", ""),
            "pass_count": state.get("pass_count", 0) + 1,
        }

    return wait_for_response
