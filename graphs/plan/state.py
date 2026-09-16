"""Plan graph state definition."""

from __future__ import annotations

from typing import TypedDict


class PlanState(TypedDict, total=False):
    goal: str  # the narrowed_goal
    language: str  # learner's language (human-readable name) for user-facing output
    boundary_map: dict  # strand -> {floor, ceiling, gap_type}
    research: dict | None  # research_topic output (first pass only)
    current_plan: dict | None  # last rendered plan (baseline + interrupt output)
    design_steps: list[dict]  # design_plan output for the current pass
    adjustment: str | None  # user's adjustment text (refinement passes)
    pass_count: int  # how many refinement passes have occurred
    response: dict | None  # resume payload {action, text?}
    approved: bool | None  # set True on approve
