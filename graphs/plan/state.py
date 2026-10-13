"""Plan graph state definition."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


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

    # Per-stage wall-clock timings, one entry per stage this request ran
    # (see graphs.common.timed_node); the router persists them to
    # graph_stage_timings. Entry shape matches the material graph's:
    # {"stage": str, "context": dict, "duration_seconds": float}.
    stage_timings: Annotated[list[dict], operator.add]
