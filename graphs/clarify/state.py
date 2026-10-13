"""Clarify graph state definition."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ClarifyState(TypedDict, total=False):
    goal: str  # original raw goal
    language: str  # learner's language (human-readable name) for user-facing output
    working_goal: str  # the goal as narrowed so far
    open_dimensions: list[str]  # axes still under-specified
    round_count: int  # clarification rounds so far (drives the safety cap)
    clarifying_questions: list[str]  # questions in the current round
    last_answer: str | None  # the learner's latest answer
    narrowed_goal: str | None  # exit output

    # Per-stage wall-clock timings, one entry per stage this request ran
    # (see graphs.common.timed_node); the router persists them to
    # graph_stage_timings. Entry shape matches the material graph's:
    # {"stage": str, "context": dict, "duration_seconds": float}.
    stage_timings: Annotated[list[dict], operator.add]
