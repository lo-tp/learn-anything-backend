"""Probe graph state definition."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ProbeState(TypedDict, total=False):
    goal: str  # the narrowed_goal
    language: str  # learner's language (human-readable name) for user-facing output
    strands: list[str]  # the fixed prerequisite-strand universe
    history: list[dict]  # prior Q&A entries
    boundary_map: dict[str, dict]  # strand -> {floor, ceiling, gap_type}
    question_count: int  # how many questions asked so far
    strand_descriptions: dict[str, str]  # strand id -> one-line description
    next_batch: list[dict] | None  # the current batch of questions (output at interrupt)
    _resume: dict | None  # transient: resume payload from wait_for_answers
    _decision: str | None  # transient: decide_next output ("continue" | "done")

    # Per-stage wall-clock timings, one entry per stage this request ran
    # (see graphs.common.timed_node); the router persists them to
    # graph_stage_timings. Entry shape matches the material graph's:
    # {"stage": str, "context": dict, "duration_seconds": float}.
    stage_timings: Annotated[list[dict], operator.add]
