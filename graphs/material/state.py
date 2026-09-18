"""Material graph state."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class MaterialState(TypedDict, total=False):
    step: dict
    established_concepts: list[dict]
    learner_context: dict
    language: str
    slide_contents: list[dict]
    slides: Annotated[list[str], operator.add]
    slide_index: int
    attempts_by_slide: list[int]
    current_slide_jsx: str | None
    # Full prompt sent to the LLM for the in-flight slide (persisted to
    # failed_slides on failure for debugging: prompt -> result -> error).
    current_slide_prompt: str | None
    last_compile_error: str | None
    failed_attempts: Annotated[list[dict], operator.add]
    # Per-stage wall-clock timings accumulated by the nodes (one entry per
    # stage execution); the driver persists them to graph_stage_timings.
    # Entry shape: {"stage": str, "context": dict, "duration_seconds": float}.
    stage_timings: Annotated[list[dict], operator.add]
    compile_result: str  # "success" | "retry" | "exhausted"
    # True when the just-compiled slide is a placeholder (the failed slide
    # exhausted all attempts). The driver persists it on the SlideContent
    # row; endpoints return placeholder rows only in dev mode.
    current_slide_is_placeholder: bool
    questions: list[dict]
    summary: dict
