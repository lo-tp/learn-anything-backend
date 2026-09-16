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
    compile_result: str  # "success" | "retry" | "exhausted"
    questions: list[dict]
    summary: dict
