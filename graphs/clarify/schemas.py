"""Pydantic schemas for the clarify graph's structured LLM outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AssessOut(BaseModel):
    verdict: Literal["specific", "too_broad"]
    narrowed_goal: str = Field(
        description=(
            "A one- or two-sentence precise statement of what to teach. "
            "Always provided — best-effort even when still too broad."
        )
    )
    open_dimensions: list[str] = Field(
        default_factory=list,
        description="1-3 short noun phrases for axes still under-specified.",
    )


class QuestionsOut(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=3)


class RefineOut(BaseModel):
    working_goal: str = Field(
        description="The tightened goal in 1-2 sentences."
    )
    open_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions still under-specified after this answer.",
    )
