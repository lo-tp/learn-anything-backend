"""Structured output schemas for the material graph."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SlideContentSpec(BaseModel):
    title: str
    key_points: list[str]
    visual_hint: str


class SlideContentsOut(BaseModel):
    slide_contents: list[SlideContentSpec] = Field(min_length=3, max_length=7)


class SlideOut(BaseModel):
    slide: str


class QuestionDraft(BaseModel):
    text: str = Field(
        description=(
            "Stem of the question only — a self-contained prompt that does "
            "not enumerate, list, or restate any of the answer options."
        )
    )
    options: list[str] = Field(description="The answer options (bare claims).")
    correct_index: int = Field(description="0-based index of the correct option.")
    explanation: str = Field(
        description="Explanation shown after answering; all reasoning lives here."
    )


class QuestionsOut(BaseModel):
    questions: list[QuestionDraft] = Field(min_length=3, max_length=5)


class SummaryOut(BaseModel):
    key_points: list[str] = Field(
        min_length=1,
        description=(
            "3-5 key points established by this step: definitions, formulas, "
            "core insights — one short self-contained line each."
        ),
    )
