"""Pydantic schemas for the probe graph's structured LLM outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StrandItem(BaseModel):
    id: str = Field(description="Short snake_case identifier for the prerequisite strand.")
    description: str = Field(
        description="One-line description of what this strand covers."
    )


class StrandsOut(BaseModel):
    strands: list[StrandItem] = Field(min_length=1)


class QuestionOut(BaseModel):
    text: str = Field(
        description=(
            "Stem of the question only — a self-contained prompt that does "
            "not enumerate, list, or restate any of the answer options."
        )
    )
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, lt=4)
    explanation: str
    strand: str = Field(description="Short snake_case identifier for the prerequisite strand.")
    difficulty: int = Field(ge=1, le=5)


class BatchQuestionsOut(BaseModel):
    """LLM output for generating a batch of probe questions in one call."""

    questions: list[QuestionOut] = Field(min_length=1)


class SingleEvaluation(BaseModel):
    """Per-question evaluation result."""

    question_id: str
    is_correct: bool


class BatchEvaluateOut(BaseModel):
    """LLM output for evaluating a full batch of answers in one call."""

    evaluations: list[SingleEvaluation]
    updated_boundary_map: dict[str, dict] = Field(
        description=(
            "The full boundary_map with every strand touched by this batch "
            "updated. Each value is {floor: str|null, ceiling: str|null, "
            "gap_type: str}."
        )
    )
    gap_summary: str = Field(description="One sentence summarising the gap.")
