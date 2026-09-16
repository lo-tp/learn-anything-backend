"""Pydantic schemas for the plan graph's structured LLM outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchOut(BaseModel):
    unconditional_truths: list[str] = Field(
        description=(
            "Facts the learner can accept at face value with no caveats. "
            "These are the root nodes of the dependency graph."
        )
    )
    core_concepts: list[str] = Field(
        description="The main concepts in the topic, roughly ordered by dependency."
    )
    standard_framing: str = Field(
        description="How this topic is most commonly introduced and structured."
    )
    common_gotchas: list[str] = Field(
        description="Common misconceptions or traps learners run into."
    )


class StepDraft(BaseModel):
    title: str = Field(description="Short title for the step.")
    description: str = Field(description="One sentence describing what this step covers.")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Titles of prerequisite steps this step depends on.",
    )
    depth: int = Field(ge=1, le=5, description="Approximate DAG level (1=shallowest).")


class DesignOut(BaseModel):
    steps: list[StepDraft] = Field(min_length=1)


class RenderedStep(BaseModel):
    id: str = Field(description="Step ID, e.g. 's0', 's1', etc.")
    title: str
    description: str = Field(description="One sentence (copied from the design step; do not expand it).")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Step IDs of prerequisite steps.",
    )
    depth: int = Field(ge=1, le=5)


class RenderOut(BaseModel):
    prose_summary: str = Field(
        description=(
            "A 2-3 sentence summary of the plan: what we will cover, in what "
            "order, and why — given the learner's starting point and goal. "
            "No filler."
        )
    )
    dependency_dag: str = Field(
        description=(
            "A mermaid DAG (graph LR) showing the step dependencies. "
            "Few nodes, short labels, edges = dependencies."
        )
    )
    steps: list[RenderedStep] = Field(min_length=1)
