"""Tests for graphs/plan/schemas.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphs.plan.schemas import (
    DesignOut,
    RenderedStep,
    RenderOut,
    ResearchOut,
    StepDraft,
)


class TestResearchOut:
    def test_valid(self):
        out = ResearchOut(
            unconditional_truths=["Facts"],
            core_concepts=["Concept A", "Concept B"],
            standard_framing="Start with basics.",
            common_gotchas=["Misconception X"],
        )
        assert len(out.core_concepts) == 2
        assert out.standard_framing == "Start with basics."


class TestStepDraft:
    def test_valid(self):
        s = StepDraft(
            title="Learn X",
            description="Covers X.",
            depends_on=["Learn Y"],
            depth=2,
        )
        assert s.title == "Learn X"
        assert s.depends_on == ["Learn Y"]
        assert s.depth == 2

    def test_reject_depth_below_min(self):
        with pytest.raises(ValidationError):
            StepDraft(title="T", description="D", depth=0)

    def test_reject_depth_above_max(self):
        with pytest.raises(ValidationError):
            StepDraft(title="T", description="D", depth=6)

    def test_depends_on_defaults_to_empty(self):
        s = StepDraft(title="T", description="D", depth=1)
        assert s.depends_on == []


class TestDesignOut:
    def test_valid(self):
        out = DesignOut(
            steps=[
                StepDraft(title="A", description="d", depth=1),
                StepDraft(title="B", description="d", depth=2),
            ]
        )
        assert len(out.steps) == 2

    def test_reject_empty(self):
        with pytest.raises(ValidationError):
            DesignOut(steps=[])


class TestRenderedStep:
    def test_valid(self):
        s = RenderedStep(
            id="s0", letter="A", title="A", description="d",
            depends_on=[], depth=1
        )
        assert s.id == "s0"
        assert s.letter == "A"

    def test_reject_depth_below_min(self):
        with pytest.raises(ValidationError):
            RenderedStep(id="s0", letter="A", title="A", description="d", depth=0)


class TestRenderOut:
    def test_valid(self):
        out = RenderOut(
            prose_summary="Summary.",
            dependency_dag="graph LR\n  s0 --> s1",
            steps=[RenderedStep(id="s0", letter="A", title="A", description="d", depth=1)],
        )
        assert len(out.steps) == 1
        assert out.dependency_dag.startswith("graph LR")

    def test_reject_empty_steps(self):
        with pytest.raises(ValidationError):
            RenderOut(prose_summary="x", dependency_dag="y", steps=[])
