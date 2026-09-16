"""Tests for graphs/clarify/schemas.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphs.clarify.schemas import AssessOut, QuestionsOut, RefineOut


class TestAssessOut:
    def test_valid_specific(self):
        out = AssessOut(
            verdict="specific",
            narrowed_goal="Learn Newton's second law with basic algebra.",
            open_dimensions=[],
        )
        assert out.verdict == "specific"
        assert out.open_dimensions == []

    def test_valid_too_broad(self):
        out = AssessOut(
            verdict="too_broad",
            narrowed_goal="Learn physics.",
            open_dimensions=["which branch", "depth level"],
        )
        assert out.verdict == "too_broad"
        assert len(out.open_dimensions) == 2

    def test_reject_invalid_verdict(self):
        with pytest.raises(ValidationError):
            AssessOut(verdict="maybe", narrowed_goal="x")  # type: ignore[type-arg]

    def test_open_dimensions_defaults_to_empty(self):
        out = AssessOut(verdict="specific", narrowed_goal="x")
        assert out.open_dimensions == []


class TestQuestionsOut:
    def test_valid_single(self):
        out = QuestionsOut(questions=["What level are you at?"])
        assert len(out.questions) == 1

    def test_valid_multiple(self):
        out = QuestionsOut(questions=["Q1?", "Q2?", "Q3?"])
        assert len(out.questions) == 3

    def test_reject_empty(self):
        with pytest.raises(ValidationError):
            QuestionsOut(questions=[])

    def test_reject_too_many(self):
        with pytest.raises(ValidationError):
            QuestionsOut(questions=["Q1?", "Q2?", "Q3?", "Q4?"])


class TestRefineOut:
    def test_valid(self):
        out = RefineOut(
            working_goal="Learn Newton's second law for high school.",
            open_dimensions=["depth"],
        )
        assert out.working_goal == "Learn Newton's second law for high school."
        assert out.open_dimensions == ["depth"]

    def test_empty_open_dimensions(self):
        out = RefineOut(working_goal="Done.", open_dimensions=[])
        assert out.open_dimensions == []

    def test_open_dimensions_defaults_to_empty(self):
        out = RefineOut(working_goal="Done.")
        assert out.open_dimensions == []
