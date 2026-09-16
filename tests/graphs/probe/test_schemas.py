"""Tests for graphs/probe/schemas.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphs.probe.schemas import (
    BatchEvaluateOut,
    BatchQuestionsOut,
    QuestionOut,
    SingleEvaluation,
    StrandItem,
    StrandsOut,
)


class TestStrandItem:
    def test_valid(self):
        item = StrandItem(id="algebra_basics", description="Solving linear equations.")
        assert item.id == "algebra_basics"
        assert item.description == "Solving linear equations."


class TestStrandsOut:
    def test_valid(self):
        out = StrandsOut(
            strands=[
                StrandItem(id="s1", description="d1"),
                StrandItem(id="s2", description="d2"),
            ]
        )
        assert len(out.strands) == 2

    def test_reject_empty(self):
        with pytest.raises(ValidationError):
            StrandsOut(strands=[])


class TestQuestionOut:
    def test_valid(self):
        q = QuestionOut(
            text="What is 2+2?",
            options=["3", "4", "5", "6"],
            correct_index=1,
            explanation="2+2=4",
            strand="arithmetic",
            difficulty=1,
        )
        assert q.correct_index == 1
        assert len(q.options) == 4

    def test_reject_too_few_options(self):
        with pytest.raises(ValidationError):
            QuestionOut(
                text="Q", options=["A", "B", "C"], correct_index=0,
                explanation="e", strand="s", difficulty=1,
            )

    def test_reject_too_many_options(self):
        with pytest.raises(ValidationError):
            QuestionOut(
                text="Q", options=["A", "B", "C", "D", "E"], correct_index=0,
                explanation="e", strand="s", difficulty=1,
            )

    def test_reject_correct_index_out_of_range(self):
        with pytest.raises(ValidationError):
            QuestionOut(
                text="Q", options=["A", "B", "C", "D"], correct_index=4,
                explanation="e", strand="s", difficulty=1,
            )

    def test_reject_negative_correct_index(self):
        with pytest.raises(ValidationError):
            QuestionOut(
                text="Q", options=["A", "B", "C", "D"], correct_index=-1,
                explanation="e", strand="s", difficulty=1,
            )

    def test_reject_difficulty_below_min(self):
        with pytest.raises(ValidationError):
            QuestionOut(
                text="Q", options=["A", "B", "C", "D"], correct_index=0,
                explanation="e", strand="s", difficulty=0,
            )

    def test_reject_difficulty_above_max(self):
        with pytest.raises(ValidationError):
            QuestionOut(
                text="Q", options=["A", "B", "C", "D"], correct_index=0,
                explanation="e", strand="s", difficulty=6,
            )


class TestBatchQuestionsOut:
    def test_valid(self):
        qs = [
            QuestionOut(
                text=f"Q{i}", options=["A", "B", "C", "D"], correct_index=0,
                explanation="e", strand="s", difficulty=1,
            )
            for i in range(3)
        ]
        out = BatchQuestionsOut(questions=qs)
        assert len(out.questions) == 3

    def test_reject_empty(self):
        with pytest.raises(ValidationError):
            BatchQuestionsOut(questions=[])


class TestSingleEvaluation:
    def test_valid(self):
        e = SingleEvaluation(question_id="q1", is_correct=True)
        assert e.is_correct is True

    def test_valid_incorrect(self):
        e = SingleEvaluation(question_id="q1", is_correct=False)
        assert e.is_correct is False


class TestBatchEvaluateOut:
    def test_valid(self):
        out = BatchEvaluateOut(
            evaluations=[
                SingleEvaluation(question_id="q1", is_correct=True),
                SingleEvaluation(question_id="q2", is_correct=False),
            ],
            updated_boundary_map={
                "arithmetic": {"floor": "addition", "ceiling": None, "gap_type": "none"}
            },
            gap_summary="Learner is solid on basics.",
        )
        assert len(out.evaluations) == 2
        assert out.gap_summary == "Learner is solid on basics."

    def test_empty_evaluations_allowed(self):
        """Schema doesn't enforce min_length on evaluations."""
        out = BatchEvaluateOut(
            evaluations=[],
            updated_boundary_map={},
            gap_summary="n/a",
        )
        assert out.evaluations == []
