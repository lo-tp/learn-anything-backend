"""Tests for graphs/material/schemas.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphs.material.schemas import (
    QuestionDraft,
    QuestionsOut,
    SlideContentsOut,
    SlideContentSpec,
    SlideOut,
    SummaryOut,
)


class TestSlideContentSpec:
    def test_valid(self):
        spec = SlideContentSpec(
            title="Intro", key_points=["fact1"], visual_hint="diagram"
        )
        assert spec.title == "Intro"
        assert spec.key_points == ["fact1"]
        assert spec.visual_hint == "diagram"


class TestSlideContentsOut:
    def test_valid_minimum(self):
        specs = [
            SlideContentSpec(title=f"s{i}", key_points=["k"], visual_hint="v")
            for i in range(3)
        ]
        out = SlideContentsOut(slide_contents=specs)
        assert len(out.slide_contents) == 3

    def test_valid_maximum(self):
        specs = [
            SlideContentSpec(title=f"s{i}", key_points=["k"], visual_hint="v")
            for i in range(7)
        ]
        out = SlideContentsOut(slide_contents=specs)
        assert len(out.slide_contents) == 7

    def test_reject_too_few(self):
        specs = [
            SlideContentSpec(title="s1", key_points=["k"], visual_hint="v"),
            SlideContentSpec(title="s2", key_points=["k"], visual_hint="v"),
        ]
        with pytest.raises(ValidationError):
            SlideContentsOut(slide_contents=specs)

    def test_reject_too_many(self):
        specs = [
            SlideContentSpec(title=f"s{i}", key_points=["k"], visual_hint="v")
            for i in range(8)
        ]
        with pytest.raises(ValidationError):
            SlideContentsOut(slide_contents=specs)


class TestSlideOut:
    def test_valid(self):
        out = SlideOut(slide="export default function S() { return null; }")
        assert out.slide.startswith("export")


class TestQuestionDraft:
    def test_valid(self):
        q = QuestionDraft(
            text="What is X?",
            options=["A", "B", "C", "D"],
            correct_index=0,
            explanation="A is correct because...",
        )
        assert q.correct_index == 0
        assert len(q.options) == 4


class TestQuestionsOut:
    def test_valid_minimum(self):
        qs = [
            QuestionDraft(
                text=f"Q{i}", options=["A", "B"], correct_index=0, explanation="e"
            )
            for i in range(3)
        ]
        out = QuestionsOut(questions=qs)
        assert len(out.questions) == 3

    def test_valid_maximum(self):
        qs = [
            QuestionDraft(
                text=f"Q{i}", options=["A", "B"], correct_index=0, explanation="e"
            )
            for i in range(5)
        ]
        out = QuestionsOut(questions=qs)
        assert len(out.questions) == 5

    def test_reject_too_few(self):
        qs = [
            QuestionDraft(
                text=f"Q{i}", options=["A", "B"], correct_index=0, explanation="e"
            )
            for i in range(2)
        ]
        with pytest.raises(ValidationError):
            QuestionsOut(questions=qs)

    def test_reject_too_many(self):
        qs = [
            QuestionDraft(
                text=f"Q{i}", options=["A", "B"], correct_index=0, explanation="e"
            )
            for i in range(6)
        ]
        with pytest.raises(ValidationError):
            QuestionsOut(questions=qs)


class TestSummaryOut:
    def test_valid(self):
        out = SummaryOut(key_points=["point1", "point2", "point3"])
        assert len(out.key_points) == 3

    def test_valid_single(self):
        out = SummaryOut(key_points=["only one"])
        assert len(out.key_points) == 1

    def test_reject_empty(self):
        with pytest.raises(ValidationError):
            SummaryOut(key_points=[])
