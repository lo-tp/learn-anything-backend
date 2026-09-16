"""Tests for graphs/clarify/nodes.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from graphs.clarify.nodes import (
    MAX_CLARIFY_ROUNDS,
    make_assess_goal,
    make_generate_questions,
    make_refine_goal,
    make_wait_for_answer,
)
from graphs.clarify.schemas import AssessOut, QuestionsOut, RefineOut


def _make_llm() -> MagicMock:
    return MagicMock()


# --- make_assess_goal ---


class TestAssessGoal:
    def test_specific_verdict_returns_narrowed_goal(self):
        llm = _make_llm()
        mock_out = AssessOut(
            verdict="specific",
            narrowed_goal="Learn F=ma with algebra.",
            open_dimensions=[],
        )
        with patch(
            "graphs.clarify.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_assess_goal(llm)
            result = node({"goal": "physics", "round_count": 0})

        assert result["narrowed_goal"] == "Learn F=ma with algebra."
        assert result["open_dimensions"] == []

    def test_too_broad_returns_open_dimensions(self):
        llm = _make_llm()
        mock_out = AssessOut(
            verdict="too_broad",
            narrowed_goal="Learn physics.",
            open_dimensions=["which branch", "level"],
        )
        with patch(
            "graphs.clarify.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_assess_goal(llm)
            result = node({"goal": "physics", "round_count": 0})

        assert "narrowed_goal" not in result
        assert result["open_dimensions"] == ["which branch", "level"]

    def test_cap_forces_specific(self):
        llm = _make_llm()
        # Even if LLM says too_broad, at cap it should return narrowed_goal
        mock_out = AssessOut(
            verdict="too_broad",
            narrowed_goal="Best-effort goal.",
            open_dimensions=["stuff"],
        )
        with patch(
            "graphs.clarify.nodes.structured_invoke", return_value=mock_out
        ) as mock_invoke:
            node = make_assess_goal(llm)
            result = node({"goal": "physics", "round_count": MAX_CLARIFY_ROUNDS})

        assert result["narrowed_goal"] == "Best-effort goal."
        assert result["open_dimensions"] == []
        # The cap message should be in the system prompt
        assert "limit has been reached" in str(mock_invoke.call_args)

    def test_no_cap_message_below_round_limit(self):
        llm = _make_llm()
        mock_out = AssessOut(
            verdict="specific", narrowed_goal="x", open_dimensions=[]
        )
        with patch(
            "graphs.clarify.nodes.structured_invoke", return_value=mock_out
        ) as mock_invoke:
            node = make_assess_goal(llm)
            node({"goal": "x", "round_count": 0})
        assert "limit has been reached" not in str(mock_invoke.call_args)


# --- make_generate_questions ---


class TestGenerateQuestions:
    def test_returns_questions(self):
        llm = _make_llm()
        mock_out = QuestionsOut(
            questions=["What level?", "Which branch?"]
        )
        with patch(
            "graphs.clarify.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_generate_questions(llm)
            result = node({"working_goal": "physics", "open_dimensions": ["level"]})

        assert result["clarifying_questions"] == ["What level?", "Which branch?"]

    def test_includes_open_dimensions_in_prompt(self):
        llm = _make_llm()
        mock_out = QuestionsOut(questions=["Q?"])
        with patch(
            "graphs.clarify.nodes.structured_invoke", return_value=mock_out
        ) as mock_invoke:
            node = make_generate_questions(llm)
            node({"working_goal": "x", "open_dimensions": ["depth", "level"]})
        assert "depth" in str(mock_invoke.call_args)
        assert "level" in str(mock_invoke.call_args)


# --- make_wait_for_answer ---


class TestWaitForAnswer:
    def test_calls_interrupt_and_stores_answer(self):
        llm = _make_llm()
        node = make_wait_for_answer(llm)
        with patch(
            "graphs.clarify.nodes.interrupt", return_value={"answer": "I want algebra"}
        ):
            result = node({})
        assert result == {"last_answer": "I want algebra"}


# --- make_refine_goal ---


class TestRefineGoal:
    def test_returns_working_goal_and_dimensions(self):
        llm = _make_llm()
        mock_out = RefineOut(
            working_goal="Learn algebra for high school.",
            open_dimensions=["depth"],
        )
        with patch(
            "graphs.clarify.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_refine_goal(llm)
            result = node(
                {
                    "working_goal": "Learn algebra",
                    "open_dimensions": ["level", "depth"],
                    "last_answer": "High school",
                    "round_count": 0,
                }
            )

        assert result["working_goal"] == "Learn algebra for high school."
        assert result["open_dimensions"] == ["depth"]
        assert result["round_count"] == 1

    def test_increments_round_count(self):
        llm = _make_llm()
        mock_out = RefineOut(working_goal="x", open_dimensions=[])
        with patch(
            "graphs.clarify.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_refine_goal(llm)
            result = node({"round_count": 2})
        assert result["round_count"] == 3

    def test_includes_last_answer_in_prompt(self):
        llm = _make_llm()
        mock_out = RefineOut(working_goal="x", open_dimensions=[])
        with patch(
            "graphs.clarify.nodes.structured_invoke", return_value=mock_out
        ) as mock_invoke:
            node = make_refine_goal(llm)
            node({"last_answer": "I want to learn calculus"})
        assert "I want to learn calculus" in str(mock_invoke.call_args)
