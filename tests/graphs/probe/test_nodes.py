"""Tests for graphs/probe/nodes.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from graphs.probe.nodes import (
    MAX_PROBE_QUESTIONS,
    make_decide_next,
    make_decompose_strands,
    make_evaluate_batch,
    make_generate_batch,
    make_wait_for_answers,
)
from graphs.probe.schemas import (
    BatchEvaluateOut,
    BatchQuestionsOut,
    QuestionOut,
    SingleEvaluation,
    StrandItem,
    StrandsOut,
)


def _make_llm() -> MagicMock:
    return MagicMock()


# --- make_decompose_strands ---


class TestDecomposeStrands:
    def test_returns_strands_and_boundary_map(self):
        llm = _make_llm()
        mock_out = StrandsOut(
            strands=[
                StrandItem(id="algebra", description="Basic algebra."),
                StrandItem(id="geometry", description="Shapes and angles."),
            ]
        )
        with patch(
            "graphs.probe.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_decompose_strands(llm)
            result = node({"goal": "Learn calculus"})

        assert result["strands"] == ["algebra", "geometry"]
        assert result["strand_descriptions"]["algebra"] == "Basic algebra."
        assert result["boundary_map"]["algebra"] == {
            "floor": None,
            "ceiling": None,
            "gap_type": "unknown",
        }
        assert result["boundary_map"]["geometry"] == {
            "floor": None,
            "ceiling": None,
            "gap_type": "unknown",
        }

    def test_includes_goal_in_prompt(self):
        llm = _make_llm()
        mock_out = StrandsOut(
            strands=[StrandItem(id="s1", description="d1")]
        )
        with patch(
            "graphs.probe.nodes.structured_invoke", return_value=mock_out
        ) as mock_invoke:
            node = make_decompose_strands(llm)
            node({"goal": "Learn quantum mechanics"})
        assert "Learn quantum mechanics" in str(mock_invoke.call_args)


# --- make_generate_batch ---


class TestGenerateBatch:
    def _state(self, **overrides) -> dict:
        base = {
            "goal": "Learn calculus",
            "strands": ["algebra", "geometry"],
            "strand_descriptions": {"algebra": "d1", "geometry": "d2"},
            "boundary_map": {
                "algebra": {"floor": None, "ceiling": None, "gap_type": "unknown"},
                "geometry": {"floor": None, "ceiling": None, "gap_type": "unknown"},
            },
            "question_count": 0,
            "history": [],
            "language": "English",
        }
        base.update(overrides)
        return base

    def test_returns_batch_of_questions(self):
        llm = _make_llm()
        mock_qs = [
            QuestionOut(
                text="What is 2+2?", options=["3", "4", "5", "6"],
                correct_index=1, explanation="4", strand="algebra", difficulty=1,
            ),
            QuestionOut(
                text="What is 3x3?", options=["6", "9", "12", "15"],
                correct_index=1, explanation="9", strand="algebra", difficulty=2,
            ),
            QuestionOut(
                text="Sum of angles in triangle?", options=["90", "180", "270", "360"],
                correct_index=1, explanation="180", strand="geometry", difficulty=1,
            ),
        ]
        mock_out = BatchQuestionsOut(questions=mock_qs)
        with (
            patch("graphs.probe.nodes.structured_invoke", return_value=mock_out),
            patch(
                "graphs.probe.nodes.with_unknown_option",
                side_effect=lambda llm, lang, opts: opts + ["I don't know"],
            ),
        ):
            node = make_generate_batch(llm)
            result = node(self._state())

        assert len(result["next_batch"]) == 3
        assert result["question_count"] == 3
        q = result["next_batch"][0]
        assert q["text"] == "What is 2+2?"
        assert q["correct_index"] == 1
        assert "I don't know" in q["options"]
        assert len(q["options"]) == 5  # 4 + unknown
        assert q["id"]  # has a uuid

    def test_increments_question_count(self):
        llm = _make_llm()
        mock_qs = [
            QuestionOut(
                text="Q?", options=["A", "B", "C", "D"], correct_index=0,
                explanation="e", strand="algebra", difficulty=1,
            ),
        ]
        mock_out = BatchQuestionsOut(questions=mock_qs)
        with (
            patch("graphs.probe.nodes.structured_invoke", return_value=mock_out),
            patch(
                "graphs.probe.nodes.with_unknown_option",
                side_effect=lambda llm, lang, opts: opts + ["I don't know"],
            ),
        ):
            node = make_generate_batch(llm)
            result = node(self._state(question_count=5))
        assert result["question_count"] == 6

    def test_batch_size_capped_by_unbracketed_strands(self):
        llm = _make_llm()
        # Only 1 unbracketed strand
        state = self._state(
            boundary_map={
                "algebra": {"floor": "ok", "ceiling": None, "gap_type": "narrow"},
                "geometry": {"floor": "ok", "ceiling": "ok", "gap_type": "none"},
            }
        )
        mock_qs = [
            QuestionOut(
                text="Q?", options=["A", "B", "C", "D"], correct_index=0,
                explanation="e", strand="algebra", difficulty=1,
            ),
        ]
        mock_out = BatchQuestionsOut(questions=mock_qs)
        with (
            patch("graphs.probe.nodes.structured_invoke", return_value=mock_out) as mock_invoke,
            patch(
                "graphs.probe.nodes.with_unknown_option",
                side_effect=lambda llm, lang, opts: opts + ["I don't know"],
            ),
        ):
            node = make_generate_batch(llm)
            node(state)
        # The prompt should have mentioned batch size 1 (capped)
        assert "Batch size: 1" in str(mock_invoke.call_args)


# --- make_wait_for_answers ---


class TestWaitForAnswers:
    def test_calls_interrupt_and_stores_resume(self):
        llm = _make_llm()
        node = make_wait_for_answers(llm)
        with patch("graphs.probe.nodes.interrupt", return_value={"answers": []}):
            result = node({})
        assert result == {"_resume": {"answers": []}}


# --- make_evaluate_batch ---


class TestEvaluateBatch:
    def _state(self, **overrides) -> dict:
        base = {
            "goal": "Learn calculus",
            "strands": ["algebra", "geometry"],
            "boundary_map": {
                "algebra": {"floor": None, "ceiling": None, "gap_type": "unknown"},
                "geometry": {"floor": None, "ceiling": None, "gap_type": "unknown"},
            },
            "history": [],
            "next_batch": [
                {
                    "id": "q1",
                    "text": "What is 2+2?",
                    "options": ["3", "4", "5", "6", "I don't know"],
                    "correct_index": 1,
                    "explanation": "4",
                    "strand": "algebra",
                    "difficulty": 1,
                },
                {
                    "id": "q2",
                    "text": "Sum of angles?",
                    "options": ["90", "180", "270", "360", "I don't know"],
                    "correct_index": 1,
                    "explanation": "180",
                    "strand": "geometry",
                    "difficulty": 1,
                },
            ],
            "_resume": {
                "answers": [
                    {"question_id": "q1", "selected_index": 1},
                    {"question_id": "q2", "selected_index": 3},
                ]
            },
            "language": "English",
        }
        base.update(overrides)
        return base

    def test_updates_boundary_map_and_history(self):
        llm = _make_llm()
        mock_out = BatchEvaluateOut(
            evaluations=[
                SingleEvaluation(question_id="q1", is_correct=True),
                SingleEvaluation(question_id="q2", is_correct=False),
            ],
            updated_boundary_map={
                "algebra": {"floor": "addition", "ceiling": None, "gap_type": "none"},
                "geometry": {"floor": None, "ceiling": "angles", "gap_type": "narrow"},
            },
            gap_summary="Solid on algebra, struggling with geometry.",
        )
        with patch(
            "graphs.probe.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_evaluate_batch(llm)
            result = node(self._state())

        assert result["boundary_map"]["algebra"]["floor"] == "addition"
        assert result["boundary_map"]["geometry"]["ceiling"] == "angles"
        assert len(result["history"]) == 2
        assert result["history"][0]["is_correct"] is True
        assert result["history"][1]["is_correct"] is False
        assert result["history"][0]["selected_index"] == 1

    def test_falls_back_to_index_comparison_if_eval_missing(self):
        llm = _make_llm()
        # LLM returns no evaluations for q2
        mock_out = BatchEvaluateOut(
            evaluations=[SingleEvaluation(question_id="q1", is_correct=True)],
            updated_boundary_map={
                "algebra": {"floor": "x", "ceiling": None, "gap_type": "none"},
            },
            gap_summary="ok",
        )
        with patch(
            "graphs.probe.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_evaluate_batch(llm)
            result = node(self._state())
        # q2: selected_index=3, correct_index=1 → not correct
        assert result["history"][1]["is_correct"] is False

    def test_merges_boundary_map_defensively(self):
        llm = _make_llm()
        # LLM only returns algebra in updated_boundary_map, not geometry
        mock_out = BatchEvaluateOut(
            evaluations=[
                SingleEvaluation(question_id="q1", is_correct=True),
                SingleEvaluation(question_id="q2", is_correct=False),
            ],
            updated_boundary_map={
                "algebra": {"floor": "x", "ceiling": None, "gap_type": "none"},
            },
            gap_summary="ok",
        )
        with patch(
            "graphs.probe.nodes.structured_invoke", return_value=mock_out
        ):
            node = make_evaluate_batch(llm)
            result = node(self._state())
        # geometry should be preserved from the old boundary_map
        assert result["boundary_map"]["geometry"] == {
            "floor": None,
            "ceiling": None,
            "gap_type": "unknown",
        }


# --- make_decide_next ---


class TestDecideNext:
    def test_continue_when_strands_unbracketed(self):
        llm = _make_llm()
        node = make_decide_next(llm)
        result = node(
            {
                "strands": ["algebra", "geometry"],
                "boundary_map": {
                    "algebra": {"floor": None, "ceiling": None, "gap_type": "unknown"},
                    "geometry": {"floor": "x", "ceiling": "y", "gap_type": "narrow"},
                },
                "question_count": 2,
            }
        )
        assert result["_decision"] == "continue"

    def test_done_when_all_bracketed(self):
        llm = _make_llm()
        node = make_decide_next(llm)
        result = node(
            {
                "strands": ["algebra", "geometry"],
                "boundary_map": {
                    "algebra": {"floor": "a", "ceiling": "b", "gap_type": "narrow"},
                    "geometry": {"floor": "x", "ceiling": None, "gap_type": "none"},
                },
                "question_count": 2,
            }
        )
        assert result["_decision"] == "done"

    def test_done_when_cap_reached(self):
        llm = _make_llm()
        node = make_decide_next(llm)
        result = node(
            {
                "strands": ["algebra"],
                "boundary_map": {
                    "algebra": {"floor": None, "ceiling": None, "gap_type": "unknown"},
                },
                "question_count": MAX_PROBE_QUESTIONS,
            }
        )
        assert result["_decision"] == "done"

    def test_continue_when_no_strands(self):
        llm = _make_llm()
        node = make_decide_next(llm)
        result = node(
            {
                "strands": [],
                "boundary_map": {},
                "question_count": 1,
            }
        )
        # No strands → all_bracketed is False (len(strands) > 0 check)
        assert result["_decision"] == "continue"

    def test_bracketed_requires_floor(self):
        llm = _make_llm()
        node = make_decide_next(llm)
        result = node(
            {
                "strands": ["algebra"],
                "boundary_map": {
                    "algebra": {"floor": None, "ceiling": "x", "gap_type": "none"},
                },
                "question_count": 1,
            }
        )
        # Floor is None → not bracketed → continue
        assert result["_decision"] == "continue"
