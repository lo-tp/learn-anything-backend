"""Tests for core/mock_llm.py (#118): is_mock_mode() and MockChatModel.

Acceptance criteria:
- is_mock_mode() parses MOCK_LLM truthy (true/1, case-insensitive).
- MockChatModel.with_structured_output(schema) returns fixed valid instances
  for every pre-material schema (Clarify, Probe, Plan + language detection).
- MockChatModel.invoke() echoes the input so localize_status/unknown_option
  return the English text unchanged.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from core.mock_llm import MockChatModel, is_mock_mode
from graphs.clarify.schemas import AssessOut, QuestionsOut, RefineOut
from graphs.plan.schemas import DesignOut, RenderOut, ResearchOut
from graphs.probe.schemas import BatchEvaluateOut, BatchQuestionsOut, StrandsOut

# --- is_mock_mode() ---


class TestIsMockMode:
    @pytest.mark.parametrize("value", ["true", "TRUE", "True", "1"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("MOCK_LLM", value)
        assert is_mock_mode() is True

    @pytest.mark.parametrize("value", ["", "false", "0", "yes", "2", "True!"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("MOCK_LLM", value)
        assert is_mock_mode() is False

    def test_unset_is_off(self, monkeypatch):
        monkeypatch.delenv("MOCK_LLM", raising=False)
        assert is_mock_mode() is False


# --- MockChatModel: canned structured outputs ---


def _structured_invoke[SchemaT: BaseModel](
    model, schema: type[SchemaT], system: str = "sys", human: str = "user"
) -> SchemaT:
    """Call the mock the way the graph nodes do (structured_invoke)."""
    return model.with_structured_output(schema).invoke(
        [SystemMessage(content=system), HumanMessage(content=human)]
    )  # type: ignore[return-value]


class TestMockStructuredOutputs:
    def test_assess_goal_is_specific(self):
        out = _structured_invoke(MockChatModel(), AssessOut)
        assert isinstance(out, AssessOut)
        assert out.verdict == "specific"
        assert out.narrowed_goal
        assert out.open_dimensions == []

    def test_questions(self):
        out = _structured_invoke(MockChatModel(), QuestionsOut)
        assert isinstance(out, QuestionsOut)
        assert 1 <= len(out.questions) <= 3

    def test_refine_goal(self):
        out = _structured_invoke(MockChatModel(), RefineOut)
        assert isinstance(out, RefineOut)
        assert out.working_goal
        assert out.open_dimensions == []

    def test_strands(self):
        out = _structured_invoke(MockChatModel(), StrandsOut)
        assert isinstance(out, StrandsOut)
        assert len(out.strands) == 1
        assert out.strands[0].id
        assert out.strands[0].description

    def test_batch_questions(self):
        out = _structured_invoke(MockChatModel(), BatchQuestionsOut)
        assert isinstance(out, BatchQuestionsOut)
        assert len(out.questions) == 1
        q = out.questions[0]
        assert len(q.options) == 4
        assert 0 <= q.correct_index < 4
        assert q.strand
        assert 1 <= q.difficulty <= 5

    def test_evaluate_brackets_the_mock_strand(self):
        """The canned evaluation must bracket the canned strand so the probe
        graph ends (decide_next: floor + ceiling set) after one answer."""
        strands = _structured_invoke(MockChatModel(), StrandsOut)
        evaluate = _structured_invoke(MockChatModel(), BatchEvaluateOut)
        assert isinstance(evaluate, BatchEvaluateOut)
        strand_id = strands.strands[0].id
        entry = evaluate.updated_boundary_map[strand_id]
        assert entry["floor"] is not None
        # bracketed() needs a ceiling OR gap_type == "none".
        assert entry["ceiling"] is not None or entry["gap_type"] == "none"
        assert evaluate.gap_summary

    def test_research(self):
        out = _structured_invoke(MockChatModel(), ResearchOut)
        assert isinstance(out, ResearchOut)
        assert out.core_concepts
        assert out.standard_framing

    def test_design_steps(self):
        out = _structured_invoke(MockChatModel(), DesignOut)
        assert isinstance(out, DesignOut)
        assert out.steps

    def test_rendered_plan_passes_validate_plan(self):
        """The canned plan must satisfy the router's validate_plan."""
        out = _structured_invoke(MockChatModel(), RenderOut)
        assert isinstance(out, RenderOut)
        assert out.prose_summary
        assert out.dependency_dag
        from routers.plan import validate_plan

        validate_plan(
            {
                "prose_summary": out.prose_summary,
                "dependency_dag": out.dependency_dag,
                "steps": [s.model_dump() for s in out.steps],
            }
        )


class TestCoreLlmMockStartup:
    """MOCK_LLM=1 starts without OPENAI_API_KEY (no import-time failure)."""

    def test_core_llm_constructs_without_api_key(self, monkeypatch):
        import importlib

        import core.llm as core_llm

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MOCK_LLM", "1")
        try:
            reloaded = importlib.reload(core_llm)
            assert reloaded.llm is not None
        finally:
            monkeypatch.undo()
            importlib.reload(core_llm)  # restore original construction


class TestMockEchoInvoke:
    def test_invoke_echoes_last_message(self):
        model = MockChatModel()
        text = "Plan approved. Material generation started in the background."
        resp = model.invoke(
            [SystemMessage(content="translate"), HumanMessage(content=text)]
        )
        assert resp.content == text
