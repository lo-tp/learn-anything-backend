"""Tests for core/mock_llm.py (#118): is_mock_mode() and MockChatModel.

Acceptance criteria:
- is_mock_mode() parses MOCK_LLM truthy (true/1, case-insensitive).
- MockChatModel.with_structured_output(schema) returns fixed valid instances
  for every pre-material schema (Clarify, Probe, Plan + language detection).
- The Clarify pair is scripted rather than fixed: two rounds of questions,
  3 then 2, and only then the narrowed goal.
- The Probe batch is the batch a real run produces: three questions across
  three strands, bracketed by the evaluation in one round.
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


def _assess_prompt(round_count: int) -> str:
    """The human prompt ``graphs/clarify/nodes.py:assess_goal`` sends.

    The mock reads the round out of it. Only that line is load-bearing here —
    the rest is the shape the node actually sends, so this test moves with the
    node. The mock driven by the real graph is walked in
    :mod:`tests.graphs.test_mock_llm_wiring`.
    """
    return (
        "Original goal: Learn calculus\n"
        "Working goal: Learn calculus\n"
        "Open dimensions: none\n"
        f"Clarification round: {round_count}"
    )


def _questions_prompt(open_dimensions: list[str]) -> str:
    """The human prompt ``graphs/clarify/nodes.py:generate_questions`` sends —
    the state's open dimensions, and no round number."""
    return f"Working goal: Learn calculus\nOpen dimensions: {open_dimensions!r}"


class TestMockStructuredOutputs:
    def test_assess_asks_twice_then_accepts(self):
        """Clarify runs the mock's script: name what is still open, ask about
        it, and accept only at the end of the script.

        The canned answer used to be "specific" on every call, so a mock run
        asked no clarifying question at all and the new-session form's first
        step had nothing to render.
        """
        model = MockChatModel()
        verdicts = [
            _structured_invoke(model, AssessOut, human=_assess_prompt(round_index))
            for round_index in range(3)
        ]
        assert [v.verdict for v in verdicts] == [
            "too_broad",
            "too_broad",
            "specific",
        ]
        # The schema always asks for a narrowed_goal, so one is stated every
        # time; only the last one is the answer the graph takes.
        assert all(v.narrowed_goal for v in verdicts)
        assert len(verdicts[0].open_dimensions) == 3
        assert len(verdicts[1].open_dimensions) == 2
        assert verdicts[2].open_dimensions == []

    def test_questions_are_one_per_dimension_and_five_across_the_rounds(self):
        """3 + 2 = 5 clarifying questions, each round still inside the real
        1-3-per-round limit."""
        model = MockChatModel()
        asked: list[list[str]] = []
        for round_index in range(2):
            assess = _structured_invoke(
                model, AssessOut, human=_assess_prompt(round_index)
            )
            out = _structured_invoke(
                model,
                QuestionsOut,
                human=_questions_prompt(list(assess.open_dimensions)),
            )
            assert isinstance(out, QuestionsOut)
            assert 1 <= len(out.questions) <= 3  # the per-round limit is unchanged
            # One question per dimension the same round just named.
            assert len(out.questions) == len(assess.open_dimensions)
            assert all(q.strip() for q in out.questions)
            asked.append(list(out.questions))
        flat = [question for round_questions in asked for question in round_questions]
        assert len(flat) == 5
        # The second round asks something different: five questions, not the
        # same three asked again.
        assert len(set(flat)) == 5

    def test_questions_refuse_a_prompt_that_names_no_round(self):
        """Script and graph have to agree about which round this is.

        Falling back quietly would re-ask a round the learner already answered.
        """
        with pytest.raises(LookupError, match="no round named"):
            _structured_invoke(MockChatModel(), QuestionsOut)

    def test_refine_goal(self):
        out = _structured_invoke(MockChatModel(), RefineOut)
        assert isinstance(out, RefineOut)
        assert out.working_goal
        assert out.open_dimensions == []

    def test_strands(self):
        out = _structured_invoke(MockChatModel(), StrandsOut)
        assert isinstance(out, StrandsOut)
        # Three, because that is the batch the new-session form shows (see
        # _MOCK_PROBE_BATCH) and a strand per question is what makes a batch of
        # three one round rather than three rounds.
        assert len(out.strands) == 3
        assert len({s.id for s in out.strands}) == 3
        assert all(s.id and s.description for s in out.strands)

    def test_batch_questions(self):
        """The canned batch is the batch a learner sees: ``PROBE_BATCH_SIZE``
        defaults to 3, so the Probing step always presents three questions."""
        out = _structured_invoke(MockChatModel(), BatchQuestionsOut)
        assert isinstance(out, BatchQuestionsOut)
        assert len(out.questions) == 3
        seeded = {s.id for s in _structured_invoke(MockChatModel(), StrandsOut).strands}
        # Each question targets a seeded strand, and the batch spans them.
        assert {q.strand for q in out.questions} == seeded
        for q in out.questions:
            assert len(q.options) == 4
            assert 0 <= q.correct_index < 4
            assert q.text and q.explanation
            assert 1 <= q.difficulty <= 5

    def test_evaluate_brackets_the_mock_strands(self):
        """The canned evaluation must bracket every canned strand so the probe
        graph ends (decide_next: floor + ceiling set) after one answered batch.

        Asserted against the strands as seeded, not against the ids written in
        ``core/mock_llm.py``: the three canned probe outputs must stay one
        round's worth of consistent content, which is what a hand-written
        triple drifts away from.
        """
        strands = _structured_invoke(MockChatModel(), StrandsOut)
        batch = _structured_invoke(MockChatModel(), BatchQuestionsOut)
        evaluate = _structured_invoke(MockChatModel(), BatchEvaluateOut)
        assert isinstance(evaluate, BatchEvaluateOut)
        assert set(evaluate.updated_boundary_map) == {s.id for s in strands.strands}
        assert {q.strand for q in batch.questions} <= set(evaluate.updated_boundary_map)
        for entry in evaluate.updated_boundary_map.values():
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

    def test_a_bound_reasoning_budget_changes_nothing(self):
        """Every graph node now binds a per-phase reasoning budget (#194). The
        mock is the same canned answer whatever it is asked to think with —
        including the Clarify script, whose answer turns on the prompt it was
        sent and not on what the call was told to reason with.
        """
        model = MockChatModel()
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content=_assess_prompt(0)),
        ]
        bound = (
            model.with_structured_output(AssessOut)
            .bind(reasoning_effort="low", max_tokens=1536)
            .invoke(messages)
        )
        unbound = model.with_structured_output(AssessOut).invoke(messages)
        assert isinstance(bound, AssessOut)
        assert bound == unbound

    def test_echo_invoke_ignores_bound_kwargs_too(self):
        """``localize_status`` binds its own budget (#194); echoing is unaffected."""
        model = MockChatModel()
        text = "I don't know"
        resp = (
            model.bind(reasoning_effort="minimal", max_tokens=256)
            .invoke([SystemMessage(content="translate"), HumanMessage(content=text)])
        )
        assert resp.content == text
