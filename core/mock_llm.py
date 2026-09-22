"""Mock LLM for MOCK_LLM mode (#118).

When the ``MOCK_LLM`` env flag is set to ``true``/``1`` (case-insensitive),
the pre-material phases (Clarify, Probe, Plan) run on :class:`MockChatModel`
instead of the real LLM so a fresh session can be spun up without any real
LLM calls. Mock mode never advances into slide (material) generation — the
material graph always stays wired to the real ``core.llm.llm`` (and is
simply never invoked).

:func:`is_mock_mode` is the single source of truth for the flag. It is read
from the environment per call (same pattern as ``DEV_MODE``); the startup
decisions that consume it (graph wiring in ``graphs``, the engine in
``db.models``) are taken once at import time, i.e. at startup.

The pre-material *schemas* are imported lazily (first structured call),
because importing ``graphs.*`` executes the ``graphs`` package init, which
imports ``core.llm`` — a module-level import here would create a
``core.llm <-> core.mock_llm <-> graphs`` import cycle.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

# --- Flag (single source of truth) ---


def is_mock_mode() -> bool:
    """True when ``MOCK_LLM`` is set to ``true``/``1`` (case-insensitive)."""
    return os.getenv("MOCK_LLM", "").strip().lower() in ("true", "1")


# --- Canned pre-material outputs ---

# Minimal fixed content (per the #118 design): the clarify goal is specific
# immediately (no questions), one probe question that brackets its strand
# in a single round, one small plan step.

_MOCK_STRAND_ID = "mock_strand"

_CANNED_OUTPUTS: dict[type[BaseModel], BaseModel] | None = None


def make_canned_output(schema: type[BaseModel]) -> BaseModel:
    """Return the fixed valid instance for a pre-material schema.

    Raises ``KeyError`` for unknown schemas — the pre-material graphs and
    the language helpers are the only intended consumers.
    """
    if _CANNED_OUTPUTS is None:
        _build_canned_outputs()
    return _CANNED_OUTPUTS[schema]  # type: ignore[index]


def _build_canned_outputs() -> None:
    """Build the schema -> canned-instance table (imported lazily)."""
    global _CANNED_OUTPUTS

    # Imported here (not at module level) to avoid the graphs import cycle.
    from core.language import _LanguageDetectionOut
    from graphs.clarify.schemas import AssessOut, QuestionsOut, RefineOut
    from graphs.plan.schemas import (
        DesignOut,
        RenderedStep,
        RenderOut,
        ResearchOut,
        StepDraft,
    )
    from graphs.probe.schemas import (
        BatchEvaluateOut,
        BatchQuestionsOut,
        QuestionOut,
        StrandItem,
        StrandsOut,
    )

    _CANNED_OUTPUTS = {
        # Clarify: the goal is "specific" immediately -> one canned
        # narrowed_goal, zero questions.
        AssessOut: AssessOut(
            verdict="specific",
            narrowed_goal="Mock narrowed goal: the core ideas of the topic.",
            open_dimensions=[],
        ),
        QuestionsOut: QuestionsOut(questions=["What outcome do you want?"]),
        RefineOut: RefineOut(
            working_goal="Mock refined goal.",
            open_dimensions=[],
        ),
        # Probe: one fixed strand, one fixed MCQ; the evaluation brackets
        # the strand -> the probe graph is done after a single learner
        # answer.
        StrandsOut: StrandsOut(
            strands=[
                StrandItem(
                    id=_MOCK_STRAND_ID,
                    description="Mock prerequisite strand.",
                )
            ]
        ),
        BatchQuestionsOut: BatchQuestionsOut(
            questions=[
                QuestionOut(
                    text="Mock probe question: pick the correct option.",
                    options=[
                        "Mock option A",
                        "Mock option B",
                        "Mock option C",
                        "Mock option D",
                    ],
                    correct_index=0,
                    explanation="Mock explanation.",
                    strand=_MOCK_STRAND_ID,
                    difficulty=1,
                )
            ]
        ),
        BatchEvaluateOut: BatchEvaluateOut(
            # evaluations may be empty: the probe node falls back to
            # selected == correct_index per question.
            evaluations=[],
            updated_boundary_map={
                _MOCK_STRAND_ID: {
                    "floor": "mock floor",
                    "ceiling": "mock ceiling",
                    "gap_type": "closed",
                }
            },
            gap_summary="Mock gap summary.",
        ),
        # Plan: a fixed canned plan that passes validate_plan.
        ResearchOut: ResearchOut(
            unconditional_truths=["Mock unconditional truth."],
            core_concepts=["Mock core concept"],
            standard_framing="Mock standard framing.",
            common_gotchas=["Mock common gotcha."],
        ),
        DesignOut: DesignOut(
            steps=[
                StepDraft(
                    title="Mock Step",
                    description="A single mock plan step.",
                    depends_on=[],
                    depth=1,
                )
            ]
        ),
        RenderOut: RenderOut(
            prose_summary="Mock plan summary: one step covering the mock topic.",
            dependency_dag="graph LR\n  s0[Mock Step]",
            steps=[
                RenderedStep(
                    id="s0",
                    letter="A",
                    title="Mock Step",
                    description="A single mock plan step.",
                    depends_on=[],
                    depth=1,
                )
            ],
        ),
        # Language detection: English.
        _LanguageDetectionOut: _LanguageDetectionOut(language="en"),
    }


# --- Mock chat model (graph/LLM seam) ---


class MockChatModel(BaseChatModel):
    """Canned pre-material LLM for MOCK_LLM mode (#118).

    - ``with_structured_output(schema)`` yields the fixed valid instance for
      the pre-material schemas (Clarify / Probe / Plan + language detection).
    - ``invoke`` echoes the last message, so ``localize_status`` and
      ``unknown_option`` return the English text unchanged.

    The prod ``core.llm.llm`` is never mutated; the pre-material graphs are
    simply built with this model in mock mode.
    """

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1].content if messages else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=last))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        del include_raw  # structured-binding kwargs are not meaningful for the mock
        # The mock only supports pydantic schemas (the pre-material graphs
        # and the language helpers are its only consumers).
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("MockChatModel schemas must be pydantic models")
        return RunnableLambda(lambda _input: make_canned_output(schema))  # type: ignore[arg-type]
