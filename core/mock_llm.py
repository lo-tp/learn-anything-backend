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
import re
from collections.abc import Callable
from typing import Any, NamedTuple

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

# Minimal fixed content (per the #118 design): a Clarify script that narrows
# the goal over two rounds, a probe batch that brackets its strands in a single
# round, one small plan step.

_MOCK_NARROWED_GOAL = "Mock narrowed goal: the core ideas of the topic."


class _MockClarifyRound(NamedTuple):
    """One round of the mock's Clarify script.

    ``open_dimensions`` is what ``assess_goal`` names as still under-specified,
    and ``questions`` is one question per named dimension — the same relation the
    real prompt draws between the two ("write the questions that most reduce the
    remaining ambiguity"). Each round stays inside the real limits (1-3 questions
    per round, ``MAX_CLARIFY_ROUNDS = 3``), and together the two rounds ask
    3 + 2 = 5 — so a mock run walks the intake the real flow walks: the
    new-session form is asked to render two rounds of answers, not the old case
    of no questions at all.

    The dimensions double as the marker that says which round is being asked
    about. Only ``assess_goal`` states a round number in its prompt;
    ``generate_questions`` states the state's open dimensions, and in a mock run
    those are the mock's own words.
    """

    open_dimensions: tuple[str, ...]
    questions: tuple[str, ...]


_MOCK_CLARIFY_SCRIPT: tuple[_MockClarifyRound, ...] = (
    _MockClarifyRound(
        open_dimensions=(
            "mock dimension: the outcome the learner wants",
            "mock dimension: how deep this should go",
            "mock dimension: what the learner already knows",
        ),
        questions=(
            "Mock clarifying question 1: what should the learner be able to do at the end?",
            "Mock clarifying question 2: how deep should this go — a working idea, or the mechanism?",
            "Mock clarifying question 3: what does the learner already know about it?",
        ),
    ),
    _MockClarifyRound(
        open_dimensions=(
            "mock dimension: what kind of material suits the learner",
            "mock dimension: what to leave out",
        ),
        questions=(
            "Mock clarifying question 4: which material works best — worked examples, or the concepts first?",
            "Mock clarifying question 5: what should this session leave out?",
        ),
    ),
)

_CLARIFY_ROUND_RE = re.compile(r"Clarification round: (\d+)")


def _clarify_round(prompt: str) -> int:
    """Which round of the Clarify loop a node is asking about.

    ``graphs/clarify/nodes.py`` states the round in the prompt it sends — the
    only place a stateless mock can learn it, and unstated reads as round 0.
    The Clarify walk in ``tests/graphs/test_mock_llm_wiring.py`` runs the real
    graph against this mock, so a node that stopped stating its round fails
    there rather than quietly asking its first round again.
    """
    match = _CLARIFY_ROUND_RE.search(prompt)
    return int(match.group(1)) if match else 0


def _content_text(content: Any) -> str:
    """Message content as text, including the content-block list form."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _prompt_text(model_input: Any) -> str:
    """The text a graph node last asked the model with.

    Every pre-material node calls ``with_structured_output(schema)`` on a
    ``[SystemMessage, HumanMessage]`` pair (``graphs/common.py:structured_invoke``)
    and puts the state the model is meant to read in the human message — the
    last one. Anything that is not a message list is answered as itself.
    """
    if isinstance(model_input, list) and model_input:
        last = model_input[-1]
        return _content_text(getattr(last, "content", last))
    return _content_text(model_input)


def _always(instance: BaseModel) -> Callable[[str], BaseModel]:
    """A canned answer that is the same whatever it was asked."""
    return lambda _prompt: instance


class _MockProbeQuestion(NamedTuple):
    """One canned probe round: a strand to seed, and the MCQ that brackets it.

    One table feeds all three probe schemas, so the strands the mock asks
    about, the questions it asks, and the boundary map it writes back are
    necessarily the same strands — the mismatch that a hand-written triple of
    canned outputs drifts into.
    """

    strand_id: str
    strand_description: str
    text: str
    options: tuple[str, str, str, str]
    correct_index: int
    explanation: str
    difficulty: int


# Three strands, one MCQ per strand, difficulty escalating across the batch.
#
# Three is the batch size a learner actually sees: ``PROBE_BATCH_SIZE`` defaults
# to 3 and the generator asks for min(that, unbracketed strands), so the new-
# session form's Probing step always presents a batch of three. The old single
# canned question made a mock run look like a different flow from the real one —
# one card instead of three, an answer payload of one, an evaluate step that only
# ever saw one question — so the thing mock mode exists for (walking the new-
# session form without waiting on a model) could not be checked against it.
#
# The mock answers with this batch whatever batch size the prompt asked for,
# exactly as it ignores the rest of the prompt.
_MOCK_PROBE_BATCH: tuple[_MockProbeQuestion, ...] = (
    _MockProbeQuestion(
        strand_id="mock_foundations",
        strand_description="Mock prerequisite strand: the basic idea the goal builds on.",
        text="Mock probe question 1: which statement states the basic idea?",
        options=(
            "Mock correct claim about the basic idea.",
            "Mock distractor: the basic idea confused with one example of it.",
            "Mock distractor: the basic idea stated in reverse.",
            "Mock distractor: a neighbouring idea mistaken for it.",
        ),
        correct_index=0,
        explanation="Mock explanation: the first option states the basic idea; the "
        "others confuse it with an example, invert it, or name a neighbouring idea.",
        difficulty=1,
    ),
    _MockProbeQuestion(
        strand_id="mock_mechanics",
        strand_description="Mock prerequisite strand: how those ideas combine.",
        text="Mock probe question 2: how do these ideas combine?",
        options=(
            "Mock correct claim about how they combine.",
            "Mock distractor: one idea applied on its own.",
            "Mock distractor: the two ideas swapped.",
            "Mock distractor: a combination that no mock case supports.",
        ),
        correct_index=0,
        explanation="Mock explanation: the first option states how they combine; the "
        "others isolate, invert, or invent a combination.",
        difficulty=2,
    ),
    _MockProbeQuestion(
        strand_id="mock_limits",
        strand_description="Mock prerequisite strand: where those combinations stop holding.",
        text="Mock probe question 3: where does this stop holding?",
        options=(
            "Mock correct claim about the limit.",
            "Mock distractor: the limit denied.",
            "Mock distractor: a limit belonging to a different strand.",
            "Mock distractor: the limit overstated so it is never true.",
        ),
        correct_index=0,
        explanation="Mock explanation: the first option states the limit; the others "
        "deny it, borrow it from elsewhere, or overstate it.",
        difficulty=3,
    ),
)

_CANNED_OUTPUTS: dict[type[BaseModel], Callable[[str], BaseModel]] | None = None


def make_canned_output(schema: type[BaseModel], model_input: Any = None) -> BaseModel:
    """Return the valid canned instance for a pre-material schema.

    ``model_input`` is what the graph asked with — the message list a node
    built. It decides which step of the Clarify script answers; every other
    schema is answered by :func:`_always`, the same whatever it was asked.

    Raises ``KeyError`` for unknown schemas — the pre-material graphs and
    the language helpers are the only intended consumers.
    """
    if _CANNED_OUTPUTS is None:
        _build_canned_outputs()
    answer = _CANNED_OUTPUTS[schema]  # type: ignore[index]
    return answer(_prompt_text(model_input))


def _build_canned_outputs() -> None:
    """Build the schema -> canned-answer table (imported lazily)."""
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

    def assess_out(prompt: str) -> AssessOut:
        """The script's step for this round: name what is still open, and
        accept only once the script has run."""
        round_index = _clarify_round(prompt)
        if round_index >= len(_MOCK_CLARIFY_SCRIPT):
            return AssessOut(
                verdict="specific",
                narrowed_goal=_MOCK_NARROWED_GOAL,
                open_dimensions=[],
            )
        step = _MOCK_CLARIFY_SCRIPT[round_index]
        return AssessOut(
            verdict="too_broad",
            narrowed_goal=_MOCK_NARROWED_GOAL,
            open_dimensions=list(step.open_dimensions),
        )

    def questions_out(prompt: str) -> QuestionsOut:
        """The questions belonging to the round this prompt's dimensions name.

        A prompt naming none of them means the script and the graph disagree
        about which round this is. That is said loudly: in a mock run, a silent
        fallback would re-ask a round the learner already answered.
        """
        for step in _MOCK_CLARIFY_SCRIPT:
            if any(dimension in prompt for dimension in step.open_dimensions):
                return QuestionsOut(questions=list(step.questions))
        raise LookupError(
            "Mock ChatModel was asked for clarifying questions with no round "
            "named in the prompt: the mock's Clarify script answers the round "
            "whose open dimensions "
            "graphs/clarify/nodes.py:generate_questions sends."
        )

    _CANNED_OUTPUTS = {
        # Clarify: a two-round script — 3 questions, then 2 — followed by the
        # narrowed goal (see _MOCK_CLARIFY_SCRIPT).
        AssessOut: assess_out,
        QuestionsOut: questions_out,
        RefineOut: _always(
            RefineOut(
                working_goal="Mock refined goal.",
                open_dimensions=[],
            )
        ),
        # Probe: the fixed strands and the batch that spans them; the
        # evaluation brackets every strand -> the probe graph is done after the
        # learner answers one batch.
        StrandsOut: _always(
            StrandsOut(
                strands=[
                    StrandItem(id=q.strand_id, description=q.strand_description)
                    for q in _MOCK_PROBE_BATCH
                ]
            )
        ),
        BatchQuestionsOut: _always(
            BatchQuestionsOut(
                questions=[
                    QuestionOut(
                        text=q.text,
                        options=list(q.options),
                        correct_index=q.correct_index,
                        explanation=q.explanation,
                        strand=q.strand_id,
                        difficulty=q.difficulty,
                    )
                    for q in _MOCK_PROBE_BATCH
                ]
            )
        ),
        BatchEvaluateOut: _always(
            BatchEvaluateOut(
                # evaluations may be empty: the probe node falls back to
                # selected == correct_index per question.
                evaluations=[],
                updated_boundary_map={
                    q.strand_id: {
                        "floor": f"mock floor for {q.strand_id}",
                        "ceiling": f"mock ceiling for {q.strand_id}",
                        "gap_type": "closed",
                    }
                    for q in _MOCK_PROBE_BATCH
                },
                gap_summary="Mock gap summary: every mock strand bracketed in one batch.",
            )
        ),
        # Plan: a fixed canned plan that passes validate_plan.
        ResearchOut: _always(
            ResearchOut(
                unconditional_truths=["Mock unconditional truth."],
                core_concepts=["Mock core concept"],
                standard_framing="Mock standard framing.",
                common_gotchas=["Mock common gotcha."],
            )
        ),
        DesignOut: _always(
            DesignOut(
                steps=[
                    StepDraft(
                        title="Mock Step",
                        description="A single mock plan step.",
                        depends_on=[],
                        depth=1,
                    )
                ]
            )
        ),
        RenderOut: _always(
            RenderOut(
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
            )
        ),
        # Language detection: English.
        _LanguageDetectionOut: _always(_LanguageDetectionOut(language="en")),
    }


# --- Mock chat model (graph/LLM seam) ---


class MockChatModel(BaseChatModel):
    """Canned pre-material LLM for MOCK_LLM mode (#118).

    - ``with_structured_output(schema)`` yields the fixed valid instance for
      the pre-material schemas (Clarify / Probe / Plan + language detection).
      The Clarify pair is the exception in kind, not in degree: it is a scripted
      run — a round of questions, then the narrowed goal — so which step answers
      is read out of the prompt the node sent (see :data:`_MOCK_CLARIFY_SCRIPT`).
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
        # ``**_`` because every node now binds a per-phase reasoning budget onto
        # its call (#194) and LangGraph hands those kwargs through to whatever a
        # structured-output binding ends up being. The canned answer is the same
        # whatever the model was asked to think with; the message list is what
        # the Clarify script reads its round from.
        return RunnableLambda(
            lambda model_input, **_: make_canned_output(schema, model_input)
        )  # type: ignore[arg-type]
