"""Node factories for the clarify graph."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.types import interrupt

from core.language import DEFAULT_LANGUAGE, language_instruction

from ..common import structured_invoke
from .prompts import (
    ASSESS_GOAL_CAP_REACHED,
    ASSESS_GOAL_SYSTEM,
    GENERATE_QUESTIONS_SYSTEM,
    REFINE_GOAL_SYSTEM,
)
from .schemas import AssessOut, QuestionsOut, RefineOut
from .state import ClarifyState

MAX_CLARIFY_ROUNDS = 3


def make_assess_goal(llm: BaseChatModel):
    """Judge whether the goal is specific enough; emit narrowed_goal when it is."""

    def assess_goal(state: ClarifyState) -> dict:
        at_cap = state.get("round_count", 0) >= MAX_CLARIFY_ROUNDS
        system = ASSESS_GOAL_SYSTEM
        if at_cap:
            system += ASSESS_GOAL_CAP_REACHED
        # narrowed_goal is user-facing — produce it in the learner's language.
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)
        out = structured_invoke(
            llm,
            AssessOut,
            system,
            (
                f"Original goal: {state.get('goal', '')}\n"
                f"Working goal: {state.get('working_goal', '')}\n"
                f"Open dimensions: {state.get('open_dimensions') or 'none'}\n"
                f"Clarification round: {state.get('round_count', 0)}"
            ),
        )
        if out.verdict == "specific" or at_cap:
            return {"narrowed_goal": out.narrowed_goal, "open_dimensions": []}
        return {"open_dimensions": out.open_dimensions}

    return assess_goal


def make_generate_questions(llm: BaseChatModel):
    """Produce 1-3 targeted clarifying questions."""

    def generate_questions(state: ClarifyState) -> dict:
        system = GENERATE_QUESTIONS_SYSTEM
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)
        out = structured_invoke(
            llm,
            QuestionsOut,
            system,
            (
                f"Working goal: {state.get('working_goal', '')}\n"
                f"Open dimensions: "
                f"{state.get('open_dimensions') or 'the vaguest part of the goal'}"
            ),
        )
        return {"clarifying_questions": out.questions}

    return generate_questions


def make_wait_for_answer(llm: BaseChatModel):
    """Pause (interrupt); resumes with the learner's answer."""

    def wait_for_answer(state: ClarifyState) -> dict:
        answer = interrupt(None)
        return {"last_answer": answer["answer"]}

    return wait_for_answer


def make_refine_goal(llm: BaseChatModel):
    """Fold the answer into a tighter working_goal and update open dimensions."""

    def refine_goal(state: ClarifyState) -> dict:
        system = REFINE_GOAL_SYSTEM
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)
        out = structured_invoke(
            llm,
            RefineOut,
            system,
            (
                f"Working goal: {state.get('working_goal', '')}\n"
                f"Open dimensions: {state.get('open_dimensions') or 'none'}\n"
                f"Learner's answer: {state.get('last_answer', '')}"
            ),
        )
        return {
            "working_goal": out.working_goal,
            "open_dimensions": out.open_dimensions,
            "round_count": state.get("round_count", 0) + 1,
        }

    return refine_goal
