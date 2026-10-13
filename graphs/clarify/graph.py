"""Clarify graph: iteratively narrow a possibly-broad goal into a precise,
learnable ``narrowed_goal``.

The graph loops on user clarification until the goal is specific enough (or
the safety cap is hit), then exits with ``narrowed_goal``.

Nodes:
- ``assess_goal``        — judge whether the goal is specific enough; emit
                           ``narrowed_goal`` when it is (or the cap is hit).
- ``generate_questions`` — produce 1-3 targeted clarifying questions.
- ``wait_for_answer``    — pause (interrupt) and hand the questions to the
                           client; resumes with the user's answer.
- ``refine_goal``        — fold the answer into a tighter ``working_goal``
                           and update the open dimensions.

Control flow:
    assess_goal --(specific)--> END
    assess_goal --(too broad)--> generate_questions --> wait_for_answer
        ^                                                        |
        +------------------- refine_goal <----------------------+

State (``working_goal``, ``open_dimensions``, ``round_count``) lives in the
checkpointer across rounds, keyed by ``thread_id = f"{session_id}:clarify"``.

Each LLM node is wrapped in :func:`graphs.common.timed_node` so its wall time is
recorded (#194); ``wait_for_answer`` is not, because the time it spans is the
learner's.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from ..common import timed_node
from .nodes import (
    make_assess_goal,
    make_generate_questions,
    make_refine_goal,
    make_wait_for_answer,
)
from .state import ClarifyState


def build_clarify_graph(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver | None = None
):
    def route_after_assess(state: ClarifyState) -> str:
        return END if state.get("narrowed_goal") else "generate_questions"

    def round_context(state: ClarifyState) -> dict:
        """Which round of the clarify loop a recorded stage belongs to."""
        return {"round_count": state.get("round_count", 0)}

    graph = StateGraph(ClarifyState)
    graph.add_node(
        "assess_goal",
        timed_node("assess_goal", make_assess_goal(llm), round_context),
    )
    graph.add_node(
        "generate_questions",
        timed_node("generate_questions", make_generate_questions(llm), round_context),
    )
    # Not timed: an interrupt spans the learner's answer, not this service's work.
    graph.add_node("wait_for_answer", make_wait_for_answer(llm))
    graph.add_node(
        "refine_goal",
        timed_node("refine_goal", make_refine_goal(llm), round_context),
    )
    graph.set_entry_point("assess_goal")
    graph.add_conditional_edges("assess_goal", route_after_assess)
    graph.add_edge("generate_questions", "wait_for_answer")
    graph.add_edge("wait_for_answer", "refine_goal")
    graph.add_edge("refine_goal", "assess_goal")
    return graph.compile(checkpointer=checkpointer)
