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
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

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

    graph = StateGraph(ClarifyState)
    graph.add_node("assess_goal", make_assess_goal(llm))
    graph.add_node("generate_questions", make_generate_questions(llm))
    graph.add_node("wait_for_answer", make_wait_for_answer(llm))
    graph.add_node("refine_goal", make_refine_goal(llm))
    graph.set_entry_point("assess_goal")
    graph.add_conditional_edges("assess_goal", route_after_assess)
    graph.add_edge("generate_questions", "wait_for_answer")
    graph.add_edge("wait_for_answer", "refine_goal")
    graph.add_edge("refine_goal", "assess_goal")
    return graph.compile(checkpointer=checkpointer)
