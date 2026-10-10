"""Plan graph: generate a learning plan, then loop on user adjustments until
approved.

The graph runs: research_topic (first pass only) → design_plan → render_plan
→ interrupt (show plan to user). On resume, the user either approves (END) or
adjusts (loop back to design_plan, skipping research).

Nodes:
- ``research_topic``   — *(first pass only)* scope the field: core concepts,
                         first principles, standard framing, common gotchas,
                         unconditional truths. It is the only node — in any
                         graph — that is given the external search tool (#166).
- ``design_plan``      — *(every pass)* build/refine the dependency-ordered
                         step list from the learner's boundary to the goal.
- ``render_plan``      — *(every pass)* format: prose summary, mermaid DAG,
                         assign step IDs, convert deps to IDs.
- ``wait_for_response``— pause (interrupt); resumes with
                         ``{"action": "approve"}`` or ``{"action": "adjust", "text": ...}``.

Control flow:
    START --(research is None)--> research_topic --> design_plan --> render_plan
    START --(research exists)   ------------------------------------------------+
                                                                              |
                                                          wait_for_response <--+
                                                                |
                                                     (user response)
                                                     /           \
                                                "approve"     "adjust"
                                                    |             |
                                                   END       (loop to design_plan)

State (``research``, ``current_plan``, ``pass_count``) lives in the
checkpointer across passes, keyed by ``thread_id = f"{session_id}:plan"``.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from core.external_tools import SearchToolLoader

from .nodes import (
    make_design_plan,
    make_render_plan,
    make_research_topic,
    make_wait_for_response,
)
from .state import PlanState


def build_plan_graph(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    search_tools: SearchToolLoader | None = None,
):
    """Compile the plan graph.

    ``search_tools`` is the external-search boundary (#166) and is handed to
    ``research_topic`` alone — no other node in this graph, and no other graph,
    is given it. Omitting it (or a loader that returns nothing) leaves the
    graph exactly as it was before external tools: the search path is a no-op.
    """

    def route_entry(state: PlanState) -> str:
        return "design_plan" if state.get("research") is not None else "research_topic"

    def route_after_response(state: PlanState) -> str:
        return END if state.get("approved") else "design_plan"

    graph = StateGraph(PlanState)
    graph.add_node("research_topic", make_research_topic(llm, search_tools))
    graph.add_node("design_plan", make_design_plan(llm))
    graph.add_node("render_plan", make_render_plan(llm))
    graph.add_node("wait_for_response", make_wait_for_response(llm))

    graph.add_conditional_edges("research_topic", lambda _: "design_plan")
    graph.add_edge("design_plan", "render_plan")
    graph.add_edge("render_plan", "wait_for_response")
    graph.add_conditional_edges("wait_for_response", route_after_response)
    graph.add_conditional_edges("__start__", route_entry)

    return graph.compile(checkpointer=checkpointer)
