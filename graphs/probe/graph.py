"""Probe graph: adaptive batched-questioning loop that brackets the learner's boundary.

The graph loops: generate a batch of questions → interrupt (hand to client) →
evaluate the whole batch of answers → decide whether to continue or exit with
the boundary_map. Adaptivity is preserved *between* batches via the updated
boundary map; within a batch, questions are independent.

Nodes:
- ``decompose_strands`` — one-shot: enumerate the prerequisite strands for the
  goal and seed ``boundary_map`` with the full set.
- ``generate_batch``    — produce a batch of MCQs targeting the unbracketed
  strands (size = min(PROBE_BATCH_SIZE, unbracketed_strand_count)).
- ``wait_for_answers``  — pause (interrupt); resumes with
  {"answers": [{question_id, selected_index}, ...]}.
- ``evaluate_batch``    — mark each answer correct/incorrect, update boundary
  estimates for every strand touched.
- ``decide_next``       — deterministic: all strands bracketed or cap hit?

Control flow:
    decompose_strands → generate_batch → wait_for_answers (interrupt)
        → evaluate_batch → decide_next
            → "continue" → generate_batch (loop)
            → "done"     → END (output boundary_map)

Each LLM node is wrapped in :func:`graphs.common.timed_node` so its wall time is
recorded (#194). Two nodes are not: ``wait_for_answers``, because the time it
spans is the learner's, and ``decide_next``, which is deterministic and asks
nothing of the model — a row for it would be noise.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from ..common import timed_node
from .nodes import (
    DECOMPOSE_STRANDS,
    EVALUATE_BATCH,
    GENERATE_BATCH,
    make_decide_next,
    make_decompose_strands,
    make_evaluate_batch,
    make_generate_batch,
    make_wait_for_answers,
)
from .state import ProbeState


def build_probe_graph(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver | None = None
):
    def route_after_decide(state: ProbeState) -> str:
        return GENERATE_BATCH if state.get("_decision") == "continue" else END

    def batch_context(state: ProbeState) -> dict:
        """Which batch of questions a recorded stage belongs to."""
        return {"question_count": state.get("question_count", 0)}

    graph = StateGraph(ProbeState)
    graph.add_node(
        DECOMPOSE_STRANDS,
        timed_node(DECOMPOSE_STRANDS, make_decompose_strands(llm)),
    )
    graph.add_node(
        GENERATE_BATCH,
        timed_node(GENERATE_BATCH, make_generate_batch(llm), batch_context),
    )
    # Not timed: an interrupt spans the learner's answers, not this service's work.
    graph.add_node("wait_for_answers", make_wait_for_answers(llm))
    graph.add_node(
        EVALUATE_BATCH,
        timed_node(EVALUATE_BATCH, make_evaluate_batch(llm), batch_context),
    )
    graph.add_node("decide_next", make_decide_next(llm))

    # A model stage is named by its qualified stage name throughout — the same
    # string core/reasoning.py's budget table is keyed by, so no stage name here
    # is spelled twice. The interrupt and the routing decision are not stages.
    graph.set_entry_point(DECOMPOSE_STRANDS)
    graph.add_edge(DECOMPOSE_STRANDS, GENERATE_BATCH)
    graph.add_edge(GENERATE_BATCH, "wait_for_answers")
    graph.add_edge("wait_for_answers", EVALUATE_BATCH)
    graph.add_edge(EVALUATE_BATCH, "decide_next")
    graph.add_conditional_edges("decide_next", route_after_decide)

    return graph.compile(checkpointer=checkpointer)
