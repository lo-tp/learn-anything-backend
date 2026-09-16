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
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from .nodes import (
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
        return "generate_batch" if state.get("_decision") == "continue" else END

    graph = StateGraph(ProbeState)
    graph.add_node("decompose_strands", make_decompose_strands(llm))
    graph.add_node("generate_batch", make_generate_batch(llm))
    graph.add_node("wait_for_answers", make_wait_for_answers(llm))
    graph.add_node("evaluate_batch", make_evaluate_batch(llm))
    graph.add_node("decide_next", make_decide_next(llm))

    graph.set_entry_point("decompose_strands")
    graph.add_edge("decompose_strands", "generate_batch")
    graph.add_edge("generate_batch", "wait_for_answers")
    graph.add_edge("wait_for_answers", "evaluate_batch")
    graph.add_edge("evaluate_batch", "decide_next")
    graph.add_conditional_edges("decide_next", route_after_decide)

    return graph.compile(checkpointer=checkpointer)
