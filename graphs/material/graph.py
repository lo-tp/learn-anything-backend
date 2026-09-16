"""Material graph: for one approved plan step, produce React JSX
slides, quiz questions, and a compact summary of the step's established
concepts.

Three-phase pipeline per step, run in dependency order by the driver in
``routers/plan.py``:

1. ``plan_slide_contents`` — one structured LLM call produces a content spec
   for each slide (3-7) of the step.
2. Per-slide generation loop — ``write_slide`` generates one slide's JSX
   from its content spec; ``compile_slide`` compiles it in the sandbox.
   A failed slide is regenerated (with the compile error as feedback) on
   its own, up to ``MAX_MATERIAL_ATTEMPTS`` attempts, without touching the
   other slides.
3. ``write_questions`` + ``summarize_step`` — quiz and step summary.

Resume lives in the graph checkpointer (per-step thread) for in-process
resilience; on server restart the domain DB (one committed ``StepMaterial``
row per step) provides step-granularity skip. See the ``generate_materials``
driver in ``routers/plan.py``.

State flows:
    START → plan_slide_contents → write_slide → compile_slide
        compile_slide --(success)-----------> pause_after_compile
        compile_slide --(retry)--------------> write_slide
        compile_slide --(exhausted)-----------> write_slide / write_questions
        pause_after_compile --(interrupt: driver saves slide)---> write_slide / write_questions
        write_questions → summarize_step → END
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from .nodes import (
    make_compile_slide,
    make_pause_after_compile,
    make_plan_slide_contents,
    make_summarize_step,
    make_write_questions,
    make_write_slide,
)
from .state import MaterialState


def build_material_graph(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver | None = None
):
    def route_after_compile(state: MaterialState) -> str:
        result = state.get("compile_result", "")
        if result == "success":
            return "pause_after_compile"
        if result == "retry":
            return "write_slide"
        # exhausted — skip this slide, move on
        slide_index = state.get("slide_index", 0)
        slide_contents = state.get("slide_contents") or []
        if slide_index < len(slide_contents):
            return "write_slide"
        return "write_questions"

    def route_after_pause(state: MaterialState) -> str:
        slide_index = state.get("slide_index", 0)
        slide_contents = state.get("slide_contents") or []
        if slide_index < len(slide_contents):
            return "write_slide"
        return "write_questions"

    # --- Build the graph ---

    graph = StateGraph(MaterialState)
    graph.add_node("plan_slide_contents", make_plan_slide_contents(llm))
    graph.add_node("write_slide", make_write_slide(llm))
    graph.add_node("compile_slide", make_compile_slide(llm))
    graph.add_node("pause_after_compile", make_pause_after_compile(llm))
    graph.add_node("write_questions", make_write_questions(llm))
    graph.add_node("summarize_step", make_summarize_step(llm))

    graph.add_edge("__start__", "plan_slide_contents")
    graph.add_edge("plan_slide_contents", "write_slide")
    graph.add_edge("write_slide", "compile_slide")
    graph.add_conditional_edges("compile_slide", route_after_compile)
    graph.add_conditional_edges("pause_after_compile", route_after_pause)
    graph.add_edge("write_questions", "summarize_step")
    graph.add_edge("summarize_step", END)

    return graph.compile(checkpointer=checkpointer)
