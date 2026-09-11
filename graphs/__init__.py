"""LangGraph checkpointer, run config, and compiled graphs."""

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from llm import llm

from .clarify import build_clarify_graph
from .material import build_material_graph
from .plan import build_plan_graph
from .probe import build_probe_graph

# One in-memory checkpoint store shared by every graph (MVP).
# Lives for the process lifetime; lost on restart — same as the domain DB.
checkpointer = MemorySaver()

# --- Compiled graphs ---

clarify_graph = build_clarify_graph(llm, checkpointer=checkpointer)
probe_graph = build_probe_graph(llm, checkpointer=checkpointer)
plan_graph = build_plan_graph(llm, checkpointer=checkpointer)

# The Material graph is one-shot per step with no interrupts, so it needs no
# checkpointer — resume state lives in the domain DB (one StepMaterial row per
# step), keeping the two stores independent.
material_graph = build_material_graph(llm)


def graph_config(session_id: str, graph: str) -> RunnableConfig:
    """Build the run config for a (session, graph) pair.

    One distinct checkpoint thread per (session, graph). A session runs the
    Clarify, Probe, and Plan graphs, so they must NOT share a thread — each
    gets its own namespace, e.g. "abc123:clarify", "abc123:probe",
    "abc123:plan".
    """
    return {"configurable": {"thread_id": f"{session_id}:{graph}"}}
