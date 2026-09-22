"""LangGraph checkpointer, run config, and compiled graphs."""

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from core.llm import llm
from core.mock_llm import MockChatModel, is_mock_mode

from .clarify import build_clarify_graph
from .material import build_material_graph
from .plan import build_plan_graph
from .probe import build_probe_graph

# One in-memory checkpoint store shared by every graph (MVP).
# Lives for the process lifetime; lost on restart — same as the domain DB.
checkpointer = MemorySaver()

# --- LLM seam (#118) ---

# The pre-material phases (Clarify, Probe, Plan) run on the pre-material
# LLM: the canned MockChatModel in mock mode, the real llm otherwise. The
# material graph is always built with the real llm — mock mode simply never
# invokes it.
if is_mock_mode():
    pre_material_llm = MockChatModel()
else:
    pre_material_llm = llm

# --- Compiled graphs ---

clarify_graph = build_clarify_graph(pre_material_llm, checkpointer=checkpointer)
probe_graph = build_probe_graph(pre_material_llm, checkpointer=checkpointer)
plan_graph = build_plan_graph(pre_material_llm, checkpointer=checkpointer)

# The Material graph uses the checkpointer for per-step in-process resume
# (one thread per step: "session_id:material:step_id"). On server restart the
# checkpoint is empty and the domain DB (one StepMaterial row per step)
# provides step-granularity skip.
material_graph = build_material_graph(llm, checkpointer=checkpointer)


def graph_config(session_id: str, graph: str) -> RunnableConfig:
    """Build the run config for a (session, graph) pair.

    One distinct checkpoint thread per (session, graph). A session runs the
    Clarify, Probe, and Plan graphs, so they must NOT share a thread — each
    gets its own namespace, e.g. "abc123:clarify", "abc123:probe",
    "abc123:plan".
    """
    return {"configurable": {"thread_id": f"{session_id}:{graph}"}}
