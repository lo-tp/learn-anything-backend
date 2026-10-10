"""LangGraph checkpointer, run config, and compiled graphs."""

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from core.external_tools import external_search_tools
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

# External search (#166) reaches one node: the plan graph's research_topic.
# No other graph is handed a search tool, and internal tools never come through
# this boundary — they are in-process functions (see graphs/material).
# Mock mode (#118) is wired with none at all, so a mock run has no path to the
# network to forget about. The boundary checks the flag too, deliberately: this
# wiring is what makes it structural, the boundary's own check is what keeps any
# other caller honest. A real run with no Tavily key, or one where Tavily
# cannot be reached, gets no search and a plan built exactly as before.
plan_search_tools = None if is_mock_mode() else external_search_tools
plan_graph = build_plan_graph(
    pre_material_llm, checkpointer=checkpointer, search_tools=plan_search_tools
)

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
