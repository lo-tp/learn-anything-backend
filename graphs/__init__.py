"""Shared LangGraph checkpointer and per-session config helpers.

One in-memory checkpoint store shared by every graph (MVP). Lives for the
process lifetime; lost on restart — same as the domain DB.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()


def graph_config(session_id: str, graph: str) -> RunnableConfig:
    """Build the run config for a (session, graph) pair.

    One distinct checkpoint thread per (session, graph). A session runs the
    Clarify, Probe, and Plan graphs, so they must NOT share a thread — each
    gets its own namespace, e.g. "abc123:clarify", "abc123:probe",
    "abc123:plan".
    """
    return {"configurable": {"thread_id": f"{session_id}:{graph}"}}
