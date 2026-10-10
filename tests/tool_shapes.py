"""Test doubles shaped like real MCP-adapted tools.

The MCP adapter builds its tools async-only — no synchronous path, with
``response_format="content_and_artifact"`` — and that shape is what the graph
side has to cope with (#166). A stub that shares the shape is what keeps
tool-calling tests honest about the real thing rather than about a convenient
fake.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool


def mcp_shaped_tool(
    name: str, calls: list[Any] | None = None, outcome: object = "searched"
) -> StructuredTool:
    """A tool named ``name`` that records the queries it is called with.

    ``outcome`` is the text it returns, or an exception to raise instead — the
    shape of a call for a tool that is not there.
    """

    async def _call(query: str) -> tuple[list[dict], dict]:
        if calls is not None:
            calls.append(query)
        if isinstance(outcome, BaseException):
            raise outcome
        # The adapter returns (content blocks, raw artifact) for a tool declared
        # response_format="content_and_artifact".
        return [{"type": "text", "text": str(outcome)}], {}

    return StructuredTool(
        name=name,
        description=f"search the web for {name}",
        args_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        coroutine=_call,
        response_format="content_and_artifact",
    )
