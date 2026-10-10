"""The external-tool boundary: capabilities that live outside this process (#166).

An **external tool** is one this app does not implement — its capability lives
in another service, and it is reached through that service's MCP server. This
module is the only place that crosses the boundary. Internal tools (e.g. slide
JSX validation) are plain in-process functions and never come through here: the
MCP boundary covers external tools only.

What is external today is one thing — Tavily web search — and it is bound to
the plan graph's ``research_topic`` node alone (#166). Nothing else is handed
these tools: ``graphs/__init__.py`` passes this loader to ``build_plan_graph``,
which passes it to that one node.

Every path here is optional-by-design. Mock mode (#118), a missing
``TAVILY_API_KEY``, or a Tavily server that cannot be reached all give an empty
list, which the graphs read as "no external search — answer as before". Search
can degrade a plan's grounding; it must never fail a request.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from functools import lru_cache

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from core.blocking import await_blocking
from core.mock_llm import is_mock_mode

logger = logging.getLogger(__name__)

# Tavily hosts its own MCP server, so the app needs no local MCP process and no
# node runtime in the image — only the adapter and an egress rule.
DEFAULT_TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"

# The name this app gives the server in its own MCP configuration.
_TAVILY_SERVER = "tavily"

# #166 binds Tavily's *search*. Its hosted server also serves
# ``tavily_research`` — a heavier, rate-limited research capability this app
# has not asked for — so the match is on the exact search tool name, not on the
# substring "search". A name change shows up in the log below, not silently.
_SEARCH_TOOL_NAMES = frozenset({"tavily_search"})

# A search is a step inside plan generation, which the learner is waiting on:
# bound the MCP round trip rather than let it hang the request.
_MCP_TIMEOUT_SECONDS = 30.0

# What a graph node is given to reach external search: a loader, not a live
# connection. Calling it is what performs the (cached, fail-soft) MCP handshake.
SearchToolLoader = Callable[[], Sequence[BaseTool]]


def is_external_search_enabled() -> bool:
    """True when external search may be used: not mock mode, and a Tavily key.

    Mock mode (#118) is always false — a mock run makes no external calls.
    """
    return not is_mock_mode() and bool(os.getenv("TAVILY_API_KEY", "").strip())


def external_search_tools() -> list[BaseTool]:
    """Tavily's search tools, adapted from MCP to LangChain tools.

    An empty list means "no external search", and every caller treats it that
    way: mock mode, no key, or an unreachable server. Loading is lazy (first
    use, not import time) and a successful load is reused; a failure is not
    cached, so the next plan generation tries again.
    """
    if not is_external_search_enabled():
        return []

    api_key = os.environ["TAVILY_API_KEY"].strip()
    url = os.getenv("TAVILY_MCP_URL", DEFAULT_TAVILY_MCP_URL).strip()
    try:
        return list(_load_search_tools(url, api_key))
    except Exception:  # an unreachable Tavily must not be a failed request
        logger.warning(
            "Tavily MCP search is unavailable; this plan will not be grounded",
            exc_info=True,
        )
        return []


# Cached by (url, key) so a key rotation or an .env change is a different entry,
# and so tests that use different keys cannot see each other's cache.
@lru_cache(maxsize=4)
def _load_search_tools(url: str, api_key: str) -> tuple[BaseTool, ...]:
    """One ``tools/list`` handshake against the Tavily MCP server.

    The returned tools open their own session per call (that is how the
    adapter works), so nothing here holds a connection open between searches.

    The hosted server answers ``tools/list`` even when the key is wrong, so a
    bad key surfaces as an error result from the search itself — not as a load
    failure here. A search that fails is never fatal: the graph side hands the
    error back to the model (see :func:`graphs.common.structured_invoke_with_tools`).
    """
    client = MultiServerMCPClient(
        {
            _TAVILY_SERVER: {
                "transport": "streamable_http",
                "url": url,
                "headers": {"Authorization": f"Bearer {api_key}"},
                "timeout": _MCP_TIMEOUT_SECONDS,
            }
        }
    )
    served = await_blocking(client.get_tools(server_name=_TAVILY_SERVER))
    search_tools = tuple(t for t in served if t.name.lower() in _SEARCH_TOOL_NAMES)
    if not search_tools:
        logger.warning(
            "Tavily MCP server offered no search tool (got: %s)",
            ", ".join(t.name for t in served) or "nothing",
        )
    return search_tools
