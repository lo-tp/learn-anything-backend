"""Tests for core/external_tools.py — the external-tool (MCP) boundary (#166).

Acceptance criteria (issue #166):
- Tavily search is reachable through the MCP adapter.
- Mock mode stays green: with no external search configured, the boundary is a
  no-op that touches neither the network nor Tavily.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import patch

import pytest

from core.external_tools import external_search_tools
from tests.tool_shapes import mcp_shaped_tool


def _fake_client(tools: list[Any] | Exception) -> type:
    """Stand in for MultiServerMCPClient; ``get_tools`` returns or raises."""

    class _Client:
        def __init__(self, connections, **kwargs):
            self.connections = connections

        async def get_tools(self, *, server_name=None):
            if isinstance(tools, Exception):
                raise tools
            return list(tools)

    return _Client


# --- The no-op paths (mock mode, no key) ---


class TestSearchIsOffWithoutConfiguration:
    def test_mock_mode_has_no_external_tools(self, monkeypatch):
        """#166: mock mode stays green — no external tool, no MCP connection."""
        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-mock-mode-key")
        with patch("core.external_tools.MultiServerMCPClient") as client_cls:
            assert external_search_tools() == []
        client_cls.assert_not_called()

    def test_no_api_key_has_no_external_tools(self, monkeypatch):
        monkeypatch.delenv("MOCK_LLM", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with patch("core.external_tools.MultiServerMCPClient") as client_cls:
            assert external_search_tools() == []
        client_cls.assert_not_called()


# --- The configured path: Tavily through the MCP adapter ---


class TestSearchToolsThroughTheMcpAdapter:
    def test_tavily_search_is_reachable(self, monkeypatch):
        monkeypatch.delenv("MOCK_LLM", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-key-one")
        client = _fake_client([mcp_shaped_tool("tavily_search")])
        with patch("core.external_tools.MultiServerMCPClient", client):
            tools = external_search_tools()
        assert [t.name for t in tools] == ["tavily_search"]

    def test_only_the_search_tool_is_bound(self, monkeypatch):
        """#166 binds Tavily *search*; the rest stays unbound.

        ``tavily_search`` and ``tavily_research`` are what the hosted MCP
        server served when this was written (checked against it); research is a
        heavier, rate-limited capability this app has not asked for. The rest
        are the tools the self-hosted tavily-mcp package adds.
        """
        monkeypatch.delenv("MOCK_LLM", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-key-two")
        served = [
            mcp_shaped_tool("tavily_search"),
            mcp_shaped_tool("tavily_research"),
            mcp_shaped_tool("tavily_extract"),
            mcp_shaped_tool("tavily_crawl"),
            mcp_shaped_tool("tavily_map"),
        ]
        with patch("core.external_tools.MultiServerMCPClient", _fake_client(served)):
            tools = external_search_tools()
        assert [t.name for t in tools] == ["tavily_search"]

    def test_the_tavily_mcp_server_is_configured_with_the_key(self, monkeypatch):
        monkeypatch.delenv("MOCK_LLM", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-key-three")
        captured: dict = {}

        class _Client(_fake_client([])):  # type: ignore[misc]
            def __init__(self, connections, **kwargs):
                captured["connections"] = connections

        with patch("core.external_tools.MultiServerMCPClient", _Client):
            external_search_tools()

        connection = captured["connections"]["tavily"]
        assert connection["transport"] == "streamable_http"
        assert connection["url"] == "https://mcp.tavily.com/mcp/"
        assert connection["headers"] == {"Authorization": "Bearer tvly-key-three"}

    def test_mcp_url_is_overridable(self, monkeypatch):
        """The hosted endpoint is config, not a constant in the image."""
        monkeypatch.delenv("MOCK_LLM", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-key-four")
        monkeypatch.setenv("TAVILY_MCP_URL", "https://mcp.example.test/mcp/")
        captured: dict = {}

        class _Client(_fake_client([])):  # type: ignore[misc]
            def __init__(self, connections, **kwargs):
                captured["connections"] = connections

        with patch("core.external_tools.MultiServerMCPClient", _Client):
            external_search_tools()

        assert (
            captured["connections"]["tavily"]["url"] == "https://mcp.example.test/mcp/"
        )


# --- Search is optional: it must never break a plan ---


class TestSearchFailsSoft:
    def test_unreachable_server_yields_no_tools_and_says_so(self, monkeypatch, caplog):
        monkeypatch.delenv("MOCK_LLM", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-key-unreachable")
        client = _fake_client(ConnectionError("tavily is not listening"))
        with (
            patch("core.external_tools.MultiServerMCPClient", client),
            caplog.at_level(logging.WARNING, logger="core.external_tools"),
        ):
            assert external_search_tools() == []
        assert "unavailable" in caplog.text.lower()

    def test_a_failure_is_not_cached(self, monkeypatch):
        """The next plan generation retries a server that was down once."""
        monkeypatch.delenv("MOCK_LLM", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-key-retry")
        failures = iter([ConnectionError("down")])

        calls = {"n": 0}

        class _Client:
            def __init__(self, connections, **kwargs):
                pass

            async def get_tools(self, *, server_name=None):
                calls["n"] += 1
                if next(failures, None) is not None:
                    raise ConnectionError("down")
                return [mcp_shaped_tool("tavily_search")]

        with patch("core.external_tools.MultiServerMCPClient", _Client):
            assert external_search_tools() == []
            assert [t.name for t in external_search_tools()] == ["tavily_search"]
        assert calls["n"] == 2

    def test_a_successful_load_is_reused(self, monkeypatch):
        """One handshake per key, not one per plan generation."""
        monkeypatch.delenv("MOCK_LLM", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-key-cached")
        calls = {"n": 0}

        class _Client:
            def __init__(self, connections, **kwargs):
                pass

            async def get_tools(self, *, server_name=None):
                calls["n"] += 1
                return [mcp_shaped_tool("tavily_search")]

        with patch("core.external_tools.MultiServerMCPClient", _Client):
            first = external_search_tools()
            second = external_search_tools()
        assert calls["n"] == 1
        assert first[0].name == second[0].name == "tavily_search"

    def test_a_server_with_no_search_tool_says_so(self, monkeypatch, caplog):
        """A renamed tool is visible in the log, not silently absent."""
        monkeypatch.delenv("MOCK_LLM", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-key-no-search-tool")
        client = _fake_client([mcp_shaped_tool("tavily_web_lookup")])
        with (
            patch("core.external_tools.MultiServerMCPClient", client),
            caplog.at_level(logging.WARNING, logger="core.external_tools"),
        ):
            assert external_search_tools() == []
        assert "tavily_web_lookup" in caplog.text


def test_search_tools_are_callable_from_this_side():
    """The adapted tool answers through its LangChain interface."""
    tool = mcp_shaped_tool("tavily_search", outcome="how the field is framed")
    result = asyncio.run(tool.ainvoke({"query": "calculus pedagogy"}))
    assert result == [{"type": "text", "text": "how the field is framed"}]

    # The adapter gives these tools no synchronous path; that is why the app
    # side drives them through core.blocking.await_blocking.
    with pytest.raises(NotImplementedError):
        tool.invoke({"query": "calculus pedagogy"})
