"""Tests for core/blocking.py — awaiting an awaitable from a sync caller."""

from __future__ import annotations

import asyncio

import pytest

from core.blocking import await_blocking


async def _eventual_answer() -> str:
    await asyncio.sleep(0)
    return "answered"


def test_it_returns_the_awaitable_result():
    assert await_blocking(_eventual_answer()) == "answered"


def test_it_works_inside_an_already_running_event_loop():
    """The reason this is not a bare asyncio.run (which raises here)."""

    def sync_node() -> str:
        return await_blocking(_eventual_answer())

    async def graph_runner() -> str:
        return sync_node()

    assert asyncio.run(graph_runner()) == "answered"


def test_the_awaitable_error_reaches_the_caller():
    async def _fails() -> None:
        raise ConnectionError("server is down")

    with pytest.raises(ConnectionError):
        await_blocking(_fails())
