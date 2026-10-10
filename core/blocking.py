"""Waiting for an awaitable from a synchronous caller.

The graphs run their nodes synchronously, and the tools that matter to them —
the MCP-adapted external tools (#166) — are async-only: the adapter gives them
no synchronous path. This helper is how a synchronous node waits for one.

A bare :func:`asyncio.run` would do today (the request handlers are plain
``def``, so they run in a worker thread with no event loop), but it fails
inside a caller that already has a running loop. The helper does not care, so
the day a graph runs asynchronously nothing here has to be found and fixed —
which matters because every consumer of this helper treats a failure as
"degrade, carry on", and a silent ``RuntimeError`` would hide that.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def await_blocking[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion on its own event loop and return its result.

    The loop is created in a worker thread and closed again when the awaitable
    settles: nothing outlives the call, and the caller may be inside a loop of
    its own.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
