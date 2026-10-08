"""Shared test configuration for the whole suite.

Probe tuning knobs (``MAX_PROBE_QUESTIONS``, ``PROBE_BATCH_SIZE``) are read
by ``graphs.probe.nodes`` at import time. A developer's local ``.env``
(loaded via ``load_dotenv()`` in ``llm.py`` during test imports) must not
change test behavior, so the code defaults are pinned *before* any test
module is imported — same strategy as ``DATABASE_URL`` in
``tests/routers/conftest.py``.
"""

from __future__ import annotations

import importlib
import os

import pytest

os.environ["MOCK_LLM"] = "false"
# core/llm.py constructs a real ChatOpenAI at import time whenever MOCK_LLM is off,
# and the SDK rejects an empty ``api_key`` — so on a machine with no ``.env``
# (a CI runner, a fresh clone), *collection itself* aborts with
# ``OpenAIError: Missing credentials`` before a single test runs. No test reaches
# the network; the client only has to construct. Pinning a non-secret placeholder
# makes the suite hermetic: a laptop and a CI runner now collect the same tests from
# the same environment, which is the whole point of this block. The one test that
# asserts startup with *no* credential deletes the variable itself (see
# tests/core/test_mock_llm.py).
os.environ["OPENAI_API_KEY"] = "test-only-placeholder-not-a-secret"
# Allow the prompt loader to fall back to public stubs when the private
# ``prompts`` submodule is not checked out (e.g. a fresh public clone).
os.environ.setdefault("LA_PROMPTS_STUB", "1")
os.environ["MAX_PROBE_QUESTIONS"] = "10"
os.environ["PROBE_BATCH_SIZE"] = "3"


def _real_prompts_available() -> bool:
    """True if the private ``prompts`` submodule is checked out."""
    try:
        importlib.import_module("prompts.probe")
        return True
    except ModuleNotFoundError:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip prompt-content tests when the private submodule is absent.

    The public repo must stay self-sufficient: on a fresh public clone the
    ``prompts`` submodule is not initialized, so tests that assert on the
    real prompt text are skipped (they run when the submodule is present).
    """
    if _real_prompts_available():
        return
    skip = pytest.mark.skip(reason="requires the private prompts submodule")
    for item in items:
        if "real_prompts" in item.keywords:
            item.add_marker(skip)
