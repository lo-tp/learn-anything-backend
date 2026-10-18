"""Shared test configuration for the whole suite.

Probe tuning knobs (``MAX_PROBE_QUESTIONS``, ``PROBE_BATCH_SIZE``) are read
by ``graphs.probe.nodes`` at import time. A developer's local ``.env``
(loaded via ``load_dotenv()`` in ``llm.py`` during test imports) must not
change test behavior, so the code defaults are pinned *before* any test
module is imported — same strategy as ``DATABASE_URL`` in
``tests/routers/conftest.py``. The model-call knobs (``REASONING_EFFORT*``,
``LLM_MAX_OUTPUT_TOKENS*``, and every name in
``core.sampling.SAMPLING_ENV_NAMES``) are read *per call* instead, and their
names only appear in the environment once ``load_dotenv`` has run, so they are
blanked at session start — see
:meth:`_stage_defaults_win_over_local_env`.
"""

from __future__ import annotations

import importlib
import os

import pytest

from core.model_answers import drain_model_recoveries
from core.sampling import SAMPLING_ENV_NAMES

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


@pytest.fixture(autouse=True, scope="session")
def _stage_defaults_win_over_local_env():
    """Blank every model-call knob the local ``.env`` injected.

    ``core/settings.py`` reads the reasoning-policy knobs
    (``REASONING_EFFORT*`` / ``LLM_MAX_OUTPUT_TOKENS*``) and the decoding knobs
    (every name in :data:`core.sampling.SAMPLING_ENV_NAMES`) per call, each one
    also readable with a stage appended to its name, and ``load_dotenv()`` (run
    when a test module imports ``core.llm``) adds those names *after* the
    pinning above could blank them — a module-level sweep would see nothing to
    blank. Blanking here, at session start (all imports done, no test body yet),
    makes the defaults in ``core/reasoning.py``, ``core/sampling.py`` and
    ``graphs/material/sandbox.py`` what the suite asserts whatever a developer's
    ``.env`` is carrying (e.g. a raised cap or a raised temperature
    mid-experiment). ``core/settings.py`` treats an empty value as unset. A test
    that wants an override sets one with ``monkeypatch``, which runs after this
    fixture.
    """
    pinned = ("REASONING_EFFORT", "LLM_MAX_OUTPUT_TOKENS", *SAMPLING_ENV_NAMES)
    for name in list(os.environ):
        if name.startswith(pinned):
            os.environ[name] = ""


@pytest.fixture(autouse=True)
def _no_answer_recovery_carries_over():
    """Drain the recovery accumulator around every test.

    ``core.model_answers`` attributes a recovered answer to the stage row written
    next — correct when one task runs one stage, and wrong between two tests
    sharing one thread, where a record left by one would be blamed on the next.
    """
    drain_model_recoveries()
    yield
    drain_model_recoveries()


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
