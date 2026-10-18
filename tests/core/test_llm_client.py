"""Tests for core/llm.py — the client's own request policy.

Seam: ``core.llm.llm`` re-built from a controlled environment. Its timeout, its
retry count, and the output cap a call inherits when it states none are the
difference between a stalled model and a request that eventually answers, and all
three were left to the SDK's defaults: 600 seconds, retried twice — up to half an
hour of a learner waiting on a call that was never going to come back.

Every assertion rebuilds the client rather than reading the one the test run
imported: ``core/llm.py`` runs ``load_dotenv()`` when it is imported, so a
developer's ``.env`` would otherwise be part of what a test asserts.
"""

from __future__ import annotations

import importlib
import logging

import pytest

import core.llm as core_llm

# Every name the client reads, and the values the module falls back to. Stated
# as literals here: the defaults are the policy, and a test that re-derived them
# from the module would prove nothing about them.
_CLIENT_SETTINGS = (
    "LLM_REQUEST_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLM_MAX_OUTPUT_TOKENS",
)


@pytest.fixture
def client_built(monkeypatch):
    """Re-import ``core.llm`` with only what a test names in the environment.

    An unnamed client setting is set to empty, which ``core/settings.py`` reads as
    unset — the way a machine with no ``.env`` would. The real client is put back
    afterwards, because every other module holds the object the first import made.
    """

    def build(**settings: str):
        for name in _CLIENT_SETTINGS:
            monkeypatch.setenv(name, "")
        for name, value in settings.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(core_llm)

    yield build
    importlib.reload(core_llm)


class TestRequestPolicy:
    def test_a_request_gives_up_well_before_the_sdk_default(self, client_built):
        """The SDK default is 600s; this backend answers a normal call in seconds."""
        assert client_built().llm.request_timeout == 120

    def test_a_dead_client_is_not_asked_three_times(self, client_built):
        assert client_built().llm.max_retries == 2

    def test_a_call_that_states_no_cap_gets_one(self, client_built):
        """Every stage call states a cap (core/reasoning.py); this is the ceiling
        for anything that does not, and it is not the SDK's "as much as fits"."""
        assert client_built().llm.max_tokens == 65536

    def test_the_policy_is_configurable_without_a_code_change(self, client_built):
        """An experiment at reasoning_effort=high is exactly the case that wants a
        longer timeout, and LLM_MAX_RETRIES=0 the one that wants a failure now."""
        rebuilt = client_built(
            LLM_REQUEST_TIMEOUT_SECONDS="45",
            LLM_MAX_RETRIES="0",
            LLM_MAX_OUTPUT_TOKENS="8192",
        )
        assert rebuilt.llm.request_timeout == 45
        assert rebuilt.llm.max_retries == 0
        assert rebuilt.llm.max_tokens == 8192

    @pytest.mark.parametrize("value", ["120", "30.5"])
    def test_seconds_are_accepted_as_written(self, client_built, value):
        assert client_built(
            LLM_REQUEST_TIMEOUT_SECONDS=value
        ).llm.request_timeout == float(value)

    def test_a_value_that_is_not_a_number_keeps_the_default(self, client_built, caplog):
        """A typo in a knob is not a reason the app cannot start (#194 review)."""
        with caplog.at_level(logging.WARNING, logger="core.settings"):
            rebuilt = client_built(LLM_REQUEST_TIMEOUT_SECONDS="ten minutes")
        assert rebuilt.llm.request_timeout == 120
        assert "ten minutes" in caplog.text

    def test_retuning_one_of_these_is_a_restart_not_a_call(self, client_built):
        """Unlike the per-call knobs in core/reasoning.py and core/sampling.py,
        these belong to the client object, which is built once at startup — so
        changing one means starting the app again, and a test has to re-import it
        to see the effect."""
        before = client_built().llm
        after = client_built(LLM_MAX_RETRIES="0").llm
        assert before is not after
        assert after.max_retries == 0
