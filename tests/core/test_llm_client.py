"""Tests for core/llm.py — the client's request policy (#194).

Seam: the constructed ``core.llm.llm``. Its timeout and retry count are the
difference between a stalled model and a request that eventually answers, and
both were left to the SDK's defaults: 600 seconds, retried twice — up to half an
hour of a learner waiting on a call that was never going to come back.
"""

from __future__ import annotations

import importlib

import pytest

import core.llm as core_llm
from core.llm import llm


class TestRequestPolicy:
    def test_a_request_gives_up_well_before_the_sdk_default(self):
        """The SDK default is 600s; this backend answers a normal call in ~10s."""
        assert llm.request_timeout == 120

    def test_a_dead_client_is_not_asked_three_times(self):
        assert llm.max_retries == 2

    def test_the_policy_is_configurable_without_a_code_change(self, monkeypatch):
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "45")
        monkeypatch.setenv("LLM_MAX_RETRIES", "0")
        try:
            reloaded = importlib.reload(core_llm)
            assert reloaded.llm.request_timeout == 45
            assert reloaded.llm.max_retries == 0
        finally:
            monkeypatch.undo()
            importlib.reload(core_llm)

    def test_a_value_that_is_not_a_number_uses_the_default(self, monkeypatch, caplog):
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "ten minutes")
        try:
            with caplog.at_level("WARNING", logger="core.llm"):
                reloaded = importlib.reload(core_llm)
            assert reloaded.llm.request_timeout == 120
            assert "ten minutes" in caplog.text
        finally:
            monkeypatch.undo()
            importlib.reload(core_llm)

    @pytest.mark.parametrize("value", ["120", "30.5"])
    def test_seconds_are_accepted_as_written(self, monkeypatch, value):
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", value)
        try:
            reloaded = importlib.reload(core_llm)
            assert reloaded.llm.request_timeout == float(value)
        finally:
            monkeypatch.undo()
            importlib.reload(core_llm)
