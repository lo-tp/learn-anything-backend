"""Tests for core/settings.py — how a tunable setting is read (#194 review).

Seam: the two functions, because the rule they encode is the one that matters:
which name wins, and what happens to a value this app cannot use.
"""

from __future__ import annotations

import logging

from core.settings import enumerated_setting, numeric_setting


class TestFirstNameWins:
    """A per-call setting sits ahead of a global one — that ordering is the
    whole reason these functions take a list of names."""

    def test_the_first_name_that_is_set_is_the_one_used(self, monkeypatch):
        monkeypatch.setenv("GLOBAL_KNOB", "1")
        monkeypatch.setenv("PER_CALL_KNOB", "7")
        assert numeric_setting(("PER_CALL_KNOB", "GLOBAL_KNOB"), 0, int) == 7

    def test_a_name_that_is_only_whitespace_does_not_win(self, monkeypatch):
        monkeypatch.setenv("PER_CALL_KNOB", "   ")
        monkeypatch.setenv("GLOBAL_KNOB", "7")
        assert numeric_setting(("PER_CALL_KNOB", "GLOBAL_KNOB"), 0, int) == 7

    def test_nothing_set_gives_the_default(self, monkeypatch):
        monkeypatch.delenv("GLOBAL_KNOB", raising=False)
        assert numeric_setting(("PER_CALL_KNOB", "GLOBAL_KNOB"), 42, int) == 42


class TestAValueThatCannotBeUsed:
    def test_a_number_that_is_not_a_number_keeps_the_default(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "huge")
        with caplog.at_level(logging.WARNING):
            assert numeric_setting(("LLM_MAX_OUTPUT_TOKENS",), 3072, int) == 3072
        assert "LLM_MAX_OUTPUT_TOKENS" in caplog.text
        assert "huge" in caplog.text

    def test_a_value_outside_the_fixed_set_keeps_the_default(self, monkeypatch, caplog):
        monkeypatch.setenv("REASONING_EFFORT", "extra-high")
        with caplog.at_level(logging.WARNING):
            assert enumerated_setting(
                ("REASONING_EFFORT",), {"none", "minimal", "low"}, "low"
            ) == "low"
        assert "extra-high" in caplog.text

    def test_the_fixed_set_is_matched_whatever_case_it_arrives_in(self, monkeypatch):
        monkeypatch.setenv("REASONING_EFFORT", "  MINIMAL ")
        assert enumerated_setting(
            ("REASONING_EFFORT",), ("none", "minimal", "low"), "low"
        ) == "minimal"

    def test_a_float_setting_reads_a_float(self, monkeypatch):
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "30.5")
        assert (
            numeric_setting(("LLM_REQUEST_TIMEOUT_SECONDS",), 120.0, float) == 30.5
        )
