"""Tests for core/settings.py — how a tunable setting is read (#194 review).

Seam: the functions every tunable knob is read through, because the rule they
encode is the one that matters: which name wins, what the two names a per-stage
knob are, and what happens to a value this app cannot use — not a number, not
one of a fixed set, or outside the range a backend will accept.
"""

from __future__ import annotations

import logging

from core.settings import (
    bounded_number_setting,
    enumerated_setting,
    numeric_setting,
    stage_setting_names,
)


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


class TestAValueOutsideTheRange:
    """A number a backend would answer 400 to is the same failure as an unknown
    enum value: reported, named, and the default stands.

    The seam is the range, because that is the rule a caller cannot be trusted to
    remember — every sampling knob carries its own bounds in core/sampling.py.
    """

    def test_a_value_inside_the_range_is_the_value_used(self, monkeypatch):
        monkeypatch.setenv("LLM_TEMPERATURE", "0.8")
        assert (
            bounded_number_setting(
                ("LLM_TEMPERATURE",), None, float, minimum=0.0, maximum=2.0
            )
            == 0.8
        )

    def test_a_value_outside_the_range_keeps_the_default(self, monkeypatch, caplog):
        monkeypatch.setenv("LLM_TEMPERATURE", "5")
        with caplog.at_level(logging.WARNING):
            assert (
                bounded_number_setting(
                    ("LLM_TEMPERATURE",), 0.4, float, minimum=0.0, maximum=2.0
                )
                == 0.4
            )
        assert "LLM_TEMPERATURE" in caplog.text
        assert "5" in caplog.text
        assert "between 0.0 and 2.0" in caplog.text

    def test_a_floor_and_no_ceiling(self, monkeypatch, caplog):
        monkeypatch.setenv("LLM_TOP_K", "0")
        with caplog.at_level(logging.WARNING):
            assert (
                bounded_number_setting(("LLM_TOP_K",), 25, int, minimum=1) == 25
            )
        assert "at least 1" in caplog.text

    def test_a_ceiling_and_no_floor(self, monkeypatch, caplog):
        monkeypatch.setenv("LLM_PRESENCE_PENALTY", "9")
        with caplog.at_level(logging.WARNING):
            assert (
                bounded_number_setting(
                    ("LLM_PRESENCE_PENALTY",), None, float, maximum=2.0
                )
                is None
            )
        assert "at most 2.0" in caplog.text

    def test_a_default_of_none_means_send_nothing(self, monkeypatch):
        """The fallback every sampling knob has: the backend's own default."""
        assert (
            bounded_number_setting(("LLM_NOT_SET",), None, float, minimum=0.0) is None
        )

    def test_a_number_that_is_not_a_number_keeps_the_default(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("LLM_TOP_K", "many")
        with caplog.at_level(logging.WARNING):
            assert bounded_number_setting(("LLM_TOP_K",), 25, int, minimum=1) == 25
        assert "many" in caplog.text

    def test_the_first_name_that_is_set_still_wins(self, monkeypatch):
        monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
        monkeypatch.setenv("LLM_TEMPERATURE_PLAN_DESIGN_PLAN", "1.1")
        assert (
            bounded_number_setting(
                ("LLM_TEMPERATURE_PLAN_DESIGN_PLAN", "LLM_TEMPERATURE"),
                None,
                float,
                minimum=0.0,
                maximum=2.0,
            )
            == 1.1
        )


class TestAStageNameBecomesASettingName:
    """One rule for the two names a per-stage knob has, shared by every module
    that has knobs on a stage (core/reasoning.py, core/sampling.py)."""

    def test_the_dots_of_a_stage_name_become_underscores(self):
        assert stage_setting_names("LLM_TEMPERATURE", "plan.design_plan") == (
            "LLM_TEMPERATURE_PLAN_DESIGN_PLAN",
            "LLM_TEMPERATURE",
        )

    def test_the_per_stage_name_comes_first(self):
        per_stage, global_name = stage_setting_names(
            "REASONING_EFFORT", "probe.evaluate_batch"
        )
        assert per_stage == "REASONING_EFFORT_PROBE_EVALUATE_BATCH"
        assert global_name == "REASONING_EFFORT"
