"""Tests for core/reasoning.py — the per-phase reasoning budget (#194).

Seam: :func:`core.reasoning.phase_call_kwargs`, the one function a graph node
calls to get what it may think and how much it may say. Every value here is a
literal, not a recomputation of the module's table, because the table *is* the
policy and a test that mirrors it proves nothing.
"""

from __future__ import annotations

import logging

import pytest

from core.reasoning import (
    PHASES,
    REASONING_EFFORTS,
    max_output_tokens_for,
    phase_call_kwargs,
    reasoning_effort_for,
)


class TestPhaseBudgets:
    def test_every_phase_asks_for_an_effort_the_backend_accepts(self):
        """A value outside this set is a rejected request (HTTP 400)."""
        for phase in PHASES:
            assert reasoning_effort_for(phase) in REASONING_EFFORTS

    def test_every_phase_bounds_what_the_call_may_generate(self):
        """The point of #194: nothing may run to the client's 65536 cap."""
        for phase in PHASES:
            assert max_output_tokens_for(phase) <= 8192
            assert max_output_tokens_for(phase) >= 256

    def test_a_phase_returns_exactly_the_two_knobs_it_sends(self):
        """Ready to spread straight into ``structured_invoke``."""
        kwargs = phase_call_kwargs("plan.design_plan")
        assert set(kwargs) == {"reasoning_effort", "max_tokens"}
        assert kwargs["reasoning_effort"] in REASONING_EFFORTS
        assert isinstance(kwargs["max_tokens"], int)
        assert kwargs["max_tokens"] > 0

    def test_deciding_phases_think_within_a_budget(self):
        """Phases that decide something keep thinking, but capped: the backend
        default is uncapped, which is what made these calls slow."""
        assert reasoning_effort_for("clarify.assess_goal") == "low"
        assert reasoning_effort_for("probe.decompose_strands") == "low"
        assert reasoning_effort_for("probe.evaluate_batch") == "low"
        assert reasoning_effort_for("plan.design_plan") == "low"
        assert reasoning_effort_for("material.write_questions") == "low"

    def test_formatting_phases_do_not_think_at_all(self):
        """Phases that only restate a decision already made need no reasoning."""
        assert reasoning_effort_for("clarify.generate_questions") == "minimal"
        assert reasoning_effort_for("clarify.refine_goal") == "minimal"
        assert reasoning_effort_for("plan.render_plan") == "minimal"
        assert reasoning_effort_for("language.detect") == "minimal"
        assert reasoning_effort_for("language.localize_status") == "minimal"

    def test_an_unknown_phase_is_named_not_silently_defaulted(self):
        with pytest.raises(KeyError) as exc_info:
            reasoning_effort_for("plan.design_plans")
        assert "plan.design_plans" in str(exc_info.value)
        assert "plan.design_plan" in str(exc_info.value)


class TestEnvironmentOverrides:
    def test_one_variable_sets_every_phase(self, monkeypatch):
        monkeypatch.setenv("REASONING_EFFORT", "minimal")
        for phase in PHASES:
            assert reasoning_effort_for(phase) == "minimal"

    def test_one_phase_can_be_set_on_its_own(self, monkeypatch):
        monkeypatch.setenv("REASONING_EFFORT_PROBE_EVALUATE_BATCH", "medium")
        assert reasoning_effort_for("probe.evaluate_batch") == "medium"
        assert reasoning_effort_for("probe.generate_batch") == "low"

    def test_a_phase_override_beats_the_global_one(self, monkeypatch):
        monkeypatch.setenv("REASONING_EFFORT", "minimal")
        monkeypatch.setenv("REASONING_EFFORT_PLAN_DESIGN_PLAN", "high")
        assert reasoning_effort_for("plan.design_plan") == "high"
        assert reasoning_effort_for("plan.render_plan") == "minimal"

    def test_an_effort_the_backend_would_reject_is_not_sent(self, monkeypatch, caplog):
        monkeypatch.setenv("REASONING_EFFORT", "extra-high")
        with caplog.at_level(logging.WARNING):
            assert reasoning_effort_for("plan.design_plan") == "low"
        assert "extra-high" in caplog.text

    def test_the_output_cap_can_be_raised_for_a_whole_run(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "16384")
        for phase in PHASES:
            assert max_output_tokens_for(phase) == 16384

    def test_a_cap_that_is_not_a_number_falls_back_to_the_phase_default(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "huge")
        with caplog.at_level(logging.WARNING):
            assert max_output_tokens_for("probe.generate_batch") == 3072
        assert "huge" in caplog.text

    def test_knobs_are_read_per_call_not_at_import(self, monkeypatch):
        """Retuning a phase is an environment change, not a code change."""
        assert reasoning_effort_for("plan.render_plan") == "minimal"
        monkeypatch.setenv("REASONING_EFFORT", "medium")
        assert reasoning_effort_for("plan.render_plan") == "medium"
