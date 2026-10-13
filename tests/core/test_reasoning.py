"""Tests for core/reasoning.py — the per-stage reasoning budget (#194).

Seam: :func:`core.reasoning.stage_call_kwargs`, the one function a graph node
calls to get what it may think and how much it may say. Every value here is a
literal, not a recomputation of the module's table, because the table *is* the
policy and a test that mirrors it proves nothing.
"""

from __future__ import annotations

import logging

import pytest

from core.reasoning import (
    STAGES,
    max_output_tokens_for,
    reasoning_effort_for,
    require_stage,
    stage_call_kwargs,
    stage_name,
)
from core.settings import REASONING_EFFORTS


class TestStageBudgets:
    def test_every_stage_asks_for_an_effort_the_backend_accepts(self):
        """A value outside this set is a rejected request (HTTP 400)."""
        for stage in STAGES:
            assert reasoning_effort_for(stage) in REASONING_EFFORTS

    def test_every_stage_bounds_what_the_call_may_generate(self):
        """The point of #194: nothing may run to the client's 65536 cap."""
        for stage in STAGES:
            assert max_output_tokens_for(stage) <= 8192
            assert max_output_tokens_for(stage) >= 256

    def test_a_stage_returns_exactly_the_two_knobs_it_sends(self):
        """Ready to spread straight into ``structured_invoke``."""
        kwargs = stage_call_kwargs("plan.design_plan")
        assert set(kwargs) == {"reasoning_effort", "max_tokens"}
        assert kwargs["reasoning_effort"] in REASONING_EFFORTS
        assert isinstance(kwargs["max_tokens"], int)
        assert kwargs["max_tokens"] > 0

    def test_deciding_stages_think_within_a_budget(self):
        """Stages that decide something keep thinking, but capped: the backend
        default is uncapped, which is what made these calls slow."""
        assert reasoning_effort_for("clarify.assess_goal") == "low"
        assert reasoning_effort_for("probe.decompose_strands") == "low"
        assert reasoning_effort_for("probe.generate_batch") == "low"
        assert reasoning_effort_for("probe.evaluate_batch") == "low"
        assert reasoning_effort_for("plan.research_topic") == "low"
        assert reasoning_effort_for("plan.design_plan") == "low"
        assert reasoning_effort_for("material.write_questions") == "low"
        assert reasoning_effort_for("material.summarize_step") == "low"

    def test_restating_stages_do_not_think_at_all(self):
        """Stages that only restate a decision the stage before already made."""
        assert reasoning_effort_for("clarify.generate_questions") == "minimal"
        assert reasoning_effort_for("clarify.refine_goal") == "minimal"
        assert reasoning_effort_for("plan.render_plan") == "minimal"
        assert reasoning_effort_for("language.detect") == "minimal"
        assert reasoning_effort_for("language.localize_status") == "minimal"

    def test_a_stage_is_named_by_its_graph_and_node(self):
        """``timed_node`` and the node body name the same call through this."""
        assert stage_name("clarify", "assess_goal") == "clarify.assess_goal"
        assert stage_name("clarify", "assess_goal") in STAGES

    def test_the_language_helpers_are_named_by_the_helper_not_a_graph(self):
        """They are called from inside stages, not built as graph nodes, so they
        are named by what they are rather than borrowed from a graph."""
        assert {s for s in STAGES if s.startswith("language.")} == {
            "language.detect",
            "language.localize_status",
        }

    def test_an_unknown_stage_is_named_not_silently_defaulted(self):
        with pytest.raises(KeyError) as exc_info:
            reasoning_effort_for("plan.design_plans")
        assert "plan.design_plans" in str(exc_info.value)
        assert "plan.design_plan" in str(exc_info.value)

    def test_require_stage_returns_the_name_it_validated(self):
        assert require_stage("probe.evaluate_batch") == "probe.evaluate_batch"


class TestEnvironmentOverrides:
    def test_one_variable_sets_every_stage(self, monkeypatch):
        monkeypatch.setenv("REASONING_EFFORT", "minimal")
        for stage in STAGES:
            assert reasoning_effort_for(stage) == "minimal"

    def test_one_stage_can_be_set_on_its_own(self, monkeypatch):
        monkeypatch.setenv("REASONING_EFFORT_PROBE_EVALUATE_BATCH", "medium")
        assert reasoning_effort_for("probe.evaluate_batch") == "medium"
        assert reasoning_effort_for("probe.generate_batch") == "low"

    def test_a_stage_override_beats_the_global_one(self, monkeypatch):
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
        for stage in STAGES:
            assert max_output_tokens_for(stage) == 16384

    def test_one_stages_cap_can_be_raised_on_its_own(self, monkeypatch):
        """Raising a stage's effort without raising its cap is how it truncates;
        both knobs have a per-stage setting for exactly that experiment."""
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS_PLAN_RENDER_PLAN", "8192")
        assert max_output_tokens_for("plan.render_plan") == 8192
        assert max_output_tokens_for("plan.design_plan") == 4096

    def test_a_stage_cap_override_beats_the_global_one(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "1024")
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS_PROBE_GENERATE_BATCH", "6144")
        assert max_output_tokens_for("probe.generate_batch") == 6144
        assert max_output_tokens_for("probe.evaluate_batch") == 1024

    def test_a_cap_that_is_not_a_number_falls_back_to_the_stage_default(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "huge")
        with caplog.at_level(logging.WARNING):
            assert max_output_tokens_for("probe.generate_batch") == 3072
        assert "huge" in caplog.text

    def test_knobs_are_read_per_call_not_at_import(self, monkeypatch):
        """Retuning a stage is an environment change, not a code change."""
        assert reasoning_effort_for("plan.render_plan") == "minimal"
        monkeypatch.setenv("REASONING_EFFORT", "medium")
        assert reasoning_effort_for("plan.render_plan") == "medium"
