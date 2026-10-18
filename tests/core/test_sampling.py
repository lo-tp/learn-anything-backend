"""Tests for core/sampling.py — how a stage decodes, and what configures it.

Seam: :func:`core.sampling.apply_configured_sampling`, the one place that joins
what a stage states in code with what an operator set in ``.env``, and the table
it reads. The values below are what a knob is expected to produce, not a
recomputation of the table: the table *is* the policy.
"""

from __future__ import annotations

import logging

import pytest
from langchain_openai import ChatOpenAI

from core.sampling import (
    EXTRA_BODY_PARAMS,
    SAMPLING_ENV_NAMES,
    SAMPLING_PARAMS,
    apply_configured_sampling,
    require_param,
    sampling_value,
    stage_sampling_values,
)

STAGE = "plan.design_plan"


# The parameters that are standard chat-completion fields, spelled here rather
# than read from the module: which side of the wire a parameter travels on is the
# fact a mistake would cost, so the test states it.
STANDARD_PARAMS = {
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
}
BACKEND_EXTENSIONS = {"top_k", "min_p", "repeat_penalty"}


def bind_for(stage: str) -> dict:
    """What a call that states no sampling of its own would be bound with."""
    return apply_configured_sampling({}, stage)


class TestTheTable:
    def test_every_parameter_is_either_standard_or_an_extension(self):
        assert set(SAMPLING_PARAMS) == STANDARD_PARAMS | BACKEND_EXTENSIONS
        assert not (STANDARD_PARAMS & BACKEND_EXTENSIONS)

    def test_the_split_matches_what_the_openai_client_can_carry(self):
        """A standard field nested is a field the backend never sees.

        ``extra_body`` is forwarded into the request body; a field ChatOpenAI
        already knows about does not need it, and a field it does not know
        cannot be sent any other way.
        """
        fields = ChatOpenAI.model_fields
        for name in STANDARD_PARAMS:
            assert name in fields
        for name in EXTRA_BODY_PARAMS:
            assert name not in fields

    def test_the_split_agrees_with_the_one_the_slide_ladder_relies_on(self):
        from graphs.material.sandbox import SLIDE_EXTRA_BODY_PARAMS

        assert SLIDE_EXTRA_BODY_PARAMS == EXTRA_BODY_PARAMS == BACKEND_EXTENSIONS

    def test_every_setting_name_is_an_env_variable_this_app_reads(self):
        assert set(SAMPLING_ENV_NAMES) == {p.env_name for p in SAMPLING_PARAMS.values()}
        assert all(name.startswith("LLM_") for name in SAMPLING_ENV_NAMES)

    def test_an_unknown_parameter_is_named_not_sent(self):
        with pytest.raises(KeyError) as exc_info:
            require_param("top_j")
        assert "top_j" in str(exc_info.value)
        assert "temperature" in str(exc_info.value)


class TestNothingIsSentUnlessSomeoneConfiguresIt:
    """No knob in this module has a value of its own.

    That is what makes adding one a no-op: every call keeps the sampling it had
    before, and only a call whose stage an operator retuned changes.
    """

    @pytest.mark.parametrize("name", sorted(SAMPLING_PARAMS))
    def test_an_unset_parameter_has_no_value(self, monkeypatch, name):
        monkeypatch.delenv(SAMPLING_PARAMS[name].env_name, raising=False)
        assert sampling_value(name, STAGE) is None

    def test_an_unconfigured_stage_configures_nothing(self):
        assert stage_sampling_values(STAGE) == {}
        assert bind_for(STAGE) == {}

    def test_layering_onto_a_call_changes_nothing_it_already_said(self):
        call = {"reasoning_effort": "low", "max_tokens": 4096, "temperature": 0.1}
        layered = apply_configured_sampling(call, STAGE)
        assert layered == call
        assert layered is not call

    def test_the_call_kwargs_are_not_mutated(self):
        """A stage's own bind dict is not edited by a layering that added nothing."""
        call = {"temperature": 0.4}
        apply_configured_sampling(call, STAGE)
        assert call == {"temperature": 0.4}


class TestWhichNameWins:
    def test_one_variable_sets_every_stage(self, monkeypatch):
        monkeypatch.setenv("LLM_TEMPERATURE", "1.1")
        for stage in (
            "plan.design_plan",
            "clarify.assess_goal",
            "material.write_slide",
        ):
            assert sampling_value("temperature", stage) == 1.1

    def test_one_stage_can_be_set_on_its_own(self, monkeypatch):
        monkeypatch.setenv("LLM_TOP_P_PLAN_DESIGN_PLAN", "0.3")
        assert sampling_value("top_p", "plan.design_plan") == 0.3
        assert sampling_value("top_p", "plan.render_plan") is None

    def test_a_stage_setting_beats_the_global_one(self, monkeypatch):
        monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
        monkeypatch.setenv("LLM_TEMPERATURE_PROBE_EVALUATE_BATCH", "0.9")
        assert sampling_value("temperature", "probe.evaluate_batch") == 0.9
        assert sampling_value("temperature", "probe.generate_batch") == 0.2

    def test_a_stage_name_becomes_a_setting_name_the_same_way_every_knob_does(self):
        """core/settings.py derives this for reasoning effort and output caps too."""
        from core.settings import stage_setting_names

        assert stage_setting_names("LLM_TOP_K", "material.write_slide") == (
            "LLM_TOP_K_MATERIAL_WRITE_SLIDE",
            "LLM_TOP_K",
        )

    def test_knobs_are_read_per_call_not_at_import(self, monkeypatch):
        assert bind_for(STAGE) == {}
        monkeypatch.setenv("LLM_MIN_P", "0.2")
        assert bind_for(STAGE) == {"extra_body": {"min_p": 0.2}}

    def test_integer_and_float_knobs_are_read_as_themselves(self, monkeypatch):
        monkeypatch.setenv("LLM_TOP_K", "40")
        assert sampling_value("top_k", STAGE) == 40
        monkeypatch.setenv("LLM_REPEAT_PENALTY", "1.05")
        assert sampling_value("repeat_penalty", STAGE) == 1.05


class TestWhatReachesTheRequest:
    def test_standard_fields_travel_top_level(self, monkeypatch):
        monkeypatch.setenv("LLM_TEMPERATURE", "0.8")
        monkeypatch.setenv("LLM_FREQUENCY_PENALTY", "0.5")
        kwargs = bind_for(STAGE)
        assert kwargs["temperature"] == 0.8
        assert kwargs["frequency_penalty"] == 0.5
        assert "extra_body" not in kwargs

    def test_backend_extensions_travel_in_extra_body(self, monkeypatch):
        monkeypatch.setenv("LLM_TOP_K", "30")
        monkeypatch.setenv("LLM_MIN_P", "0.05")
        assert bind_for(STAGE) == {
            "extra_body": {"top_k": 30, "min_p": 0.05}
        }

    def test_a_stage_that_states_its_own_sampling_keeps_it(self, monkeypatch):
        call = {"temperature": 0.4, "extra_body": {"top_k": 25, "min_p": 0.06}}
        layered = apply_configured_sampling(call, STAGE)
        assert layered == call

    def test_setting_one_nested_knob_does_not_drop_the_others_the_stage_stated(
        self, monkeypatch
    ):
        """The merge is per parameter, not per transport.

        A slide call carries three parameters in ``extra_body``. Setting one of
        them from the environment must not send a body that lost the other two.
        """
        monkeypatch.setenv("LLM_TOP_K_MATERIAL_WRITE_SLIDE", "5")
        call = {"extra_body": {"top_k": 25, "min_p": 0.06, "repeat_penalty": 1.02}}
        layered = apply_configured_sampling(call, "material.write_slide")
        assert layered["extra_body"] == {
            "top_k": 5,
            "min_p": 0.06,
            "repeat_penalty": 1.02,
        }

    def test_a_configured_value_sits_above_what_a_stage_states_in_code(
        self, monkeypatch
    ):
        """The order is the one core/settings.py defines for every other knob.

        This is what flattens the slide ladder for the parameter someone set:
        a number a person typed beats a number in a source file.
        """
        monkeypatch.setenv("LLM_TEMPERATURE", "1.2")
        layered = apply_configured_sampling(
            {"temperature": 0.4, "top_p": 0.75}, "material.write_slide"
        )
        assert layered == {"temperature": 1.2, "top_p": 0.75}

    def test_nothing_configured_for_a_stage_leaves_a_code_stated_value_alone(
        self, monkeypatch
    ):
        monkeypatch.setenv("LLM_TEMPERATURE_OTHER_GRAPH_OTHER_NODE", "1.5")
        call = {"temperature": 0.4}
        assert apply_configured_sampling(call, STAGE) == call


class TestAValueTheBackendWouldReject:
    """An out-of-range knob is a request the backend answers with 400.

    Treating it as unset keeps the stage's own default — or the backend's — which
    is the same rule an unknown reasoning effort follows.
    """

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("temperature", "5"),
            ("temperature", "-0.1"),
            ("top_p", "1.5"),
            ("top_p", "-1"),
            ("min_p", "2"),
            ("top_k", "0"),
            ("top_k", "-1"),
            ("repeat_penalty", "-2"),
            ("presence_penalty", "3"),
            ("frequency_penalty", "-3"),
        ],
    )
    def test_a_value_outside_the_range_is_not_sent(
        self, monkeypatch, caplog, name, value
    ):
        monkeypatch.setenv(SAMPLING_PARAMS[name].env_name, value)
        with caplog.at_level(logging.WARNING):
            assert sampling_value(name, STAGE) is None
            assert stage_sampling_values(STAGE) == {}
        assert value in caplog.text
        assert SAMPLING_PARAMS[name].env_name in caplog.text

    def test_a_value_that_is_not_a_number_is_not_sent(self, monkeypatch, caplog):
        monkeypatch.setenv("LLM_TEMPERATURE", "hot")
        with caplog.at_level(logging.WARNING):
            assert sampling_value("temperature", STAGE) is None
        assert "hot" in caplog.text

    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [
            ("temperature", "0", 0),
            ("temperature", "2", 2.0),
            ("top_p", "1", 1.0),
            ("top_p", "0", 0),
            ("top_k", "1", 1),
            ("min_p", "0", 0),
            ("repeat_penalty", "0", 0),
            ("presence_penalty", "-2", -2.0),
            ("frequency_penalty", "2", 2.0),
        ],
    )
    def test_the_ends_of_a_range_are_usable_values(
        self, monkeypatch, name, value, expected
    ):
        """An open or closed end is a policy, not an accident: temperature 0 and
        temperature 2 are both requests this backend serves."""
        monkeypatch.setenv(SAMPLING_PARAMS[name].env_name, value)
        assert sampling_value(name, STAGE) == expected

    def test_whitespace_and_a_blank_value_are_unset_not_zero(self, monkeypatch):
        monkeypatch.setenv("LLM_TEMPERATURE", "   ")
        assert sampling_value("temperature", STAGE) is None
