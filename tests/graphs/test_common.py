"""Tests for graphs/common.py."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai.chat_models.base import OpenAIAPIError
from pydantic import BaseModel, ValidationError

from core.model_answers import (
    ModelAnswerUnparseable,
    ModelRecovery,
    drain_model_recoveries,
    record_model_recovery,
)
from core.reasoning import stage_name
from graphs.common import (
    _MAX_TOOL_CALL_ROUNDS,
    structured_invoke,
    structured_invoke_messages,
    structured_invoke_with_tools,
    timed_node,
    unknown_option,
    with_unknown_option,
)
from tests.error_shapes import model_unavailable, structured_output_refused
from tests.tool_shapes import mcp_shaped_tool as _mcp_shaped_tool

# Every structured call names the stage it belongs to (#195): the same name the
# node passes to core/reasoning.py and the graph passes to timed_node.
STAGE = "clarify.assess_goal"

# --- structured_invoke ---


class _DummySchema(BaseModel):
    value: str


class TestStructuredInvoke:
    def test_builds_system_and_human_messages(self):
        llm = MagicMock()
        mock_result = _DummySchema(value="hello")
        with patch(
            "graphs.common.structured_invoke_messages", return_value=mock_result
        ) as mock_fn:
            result = structured_invoke(
                llm, _DummySchema, "sys prompt", "user prompt", stage=STAGE
            )
            assert result == mock_result
            # Verify the messages passed
            args = mock_fn.call_args
            messages = args[0][2]
            assert isinstance(messages[0], SystemMessage)
            assert messages[0].content == "sys prompt"
            assert isinstance(messages[1], HumanMessage)
            assert messages[1].content == "user prompt"

    def test_passes_llm_and_schema(self):
        llm = MagicMock()
        mock_result = _DummySchema(value="x")
        with patch(
            "graphs.common.structured_invoke_messages", return_value=mock_result
        ) as mock_fn:
            structured_invoke(llm, _DummySchema, "s", "h", stage=STAGE)
            assert mock_fn.call_args[0][0] is llm
            assert mock_fn.call_args[0][1] is _DummySchema

    def test_forwards_bind_kwargs(self):
        """A per-call bind (e.g. reasoning_effort) must reach the invoker."""
        llm = MagicMock()
        mock_result = _DummySchema(value="x")
        with patch(
            "graphs.common.structured_invoke_messages", return_value=mock_result
        ) as mock_fn:
            structured_invoke(
                llm, _DummySchema, "s", "h", stage=STAGE, reasoning_effort="minimal"
            )
            assert mock_fn.call_args.kwargs == {
                "stage": STAGE,
                "reasoning_effort": "minimal",
            }

    def test_no_bind_kwargs_by_default(self):
        """Call sites that pass nothing must keep sending nothing."""
        llm = MagicMock()
        mock_result = _DummySchema(value="x")
        with patch(
            "graphs.common.structured_invoke_messages", return_value=mock_result
        ) as mock_fn:
            structured_invoke(llm, _DummySchema, "s", "h", stage=STAGE)
            assert mock_fn.call_args.kwargs == {"stage": STAGE}


# --- structured_invoke_messages ---


class TestStructuredInvokeMessages:
    def test_calls_with_structured_output_and_invoke(self):
        llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _DummySchema(value="ok")
        llm.with_structured_output.return_value = mock_structured

        result = structured_invoke_messages(
            llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
        )
        assert result.value == "ok"
        llm.with_structured_output.assert_called_once_with(_DummySchema)
        mock_structured.invoke.assert_called_once()

    def test_bind_kwargs_applied_after_structured_output(self):
        llm = MagicMock()
        mock_structured = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = _DummySchema(value="ok")
        mock_structured.bind.return_value = mock_bound
        llm.with_structured_output.return_value = mock_structured

        result = structured_invoke_messages(
            llm, _DummySchema, [HumanMessage(content="hi")],
            stage=STAGE,
            temperature=0.1,
        )
        assert result.value == "ok"
        # bind must wrap the structured-output model (i.e. run AFTER
        # with_structured_output) and invoke must hit the bound model.
        mock_structured.bind.assert_called_once_with(temperature=0.1)
        mock_bound.invoke.assert_called_once()
        mock_structured.invoke.assert_not_called()


# --- the structured call has a second plan (#195) ---


def _refusing_model(llm: MagicMock, refusal: BaseException | None = None) -> MagicMock:
    """A model whose structured answer the backend refuses to parse.

    Set on both shapes the strict plan takes: bound (a stage with a reasoning
    budget) and unbound. The refusal is the same either way, because the backend
    does not know which one made the text it could not parse.
    """
    structured = llm.with_structured_output.return_value
    error = refusal if refusal is not None else structured_output_refused()
    structured.invoke.side_effect = error
    structured.bind.return_value.invoke.side_effect = error
    return llm


def _answered_with(llm: MagicMock, text: str) -> MagicMock:
    """What the unconstrained second ask comes back with."""
    llm.invoke.return_value = AIMessage(content=text)
    llm.bind.return_value.invoke.return_value = AIMessage(content=text)
    return llm


class TestStructuredCallIsRecoverable:
    """A refused answer is not the end of a deck.

    The backend's grammar lets the model write `$\\int` and then refuses to
    parse it, which is #195: the failure reached a driver that could only mark
    the session ``error``. The first plan is unchanged; the second one asks for
    the same shape with nothing constraining the decode, reads the text this
    side of the wire, and validates it as strictly as the first plan did.
    """

    def test_the_schema_enforced_ask_is_still_the_first_plan(self):
        """The grammar stays on: this is a fallback, not a downgrade of every
        structured call in the app."""
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = _DummySchema(
            value="ok"
        )

        result = structured_invoke_messages(
            llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
        )

        assert result.value == "ok"
        llm.invoke.assert_not_called()
        llm.bind.assert_not_called()

    def test_a_refused_answer_is_asked_for_again_unconstrained(self):
        llm = _refusing_model(MagicMock())
        _answered_with(llm, '{"value": "the answer, with $\\frac{1}{2}$ in it"}')

        result = structured_invoke_messages(
            llm,
            _DummySchema,
            [SystemMessage(content="s"), HumanMessage(content="h")],
            stage=STAGE,
        )

        assert result.value == "the answer, with $\\frac{1}{2}$ in it"
        llm.with_structured_output.assert_called_once_with(_DummySchema)
        # The second plan has no schema enforcement: a plain call, not a second
        # structured one.
        llm.bind.assert_not_called()
        llm.invoke.assert_called_once()

    def test_the_second_ask_carries_the_same_reasoning_budget(self):
        """A recovery that escapes the stage's budget is a hole in #194: the
        fallback is a model call like any other, and is bound like one."""
        llm = _refusing_model(MagicMock())
        _answered_with(llm, '{"value": "x"}')

        structured_invoke_messages(
            llm,
            _DummySchema,
            [HumanMessage(content="hi")],
            stage=STAGE,
            reasoning_effort="low",
            max_tokens=3072,
        )

        llm.bind.assert_called_once_with(reasoning_effort="low", max_tokens=3072)

    def test_the_second_ask_is_told_the_shape_of_an_answer(self):
        """Without the grammar there is nothing else holding the contract, and a
        model left with no shape at all has been measured answering with its own
        schema. The stage's own messages come first, unchanged: the language the
        prompt asked for must still be in effect."""
        llm = _refusing_model(MagicMock())
        _answered_with(llm, '{"value": "x"}')

        structured_invoke_messages(
            llm,
            _DummySchema,
            [SystemMessage(content="s"), HumanMessage(content="h")],
            stage=STAGE,
        )

        sent = llm.invoke.call_args[0][0]
        assert [type(message) for message in sent[:2]] == [SystemMessage, HumanMessage]
        assert [message.content for message in sent[:2]] == ["s", "h"]
        asked = sent[-1].content
        assert '"value": "<string>"' in asked
        assert "additionalProperties" not in asked

    def test_the_latex_arrives_as_latex(self):
        """The stored value must be the formula, not a control character plus
        whatever letters followed it — the half of #195 that never raised."""
        llm = _refusing_model(MagicMock())
        _answered_with(llm, '{"value": "$\\frac{1}{2}$ and \\beta decay"}')

        result = structured_invoke_messages(
            llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
        )

        assert result.value == "$\\frac{1}{2}$ and \\beta decay"
        assert "\u000c" not in result.value  # form feed
        assert "\u0008" not in result.value  # backspace

    def test_a_recovered_answer_is_attributed_to_the_stage_that_recovered_it(self):
        """Which stage leaned on this is a query, not a guess from log lines."""
        llm = _refusing_model(MagicMock())
        _answered_with(llm, '{"value": "$\\int_1^2$"}')

        structured_invoke_messages(
            llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
        )

        assert drain_model_recoveries() == [
            {"reason": "backend_refused_json", "repaired_escapes": 1}
        ]

    def test_a_recovery_is_logged_naming_the_stage(self, caplog):
        llm = _refusing_model(MagicMock())
        _answered_with(llm, '{"value": "$\\int_1^2$"}')

        with caplog.at_level(logging.WARNING, logger="graphs.common"):
            structured_invoke_messages(
                llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
            )

        assert "clarify.assess_goal" in caplog.text
        assert "#195" in caplog.text

    def test_a_backend_that_is_busy_is_not_recovered_from(self):
        """A busy model is not a model whose JSON did not parse: for that one,
        repeating the request is the right plan, and this must not pretend
        otherwise by rewriting the request."""
        llm = _refusing_model(MagicMock(), refusal=model_unavailable())

        with pytest.raises(OpenAIAPIError):
            structured_invoke_messages(
                llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
            )

        llm.invoke.assert_not_called()

    def test_an_answer_that_validated_badly_is_not_recovered_from(self):
        """A shape miss is a prompt problem. Asking again with *less* enforcement
        is a guess at it, and a guess is not what this seam is for."""
        try:
            _DummySchema.model_validate({})
        except ValidationError as invalid_shape:
            llm = _refusing_model(MagicMock(), refusal=invalid_shape)

            with pytest.raises(ValidationError):
                structured_invoke_messages(
                    llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
                )

            llm.invoke.assert_not_called()

    def test_an_answer_that_cannot_be_read_fails_by_name(self):
        """When the second plan fails too, what reaches the driver says which
        stage, what was wrong, and keeps the original refusal as its cause."""
        refusal = structured_output_refused()
        llm = _refusing_model(MagicMock(), refusal=refusal)
        _answered_with(llm, "I could not decide what to teach.")

        with pytest.raises(ModelAnswerUnparseable) as exc_info:
            structured_invoke_messages(
                llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
            )

        assert STAGE in str(exc_info.value)
        assert exc_info.value.__cause__ is refusal

    def test_an_answer_that_does_not_fit_the_schema_fails_by_name(self):
        """Recovering the parse does not lower the standard the answer is held to."""
        llm = _refusing_model(MagicMock())
        _answered_with(llm, '{"not_the_field": 1}')

        with pytest.raises(ModelAnswerUnparseable) as exc_info:
            structured_invoke_messages(
                llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
            )

        assert STAGE in str(exc_info.value)

    def test_a_tool_using_call_site_gets_the_same_recovery(self):
        """``research_topic`` answers through this seam too (#166): the tool turn
        and the plain answer share the recovery."""
        llm = MagicMock()
        _refusing_model(llm)
        _answered_with(llm, '{"value": "$\\frac{1}{2}$"}')

        result = structured_invoke_with_tools(
            llm, _DummySchema, "s", "h", [], stage=STAGE
        )

        assert result.value == "$\\frac{1}{2}$"


# --- what an operator configured for this stage (the decoding knobs) ---


def _answering(llm: MagicMock) -> MagicMock:
    """A model that answers, bound or not, so the bind path can be inspected."""
    structured = llm.with_structured_output.return_value
    structured.invoke.return_value = _DummySchema(value="ok")
    structured.bind.return_value.invoke.return_value = _DummySchema(value="ok")
    return llm


def _bound_kwargs(llm: MagicMock) -> dict:
    """Exactly what the structured call was bound with."""
    return llm.with_structured_output.return_value.bind.call_args.kwargs


class TestConfiguredSamplingIsAppliedAtTheCall:
    r"""The stage name is the key, and this is the one place it is looked up.

    A node does not thread ``temperature`` through itself and a graph does not
    know which knobs exist: a setting in ``.env`` reaches the call it names and no
    other, both halves of a tool-using turn, and the unconstrained second ask of
    a recovery. With nothing configured nothing is bound, which is what keeps
    every stage's sampling exactly as it was.
    """

    def test_an_unconfigured_stage_binds_nothing_it_did_not_already_bind(self):
        llm = _answering(MagicMock())
        structured_invoke_messages(
            llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
        )
        llm.with_structured_output.return_value.bind.assert_not_called()

    def test_a_global_setting_reaches_a_call_that_never_named_it(self, monkeypatch):
        monkeypatch.setenv("LLM_TEMPERATURE", "0.9")
        llm = _answering(MagicMock())
        structured_invoke_messages(
            llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
        )
        assert _bound_kwargs(llm) == {"temperature": 0.9}

    def test_a_stage_setting_reaches_its_stage_and_not_the_next_one(self, monkeypatch):
        monkeypatch.setenv("LLM_TEMPERATURE_CLARIFY_ASSESS_GOAL", "1.2")
        named = _answering(MagicMock())
        other = _answering(MagicMock())

        structured_invoke_messages(
            named, _DummySchema, [HumanMessage("hi")], stage=STAGE
        )
        structured_invoke_messages(
            other, _DummySchema, [HumanMessage("hi")], stage="plan.render_plan"
        )

        assert _bound_kwargs(named) == {"temperature": 1.2}
        other.with_structured_output.return_value.bind.assert_not_called()

    def test_a_configured_value_sits_above_the_one_the_call_stated(self, monkeypatch):
        """The order core/settings.py defines for every knob: an operator's number
        beats a number in a source file, which is what flattens the slide ladder."""
        monkeypatch.setenv("LLM_TOP_P", "0.4")
        llm = _answering(MagicMock())
        structured_invoke_messages(
            llm,
            _DummySchema,
            [HumanMessage(content="hi")],
            stage=STAGE,
            temperature=0.1,
            top_p=0.9,
        )
        assert _bound_kwargs(llm) == {"temperature": 0.1, "top_p": 0.4}

    def test_a_backend_only_parameter_is_nested_as_it_has_to_be(self, monkeypatch):
        """The caller does not choose the transport; the table does."""
        monkeypatch.setenv("LLM_TOP_K", "12")
        llm = _answering(MagicMock())
        structured_invoke_messages(
            llm, _DummySchema, [HumanMessage(content="hi")], stage=STAGE
        )
        assert _bound_kwargs(llm) == {"extra_body": {"top_k": 12}}

    def test_the_second_ask_carries_the_configured_decoding_too(self, monkeypatch):
        """A recovery is a model call like any other (#194), and a fallback that
        ignored what an operator configured would be a hole in the policy."""
        monkeypatch.setenv("LLM_MIN_P", "0.2")
        llm = _refusing_model(_answering(MagicMock()))
        _answered_with(llm, '{"value": "x"}')

        structured_invoke_messages(
            llm,
            _DummySchema,
            [HumanMessage(content="hi")],
            stage=STAGE,
            reasoning_effort="low",
            max_tokens=3072,
        )

        llm.bind.assert_called_once_with(
            reasoning_effort="low",
            max_tokens=3072,
            extra_body={"min_p": 0.2},
        )

    def test_a_tool_using_turn_runs_under_it_before_it_searches(self, monkeypatch):
        """Deciding whether to search is where a tool-using node does its
        thinking, so the decoding applies to that call too (#194's reasoning)."""
        monkeypatch.setenv("LLM_TEMPERATURE", "0.8")
        monkeypatch.setenv("LLM_TOP_K", "12")
        llm = MagicMock()
        calls: list = []
        tool = _mcp_shaped_tool("tavily_search", calls, "unused")
        llm.bind_tools.return_value.invoke.return_value = AIMessage(content="known")
        with patch(
            "graphs.common.structured_invoke_messages",
            return_value=_DummySchema(value="x"),
        ) as structured:
            structured_invoke_with_tools(
                llm, _DummySchema, "s", "h", [tool], stage=STAGE
            )

        assert llm.bind_tools.call_args.kwargs == {
            "temperature": 0.8,
            "extra_body": {"top_k": 12},
        }
        assert structured.call_args.kwargs["stage"] == STAGE
        assert structured.call_args.kwargs["temperature"] == 0.8


# --- unknown_option ---


class TestUnknownOption:
    def test_calls_localize_status_with_language(self):
        with patch(
            "graphs.common.localize_status", return_value="Je ne sais pas"
        ) as mock_ls:
            llm = MagicMock()
            result = unknown_option(llm, "French")
            assert result == "Je ne sais pas"
            mock_ls.assert_called_once_with(llm, "French", "I don't know")

    def test_defaults_to_default_language_when_none(self):
        with patch(
            "graphs.common.localize_status", return_value="I don't know"
        ) as mock_ls:
            result = unknown_option(None)
            assert result == "I don't know"
            from core.language import DEFAULT_LANGUAGE

            mock_ls.assert_called_once_with(None, DEFAULT_LANGUAGE, "I don't know")


# --- with_unknown_option ---


class TestWithUnknownOption:
    def _patch_unknown(self, value: str = "I don't know"):
        return patch("graphs.common.unknown_option", return_value=value)

    def test_appends_unknown_option(self):
        with self._patch_unknown("I don't know"):
            result = with_unknown_option(None, None, ["A", "B"])
        assert result == ["A", "B", "I don't know"]

    def test_does_not_duplicate_exact_match(self):
        with self._patch_unknown("I don't know"):
            result = with_unknown_option(None, None, ["A", "I don't know"])
        assert result == ["A", "I don't know"]

    def test_does_not_duplicate_case_insensitive_match(self):
        with self._patch_unknown("I don't know"):
            result = with_unknown_option(None, None, ["A", "i don't know"])
        assert result == ["A", "i don't know"]

    def test_does_not_mutate_input(self):
        options = ["A", "B"]
        with self._patch_unknown("I don't know"):
            result = with_unknown_option(None, None, options)
        assert options == ["A", "B"]
        assert result is not options

    def test_empty_options_list(self):
        with self._patch_unknown("I don't know"):
            result = with_unknown_option(None, None, [])
        assert result == ["I don't know"]

    def test_localized_unknown_is_used(self):
        with self._patch_unknown("Je ne sais pas"):
            result = with_unknown_option(MagicMock(), "French", ["A"])
        assert result == ["A", "Je ne sais pas"]


# --- structured_invoke_with_tools (#166) ---


def _asking_for_a_tool(name: str = "tavily_search") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": "calculus pedagogy"}, "id": "c1"}],
    )


class TestStructuredInvokeWithTools:
    def test_no_tools_is_the_plain_structured_path(self):
        """Mock mode / search off: nothing is bound, the old path is unchanged."""
        llm = MagicMock()
        with patch(
            "graphs.common.structured_invoke_messages",
            return_value=_DummySchema(value="x"),
        ) as structured:
            result = structured_invoke_with_tools(
                llm, _DummySchema, "sys", "human", [], stage=STAGE
            )
        assert result.value == "x"
        llm.bind_tools.assert_not_called()
        assert structured.call_args[0][2] == [
            SystemMessage(content="sys"),
            HumanMessage(content="human"),
        ]

    def test_the_tools_are_offered_to_the_model(self):
        llm = MagicMock()
        calls: list = []
        tool = _mcp_shaped_tool("tavily_search", calls, "no results")
        llm.bind_tools.return_value.invoke.return_value = AIMessage(content="ok")
        with patch(
            "graphs.common.structured_invoke_messages",
            return_value=_DummySchema(value="x"),
        ):
            structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool], stage=STAGE)
        llm.bind_tools.assert_called_once_with([tool])

    def test_the_model_may_answer_without_searching(self):
        llm = MagicMock()
        calls: list = []
        tool = _mcp_shaped_tool("tavily_search", calls, "unused")
        llm.bind_tools.return_value.invoke.return_value = AIMessage(content="known")
        with patch(
            "graphs.common.structured_invoke_messages",
            return_value=_DummySchema(value="x"),
        ):
            structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool], stage=STAGE)
        assert calls == []
        assert llm.bind_tools.return_value.invoke.call_count == 1

    def test_search_results_go_back_to_the_model_before_the_answer(self):
        """The grounding the issue asks for: ask, search, then answer."""
        llm = MagicMock()
        calls: list = []
        tool = _mcp_shaped_tool("tavily_search", calls, "how the field is framed")
        # First turn: the model asks to search. Second: it has nothing to add.
        llm.bind_tools.return_value.invoke.side_effect = [
            _asking_for_a_tool(),
            AIMessage(content="enough"),
        ]
        with patch(
            "graphs.common.structured_invoke_messages",
            return_value=_DummySchema(value="x"),
        ) as structured:
            structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool], stage=STAGE)

        assert calls == ["calculus pedagogy"]
        messages = structured.call_args[0][2]
        assert isinstance(messages[2], AIMessage)  # the tool request
        assert isinstance(messages[3], ToolMessage)
        assert messages[3].tool_call_id == "c1"
        assert "how the field is framed" in str(messages[3].content)

    def test_a_failing_search_does_not_break_the_answer(self):
        """Search is optional: a dead Tavily must not fail the request."""
        llm = MagicMock()
        calls: list = []
        tool = _mcp_shaped_tool("tavily_search", calls, ConnectionError("down"))
        llm.bind_tools.return_value.invoke.side_effect = [
            _asking_for_a_tool(),
            AIMessage(content="answer anyway"),
        ]
        with patch(
            "graphs.common.structured_invoke_messages",
            return_value=_DummySchema(value="x"),
        ) as structured:
            result = structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool], stage=STAGE)

        assert result.value == "x"
        tool_message = structured.call_args[0][2][3]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.status == "error"

    def test_a_tool_name_the_app_never_offered_is_reported_not_fatal(self):
        llm = MagicMock()
        calls: list = []
        tool = _mcp_shaped_tool("tavily_search", calls, "unused")
        llm.bind_tools.return_value.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "tavily_crawl", "args": {"url": "x"}, "id": "c9"}
                ],
            ),
            AIMessage(content="answer anyway"),
        ]
        with patch(
            "graphs.common.structured_invoke_messages",
            return_value=_DummySchema(value="x"),
        ) as structured:
            result = structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool], stage=STAGE)
        assert result.value == "x"
        assert calls == []
        tool_message = structured.call_args[0][2][3]
        assert tool_message.status == "error"
        assert "tavily_crawl" in str(tool_message.content)

    def test_tool_use_is_bounded(self):
        """A model that keeps searching must not spin the request forever."""
        llm = MagicMock()
        calls: list = []
        tool = _mcp_shaped_tool("tavily_search", calls, "more")
        llm.bind_tools.return_value.invoke.side_effect = lambda _messages: (
            _asking_for_a_tool()
        )
        with patch(
            "graphs.common.structured_invoke_messages",
            return_value=_DummySchema(value="x"),
        ) as structured:
            structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool], stage=STAGE)
        assert len(calls) == _MAX_TOOL_CALL_ROUNDS
        assert llm.bind_tools.return_value.invoke.call_count == _MAX_TOOL_CALL_ROUNDS
        # The answer is still produced over the accumulated history.
        assert len(structured.call_args[0][2]) == 2 + 2 * _MAX_TOOL_CALL_ROUNDS


# --- the reasoning budget reaches the whole turn (#194) ---


class TestToolTurnCarriesTheReasoningBudget:
    """A phase's budget must cover the thinking a tool call involves, not only
    the answer that follows it: for ``research_topic`` the search decision is
    where the reasoning happens."""

    def test_the_budget_is_bound_on_the_turn_that_decides_to_search(self):
        llm = MagicMock()
        calls: list = []
        tool = _mcp_shaped_tool("tavily_search", calls, "results")
        llm.bind_tools.return_value.invoke.side_effect = [
            _asking_for_a_tool(),
            AIMessage(content="enough"),
        ]
        with patch(
            "graphs.common.structured_invoke_messages",
            return_value=_DummySchema(value="x"),
        ):
            structured_invoke_with_tools(
                llm,
                _DummySchema,
                "s",
                "h",
                [tool],
                stage=STAGE,
                reasoning_effort="low",
                max_tokens=4096,
            )
        llm.bind_tools.assert_called_once_with(
            [tool], reasoning_effort="low", max_tokens=4096
        )

    def test_the_budget_is_also_on_the_answer_that_follows(self):
        llm = MagicMock()
        calls: list = []
        tool = _mcp_shaped_tool("tavily_search", calls, "results")
        llm.bind_tools.return_value.invoke.return_value = AIMessage(content="ok")
        with patch(
            "graphs.common.structured_invoke_messages",
            return_value=_DummySchema(value="x"),
        ) as structured:
            structured_invoke_with_tools(
                llm,
                _DummySchema,
                "s",
                "h",
                [tool],
                stage=STAGE,
                reasoning_effort="low",
                max_tokens=4096,
            )
        assert structured.call_args.kwargs == {
            "stage": STAGE,
            "reasoning_effort": "low",
            "max_tokens": 4096,
        }

    def test_an_unbounded_call_is_unchanged(self):
        """The no-budget path is the one every other caller already uses."""
        llm = MagicMock()
        with patch(
            "graphs.common.structured_invoke_messages",
            return_value=_DummySchema(value="x"),
        ) as structured:
            structured_invoke_with_tools(llm, _DummySchema, "s", "h", [], stage=STAGE)
        assert structured.call_args.kwargs == {"stage": STAGE}


# --- timed_node (#194) ---


class TestTimedNode:
    """A graph node wrapped so its wall time is recorded.

    The material graph already times its own stages because its driver persists
    them between interrupts; Clarify, Probe and Plan pause between HTTP requests,
    so they get the timing at the seam instead of inside each node (#194).
    """

    def test_the_result_is_the_nodes_plus_its_own_timing(self):
        def node(_state: dict) -> dict:
            return {"answer": 1}

        result = timed_node(stage_name("clarify", "assess_goal"), node)({})

        assert result["answer"] == 1
        (entry,) = result["stage_timings"]
        # The row stores the stage without its graph prefix — the graph is in its
        # own column, exactly as the material graph stores it.
        assert entry["stage"] == "assess_goal"
        assert entry["context"] == {}
        assert entry["duration_seconds"] >= 0

    def test_a_model_stage_no_one_has_a_budget_for_is_rejected_here(self):
        """A wrapped node is a model call, so it must be a stage core/reasoning.py
        knows: a renamed node or a new call nobody budgeted fails when the graph
        is assembled, not in a timings row nobody can explain."""
        def node(_state: dict) -> dict:
            return {}

        with pytest.raises(KeyError) as exc_info:
            timed_node(stage_name("clarify", "assess_goals"), node)

        assert "clarify.assess_goals" in str(exc_info.value)
        assert "clarify.assess_goal" in str(exc_info.value)

    def test_the_cost_is_logged(self, caplog):
        def node(_state: dict) -> dict:
            return {}

        with caplog.at_level(logging.DEBUG, logger="graphs.common"):
            timed_node(stage_name("plan", "design_plan"), node)({})

        assert "plan.design_plan: " in caplog.text

    def test_a_node_that_raises_still_raises(self):
        """Timing must not swallow a failure: a node's error is the graph's."""

        def node(_state: dict) -> dict:
            raise RuntimeError("model down")

        with pytest.raises(RuntimeError, match="model down"):
            timed_node(stage_name("probe", "evaluate_batch"), node)({})

    def test_a_recovered_answer_is_recorded_on_the_stage_row(self):
        """The recovery is a fact about this stage, so it belongs in this stage's
        row: what the row already carries, plus why the answer needed help."""
        def node(_state: dict) -> dict:
            record_model_recovery(
                ModelRecovery(reason="backend_refused_json", repaired_escapes=21)
            )
            return {}

        result = timed_node(stage_name("clarify", "assess_goal"), node)({})

        (entry,) = result["stage_timings"]
        assert entry["context"] == {
            "model_recoveries": [{"reason": "backend_refused_json", "repaired_escapes": 21}]
        }

    def test_the_context_names_which_execution_it_was(self):
        """Attribution is what makes the rows usable: a batch, a round, a pass."""

        def node(_state: dict) -> dict:
            return {}

        result = timed_node(
            stage_name("probe", "generate_batch"),
            node,
            context=lambda state: {"question_count": state.get("question_count", 0)},
        )({"question_count": 3})

        (entry,) = result["stage_timings"]
        assert entry["context"] == {"question_count": 3}
