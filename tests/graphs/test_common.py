"""Tests for graphs/common.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from graphs.common import (
    _MAX_TOOL_CALL_ROUNDS,
    structured_invoke,
    structured_invoke_messages,
    structured_invoke_with_tools,
    unknown_option,
    with_unknown_option,
)
from tests.tool_shapes import mcp_shaped_tool as _mcp_shaped_tool

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
            result = structured_invoke(llm, _DummySchema, "sys prompt", "user prompt")
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
            structured_invoke(llm, _DummySchema, "s", "h")
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
                llm, _DummySchema, "s", "h", reasoning_effort="minimal"
            )
            assert mock_fn.call_args.kwargs == {"reasoning_effort": "minimal"}

    def test_no_bind_kwargs_by_default(self):
        """Call sites that pass nothing must keep sending nothing."""
        llm = MagicMock()
        mock_result = _DummySchema(value="x")
        with patch(
            "graphs.common.structured_invoke_messages", return_value=mock_result
        ) as mock_fn:
            structured_invoke(llm, _DummySchema, "s", "h")
            assert mock_fn.call_args.kwargs == {}


# --- structured_invoke_messages ---


class TestStructuredInvokeMessages:
    def test_calls_with_structured_output_and_invoke(self):
        llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _DummySchema(value="ok")
        llm.with_structured_output.return_value = mock_structured

        result = structured_invoke_messages(
            llm, _DummySchema, [HumanMessage(content="hi")]
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
            llm, _DummySchema, [HumanMessage(content="hi")], temperature=0.1
        )
        assert result.value == "ok"
        # bind must wrap the structured-output model (i.e. run AFTER
        # with_structured_output) and invoke must hit the bound model.
        mock_structured.bind.assert_called_once_with(temperature=0.1)
        mock_bound.invoke.assert_called_once()
        mock_structured.invoke.assert_not_called()


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
                llm, _DummySchema, "sys", "human", []
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
            structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool])
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
            structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool])
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
            structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool])

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
            result = structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool])

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
            result = structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool])
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
            structured_invoke_with_tools(llm, _DummySchema, "s", "h", [tool])
        assert len(calls) == _MAX_TOOL_CALL_ROUNDS
        assert llm.bind_tools.return_value.invoke.call_count == _MAX_TOOL_CALL_ROUNDS
        # The answer is still produced over the accumulated history.
        assert len(structured.call_args[0][2]) == 2 + 2 * _MAX_TOOL_CALL_ROUNDS
