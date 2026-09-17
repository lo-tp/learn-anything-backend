"""Tests for graphs/common.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from graphs.common import (
    structured_invoke,
    structured_invoke_messages,
    unknown_option,
    with_unknown_option,
)

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
