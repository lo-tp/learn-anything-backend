"""Tests for core/language.py — the status-line translation path (#194).

Seam: :func:`core.language.localize_status`, the public boundary the graphs and
routers use for short system messages. The count of LLM calls it makes is part
of its contract: it is called once per generated MCQ option set and on plan
approval, so an uncached, un-short-circuited implementation costs a full model
round trip per question.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.language import (
    DEFAULT_LANGUAGE,
    clear_status_translation_cache,
    detect_language,
    localize_status,
)


@pytest.fixture(autouse=True)
def _empty_translation_cache():
    """The translation cache is process-global; each test starts clean."""
    clear_status_translation_cache()
    yield
    clear_status_translation_cache()


def _reply(text: str) -> MagicMock:
    """A model that answers ``text``, whichever call path the helper takes.

    Every path gets its own response object and a ``side_effect``, so no test
    depends on which path the helper chose — except where that is the point (the
    reasoning budget). One shared response object would be re-parented by each
    assignment, which is how a mock starts reporting a call the code never made.
    """
    llm = MagicMock()

    def answer(*_args, **_kwargs):
        return MagicMock(content=text)

    for path in (
        llm,
        llm.bind.return_value,
        llm.with_structured_output.return_value,
        llm.with_structured_output.return_value.bind.return_value,
    ):
        path.invoke.side_effect = answer
    return llm


class TestLocalizeStatus:
    def test_english_status_never_reaches_the_llm(self):
        """A status line is already written in English; asking English back
        is a no-op, and it used to be a model call per question."""
        llm = _reply("I don't know")

        assert localize_status(llm, DEFAULT_LANGUAGE, "I don't know") == "I don't know"
        llm.bind.assert_not_called()
        llm.invoke.assert_not_called()

    def test_english_status_is_returned_whatever_its_capitalization(self):
        llm = _reply("I don't know")

        assert localize_status(llm, "english", "Plan approved.") == "Plan approved."
        llm.bind.assert_not_called()

    def test_other_languages_are_translated(self):
        llm = _reply("Je ne sais pas")

        assert localize_status(llm, "French", "I don't know") == "Je ne sais pas"
        llm.bind.assert_called_once()

    def test_a_fixed_status_is_translated_once_per_language(self):
        """Ten questions in one batch share one translation, not ten calls."""
        llm = _reply("Je ne sais pas")

        for _ in range(10):
            assert localize_status(llm, "French", "I don't know") == "Je ne sais pas"

        llm.bind.assert_called_once()

    def test_each_language_is_translated_for_itself(self):
        llm = _reply("translated")

        localize_status(llm, "French", "I don't know")
        localize_status(llm, "German", "I don't know")

        assert llm.bind.call_count == 2

    def test_a_failed_translation_falls_back_to_the_original_text(self):
        """A status message is always returned: translation is optional."""
        llm = MagicMock()
        llm.bind.return_value.invoke.side_effect = RuntimeError("model down")

        assert localize_status(llm, "French", "I don't know") == "I don't know"

    def test_an_empty_translation_falls_back_to_the_original_text(self):
        llm = _reply("   ")

        assert localize_status(llm, "French", "I don't know") == "I don't know"

    def test_no_llm_returns_the_original_text(self):
        assert localize_status(None, "French", "I don't know") == "I don't know"

    def test_blank_status_returns_blank(self):
        llm = _reply("x")

        assert localize_status(llm, "French", "   ") == ""
        llm.bind.assert_not_called()

    def test_cleared_cache_translates_again(self):
        """The cache is process-global, so clearing it is a public operation."""
        llm = _reply("Je ne sais pas")
        localize_status(llm, "French", "I don't know")

        clear_status_translation_cache()
        localize_status(llm, "French", "I don't know")

        assert llm.bind.call_count == 2


class TestLanguageCallsStateTheirBudget:
    """#194: the helpers every phase calls are bounded calls too. These two are
    the cheapest work in the system — a two-letter answer and two words — and
    they were the least defensible model round trips in it."""

    def test_the_translation_call_states_its_budget(self):
        llm = _reply("Je ne sais pas")

        localize_status(llm, "French", "I don't know")

        llm.bind.assert_called_once_with(reasoning_effort="minimal", max_tokens=256)

    def test_short_text_language_detection_states_its_budget(self):
        """Text too short for ``langid`` is the only case detection asks a model."""
        from core.language import _LanguageDetectionOut

        llm = MagicMock()
        structured = llm.with_structured_output.return_value.bind.return_value
        structured.invoke.return_value = _LanguageDetectionOut(language="es")

        assert detect_language("hola", llm) == "Spanish"

        llm.with_structured_output.return_value.bind.assert_called_once_with(
            reasoning_effort="minimal", max_tokens=256
        )
        structured.invoke.assert_called_once()
