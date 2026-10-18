"""Tests for core/language.py — the status-line translation path (#194).

Seam: :func:`core.language.localize_status`, the public boundary the graphs and
routers use for short system messages. The count of LLM calls it makes is part
of its contract: it is called once per generated MCQ option set and on plan
approval, so an uncached, un-short-circuited implementation costs a full model
round trip per question.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import core.language
from core.language import (
    DEFAULT_LANGUAGE,
    clear_short_text_language_cache,
    clear_status_translation_cache,
    detect_language,
    localize_status,
)


@pytest.fixture(autouse=True)
def _empty_caches():
    """Both caches are process-global; each test starts clean."""
    clear_status_translation_cache()
    clear_short_text_language_cache()
    yield
    clear_status_translation_cache()
    clear_short_text_language_cache()


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
        structured.invoke.return_value = _LanguageDetectionOut(language="zh")

        assert detect_language("hola", llm) == "Chinese"

        llm.with_structured_output.return_value.bind.assert_called_once_with(
            reasoning_effort="minimal", max_tokens=256
        )
        structured.invoke.assert_called_once()


def _classifier(code: str) -> tuple[MagicMock, MagicMock]:
    """A model whose language classification answers ``code``, with its invoke.

    A plain MagicMock would do for the answer except that the helper calls
    ``.lower()`` on it, so the answer is a real string in a real object.
    """
    llm = MagicMock()
    structured = llm.with_structured_output.return_value.bind.return_value
    structured.invoke.return_value = SimpleNamespace(language=code)
    return llm, structured.invoke


class TestShortTextLanguageDetection:
    """#194: every learner reply under the classifier's reliable length goes to
    the model, and a session answers with the same short replies all the way
    through it — "yes", "no", "I don't know" — one round trip per ask."""

    def test_the_same_short_reply_is_classified_once(self):
        llm, invoke = _classifier("en")
        for _ in range(4):
            assert detect_language("yes", llm) == "English"
        assert invoke.call_count == 1

    def test_a_reply_is_recognised_whatever_its_capitalisation(self):
        llm, invoke = _classifier("en")
        assert detect_language("Yes", llm) == "English"
        assert detect_language("  yes  ", llm) == "English"
        assert invoke.call_count == 1

    def test_different_replies_are_classified_separately(self):
        llm, invoke = _classifier("en")
        detect_language("yes", llm)
        detect_language("maybe not", llm)
        assert invoke.call_count == 2

    def test_a_failed_classification_is_not_remembered(self):
        """A model that was down must not become the session's language forever."""
        llm = MagicMock()
        structured = llm.with_structured_output.return_value.bind.return_value
        structured.invoke.side_effect = [
            ConnectionError("model down"),
            SimpleNamespace(language="zh"),
        ]
        assert detect_language("oui", llm) == DEFAULT_LANGUAGE
        assert detect_language("oui", llm) == "Chinese"
        assert structured.invoke.call_count == 2

    def test_text_the_classifier_can_read_never_reaches_the_model(self):
        llm, invoke = _classifier("fr")
        assert detect_language("Je veux apprendre le calcul differentiel", llm) == DEFAULT_LANGUAGE
        assert invoke.call_count == 0

    def test_the_cache_is_bounded(self):
        """Its keys are learner text, which is not a fixed set, so the number of
        remembered texts has a limit rather than growing with the session."""
        llm, _ = _classifier("en")
        for i in range(core.language.MAX_DETECTED_TEXTS + 50):
            detect_language(f"reply number {i} x", llm)
        assert len(core.language._SHORT_TEXT_LANGUAGES) <= core.language.MAX_DETECTED_TEXTS

    def test_cleared_cache_classifies_again(self):
        llm, invoke = _classifier("en")
        detect_language("yes", llm)
        clear_short_text_language_cache()
        detect_language("yes", llm)
        assert invoke.call_count == 2


class TestOnlyEnglishAndChineseAreSupported:
    """What detection is allowed to name. The app writes learner-facing copy in
    English and Chinese only; every path — classifier, fallback model, or a
    model that misbehaves — resolves into one of those two or the default."""

    def test_classifier_chinese_text_resolves_to_chinese(self):
        llm, invoke = _classifier("en")  # inverted: reaching the model says English
        assert detect_language("我想学习微积分和线性代数。", llm) == "Chinese"
        assert invoke.call_count == 0

    def test_classifier_english_text_resolves_to_english(self):
        llm, invoke = _classifier("zh")  # inverted: reaching the model says Chinese
        assert detect_language("I want to learn linear algebra and calculus.", llm) == "English"
        assert invoke.call_count == 0

    def test_a_language_outside_the_two_resolves_to_the_default(self):
        """A learner may write in any language; Chinese is the only second one."""
        assert detect_language("ありがとう、微積分を勉強したいです。") == DEFAULT_LANGUAGE
        assert detect_language("Je veux apprendre le calcul différentiel.") == DEFAULT_LANGUAGE

    def test_short_chinese_text_resolves_through_the_model_path(self):
        llm, _ = _classifier("zh")
        assert detect_language("我想学", llm) == "Chinese"

    def test_a_model_answer_outside_the_two_codes_is_the_default(self):
        """The schema asks for the two codes; the mapping keeps that promise
        even if a backend hands back something else."""
        llm, _ = _classifier("ja")
        assert detect_language("ありがとう", llm) == DEFAULT_LANGUAGE

    def test_the_detection_schema_accepts_only_the_two_codes(self):
        from core.language import _LanguageDetectionOut

        assert _LanguageDetectionOut(language="zh").language == "zh"
        assert _LanguageDetectionOut(language="en").language == "en"
        with pytest.raises(ValidationError):
            _LanguageDetectionOut.model_validate({"language": "fr"})
