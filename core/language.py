"""Language detection and user-facing-language prompt helpers.

The system detects the language the learner is using (their goal, clarification
answers, and plan adjustments) and uses that language for every user-facing
reply and for the generated learning materials (slides, quiz questions,
explanations, and concept summaries).

Detection strategy
------------------
1. ``langid`` (fast, deterministic n-gram classifier) is used for text that is
   long enough to be reliable — a full goal sentence or paragraph.
2. Very short / low-signal text (a single foreign word, a fragment) is routed
   to a small constrained LLM call, which handles those cases better than a
   classifier.
3. If detection is unavailable or fails, we default to English.

The detected language is stored on the ``Session`` row and threaded through
every graph so all user-facing output stays in the learner's language.
"""

from __future__ import annotations

from typing import cast

import langid
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.prompts import (
    DETECT_LANGUAGE_SYSTEM,
    language_instruction,
    localize_status_system,
)

#: Language used when detection is unavailable or the text has no signal.
DEFAULT_LANGUAGE = "English"

# Below this length ``langid`` is unreliable (it returns its default 'en'
# sentinel). Short text is routed to the LLM fallback instead; short *reply*
# text (e.g. "yes") is not used to switch the session language at all.
MEANINGFUL_TEXT_LEN = 12

# ISO 639-1 -> human-readable name. Names read better in a prompt than a bare
# code; unknown codes fall back to the code string itself.
_CODE_TO_NAME: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "nb": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "is": "Icelandic",
    "pl": "Polish",
    "cs": "Czech",
    "sk": "Slovak",
    "sl": "Slovenian",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "ru": "Russian",
    "uk": "Ukrainian",
    "be": "Belarusian",
    "el": "Greek",
    "tr": "Turkish",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
    "ur": "Urdu",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "my": "Burmese",
    "km": "Khmer",
    "lo": "Lao",
    "sw": "Swahili",
    "am": "Amharic",
    "ha": "Hausa",
    "yo": "Yoruba",
    "zu": "Zulu",
    "af": "Afrikaans",
    "ca": "Catalan",
    "eu": "Basque",
    "ga": "Irish",
    "cy": "Welsh",
    "mk": "Macedonian",
    "sr": "Serbian",
    "bs": "Bosnian",
    "hr": "Croatian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
}

_COMMON_CODE_HINT = ", ".join(sorted(_CODE_TO_NAME))


class _LanguageDetectionOut(BaseModel):
    language: str = Field(
        description=(
            "The language the user wrote the text in, as a 2-letter ISO 639-1 "
            f"code (one of: {_COMMON_CODE_HINT}). Return 'en' if you cannot tell."
        )
    )


def _code_to_name(code: str) -> str:
    return _CODE_TO_NAME.get((code or "").lower(), code)


def has_meaningful_signal(text: str) -> bool:
    """True if ``text`` is long enough to reliably detect a language from.

    Used to gate *re-detection* on reply text: a one-word answer like "yes"
    must not flip the session language.
    """
    return len((text or "").strip()) >= MEANINGFUL_TEXT_LEN


def detect_language(text: str, llm: BaseChatModel | None = None) -> str:
    """Detect the learner's language and return a human-readable name.

    ``llm`` is a fallback for short / low-signal text; if it is ``None`` or the
    call fails, detection falls back to :data:`DEFAULT_LANGUAGE`.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return DEFAULT_LANGUAGE

    if len(cleaned) < MEANINGFUL_TEXT_LEN:
        return _llm_detect(cleaned, llm) if llm is not None else DEFAULT_LANGUAGE

    try:
        code = langid.classify(cleaned)[0]
    except Exception:  # noqa: BLE001 — classifier may raise on odd input
        return _llm_detect(cleaned, llm) if llm is not None else DEFAULT_LANGUAGE
    return _code_to_name(code)


def _llm_detect(text: str, llm: BaseChatModel | None) -> str:
    """Classify the language of short text with a constrained LLM call."""
    if llm is None:
        return DEFAULT_LANGUAGE
    try:
        out = cast(
            _LanguageDetectionOut,
            llm.with_structured_output(_LanguageDetectionOut).invoke(
                [
                    SystemMessage(content=DETECT_LANGUAGE_SYSTEM),
                    HumanMessage(content=text),
                ]
            ),
        )
        return _code_to_name(out.language)
    except Exception:  # noqa: BLE001 — any LLM/schema failure -> default
        return DEFAULT_LANGUAGE


# ``language_instruction`` is provided by ``core.prompts`` (private prompts).
# See imports above; re-exported here for graph nodes.


def localize_status(llm: BaseChatModel | None, language: str, english: str) -> str:
    """Translate a fixed English status line into ``language``.

    Used for short system messages (e.g. "Plan approved…") that are produced by
    a router rather than a domain node. Returns the original English string if
    ``llm`` is unavailable or the call fails, so a status message is always
    returned.
    """
    text = (english or "").strip()
    if not text or llm is None:
        return text
    system = localize_status_system(language)
    try:
        resp = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=text)]
        )
        translated = (getattr(resp, "content", "") or "").strip()
        return translated or text
    except Exception:  # noqa: BLE001 — any failure -> original English
        return text
