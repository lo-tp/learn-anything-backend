"""Test stubs for language prompt snippets (no private content)."""

DETECT_LANGUAGE_SYSTEM = "STUB language.DETECT_LANGUAGE_SYSTEM"


def language_instruction(language: str) -> str:
    return f"STUB language.language_instruction({language})"


def localize_status_system(language: str) -> str:
    return f"STUB language.localize_status_system({language})"
