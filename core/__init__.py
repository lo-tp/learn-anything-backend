"""Shared infrastructure: the LLM client, language-detection, and auth/security.

Submodules are imported lazily (PEP 562): an eager import here would create
an import cycle, because ``core.security`` imports ``db`` and ``db.models``
imports ``core.mock_llm`` (the MOCK_LLM flag, #118).
"""

from importlib import import_module

_SUBMODULES = ("language", "llm", "mock_llm", "security")


def __getattr__(name: str):
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
