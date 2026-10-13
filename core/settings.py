"""One way to read a tunable environment setting.

Several parts of this app are tunable without a code change: how much a model
call may think (`core/reasoning.py`), how long it may take (`core/llm.py`), how
soon an unreachable search service is asked again (`core/external_tools.py`).
Each of them used to carry its own parse-and-fallback block, which is how three
copies of the same rule drifted apart (#194 review). The rule is short and it
should only be written once:

- the first name that is set wins, so a per-call setting can sit ahead of a
  global one;
- a value this app cannot use is reported, naming the bad value, and the default
  stands — a typo in a knob must not be the reason a request fails or the app
  does not start.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Final

logger = logging.getLogger(__name__)

LLM_MAX_OUTPUT_TOKENS_ENV: Final[str] = "LLM_MAX_OUTPUT_TOKENS"
"""Cap on what one model call may write, wherever a call has a cap."""

REASONING_EFFORT_ENV: Final[str] = "REASONING_EFFORT"
"""How much a model call may think, wherever a call has a budget."""

REASONING_EFFORTS: Final[frozenset[str]] = frozenset(
    {"none", "minimal", "low", "medium", "high"}
)
"""Accepted by this project's llama.cpp backend (mapped onto the Qwen3 think
switch) and by OpenAI. Anything else is a request the backend answers with 400."""


def _first_set_value(names: Sequence[str]) -> tuple[str, str] | None:
    """The first ``(env name, raw value)`` that is set."""
    for name in names:
        raw = os.getenv(name, "").strip()
        if raw:
            return name, raw
    return None


def numeric_setting[NumberT](
    names: Sequence[str], default: NumberT, cast: Callable[[str], NumberT]
) -> NumberT:
    """A number from the environment, or ``default`` when it is absent or unreadable."""
    found = _first_set_value(names)
    if found is None:
        return default
    name, raw = found
    try:
        return cast(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using the default %s", name, raw, default)
        return default


def enumerated_setting(
    names: Sequence[str], allowed: Iterable[str], default: str
) -> str:
    """A value from a fixed set, or ``default`` when it is absent or not one of them.

    The comparison is case-insensitive; the value returned is the lowercase one,
    which is what the backends expect.
    """
    allowed_set = {value.casefold() for value in allowed}
    found = _first_set_value(names)
    if found is None:
        return default
    name, raw = found
    value = raw.casefold()
    if value in allowed_set:
        return value
    logger.warning(
        "%s=%r is not one of %s; using the default %r",
        name,
        raw,
        ", ".join(sorted(allowed_set)),
        default,
    )
    return default
