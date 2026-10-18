"""One way to read a tunable environment setting.

Several parts of this app are tunable without a code change: how much a model
call may think (`core/reasoning.py`), how it decodes (`core/sampling.py`), how
long it may take (`core/llm.py`), how soon an unreachable search service is
asked again (`core/external_tools.py`). Each of them used to carry its own
parse-and-fallback block, which is how three copies of the same rule drifted
apart (#194 review). The rule is short and it should only be written once:

- the first name that is set wins, so a per-call setting can sit ahead of a
  global one;
- a value this app cannot use is reported, naming the bad value, and the default
  stands — a typo in a knob must not be the reason a request fails or the app
  does not start. "Cannot use" covers both shapes a bad value arrives in: not one
  of an enumerated set, or outside the range a backend will accept.
- the two names a per-stage knob has are derived from the stage name in one
  place, :func:`stage_setting_names`, so two modules with knobs on the same
  stage read the same variable.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Final

logger = logging.getLogger(__name__)

LLM_MAX_OUTPUT_TOKENS_ENV: Final[str] = "LLM_MAX_OUTPUT_TOKENS"
"""Cap on what one model call may write, wherever a call has a cap."""

LLM_REQUEST_TIMEOUT_SECONDS_ENV: Final[str] = "LLM_REQUEST_TIMEOUT_SECONDS"
"""How long one model request may take before this app gives up on it."""

LLM_MAX_RETRIES_ENV: Final[str] = "LLM_MAX_RETRIES"
"""How many times a request that never answered is asked again."""

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


def _parse_number[NumberT](
    name: str, raw: str, cast: Callable[[str], NumberT], default: NumberT | None
) -> NumberT | None:
    """``cast(raw)``, or ``default`` with the bad value named in the log."""
    try:
        return cast(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using the default %s",
            name,
            raw,
            _described(default),
        )
        return default


def _described(value: object) -> str:
    """How to say a fallback in one sentence, including "send nothing"."""
    if value is None:
        return "the backend's own default (nothing is sent)"
    return str(value)


def _range_text(minimum: float | None, maximum: float | None) -> str:
    if minimum is not None and maximum is not None:
        return f"between {minimum} and {maximum}"
    if minimum is not None:
        return f"at least {minimum}"
    if maximum is not None:
        return f"at most {maximum}"
    return "a number the backend can use"


def numeric_setting[NumberT](
    names: Sequence[str], default: NumberT, cast: Callable[[str], NumberT]
) -> NumberT:
    """A number from the environment, or ``default`` when it is absent or unreadable."""
    found = _first_set_value(names)
    if found is None:
        return default
    name, raw = found
    parsed = _parse_number(name, raw, cast, default)
    return default if parsed is None else parsed


def bounded_number_setting(
    names: Sequence[str],
    default: float | None,
    cast: Callable[[str], float],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    r"""A number the backend will actually accept, or ``default`` when it is not one.

    ``cast`` may be :func:`int` — an integer is a number a ``float`` annotation
    accepts, and the value that reaches the request is the one that was written.

    The same rule as :func:`enumerated_setting`, for a range instead of a fixed
    set: ``LLM_TEMPERATURE=5`` is a request the backend answers with HTTP 400,
    and a rejected request is a generation that failed where a knob ignored is
    not. The value that was rejected is named in the log.

    ``default`` may be ``None``, meaning *send nothing* — the backend's own
    default, which is what every sampling knob here falls back to. That is what
    makes adding a knob a no-op until someone writes it into ``.env``.
    """
    found = _first_set_value(names)
    if found is None:
        return default
    name, raw = found
    value = _parse_number(name, raw, cast, default)
    if value is None:
        return default
    too_low = minimum is not None and value < minimum
    too_high = maximum is not None and value > maximum
    if too_low or too_high:
        logger.warning(
            "%s=%s is outside %s; using the default %s",
            name,
            raw,
            _range_text(minimum, maximum),
            _described(default),
        )
        return default
    return value


def stage_setting_names(base_name: str, stage: str) -> tuple[str, str]:
    r"""The per-stage name for ``base_name``, then the global one.

    ``("LLM_TEMPERATURE", "plan.design_plan")`` →
    ``("LLM_TEMPERATURE_PLAN_DESIGN_PLAN", "LLM_TEMPERATURE")``.

    Every knob this app reads per stage — reasoning effort, output cap, decoding
    — is named this way, so the rule for turning a stage name into a setting
    name is written once, here, rather than in each module that has knobs.
    """
    return (f"{base_name}_{stage.replace('.', '_').upper()}", base_name)


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
