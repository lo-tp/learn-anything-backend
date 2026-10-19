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
- a numeric knob may be asked for as a **ladder** — one value, or several
  separated by commas (:func:`bounded_number_ladder`) — because a stage that
  repeats its call has one deliberate value per repeat, and retuning it needs the
  same shape. Which value applies to which repeat is not this module's business;
  it returns the numbers in the order they were written, together with the name
  that named them, because a warning has to be able to say which setting it is
  objecting to. Numbers that must not be lists — an output cap, a timeout, a
  retry count — go through :func:`bounded_number_setting`, which reports a list
  rather than reading half of it.
- a setting that names *origins* rather than numbers is read by
  :func:`origin_list_setting`, which keeps their order and drops the two shapes
  that could only ever produce a CORS error and no explanation.
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

CORS_ORIGINS_ENV: Final[str] = "CORS_ORIGINS"
"""Extra browser origins this API answers, beyond the one in ``FRONTEND_DOMAIN``."""

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


def _parse_number(raw: str, cast: Callable[[str], float]) -> float | None:
    """``cast(raw)``, or ``None`` when it is not a number.

    The reporting is the caller's, because what a bad value means depends on the
    knob it arrived for: one knob falls back to a default, a ladder treats the
    whole setting as unset.
    """
    try:
        return cast(raw)
    except ValueError:
        return None


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
    try:
        return cast(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using the default %s", name, raw, _described(default)
        )
        return default


def bounded_number_ladder(
    names: Sequence[str],
    cast: Callable[[str], float],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[str, tuple[float, ...]] | None:
    r"""The setting that is set, and the one or more numbers it names.

    A value is a **ladder** when it names several numbers: one per rung of a
    stage that repeats its call. ``material.write_slide`` is that stage — it
    writes a slide again, each attempt deliberately less creative than the last —
    so retuning its ladder from ``.env`` needs one value per rung, not one
    value. The same knob therefore accepts either shape:
    ``LLM_TEMPERATURE_MATERIAL_WRITE_SLIDE=0.9`` is 0.9 at every rung, and
    ``0.9,0.6,0.4`` is a new three-rung ladder. Which rung applies is not this
    function's business; it returns the values in the order they were written.

    Every number in the list is checked the way a single number is, and one bad
    one makes the whole setting unusable: half a ladder is not a ladder, and a
    setting that silently kept two of its three rungs is a behaviour change
    nobody chose. The rejected value is named in the log.

    Returns ``None`` when nothing is set or the setting is unusable, and the
    ``(name, values)`` pair when it is — the name because "which of these two
    settings did I just set?" is the question a warning has to answer.
    """
    found = _first_set_value(names)
    if found is None:
        return None
    name, raw = found
    parts = [part.strip() for part in raw.split(",")]
    if any(not part for part in parts):
        logger.warning(
            "%s=%r is not a list of numbers; treating the whole setting as unset",
            name,
            raw,
        )
        return None
    values: list[float] = []
    for part in parts:
        number = _parse_number(part, cast)
        if number is None:
            logger.warning(
                "%s=%r contains %r, which is not a number; treating the whole "
                "setting as unset",
                name,
                raw,
                part,
            )
            return None
        too_low = minimum is not None and number < minimum
        too_high = maximum is not None and number > maximum
        if too_low or too_high:
            logger.warning(
                "%s=%s is outside %s; treating the whole setting as unset",
                name,
                part,
                _range_text(minimum, maximum),
            )
            return None
        values.append(number)
    return name, tuple(values)


def bounded_number_setting(
    names: Sequence[str],
    default: float | None,
    cast: Callable[[str], float],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    r"""One number the backend will actually accept, or ``default`` when it is not.

    :func:`bounded_number_ladder` is the reading; this is the shape a knob that
    takes exactly one value wants — including the ladder-shaped mistake, which is
    reported and answered with the default rather than guessed at.

    The same rule as :func:`enumerated_setting`, for a range instead of a fixed
    set: ``LLM_TEMPERATURE=5`` is a request the backend answers with HTTP 400,
    and a rejected request is a generation that failed where a knob ignored is
    not.

    ``default`` may be ``None``, meaning *send nothing* — the backend's own
    default, which is what every sampling knob falls back to. That is what makes
    adding a knob a no-op until someone writes it into ``.env``.
    """
    found = bounded_number_ladder(names, cast, minimum=minimum, maximum=maximum)
    if found is None:
        return default
    name, numbers = found
    if len(numbers) != 1:
        logger.warning(
            "%s names %d values where this knob takes one; using the default %s",
            name,
            len(numbers),
            _described(default),
        )
        return default
    return numbers[0]


def origin_list_setting(names: Sequence[str]) -> list[str]:
    r"""A comma-separated list of browser origins, in the order they were written.

    The one setting that is a list of *names* rather than numbers, because the
    thing it lists is a policy: which origins may send a browser to this API.
    A deployment served from one address is the common case; a developer walking
    a phone through the app over their home network is the case that needs a
    second one (``CORS_ORIGINS=http://192.168.31.178:3000``), and a staging
    preview is the case that needs a third.

    Two mistakes are reported rather than carried silently, because both produce
    a request that is refused with no clue left behind:

    - an origin with no scheme (``192.168.31.178:3000``). A browser sends
      ``Origin: http://192.168.31.178:3000``, so a bare host can never match
      anything, and "my phone gets a CORS error and the log says nothing" is the
      exact confusion this warning exists to prevent.
    - ``*``. This app sends credentials (the sign-in cookie), and a browser
      rejects a wildcard origin on a credentialed response — a setting that looks
      like it opens everything and in fact answers nothing.

    An unset setting is an empty list, which means no added policy: whatever the
    caller already allows stands.
    """
    found = _first_set_value(names)
    if found is None:
        return []
    name, raw = found
    origins: list[str] = []
    for part in (piece.strip() for piece in raw.split(",")):
        if not part:
            continue
        if part == "*":
            logger.warning(
                '%s contains "*", which this app cannot use: it sends credentials, '
                "and a browser rejects a wildcard origin on a credentialed response",
                name,
            )
            continue
        if "://" not in part:
            logger.warning(
                '%s contains %r, which names no scheme; a browser sends an Origin like '
                '"http://192.168.31.178:3000", so it can never match and is dropped',
                name,
                part,
            )
            continue
        origins.append(part)
    return origins


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
