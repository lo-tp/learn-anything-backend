"""How a stage decodes, when someone has said how.

Why this module exists
---------------------
:mod:`core.reasoning` decides what a call may think and how much it may write.
What it has never decided is *how the model chooses the next token* — temperature,
top_p, top_k, min_p, the repetition penalty. Today those live in exactly one
place in this app: the slide-writing ladder in ``graphs/material/sandbox.py``.
Every other call runs on whatever defaults the server at ``OPENAI_BASE_URL``
carries, which on this one is measured below to be a default that decodes the
same way every time — no creativity outside the slide ladder, and no way to ask
for any without a code change.

The consequence is the one #194 hit for thinking: measuring what a stage does when
its temperature changes means editing a table in a graph module and redeploying,
so nobody runs the measurement. Every knob here is read per call from the
environment, in the shape :mod:`core.settings` already defines:

    LLM_TEMPERATURE                       — every call
    LLM_TEMPERATURE_<GRAPH>_<NODE>        — one stage (dots → underscores)
    LLM_TOP_P / LLM_TOP_P_<GRAPH>_<NODE>
    LLM_MIN_P  / LLM_MIN_P_<GRAPH>_<NODE>
    LLM_TOP_K  / LLM_TOP_K_<GRAPH>_<NODE>
    LLM_REPEAT_PENALTY / LLM_REPEAT_PENALTY_<GRAPH>_<NODE>
    LLM_PRESENCE_PENALTY / LLM_FREQUENCY_PENALTY (same two forms)

Each of them takes one number, or several separated by commas — see the ladder
section below.

Precedence, for one stage and one parameter
------------------------------------------
1. ``LLM_<PARAM>_<GRAPH>_<NODE>`` — that stage's own setting;
2. ``LLM_<PARAM>`` — the whole run;
3. what the stage states in code today (the ladder for ``material.write_slide``,
   nothing anywhere else);
4. nothing sent, which is the backend's own default.

The last two are the ones to read before changing anything here. **No knob in
this module has a value of its own.** An unset parameter sends nothing, so every
call keeps exactly the sampling it had before this module existed, and adding a
parameter to the table below changes no behaviour until someone writes it into
``.env``. It is also why setting one globally flattens the slide ladder: a number
a person typed beats a number in a source file, and the point of that ladder was
always to be retunable — which is why a value may also be a list, below.

A value may be a ladder: several numbers, one per repeat of the stage
----------------------------------------------------------------------------
One stage in this app repeats its call under a policy — ``material.write_slide``
writes a slide again, each attempt deliberately less creative than the last — so
retuning *that* from ``.env`` needs one value per rung, not one value. Every
sampling knob therefore accepts either shape:

    LLM_TEMPERATURE_MATERIAL_WRITE_SLIDE=0.9          # 0.9 at every attempt
    LLM_TEMPERATURE_MATERIAL_WRITE_SLIDE=0.4,0.2,0.1  # a new three-rung ladder

The values are positional. Which rung a call gets is decided by the
``stage_position`` the call states (:func:`graphs.common.structured_invoke_messages`),
clamped at both ends exactly as the code-stated ladder clamps itself: a stage
asking more times than the ladder has rungs keeps its last value, and a position
below 1 is the first. Two consequences worth knowing before writing one:

- a setting that names several values is **not** readable by a call that states no
  position, and only ``material.write_slide`` states one today. A stage a graph
  loop asks again — ``probe.generate_batch`` per round — or a stage a revision runs
  again is in the same case until its node says which round it is: reported,
  naming the setting and the stage, and treated as unset rather than guessed at.
  Picking rung 1 because it is first is how a settings mistake becomes a silent
  behaviour change.
- one value is the one-rung ladder, which is why the scalar form and the list form
  are the same mechanism. There is no "global list applies everywhere" case to
  learn: a global list is read by the calls that state a position, and reported to
  the ones that do not.

What the backend reads (measured, not assumed)
----------------------------------------------
Against the project's llama.cpp server (``OPENAI_BASE_URL``, 2026-10-18), through
the schema-enforced path a real stage call uses, not a bare completion request:

- **"unset" is not "the OpenAI defaults" — it is greedy here.** Two identical
  structured requests with no sampling parameter in the body answered with the
  same list, and the same two requests carrying ``extra_body:
  {temperature: 0, top_k: 1, min_p: 1.0}`` answered with *that same* list. This
  server's own default is already the deterministic one, which is the state every
  stage call in this app is in except the slide ladder. Reading a knob as "the
  boring safe default" would be the wrong prior;
- the parameters land when nested: the same structured request twice with
  ``extra_body: {temperature: 1.9, top_p: 0.99, top_k: 300, min_p: 0.0,
  repeat_penalty: 1.0}`` answered differently, and the same request with
  ``{temperature: 0, top_k: 1, min_p: 1.0}`` answered identically. So
  ``extra_body`` reaches the sampler on the grammar-enforced path too, which is
  what the split below is for;
- a top-level ``top_k`` of 1 produced *the same* completion on two requests at
  ``temperature=1.3`` — greedy decoding, so a parameter the backend parses and
  one it ignores are distinguishable this way;
- an unrecognised body parameter is ignored, not rejected: a request carrying
  ``totally_made_up_param`` answered 200. A knob this server does not implement
  therefore looks configured from here and is not honoured — which is why the
  table below holds only parameters this server is known to take;
- ``seed`` is deliberately **not** a knob. Two requests with ``seed=42`` returned
  different completions, so this backend does not honour a per-request seed, and
  a control that is not a control is worse than no control. It belongs here again
  if the backend changes to one that reads it.

Top-level or ``extra_body``
---------------------------
``temperature``, ``top_p``, ``presence_penalty`` and ``frequency_penalty`` are
standard chat-completion fields, so they are sent top-level, which is what both
OpenAI and this backend read. ``top_k``, ``min_p`` and ``repeat_penalty`` are
not: the OpenAI client rejects an unknown top-level parameter, so they are sent
nested under ``extra_body``, which is how they arrive flat in the body the
backend parses. The registry below is what records that split, so a stage's
caller never has to know which transport a parameter uses — see
:func:`apply_configured_sampling`.

What is not here
----------------
The output cap and the reasoning effort are :mod:`core.reasoning`'s; the request
timeout, the retry count and the model itself are the client's
(:mod:`core.llm`). All of them are read through :mod:`core.settings`, so the
whole surface of "how this app talks to a model" is names in ``.env`` and one
rule for reading them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Final, NamedTuple

from core.settings import bounded_number_ladder, stage_setting_names

logger = logging.getLogger(__name__)


class _Param(NamedTuple):
    """One row of the table below: the setting it reads, the values this backend
    will accept, and whether the OpenAI client can carry it top-level."""

    env_name: str
    cast: Callable[[str], float]
    minimum: float | None
    maximum: float | None
    extra_body: bool


SAMPLING_PARAMS: Final[dict[str, _Param]] = {
    # Standard chat-completion fields: top-level, which is what OpenAI and this
    # project's llama.cpp backend both read.
    "temperature": _Param("LLM_TEMPERATURE", float, 0.0, 2.0, False),
    "top_p": _Param("LLM_TOP_P", float, 0.0, 1.0, False),
    "presence_penalty": _Param("LLM_PRESENCE_PENALTY", float, -2.0, 2.0, False),
    "frequency_penalty": _Param("LLM_FREQUENCY_PENALTY", float, -2.0, 2.0, False),
    # Backend extensions: accepted by llama.cpp/vLLM in the request body, and by
    # the OpenAI client only nested under extra_body.
    "top_k": _Param("LLM_TOP_K", int, 1, None, True),
    "min_p": _Param("LLM_MIN_P", float, 0.0, 1.0, True),
    "repeat_penalty": _Param("LLM_REPEAT_PENALTY", float, 0.0, 2.0, True),
}

EXTRA_BODY_PARAMS: frozenset[str] = frozenset(
    name for name, param in SAMPLING_PARAMS.items() if param.extra_body
)

SAMPLING_ENV_NAMES: Final[tuple[str, ...]] = tuple(
    param.env_name for param in SAMPLING_PARAMS.values()
)
"""Every name this module reads, global form. A per-stage name is one of these
with the stage appended, which is what makes a test sweep able to blank them."""


def require_param(name: str) -> _Param:
    """Check a parameter is one this app sends, and return its row.

    A name that is not in the table is a typo in a caller or a parameter nobody
    has measured against this backend yet; naming it here is what keeps an
    invented knob from being sent to a server that would quietly ignore it.
    """
    param = SAMPLING_PARAMS.get(name)
    if param is None:
        known = ", ".join(sorted(SAMPLING_PARAMS))
        raise KeyError(
            f"unknown sampling parameter {name!r}; known parameters: {known}"
        )
    return param


def _setting_values(name: str, stage: str) -> tuple[str, tuple[float, ...]] | None:
    """The setting for ``name`` that this stage actually reaches, and its values."""
    param = require_param(name)
    return bounded_number_ladder(
        stage_setting_names(param.env_name, stage),
        param.cast,
        minimum=param.minimum,
        maximum=param.maximum,
    )


def _rung(count: int, position: int) -> int:
    """Which value of a ladder the call at ``position`` uses, clamped at both ends.

    A stage that asks more times than the ladder has rungs keeps the last one —
    the most deliberate value written — and a position below 1 is the first. This
    is exactly how ``graphs/material/sandbox.py`` already clamps its own profiles,
    so a ladder written in ``.env`` and a ladder written in code behave the same.
    """
    return max(0, min(position - 1, count - 1))


def sampling_value(
    name: str, stage: str, position: int | None = None
) -> float | None:
    r"""The configured value for ``name`` at ``stage``, or ``None`` when there is none.

    That stage's setting, then the global one, then nothing. A value this backend
    would reject — outside the range on the parameter's row, or not a number at
    all — is reported and treated as unset, so the stage's own default or the
    backend's default stands.

    A setting may name several values instead of one: ``0.4,0.2,0.1`` is the
    ``material.write_slide`` ladder retuned in one line. ``position`` is which
    repeat of the stage this call is (1-based) and it picks the rung, clamped at
    both ends by :func:`_rung`. A ladder read by a call that states no position
    is reported and treated as unset: there is nothing to index it against, and
    guessing which rung was meant is how a settings mistake becomes a silent
    behaviour change.
    """
    found = _setting_values(name, stage)
    if found is None:
        return None
    setting_name, values = found
    if len(values) == 1:
        return values[0]
    if position is None:
        logger.warning(
            "%s names %d values for %s, which is a ladder read per repeat of "
            "that stage; this call states no repeat, so it is treated as unset",
            setting_name,
            len(values),
            stage,
        )
        return None
    return values[_rung(len(values), position)]


def stage_sampling_values(
    stage: str, position: int | None = None
) -> dict[str, float]:
    """Every parameter the environment configures for ``stage``.

    Empty when nothing is set, which is the state every stage is in until someone
    writes a knob into ``.env``.
    """
    values: dict[str, float] = {}
    for name in SAMPLING_PARAMS:
        value = sampling_value(name, stage, position)
        if value is not None:
            values[name] = value
    return values


def apply_configured_sampling(
    call_kwargs: Mapping[str, Any], stage: str, position: int | None = None
) -> dict[str, Any]:
    r"""Return ``call_kwargs`` with whatever the environment configures for
    ``stage``, at the repeat ``position`` of that stage.

    This is the one place that joins what a stage states in code (its reasoning
    budget, the slide ladder) with what an operator set in ``.env``, and the
    order is the precedence at the top of this module: the environment sits
    *above* a code-stated default, and a parameter nobody configured is left
    exactly as it was.

    Layered one parameter at a time, ``extra_body`` included: a stage that puts
    three decode knobs in it must not lose two of them because the environment
    set one. Applying this twice is the same as applying it once, which is what
    lets a tool-using turn apply it for both halves of itself
    (:func:`graphs.common.structured_invoke_with_tools`).

    With nothing configured it returns a copy of ``call_kwargs``, so a stage that
    is not configured sends byte-identically what it sent before this existed.
    """
    merged = dict(call_kwargs)
    configured = stage_sampling_values(stage, position)
    if not configured:
        return merged

    extra_body = dict(merged.get("extra_body") or {})
    for name, value in configured.items():
        if require_param(name).extra_body:
            extra_body[name] = value
        else:
            merged[name] = value
    if extra_body:
        merged["extra_body"] = extra_body
    return merged
