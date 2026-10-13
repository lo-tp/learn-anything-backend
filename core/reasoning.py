"""What each phase may think, and how much it may say (#194).

The measured fact this module exists for: on this project's llama.cpp backend
(``qwen3.8-flash-next-iq3_xxs``) the backend default is thinking **ON**, and one
plain call produced 1544 completion tokens of which 6477 characters were
reasoning, against 817 with ``reasoning_effort="minimal"`` for the same prompt
— ~2x the wall time per call, for an answer the prompt asked for in a few
hundred words. The request also carried no output cap at all
(``core/llm.py`` sets ``max_tokens=65536`` globally), so nothing bounded a
runaway think loop either.

Slide generation was already fixed for this (`SLIDE_REASONING_EFFORT` in
``graphs/material/sandbox.py``); everything before it — Clarify, Probe, Plan,
the language helpers, and the quiz/summary half of Material — was not, and those
are the calls the learner waits through one at a time.

Why the table is not one value
------------------------------
The phases are not the same kind of step. A phase that *decides* something
(what must be learned first, whether an answer is right, how steps depend on
each other) keeps thinking, but within a budget: on this backend the mapping is
``minimal``/``none`` → no reasoning at all, ``low``/``medium``/``high`` → a
capped amount (see the measurement recorded in ``sandbox.py``: 148-300 reasoning
characters, against 6477 unconstrained). A phase that only *restates* a decision
made by the phase before it — rendering IDs and a mermaid DAG, turning a named
dimension into a question, folding an answer into a goal — needs none.

Each entry also carries an output cap, so a mis-set effort cannot run to the
client's global 65536. The caps sit well above the largest output each prompt
asks for; they are a runaway guard, not a budget to fill.

What lives here and what does not
---------------------------------
``write_slide`` and ``plan_slide_contents`` keep their own value in
``graphs/material/sandbox.py``, deliberately: ``SLIDE_REASONING_EFFORT`` is bound
by ``slide_sampling_bind_kwargs``, the one place that assembles a slide-writing
call, and it belongs beside the sampling ladder that measurement produced. This
module is the authority for every *other* call.

The knobs are read per call (same pattern as ``is_mock_mode``): retuning a phase
to measure its effect is an environment change, not a code change.

    REASONING_EFFORT                        — every phase
    REASONING_EFFORT_PLAN_DESIGN_PLAN       — one phase (dots → underscores)
    LLM_MAX_OUTPUT_TOKENS                   — every phase's output cap

A value the backend would reject (an unknown effort, a non-numeric cap) is
reported and the phase default is used instead: a typo must not send a request
the backend answers with HTTP 400.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Final

logger = logging.getLogger(__name__)

# Accepted by this project's llama.cpp backend (mapped onto the Qwen3 think
# switch) and by OpenAI. Anything else is a rejected request.
REASONING_EFFORTS: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high"}
)

_EFFORT_GLOBAL_ENV = "REASONING_EFFORT"
_CAP_GLOBAL_ENV = "LLM_MAX_OUTPUT_TOKENS"

# phase -> (reasoning_effort, max output tokens)
_PHASE_BUDGETS: Final[dict[str, tuple[str, int]]] = {
    # --- Clarify ---
    # Decides whether the goal is specific enough, and what is still open.
    "clarify.assess_goal": ("low", 1536),
    # Restates an open dimension as a question: the decision is already made.
    "clarify.generate_questions": ("minimal", 1024),
    # Folds the learner's answer into the working goal.
    "clarify.refine_goal": ("minimal", 1024),
    # --- Probe ---
    # Enumerates the prerequisite strands: the whole boundary map hangs on this.
    "probe.decompose_strands": ("low", 1536),
    # Writes questions aimed at a boundary map that is already known, but the
    # difficulty ladder and the distractors are a judgement.
    "probe.generate_batch": ("low", 3072),
    # Judges each answer and moves the floor/ceiling of every strand it touched.
    "probe.evaluate_batch": ("low", 3072),
    # --- Plan ---
    # Chooses whether to search and reads the results back into the topic.
    "plan.research_topic": ("low", 3072),
    # Orders the steps into a dependency DAG.
    "plan.design_plan": ("low", 4096),
    # Pure formatting: step IDs, labels, a mermaid DAG, a summary of a plan that
    # already exists.
    "plan.render_plan": ("minimal", 6144),
    # --- Language helpers (every phase calls these) ---
    "language.detect": ("minimal", 256),
    "language.localize_status": ("minimal", 256),
    # --- The non-slide half of Material ---
    # Left untouched by the slide-side fix; writing a quiz is a judgement.
    "material.write_questions": ("low", 4096),
    # Condenses the step into the key points every later step is told to connect
    # to, so its output quality is load-bearing downstream.
    "material.summarize_step": ("low", 3072),
}

PHASES: frozenset[str] = frozenset(_PHASE_BUDGETS)


def _phase_env_var(name: str, phase: str) -> str:
    """The per-phase override for a global knob name."""
    return f"{name}_{phase.replace('.', '_').upper()}"


def _budget(phase: str) -> tuple[str, int]:
    """The configured budget for a phase, naming the phase if it is unknown.

    An unknown phase is a programming error at a call site (a mistyped name),
    and the list of known phases is the correction it needs.
    """
    budget = _PHASE_BUDGETS.get(phase)
    if budget is None:
        known = ", ".join(sorted(PHASES))
        raise KeyError(f"unknown phase {phase!r}; known phases: {known}")
    return budget


def reasoning_effort_for(phase: str) -> str:
    """The reasoning effort to send for ``phase``.

    Per-phase override, then global override, then the phase default. A value
    outside :data:`REASONING_EFFORTS` is reported and the phase default stands.
    """
    effort_default, _ = _budget(phase)
    for name in (_phase_env_var(_EFFORT_GLOBAL_ENV, phase), _EFFORT_GLOBAL_ENV):
        configured = os.getenv(name, "").strip().lower()
        if not configured:
            continue
        if configured in REASONING_EFFORTS:
            return configured
        logger.warning(
            "%s=%r is not a reasoning effort this backend accepts (%s); "
            "using the %s default %r",
            name,
            configured,
            ", ".join(sorted(REASONING_EFFORTS)),
            phase,
            effort_default,
        )
    return effort_default


def max_output_tokens_for(phase: str) -> int:
    """The output cap to send for ``phase`` (``LLM_MAX_OUTPUT_TOKENS`` override)."""
    _, cap_default = _budget(phase)
    configured = os.getenv(_CAP_GLOBAL_ENV, "").strip()
    if not configured:
        return cap_default
    try:
        return int(configured)
    except ValueError:
        logger.warning(
            "%s=%r is not a token count; using the %s default %d",
            _CAP_GLOBAL_ENV,
            configured,
            phase,
            cap_default,
        )
        return cap_default


def phase_call_kwargs(phase: str) -> dict[str, Any]:
    """Bind kwargs for one call of ``phase``, ready to spread into
    :func:`graphs.common.structured_invoke`.

    Both values are always present: an unbounded call is exactly the shape #194
    is about.
    """
    return {
        "reasoning_effort": reasoning_effort_for(phase),
        "max_tokens": max_output_tokens_for(phase),
    }
