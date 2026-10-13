"""What each stage may think, and how much it may say (#194).

The measured fact this module exists for
---------------------------------------
On this project's llama.cpp backend (``qwen3.8-flash-next-iq3_xxs``) the backend
default is thinking **ON**, and the thinking is unbounded. The same request —
weigh three first-month plans against what a learner said, and pick one — run
three ways:

======================  ==========  ===============  ==================  ==========
request                 wall time   completion cap  reasoning reached   answered
======================  ==========  ===============  ==================  ==========
no ``reasoning_effort``  71.2 s      8192 tokens     37,803 characters  **no**
``reasoning_effort=low``  21.2 s     1865 tokens      2,136 characters  yes
``…=minimal``             17.3 s      1471 tokens          0 characters  yes
======================  ==========  ===============  ==================  ==========

The first row is the shape of every pre-slide call this app was making: it spent
its whole token allowance thinking, hit ``finish_reason="length"`` and returned
**no content at all**. An earlier measurement (#194) caught the same thing at a
smaller scale — 1544 completion tokens of which 6477 characters were reasoning,
against 817 with ``minimal`` for the same prompt — which is where the ~2x per
call came from.

So the fix has two halves and the caps are not a nicety: a cap alone would have
truncated that 37,803-character think loop into a *shorter* answer, and effort
alone would still have left nothing bounding a runaway loop. A call that ends on
``finish_reason="length"`` is a silent quality bug either way, and nothing here
reports it yet.

Stage, not phase
----------------
A **stage** is one named model call inside one graph — ``clarify.assess_goal``,
``plan.design_plan`` — which is what ``graph_stage_timings.stage`` already calls
it, and what the graphs pass to :func:`graphs.common.timed_node`. A **phase** in
this project's glossary (``CONTEXT.md``, and ``db.models.Phase``) is the coarser
clarifying → probing → planning → reviewing → generating sequence. Getting a
slide written is a phase; asking the model to order the step DAG is a stage, and
this module budgets stages.

Why the table is not one value
------------------------------
The stages are not the same kind of step. A stage that *decides* something (what
must be learned first, whether an answer is right, how steps depend on each
other) keeps thinking, but within a budget: on this backend the mapping is
``minimal``/``none`` → no reasoning at all, ``low``/``medium``/``high`` → a
capped amount (see the measurement recorded in ``graphs/material/sandbox.py``:
148-300 reasoning characters, against 6477 unconstrained). A stage that only
*restates* a decision made by the stage before it — rendering IDs and a mermaid
DAG, turning a named dimension into a question, folding an answer into a goal —
needs none.

Each entry also carries an output cap, so a mis-set effort cannot run to the
client's global 65536. The caps sit well above the largest output each prompt
asks for; they are a runaway guard, not a budget to fill. Raising a stage's
effort without raising its cap is how a call gets truncated — both knobs have a
per-stage override for exactly that experiment.

What lives here and what does not
---------------------------------
The slide-writing stages keep their own values in ``graphs/material/sandbox.py``,
deliberately: ``SLIDE_REASONING_EFFORT`` and ``SLIDE_MAX_OUTPUT_TOKENS`` are
bound by ``slide_sampling_bind_kwargs``, the one place that assembles a
slide-writing call, and they belong beside the sampling ladder that measurement
produced. This module is the authority for every *other* call — and for
``language.detect`` / ``language.localize_status``, which are called from inside
stages rather than being graph stages themselves, so they are named by the
helper they belong to rather than a graph.

The knobs are read per call through :mod:`core.settings` (same pattern as
``is_mock_mode``): retuning a stage to measure its effect is an environment
change, not a code change.

    REASONING_EFFORT                          — every stage
    REASONING_EFFORT_<GRAPH>_<NODE>           — one stage (dots → underscores)
    LLM_MAX_OUTPUT_TOKENS                     — every stage's output cap
    LLM_MAX_OUTPUT_TOKENS_<GRAPH>_<NODE>      — one stage's cap

A value the backend would reject (an unknown effort, a non-numeric cap) is
reported and the stage default is used instead: a typo must not send a request
the backend answers with HTTP 400.

What has not been measured
--------------------------
Everything above is a measurement of *cost*. What is unmeasured is *quality*:
whether ``evaluate_batch`` grades as well at ``low`` as it did with the backend's
unbounded default, or whether ``design_plan`` orders a DAG as well. That needs
the outputs of real sessions compared, not a latency table, and it is the reason
the judgement stages sit at ``low`` rather than ``minimal``: the cheap option was
chosen only where a stage restates a decision something else already made.
``scripts/slide_analysis.py --graph clarify|probe|plan`` now has the per-stage
numbers to compare against when someone does it.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from core.settings import (
    LLM_MAX_OUTPUT_TOKENS_ENV,
    REASONING_EFFORT_ENV,
    REASONING_EFFORTS,
    enumerated_setting,
    numeric_setting,
)

logger = logging.getLogger(__name__)

# stage -> (reasoning_effort, max output tokens)
_STAGE_BUDGETS: Final[dict[str, tuple[str, int]]] = {
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
    # --- Language helpers (called from inside stages) ---
    "language.detect": ("minimal", 256),
    "language.localize_status": ("minimal", 256),
    # --- The non-slide half of Material ---
    # Left untouched by the slide-side fix; writing a quiz is a judgement.
    "material.write_questions": ("low", 4096),
    # Condenses the step into the key points every later step is told to connect
    # to, so its output quality is load-bearing downstream.
    "material.summarize_step": ("low", 3072),
}

STAGES: frozenset[str] = frozenset(_STAGE_BUDGETS)


def stage_name(graph: str, stage: str) -> str:
    """The qualified name of a graph stage — what this module's table is keyed by.

    Both the graph builders (:func:`graphs.common.timed_node`) and the nodes
    themselves name the same call through this, so the stage that gets timed and
    the stage that gets a budget cannot drift apart (#194 review).
    """
    return f"{graph}.{stage}"


def require_stage(name: str) -> str:
    """Check a stage is known, and return its name.

    A graph that times a stage it has no budget for is a programming error — a
    renamed node, or a new model call nobody has budgeted. Failing here, at the
    point the graph is assembled, is what keeps ``graph_stage_timings.stage`` and
    this table naming the same thing.
    """
    if name not in _STAGE_BUDGETS:
        known = ", ".join(sorted(STAGES))
        raise KeyError(f"unknown stage {name!r}; known stages: {known}")
    return name


def _setting_names(default_name: str, stage: str) -> tuple[str, str]:
    """The per-stage setting, then the global one."""
    return (f"{default_name}_{stage.replace('.', '_').upper()}", default_name)


def _budget(stage: str) -> tuple[str, int]:
    """The configured budget for a stage, naming the stage if it is unknown."""
    budget = _STAGE_BUDGETS.get(require_stage(stage))
    assert budget is not None  # require_stage raises otherwise
    return budget


def reasoning_effort_for(stage: str) -> str:
    """The reasoning effort to send for ``stage``.

    Per-stage setting, then the global one, then the stage default. A value
    outside :data:`core.settings.REASONING_EFFORTS` is reported and the stage
    default stands.
    """
    effort_default, _ = _budget(stage)
    return enumerated_setting(
        _setting_names(REASONING_EFFORT_ENV, stage), REASONING_EFFORTS, effort_default
    )


def max_output_tokens_for(stage: str) -> int:
    """The output cap to send for ``stage``.

    Per-stage setting, then the global one, then the stage default.
    """
    _, cap_default = _budget(stage)
    return numeric_setting(
        _setting_names(LLM_MAX_OUTPUT_TOKENS_ENV, stage), cap_default, int
    )


def stage_call_kwargs(stage: str) -> dict[str, Any]:
    """Bind kwargs for one call of ``stage``, ready to spread into
    :func:`graphs.common.structured_invoke`.

    Both values are always present: an unbounded call is exactly the shape #194
    is about. The slide-writing stages are the one place that states its own
    values (:data:`graphs.material.sandbox.SLIDE_REASONING_EFFORT` and
    :data:`graphs.material.sandbox.SLIDE_MAX_OUTPUT_TOKENS`), and does.
    """
    return {
        "reasoning_effort": reasoning_effort_for(stage),
        "max_tokens": max_output_tokens_for(stage),
    }
