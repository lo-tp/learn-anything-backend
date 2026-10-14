"""Regenerate one stored slide with the material graph's own nodes (#197).

The material graph produces a whole step's deck: it plans the step's slide
contents, then writes them one at a time. Regenerating a *single* slide must
not re-plan anything — the spec that slide was written from is now stored on
its own row (``SlideContent.content_spec``) — so this module runs the graph's
own ``write_slide`` and ``compile_slide`` over one spec, in a plain loop, with
no graph, no checkpointer and no interrupt.

The nodes are pure functions of state, which is what makes this more than a
copy of them: the sampling ladder, the self-repair loop, the attempt bound and
the MathJax backstop all run here exactly as they run inside the graph, because
this file calls them rather than reimplementing them. What it owns is the
routing ``compile_slide``'s result would have been routed by, and the state a
graph run would have carried.

Callers rebuild the rest of that state from the domain DB — the plan step, the
prior steps' summaries, the boundary map, the session language — and persist
what comes back (``routers/plan.py``).
"""

from __future__ import annotations

from typing import Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel

from core.language import DEFAULT_LANGUAGE

from .nodes import make_compile_slide, make_write_slide
from .sandbox import MAX_MATERIAL_ATTEMPTS
from .state import MaterialState

# The state channels LangGraph accumulates with ``operator.add``. Outside a
# graph there is no reducer to do it, so :func:`_advance` accumulates them
# here — getting this wrong loses the failed attempts and the timings a
# regeneration exists to produce.
_ADDITIVE_CHANNELS: tuple[str, ...] = ("slides", "failed_attempts", "stage_timings")


class SlideRegeneration(TypedDict):
    """What one single-slide regeneration run produced.

    ``jsx`` is the compiled component source, or ``None`` when nothing this run
    wrote ever compiled. ``is_placeholder`` says that JSX is the stand-in a
    slide gets when it exhausts its attempts — a placeholder is what the deck
    already has, so a caller deciding whether to keep it should read it as a
    regeneration that did not succeed. ``failed_attempts`` and
    ``stage_timings`` are the same records the graph's driver persists.
    """

    jsx: str | None
    is_placeholder: bool
    failed_attempts: list[dict[str, Any]]
    stage_timings: list[dict[str, Any]]


def _advance(state: MaterialState, update: dict[str, Any]) -> MaterialState:
    """Apply one node's update to the synthetic state, as the graph would.

    The additive channels accumulate; every other key is written as the node
    writes it. ``slide_index`` is left exactly as ``compile_slide`` sets it: it
    advances past the slide just compiled, which is what ends a one-slide run.
    """
    merged: dict[str, Any] = dict(state)
    for key, value in update.items():
        if key in _ADDITIVE_CHANNELS:
            merged[key] = [*merged.get(key, []), *(value or [])]
        else:
            merged[key] = value
    return cast(MaterialState, merged)


def _attributed(entry: dict[str, Any]) -> dict[str, Any]:
    """Mark a stage row as the work of a regeneration, not of a generation.

    Both run the same stages under the same names, and #194's rows are read
    afterwards to ask what generating a deck costs. A regeneration's rows that
    say nothing would be counted into that number silently.
    """
    return {
        **entry,
        "context": {**(entry.get("context") or {}), "regenerated": True},
    }


def regenerate_slide(
    llm: BaseChatModel,
    *,
    step: dict[str, Any],
    content_spec: dict[str, Any],
    established_concepts: list[dict[str, Any]],
    learner_context: dict[str, Any],
    language: str | None,
    feedback: str | None = None,
) -> SlideRegeneration:
    """Write one slide again from its stored content spec.

    ``step``, ``established_concepts``, ``learner_context`` and ``language`` are
    the same values a material graph run for this step received — a caller
    rebuilds them from the plan, the prior steps' ``StepMaterial`` summaries,
    the session's boundary map and its language. ``content_spec`` is what
    ``plan_slide_contents`` produced for this one slide and what its row now
    stores; it is not re-planned here, by design: what the slide *covers* is
    the deck's plan, and changing it is a different operation.

    ``feedback`` is the learner's own words about what should be different,
    carried into the slide request (``MaterialState.learner_feedback``) the same
    way a retry carries a compile error.

    Retries are the graph's: up to ``MAX_MATERIAL_ATTEMPTS`` write attempts, each
    with the sampling profile its attempt number selects, ending the way the
    graph ends — with a placeholder when the last attempt exhausts. Returns the
    compiled JSX rather than writing it anywhere; persisting a slide is the
    caller's.
    """
    write_slide = make_write_slide(llm)
    compile_slide = make_compile_slide(llm)

    # A one-slide deck: the stored spec is the entire plan this run has, and
    # the cursor starts at its only slide. `attempts_by_slide` starts at zero so
    # the first write is the most creative profile, as a first draft is.
    state: MaterialState = {
        "step": step,
        "established_concepts": established_concepts,
        "learner_context": learner_context,
        "language": language or DEFAULT_LANGUAGE,
        "slide_contents": [content_spec],
        "slide_index": 0,
        "attempts_by_slide": [0],
        "learner_feedback": feedback or "",
        "stage_timings": [],
    }  # type: ignore[assignment]

    failed: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []

    for _ in range(MAX_MATERIAL_ATTEMPTS):
        # What each node *returned* is what this run records; the merged state
        # already carries the earlier attempts in the same channels.
        wrote = write_slide(state)
        state = _advance(state, wrote)
        compiled = compile_slide(state)
        state = _advance(state, compiled)

        failed.extend(compiled.get("failed_attempts") or [])
        timings.extend(
            _attributed(t)
            for t in [*(wrote.get("stage_timings") or []),
                      *(compiled.get("stage_timings") or [])]
        )

        if compiled.get("compile_result") == "success":
            # Same routing the graph's conditional edge would have taken on
            # "success" — except a one-slide run has no next slide to write.
            slides = state.get("slides") or []
            return {
                "jsx": slides[-1] if slides else None,
                "is_placeholder": bool(state.get("current_slide_is_placeholder")),
                "failed_attempts": failed,
                "stage_timings": timings,
            }

    # Every attempt failed to compile and even the placeholder could not be
    # compiled: there is no slide to hand back.
    return {
        "jsx": None,
        "is_placeholder": False,
        "failed_attempts": failed,
        "stage_timings": timings,
    }
