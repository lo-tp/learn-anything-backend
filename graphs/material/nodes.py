"""Node factory functions for the material graph.

Each factory takes the LLM and returns a LangGraph node closure
``(state: MaterialState) -> dict``.
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from core.language import DEFAULT_LANGUAGE, language_instruction

from ..common import structured_invoke, structured_invoke_messages, with_unknown_option
from .prompts import (
    _JSX_SYSTEM_PROMPT,
    PLAN_SLIDE_CONTENTS_SYSTEM,
    SUMMARIZE_STEP_SYSTEM,
    WRITE_QUESTIONS_SYSTEM,
)
from .sandbox import (
    MAX_MATERIAL_ATTEMPTS,
    MAX_SLIDE_SELF_REPAIR_TURNS,
    SLIDE_REASONING_EFFORT,
    _compile_slide,
    _placeholder_slide_jsx,
    classify_compile_error,
    compile_error_for_feedback,
    slide_sampling_bind_kwargs,
    validate_jsx,
)
from .schemas import (
    QuestionsOut,
    SlideContentsOut,
    SlideOut,
    SummaryOut,
)
from .state import MaterialState

logger = logging.getLogger(__name__)


def _timing_entry(stage: str, context: dict, started: float) -> dict:
    """One stage_timings entry: the stage name, its context, and the wall
    time since ``started`` (time.monotonic())."""
    return {
        "stage": stage,
        "context": context,
        "duration_seconds": time.monotonic() - started,
    }


def _established_concepts_block(established: list[dict]) -> str:
    """The shared "connect to these explicitly" established-concepts prompt
    block, used by the slide-planning and slide-writing nodes."""
    return (
        "Concepts established by prior steps (connect to these explicitly):\n"
        + "\n".join(
            f"  - {c.get('title', c.get('step_id', ''))}: "
            f"{', '.join(c.get('key_points', []))}"
            for c in established
        )
    )


def make_plan_slide_contents(llm: BaseChatModel):
    def plan_slide_contents(state: MaterialState) -> dict:
        step = state.get("step") or {}
        established = state.get("established_concepts") or []
        learner_context = state.get("learner_context") or {}

        parts = [
            f"Step: {step.get('title', '')}",
            f"Step description: {step.get('description', '')}",
            f"Learner context (boundary map): {learner_context}",
        ]
        if established:
            parts.append(_established_concepts_block(established))

        system = PLAN_SLIDE_CONTENTS_SYSTEM
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)

        started = time.monotonic()
        logger.debug(
            "plan_slide_contents: step=%s, established_concepts=%d",
            step.get("id"), len(established),
        )
        out = structured_invoke(
            llm, SlideContentsOut, system, "\n".join(parts),
            # Slide generation runs at the minimum reasoning budget; see
            # SLIDE_REASONING_EFFORT in sandbox.py for the measurement behind it.
            reasoning_effort=SLIDE_REASONING_EFFORT,
        )
        contents = [
            {
                "title": s.title,
                "key_points": s.key_points,
                "visual_hint": s.visual_hint,
            }
            for s in out.slide_contents
        ]
        logger.debug(
            "plan_slide_contents: step=%s -> %d slide spec(s) (%.1fs)",
            step.get("id"), len(contents), time.monotonic() - started,
        )
        return {
            "slide_contents": contents,
            "slide_index": 0,
            "attempts_by_slide": [0] * len(contents),
            "stage_timings": [
                _timing_entry(
                    "plan_slide_contents",
                    {"step_id": step.get("id"), "slide_index": None, "attempt": None},
                    started,
                )
            ],
        }

    return plan_slide_contents


def _slide_code(
    llm: BaseChatModel, messages: list[BaseMessage], bind_kwargs: dict
) -> str:
    """One structured slide-writing call: return the JSX code string."""
    out = structured_invoke_messages(llm, SlideOut, messages, **bind_kwargs)
    return out.code


def _repair_slide_code(
    llm: BaseChatModel,
    jsx: str,
    result: dict,
    step: dict,
    learner_context: dict,
    established: list[dict],
    spec: dict,
    language: str,
    bind_kwargs: dict,
) -> str:
    """One bounded self-repair turn.

    Feeds the reduced compile error (and any deterministic repairs that were
    already applied) back to the model and asks for the corrected, complete
    slide JSX. ``result`` is a ``validate_jsx`` result with ``ok`` False.
    """
    parts = [
        f"Step: {step.get('title', '')}",
        f"Step description: {step.get('description', '')}",
        f"Learner context (boundary map): {learner_context}",
        f"Slide: {spec.get('title', '')}",
    ]
    kp_lines = "\n".join(f"  - {kp}" for kp in spec.get("key_points", []))
    parts.append(f"Key points this slide must establish:\n{kp_lines}")
    parts.append(f"Visual hint: {spec.get('visual_hint', '')}")
    if established:
        parts.append(_established_concepts_block(established))
    parts.append("Your slide JSX failed to validate against the compiler.")
    parts.append(f"Compile error: {result['error']}")
    if result.get("repairs"):
        parts.append(
            "Deterministic repairs were already applied before the error was "
            "reported: " + ", ".join(result["repairs"])
        )
    parts.append("The JSX that failed to compile was:")
    parts.append(jsx)
    parts.append(
        "Return the corrected complete slide JSX: a single bare React module "
        "starting with `export default function` and ending with `}`."
    )
    system = _JSX_SYSTEM_PROMPT + language_instruction(language)
    messages = [SystemMessage(content=system), HumanMessage(content="\n".join(parts))]
    return _slide_code(llm, messages, bind_kwargs)


def make_write_slide(llm: BaseChatModel):
    def write_slide(state: MaterialState) -> dict:
        i = state.get("slide_index", 0)
        slide_contents = state.get("slide_contents") or []
        spec = slide_contents[i] if i < len(slide_contents) else {}
        step = state.get("step") or {}
        established = state.get("established_concepts") or []
        learner_context = state.get("learner_context") or {}
        prev_error = state.get("last_compile_error")
        language = state.get("language") or DEFAULT_LANGUAGE

        attempts = list(state.get("attempts_by_slide") or [0] * len(slide_contents))
        attempts[i] = (attempts[i] or 0) + 1
        new_attempts = attempts[i]

        # Sampling profile by attempt: attempt 1 is the most creative, and each
        # outer retry steps down toward the most deterministic profile. All
        # self-repair turns within an attempt share that attempt's profile.
        # structured_invoke_messages applies the bind AFTER
        # with_structured_output so the params actually land. Non-OpenAI
        # sampling params (top_k/min_p/repeat_penalty) are nested under
        # extra_body so the OpenAI client forwards them to the request body.
        bind = slide_sampling_bind_kwargs(new_attempts)

        kp_lines = "\n".join(f"  - {kp}" for kp in spec.get("key_points", []))
        parts = [
            f"Step: {step.get('title', '')}",
            f"Step description: {step.get('description', '')}",
            f"Learner context (boundary map): {learner_context}",
            f"Write slide {i + 1} of {len(slide_contents)}: {spec.get('title', '')}",
            f"Key points this slide must establish:\n{kp_lines}",
            f"Visual hint: {spec.get('visual_hint', '')}",
        ]
        if established:
            parts.append(_established_concepts_block(established))
        if new_attempts > 1 and prev_error:
            parts.append(
                "Your previous version of this slide failed to compile: "
                f"{compile_error_for_feedback(prev_error)}.\n"
                "Rewrite the slide fixing the problem."
            )

        system = _JSX_SYSTEM_PROMPT
        system += language_instruction(language)
        human = "\n".join(parts)
        # The exact messages sent to the LLM, serialized so a failed compile
        # can be persisted for debugging (prompt -> result -> error). The same
        # objects are invoked below, so the persisted prompt is the real one.
        initial_messages = [SystemMessage(content=system), HumanMessage(content=human)]
        prompt = json.dumps(
            [{"role": m.type, "content": m.content} for m in initial_messages]
        )

        started = time.monotonic()
        logger.debug(
            "write_slide: step=%s, slide=%d/%d, attempt=%d",
            step.get("id"), i + 1, len(slide_contents), new_attempts,
        )

        # --- Bounded self-repair loop: write -> validate_jsx -> repair ---
        # The model writes the slide, then the node binds it to validate_jsx
        # (deterministic repairs, then a sandbox compile). On failure the
        # reduced error is handed back for a repair turn. The loop is bounded
        # by MAX_SLIDE_SELF_REPAIR_TURNS; if it exhausts, the best-effort JSX
        # is returned and the outer retry routing (compile_slide) takes over.
        jsx = _slide_code(llm, initial_messages, bind)
        last_error: str | None = None
        repair_turns = 0
        for turn in range(MAX_SLIDE_SELF_REPAIR_TURNS + 1):
            result = validate_jsx(jsx)
            if result["ok"]:
                # Store the (possibly repaired) source that actually compiled;
                # compile_slide recomputes the canonical compiled code from it.
                jsx = result["source"]
                last_error = None
                break
            last_error = result["error"]
            if turn == MAX_SLIDE_SELF_REPAIR_TURNS:
                logger.info(
                    "write_slide: slide %d/%d exhausted self-repair after "
                    "%d repair turn(s); handing off to retry routing",
                    i + 1, len(slide_contents), repair_turns,
                )
                break
            repair_turns += 1
            logger.debug(
                "write_slide: slide %d/%d repair turn %d: %s",
                i + 1, len(slide_contents), repair_turns, result["error"],
            )
            jsx = _repair_slide_code(
                llm, jsx, result, step, learner_context, established, spec,
                language, bind,
            )

        logger.info(
            "write_slide: step=%s, slide=%d/%d attempt=%d -> %d chars, "
            "%d repair turn(s) (%.1fs)",
            step.get("id"), i + 1, len(slide_contents), new_attempts,
            len(jsx), repair_turns, time.monotonic() - started,
        )
        ticks = jsx.count("`")
        if ticks % 2:
            logger.warning(
                "write_slide: slide %d has %d backticks (odd) — "
                "possible broken template literal",
                i + 1, ticks,
            )
        return {
            "attempts_by_slide": attempts,
            "current_slide_jsx": jsx,
            "current_slide_prompt": prompt,
            # Carried to the outer retry routing (the next attempt's prompt
            # feeds it back); None when self-repair already resolved it.
            "last_compile_error": last_error,
            "stage_timings": [
                _timing_entry(
                    "write_slide",
                    {"step_id": step.get("id"), "slide_index": i + 1,
                     "attempt": new_attempts},
                    started,
                )
            ],
        }

    return write_slide


def make_compile_slide(llm: BaseChatModel):
    def compile_slide(state: MaterialState) -> dict:
        i = state.get("slide_index", 0)
        slide_contents = state.get("slide_contents") or []
        spec = slide_contents[i] if i < len(slide_contents) else {}
        jsx = state.get("current_slide_jsx") or ""
        prompt = state.get("current_slide_prompt") or ""
        attempts = state.get("attempts_by_slide") or []
        current_attempts = attempts[i] if i < len(attempts) else 0

        started = time.monotonic()
        compiled_code, error = _compile_slide(jsx)

        if compiled_code is not None:
            logger.debug(
                "compile_slide: step=%s, slide=%d compiled successfully",
                state.get("step", {}).get("id"), i + 1,
            )
            return {
                "slides": [compiled_code],
                "slide_index": i + 1,
                "compile_result": "success",
                "current_slide_is_placeholder": False,
                "stage_timings": [
                    _timing_entry(
                        "compile_slide",
                        {"step_id": state.get("step", {}).get("id"),
                         "slide_index": i + 1, "attempt": current_attempts},
                        started,
                    )
                ],
            }

        # Tag the failure with a coarse class so slide-skip rates are
        # measurable by class (the C1.2 classifier builds on this).
        failure_class = classify_compile_error(error)
        logger.warning(
            "compile_slide: step=%s, slide=%d attempt %d/%d failed "
            "(class=%s): %s",
            state.get("step", {}).get("id"), i + 1,
            current_attempts, MAX_MATERIAL_ATTEMPTS,
            failure_class, error,
        )
        result: dict = {
            "failed_attempts": [
                {
                    "index": i, "prompt": prompt, "jsx": jsx, "error": error,
                    "failure_class": failure_class,
                }
            ],
            "last_compile_error": error,
            "stage_timings": [
                _timing_entry(
                    "compile_slide",
                    {"step_id": state.get("step", {}).get("id"),
                     "slide_index": i + 1, "attempt": current_attempts},
                    started,
                )
            ],
        }
        if current_attempts >= MAX_MATERIAL_ATTEMPTS:
            # Exhausted: substitute a placeholder slide so the deck keeps a
            # visible slot (and still records the failed attempt). The
            # placeholder is always generated and persisted; the endpoints
            # only return it in dev mode (flagged via
            # current_slide_is_placeholder). If the placeholder itself fails
            # to compile, the slide is skipped entirely.
            placeholder = _placeholder_slide_jsx(spec.get("title", ""))
            placeholder_code, ph_error = _compile_slide(placeholder)
            if placeholder_code is not None:
                logger.info(
                    "compile_slide: step=%s, slide=%d exhausted; using "
                    "placeholder slide",
                    state.get("step", {}).get("id"), i + 1,
                )
                result["slides"] = [placeholder_code]
                result["slide_index"] = i + 1
                result["compile_result"] = "success"
                result["current_slide_is_placeholder"] = True
                return result
            logger.warning(
                "compile_slide: step=%s, slide=%d placeholder failed to "
                "compile: %s",
                state.get("step", {}).get("id"), i + 1, ph_error,
            )
            result["slide_index"] = i + 1
            result["compile_result"] = "exhausted"
        else:
            result["compile_result"] = "retry"
        return result

    return compile_slide


def make_pause_after_compile(llm: BaseChatModel):
    def pause_after_compile(state: MaterialState) -> dict:
        """Pause so the driver can persist the just-compiled slide to the DB.

        On resume the graph continues to the next ``write_slide`` (or
        ``write_questions`` if this was the last slide) — ``compile_slide``
        is NOT re-executed.
        """
        from langgraph.types import interrupt

        interrupt(None)
        return {}

    return pause_after_compile


def make_write_questions(llm: BaseChatModel):
    def write_questions(state: MaterialState) -> dict:
        step = state.get("step") or {}
        established = state.get("established_concepts") or []
        learner_context = state.get("learner_context") or {}

        parts = [
            f"Step: {step.get('title', '')}",
            f"Step description: {step.get('description', '')}",
            f"Learner context (boundary map): {learner_context}",
        ]
        if established:
            parts.append(
                "Concepts established by prior steps:\n"
                + "\n".join(
                    f"  - {c.get('title', c.get('step_id', ''))}: "
                    f"{', '.join(c.get('key_points', []))}"
                    for c in established
                )
            )

        system = WRITE_QUESTIONS_SYSTEM
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)

        started = time.monotonic()
        logger.debug(
            "write_questions: step=%s, established_concepts=%d",
            step.get("id"), len(established),
        )
        out = structured_invoke(llm, QuestionsOut, system, "\n".join(parts))
        logger.debug(
            "write_questions: step=%s -> %d question(s) (%.1fs)",
            step.get("id"), len(out.questions), time.monotonic() - started,
        )
        language = state.get("language") or DEFAULT_LANGUAGE
        questions = [
            {
                "id": f"{step.get('id', '')}_q{n}",
                "text": q.text,
                "options": with_unknown_option(llm, language, q.options),
                "correct_index": q.correct_index,
                "explanation": q.explanation,
            }
            for n, q in enumerate(out.questions, start=1)
        ]
        return {
            "questions": questions,
            "stage_timings": [
                _timing_entry(
                    "write_questions",
                    {"step_id": step.get("id"), "slide_index": None, "attempt": None},
                    started,
                )
            ],
        }

    return write_questions


def make_summarize_step(llm: BaseChatModel):
    def summarize_step(state: MaterialState) -> dict:
        step = state.get("step") or {}
        slide_contents = state.get("slide_contents") or []

        system = SUMMARIZE_STEP_SYSTEM
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)
        human = (
            f"Step: {step.get('title', '')}\n"
            f"Step description: {step.get('description', '')}\n\n"
            "Slide contents for this step:\n"
            + "\n\n".join(
                f"--- Slide {n}: {spec.get('title', '')} ---\n"
                f"Key points: {', '.join(spec.get('key_points', []))}"
                for n, spec in enumerate(slide_contents, start=1)
            )
        )

        started = time.monotonic()
        logger.debug(
            "summarize_step: step=%s, slides=%d", step.get("id"), len(slide_contents)
        )
        out = structured_invoke(llm, SummaryOut, system, human)
        logger.debug(
            "summarize_step: step=%s -> %d key point(s) (%.1fs)",
            step.get("id"), len(out.key_points), time.monotonic() - started,
        )
        return {
            "summary": {
                "step_id": step.get("id", ""),
                "title": step.get("title", ""),
                "key_points": out.key_points,
            },
            "stage_timings": [
                _timing_entry(
                    "summarize_step",
                    {"step_id": step.get("id"), "slide_index": None, "attempt": None},
                    started,
                )
            ],
        }

    return summarize_step
