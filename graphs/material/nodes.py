"""Node factory functions for the material graph.

Each factory takes the LLM and returns a LangGraph node closure
``(state: MaterialState) -> dict``.
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

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
    _compile_slide,
    _placeholder_slide_jsx,
    compile_error_for_feedback,
    slide_sampling_bind_kwargs,
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
            parts.append(
                "Concepts established by prior steps (connect to these explicitly):\n"
                + "\n".join(
                    f"  - {c.get('title', c.get('step_id', ''))}: "
                    f"{', '.join(c.get('key_points', []))}"
                    for c in established
                )
            )

        system = PLAN_SLIDE_CONTENTS_SYSTEM
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)

        started = time.monotonic()
        logger.debug(
            "plan_slide_contents: step=%s, established_concepts=%d",
            step.get("id"), len(established),
        )
        out = structured_invoke(llm, SlideContentsOut, system, "\n".join(parts))
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


def make_write_slide(llm: BaseChatModel):
    def write_slide(state: MaterialState) -> dict:
        i = state.get("slide_index", 0)
        slide_contents = state.get("slide_contents") or []
        spec = slide_contents[i] if i < len(slide_contents) else {}
        step = state.get("step") or {}
        established = state.get("established_concepts") or []
        learner_context = state.get("learner_context") or {}
        prev_error = state.get("last_compile_error")

        attempts = list(state.get("attempts_by_slide") or [0] * len(slide_contents))
        attempts[i] = (attempts[i] or 0) + 1
        new_attempts = attempts[i]

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
            parts.append(
                "Concepts established by prior steps (connect to these explicitly):\n"
                + "\n".join(
                    f"  - {c.get('title', c.get('step_id', ''))}: "
                    f"{', '.join(c.get('key_points', []))}"
                    for c in established
                )
            )
        if new_attempts > 1 and prev_error:
            parts.append(
                "Your previous version of this slide failed to compile: "
                f"{compile_error_for_feedback(prev_error)}.\n"
                "Rewrite the slide fixing the problem."
            )

        system = _JSX_SYSTEM_PROMPT
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)
        human = "\n".join(parts)
        # The exact messages sent to the LLM, serialized so a failed compile
        # can be persisted for debugging (prompt -> result -> error). The same
        # objects are invoked below, so the persisted prompt is the real one.
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        prompt = json.dumps(
            [{"role": m.type, "content": m.content} for m in messages]
        )

        started = time.monotonic()
        logger.debug(
            "write_slide: step=%s, slide=%d/%d, attempt=%d",
            step.get("id"), i + 1, len(slide_contents), new_attempts,
        )
        # Sampling profile by attempt: attempt 1 is the most creative, and
        # each retry steps down toward the most deterministic profile so a
        # slide that keeps failing to compile gets progressively stable
        # samples. structured_invoke_messages applies the bind AFTER
        # with_structured_output so the params actually land. Non-OpenAI
        # sampling params (top_k/min_p/repeat_penalty) are nested under
        # extra_body so the OpenAI client forwards them to the request body.
        out = structured_invoke_messages(
            llm, SlideOut, messages, **slide_sampling_bind_kwargs(new_attempts)
        )
        jsx = out.code
        logger.info(
            "write_slide: step=%s, slide=%d -> %d chars (%.1fs)",
            step.get("id"), i + 1, len(jsx), time.monotonic() - started,
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
            "last_compile_error": None,
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

        logger.warning(
            "compile_slide: step=%s, slide=%d attempt %d/%d failed: %s",
            state.get("step", {}).get("id"), i + 1,
            current_attempts, MAX_MATERIAL_ATTEMPTS, error,
        )
        result: dict = {
            "failed_attempts": [
                {"index": i, "prompt": prompt, "jsx": jsx, "error": error}
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
