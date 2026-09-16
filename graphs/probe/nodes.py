"""Node factories for the probe graph."""

from __future__ import annotations

import os
import uuid

from langchain_core.language_models import BaseChatModel
from langgraph.types import interrupt

from language import DEFAULT_LANGUAGE, language_instruction

from ..common import structured_invoke, with_unknown_option
from .prompts import (
    DECOMPOSE_STRANDS_SYSTEM,
    evaluate_batch_system,
    generate_batch_system,
)
from .schemas import (
    BatchEvaluateOut,
    BatchQuestionsOut,
    StrandsOut,
)
from .state import ProbeState

MAX_PROBE_QUESTIONS = int(os.getenv("MAX_PROBE_QUESTIONS", "10"))
PROBE_BATCH_SIZE = int(os.getenv("PROBE_BATCH_SIZE", "3"))


def _unbracketed_strands(state: ProbeState) -> list[str]:
    """Strands not yet fully bracketed: null floor or ceiling in boundary_map."""
    boundary_map = state.get("boundary_map") or {}
    return [
        s
        for s in (state.get("strands") or [])
        if (boundary_map.get(s) or {}).get("floor") is None
        or (boundary_map.get(s) or {}).get("ceiling") is None
    ]


def make_decompose_strands(llm: BaseChatModel):
    """One-shot: enumerate prerequisite strands and seed the boundary map."""

    def decompose_strands(state: ProbeState) -> dict:
        goal = state.get("goal", "")
        out = structured_invoke(
            llm,
            StrandsOut,
            DECOMPOSE_STRANDS_SYSTEM,
            f"Goal: {goal}",
        )
        return {
            "strands": [s.id for s in out.strands],
            "strand_descriptions": {s.id: s.description for s in out.strands},
            "boundary_map": {
                s.id: {"floor": None, "ceiling": None, "gap_type": "unknown"}
                for s in out.strands
            },
        }

    return decompose_strands


def make_generate_batch(llm: BaseChatModel):
    """Produce a batch of MCQs targeting the unbracketed strands."""

    def generate_batch(state: ProbeState) -> dict:
        count = state.get("question_count", 0)
        strands = state.get("strands") or []
        unbracketed = _unbracketed_strands(state)
        batch_size = min(PROBE_BATCH_SIZE, len(unbracketed))

        system = generate_batch_system(batch_size)
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)

        out = structured_invoke(
            llm,
            BatchQuestionsOut,
            system,
            (
                f"Goal: {state.get('goal', '')}\n"
                f"Strand set: {', '.join(strands) if strands else '(none seeded)'}\n"
                f"Strand descriptions: {state.get('strand_descriptions') or '{}'}\n"
                f"Unbracketed strands: {', '.join(unbracketed) if unbracketed else '(none)'}\n"
                f"Batch size: {batch_size} "
                f"(questions {count + 1} through {count + batch_size})\n"
                f"Boundary map so far: {state.get('boundary_map') or '{}'}\n"
                f"History: {state.get('history') or 'none'}"
            ),
        )

        questions = []
        for q in out.questions:
            question_id = str(uuid.uuid4())
            options = with_unknown_option(
                llm, state.get("language") or DEFAULT_LANGUAGE, q.options
            )
            questions.append(
                {
                    "id": question_id,
                    "text": q.text,
                    "options": options,
                    "correct_index": q.correct_index,
                    "explanation": q.explanation,
                    "strand": q.strand,
                    "difficulty": q.difficulty,
                }
            )
        return {
            "next_batch": questions,
            "question_count": count + len(questions),
        }

    return generate_batch


def make_wait_for_answers(llm: BaseChatModel):
    """Pause (interrupt); resumes with the learner's answers."""

    def wait_for_answers(state: ProbeState) -> dict:
        answer = interrupt(None)
        return {"_resume": answer}

    return wait_for_answers


def make_evaluate_batch(llm: BaseChatModel):
    """Mark each answer correct/incorrect and update boundary estimates."""

    def evaluate_batch(state: ProbeState) -> dict:
        resume = state.get("_resume") or {}
        batch = state.get("next_batch") or []

        answers = {
            a["question_id"]: a["selected_index"]
            for a in (resume.get("answers") or [])
        }

        batch_lines = []
        for i, q in enumerate(batch, start=1):
            selected_index = answers.get(q["id"], 0)
            options = q.get("options") or []
            selected_option = (
                options[selected_index] if 0 <= selected_index < len(options) else "?"
            )
            batch_lines.append(
                f"[{i}] question_id: {q['id']}\n"
                f"    Question: {q.get('text', '')}\n"
                f"    Strand: {q.get('strand', '')}\n"
                f"    Correct index: {q.get('correct_index', 0)}\n"
                f"    Learner selected index: {selected_index}\n"
                f"    Selected option: {selected_option}"
            )

        system = evaluate_batch_system(len(batch))
        system += language_instruction(state.get("language") or DEFAULT_LANGUAGE)
        out = structured_invoke(
            llm,
            BatchEvaluateOut,
            system,
            (
                f"Goal: {state.get('goal', '')}\n"
                f"Batch ({len(batch)} questions):\n"
                + "\n".join(batch_lines)
                + "\n"
                f"Strand set: {', '.join(state.get('strands') or []) or '(none seeded)'}\n"
                f"Current boundary map: {state.get('boundary_map') or '{}'}\n"
                f"History: {state.get('history') or 'none'}"
            ),
        )

        evals = {e.question_id: e for e in out.evaluations}

        history = list(state.get("history", []))
        for q in batch:
            selected_index = answers.get(q["id"], 0)
            evaluation = evals.get(q["id"])
            history.append(
                {
                    "question_id": q["id"],
                    "text": q.get("text", ""),
                    "options": q.get("options", []),
                    "correct_index": q.get("correct_index", 0),
                    "selected_index": selected_index,
                    "is_correct": (
                        evaluation.is_correct
                        if evaluation is not None
                        else selected_index == q.get("correct_index", 0)
                    ),
                    "strand": q.get("strand", ""),
                    "difficulty": q.get("difficulty", 1),
                }
            )

        updated = {**(state.get("boundary_map") or {}), **out.updated_boundary_map}

        return {"boundary_map": updated, "history": history}

    return evaluate_batch


def make_decide_next(llm: BaseChatModel):
    """Deterministic: all strands bracketed or cap hit?"""

    def decide_next(state: ProbeState) -> dict:
        strands = state.get("strands") or []
        boundary_map = state.get("boundary_map", {})
        count = state.get("question_count", 0)

        def bracketed(strand_id: str) -> bool:
            v = boundary_map.get(strand_id) or {}
            return v.get("floor") is not None and (
                v.get("ceiling") is not None or v.get("gap_type") == "none"
            )

        all_bracketed = len(strands) > 0 and all(bracketed(s) for s in strands)
        if count >= MAX_PROBE_QUESTIONS or all_bracketed:
            return {"_decision": "done"}
        return {"_decision": "continue"}

    return decide_next
