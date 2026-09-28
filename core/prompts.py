"""Public loader for the private LLM prompt templates.

The prompt text lives in a private repo mounted as the ``prompts/``
submodule. This module resolves the prompt backend and re-exports the
symbols the graph code uses, so the rest of the codebase imports from
``core.prompts`` and never sees the private content directly.

Resolution order:
  1. The ``prompts`` submodule package (production / checked-out submodule).
  2. The public stubs when ``LA_PROMPTS_STUB=1`` (tests, and any context
     where the submodule is intentionally absent).
  3. Otherwise a clear setup error.
"""

from __future__ import annotations

import importlib
import os

_pkg = None


def _resolve_backend():
    global _pkg
    if _pkg is not None:
        return _pkg
    try:
        # Import a real submodule module as the canary: an empty ``prompts/``
        # (submodule not initialized) is a namespace package, so this fails
        # and we fall through to the stub/error path below.
        importlib.import_module("prompts.probe")
        _pkg = importlib.import_module("prompts")
    except ModuleNotFoundError:
        if os.environ.get("LA_PROMPTS_STUB") == "1":
            _pkg = importlib.import_module("core.prompts_stub")
        else:
            raise RuntimeError(
                "Private LLM prompts are not available. Check out the "
                "submodule with: git submodule update --init prompts "
                "(tests set LA_PROMPTS_STUB=1)."
            ) from None
    return _pkg


def _sub(name: str):
    return importlib.import_module(f"{_resolve_backend().__name__}.{name}")


_probe = _sub("probe")
_plan = _sub("plan")
_clarify = _sub("clarify")
_material = _sub("material")
_language = _sub("language")

# --- probe ---
DECOMPOSE_STRANDS_SYSTEM = _probe.DECOMPOSE_STRANDS_SYSTEM
generate_batch_system = _probe.generate_batch_system
evaluate_batch_system = _probe.evaluate_batch_system

# --- plan ---
RESEARCH_TOPIC_SYSTEM = _plan.RESEARCH_TOPIC_SYSTEM
DESIGN_PLAN_INITIAL_SYSTEM = _plan.DESIGN_PLAN_INITIAL_SYSTEM
DESIGN_PLAN_REFINE_SYSTEM = _plan.DESIGN_PLAN_REFINE_SYSTEM
RENDER_PLAN_SYSTEM = _plan.RENDER_PLAN_SYSTEM

# --- clarify ---
ASSESS_GOAL_SYSTEM = _clarify.ASSESS_GOAL_SYSTEM
ASSESS_GOAL_CAP_REACHED = _clarify.ASSESS_GOAL_CAP_REACHED
GENERATE_QUESTIONS_SYSTEM = _clarify.GENERATE_QUESTIONS_SYSTEM
REFINE_GOAL_SYSTEM = _clarify.REFINE_GOAL_SYSTEM

# --- material ---
PLAN_SLIDE_CONTENTS_SYSTEM = _material.PLAN_SLIDE_CONTENTS_SYSTEM
SUMMARIZE_STEP_SYSTEM = _material.SUMMARIZE_STEP_SYSTEM
WRITE_QUESTIONS_SYSTEM = _material.WRITE_QUESTIONS_SYSTEM
_JSX_SYSTEM_PROMPT = _material._JSX_SYSTEM_PROMPT
_PLACEHOLDER_JSX_TEMPLATE = _material._PLACEHOLDER_JSX_TEMPLATE

# --- language ---
language_instruction = _language.language_instruction
DETECT_LANGUAGE_SYSTEM = _language.DETECT_LANGUAGE_SYSTEM
localize_status_system = _language.localize_status_system
