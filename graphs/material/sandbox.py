"""Sandbox compile and placeholder slide generation."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from .prompts import _PLACEHOLDER_JSX_TEMPLATE

logger = logging.getLogger(__name__)

# --- Constants ---

SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080")
_COMPILE_TIMEOUT_S = 30.0
MAX_MATERIAL_ATTEMPTS = max(1, int(os.getenv("MAX_MATERIAL_ATTEMPTS", "3")))

# --- Slide-generation sampling profiles ---
# One profile per write attempt: attempt 1 is the most creative (higher
# temp/top_p) to maximize the chance of a good first draft; each retry steps
# down toward the most deterministic profile so a slide that keeps failing to
# compile gets progressively stable samples.
#
# temperature/top_p are standard OpenAI params, but top_k/min_p/
# repeat_penalty are NOT part of the OpenAI API — OpenAI-compatible backends
# (e.g. vLLM) only accept them when nested under ``extra_body``. See
# SLIDE_EXTRA_BODY_PARAMS and slide_sampling_bind_kwargs.
SLIDE_SAMPLING_PROFILES: tuple[dict[str, float], ...] = (
    {  # Attempt 1 — creative but reliable
        "temperature": 0.4,
        "top_p": 0.75,
        "top_k": 25,
        "min_p": 0.06,
        "repeat_penalty": 1.02,
    },
    {  # Attempt 2 — mostly deterministic
        "temperature": 0.2,
        "top_p": 0.6,
        "top_k": 15,
        "min_p": 0.08,
        "repeat_penalty": 1.02,
    },
    {  # Attempt 3 — most reliable
        "temperature": 0.1,
        "top_p": 0.5,
        "top_k": 10,
        "min_p": 0.1,
        "repeat_penalty": 1.05,
    }
)


def slide_sampling_for_attempt(attempt: int) -> dict[str, float]:
    """Sampling params for the given 1-based write attempt.

    Attempts beyond the number of profiles reuse the last (most
    deterministic) profile; attempts below 1 reuse the first.
    """
    idx = max(0, min(attempt - 1, len(SLIDE_SAMPLING_PROFILES) - 1))
    return dict(SLIDE_SAMPLING_PROFILES[idx])


# Profile params that are not part of the standard OpenAI API and must be
# sent nested under ``extra_body`` (supported by OpenAI-compatible backends
# such as vLLM). Passing them as top-level kwargs makes the OpenAI client
# reject the request.
SLIDE_EXTRA_BODY_PARAMS: frozenset[str] = frozenset(
    {"top_k", "min_p", "repeat_penalty"}
)


def slide_sampling_bind_kwargs(attempt: int) -> dict[str, Any]:
    """Bind kwargs for ``structured_invoke_messages`` for a 1-based attempt.

    Splits the flat profile into what ``ChatOpenAI`` accepts as top-level
    kwargs (``temperature``/``top_p``) and what must be forwarded to the
    request body via ``extra_body`` (``top_k``/``min_p``/``repeat_penalty``).
    The result is ready to spread into the invoke call:
    ``structured_invoke_messages(llm, schema, msgs, **slide_sampling_bind_kwargs(n))``.
    """
    profile = slide_sampling_for_attempt(attempt)
    extra_body = {k: v for k, v in profile.items() if k in SLIDE_EXTRA_BODY_PARAMS}
    kwargs: dict[str, Any] = {
        k: v for k, v in profile.items() if k not in SLIDE_EXTRA_BODY_PARAMS
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


# --- Sandbox compile ---


def _compile_slide(code: str) -> tuple[str | None, str]:
    """Compile a slide's JSX via the sandbox service.

    Returns ``(compiled_code, "")`` on success or ``(None, error)`` on failure.
    """
    url = f"{SANDBOX_URL.rstrip('/')}/api/compile"
    try:
        with httpx.Client(timeout=_COMPILE_TIMEOUT_S) as client:
            resp = client.post(url, json={"code": code})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        msg = f"Transport/HTTP error: {exc}"
        logger.warning(
            "Slide compile failed (sandbox=%s); skipping slide: %s", url, msg,
            exc_info=True,
        )
        return None, msg
    if data.get("error"):
        logger.warning(
            "Slide compile returned error; skipping slide: %s",
            data.get("error"),
        )
        return None, data["error"]
    return (data.get("code") or code), ""


# --- Placeholder slide ---


def _placeholder_slide_jsx(title: str) -> str:
    """Build the dev-mode placeholder slide JSX for a failed slide.

    ``title`` is embedded as a JS-compatible string literal so it cannot break
    the module (quotes/backslashes are escaped by ``json.dumps``).
    """
    title_literal = json.dumps(title or "This slide")
    return (
        _PLACEHOLDER_JSX_TEMPLATE
        .replace("__TITLE__", title_literal)
        .replace("__ATTEMPTS__", str(MAX_MATERIAL_ATTEMPTS))
    )
