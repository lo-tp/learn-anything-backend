"""Sandbox compile and placeholder slide generation."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from .prompts import _PLACEHOLDER_JSX_TEMPLATE

logger = logging.getLogger(__name__)

# --- Constants ---

SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080")
_COMPILE_TIMEOUT_S = 30.0
MAX_MATERIAL_ATTEMPTS = max(1, int(os.getenv("MAX_MATERIAL_ATTEMPTS", "3")))

# --- Slide-generation sampling profiles ---
# One profile per write attempt. All attempts use the same deterministic
# profile so a slide that keeps failing to compile gets identical, stable
# samples on every retry.
#
# temperature/top_p are standard OpenAI params, but top_k/min_p/
# repeat_penalty are NOT part of the OpenAI API — OpenAI-compatible backends
# (e.g. vLLM) only accept them when nested under ``extra_body``. See
# SLIDE_EXTRA_BODY_PARAMS and slide_sampling_bind_kwargs.
SLIDE_SAMPLING_PROFILES: tuple[dict[str, float], ...] = (
    {  # Attempt 1
        "temperature": 0.1,
        "top_p": 0.5,
        "top_k": 10,
        "min_p": 0.1,
        "repeat_penalty": 1.05,
    },
    {  # Attempt 2
        "temperature": 0.1,
        "top_p": 0.5,
        "top_k": 10,
        "min_p": 0.1,
        "repeat_penalty": 1.05,
    },
    {  # Attempt 3
        "temperature": 0.1,
        "top_p": 0.5,
        "top_k": 10,
        "min_p": 0.1,
        "repeat_penalty": 1.05,
    }
)


# --- Slide-generation reasoning budget ---
#
# Every slide-generation call (planning a step's slide contents, and each
# write attempt) asks for the smallest reasoning budget. On this project's
# llama.cpp backend `reasoning_effort` is honoured and maps onto the Qwen3
# think switch: `minimal`/`none` do not think at all (a measured answer came
# back in 4 completion tokens with no reasoning), while `low`/`medium`/`high`
# do (~150-300 reasoning characters). `minimal` is also a valid OpenAI
# reasoning effort, so the same value is correct if the backend is
# api.openai.com.
#
# It lives next to the sampling profiles because `slide_sampling_bind_kwargs`
# is the single place that assembles what a slide-writing call sends.
SLIDE_REASONING_EFFORT: str = "minimal"


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
    Adds ``reasoning_effort`` (SLIDE_REASONING_EFFORT) at the top level: it is
    a standard OpenAI field, so nesting it under ``extra_body`` — the vLLM
    ``chat_template_kwargs`` dialect — is ignored by llama.cpp.
    The result is ready to spread into the invoke call:
    ``structured_invoke_messages(llm, schema, msgs, **slide_sampling_bind_kwargs(n))``.
    """
    profile = slide_sampling_for_attempt(attempt)
    extra_body = {k: v for k, v in profile.items() if k in SLIDE_EXTRA_BODY_PARAMS}
    kwargs: dict[str, Any] = {
        k: v for k, v in profile.items() if k not in SLIDE_EXTRA_BODY_PARAMS
    }
    kwargs["reasoning_effort"] = SLIDE_REASONING_EFFORT
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


# --- Sandbox compile ---

# Raw transport errors look like:
#   Transport/HTTP error: Client error '400 code must declare `export default`'
#   for url 'http://localhost:3001/api/compile' For more information check: ...
# Only the quoted status message is useful to the model on a retry.
_TRANSPORT_ERROR_RE = re.compile(
    r"^Transport/HTTP error: (?:Client|Server) error '(?P<msg>.*)' for url",
    re.DOTALL,
)


def compile_error_for_feedback(error: str) -> str:
    """Reduce a raw compile error to the meaningful message for retry feedback.

    Strips the httpx transport wrapper (URL + "For more information" footer) so
    the model sees e.g. ``500 Build failed with 1 error: compile.tsx:162:74:
    ERROR: ...`` instead of the full client error.
    """
    m = _TRANSPORT_ERROR_RE.match(error.strip())
    if m:
        return " ".join(m.group("msg").split())
    return " ".join(error.split())


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
