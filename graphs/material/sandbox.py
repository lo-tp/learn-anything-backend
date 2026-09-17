"""Sandbox compile and placeholder slide generation."""

from __future__ import annotations

import json
import logging
import os

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
# compile gets progressively stable samples. Only OpenAI-supported params are
# used (top_k/min_p/repeat_penalty are rejected by the OpenAI API).
SLIDE_SAMPLING_PROFILES: tuple[dict[str, float], ...] = (
    {  # Attempt 1 — most creative
        "temperature": 0.6,
        "top_p": 0.8,
    },
    {  # Attempt 2 — middle
        "temperature": 0.35,
        "top_p": 0.7,
    },
    {  # Attempt 3 — most reliable
        "temperature": 0.1,
        "top_p": 0.5,
    },
)


def slide_sampling_for_attempt(attempt: int) -> dict[str, float]:
    """Sampling params for the given 1-based write attempt.

    Attempts beyond the number of profiles reuse the last (most
    deterministic) profile; attempts below 1 reuse the first.
    """
    idx = max(0, min(attempt - 1, len(SLIDE_SAMPLING_PROFILES) - 1))
    return dict(SLIDE_SAMPLING_PROFILES[idx])


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
