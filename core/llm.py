"""OpenAI-compatible chat model (config from .env)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from core.mock_llm import is_mock_mode

load_dotenv()

logger = logging.getLogger(__name__)

# A stalled request has to end (#194). Left unset, the OpenAI SDK waits 600
# seconds and retries twice: half an hour of a learner watching one call that was
# never going to come back. This backend answers a normal pre-slide call in
# seconds (a full Clarify assessment with its thinking on took 18.4s), so 120 is
# several times the slowest measured call and a fraction of the default.
#
# `core/reasoning.py` bounds what a call may think; this bounds how long one is
# allowed to take. Both are environment knobs so either can be retuned for a
# measurement — an experiment at `reasoning_effort=high` is exactly the case that
# wants a larger value, and `LLM_MAX_RETRIES=0` is the case that wants a failure
# reported now rather than after three attempts.
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
_DEFAULT_MAX_RETRIES = 2


def _env_number[NumberT](name: str, default: NumberT, cast: Callable[[str], NumberT]) -> NumberT:
    """A numeric setting, or the default when it is absent or unreadable.

    A typo in a knob is reported and the default stands: a bad number must not
    be the reason the app cannot start.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using the default %s", name, raw, default)
        return default


if is_mock_mode():
    # #118: mock mode must start without OPENAI_API_KEY. A real ChatOpenAI is
    # still constructed (so the material graph stays wired to the real llm)
    # with a placeholder key — it is never invoked in mock mode.
    _api_key = SecretStr("mock")
else:
    _api_key = SecretStr(os.getenv("OPENAI_API_KEY", ""))

_request_timeout = _env_number(
    "LLM_REQUEST_TIMEOUT_SECONDS",
    _DEFAULT_REQUEST_TIMEOUT_SECONDS,
    float,
)
_max_retries = _env_number("LLM_MAX_RETRIES", _DEFAULT_MAX_RETRIES, int)

llm = ChatOpenAI(
    api_key=_api_key,
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    max_tokens=65536,  # type: ignore[call-arg]
    timeout=_request_timeout,
    max_retries=_max_retries,
)
