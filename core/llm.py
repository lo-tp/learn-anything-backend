"""OpenAI-compatible chat model (config from .env).

What this module decides is the *client*: which model to ask, at which address,
how long one request is allowed to take, how many times a request that never
answered is asked again, and the output cap a call gets when it states none of
its own. Everything about a *call* — what it may think
(:mod:`core.reasoning`), how it decodes (:mod:`core.sampling`) — is decided per
call instead, and the values here are read once, at startup, because the client
is built once.

Every one of them is a setting, read through :mod:`core.settings`: a typo is
reported and the default stands, so a bad number is never the reason the app
cannot start.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from core.mock_llm import is_mock_mode
from core.settings import (
    LLM_MAX_OUTPUT_TOKENS_ENV,
    LLM_MAX_RETRIES_ENV,
    LLM_REQUEST_TIMEOUT_SECONDS_ENV,
    numeric_setting,
)

load_dotenv()

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
# The ceiling for a call that states no cap of its own. Every stage call states
# one (`core/reasoning.py`), so this is what an uncapped call would get — and it
# honours `LLM_MAX_OUTPUT_TOKENS`, the same name every stage's cap honours, so a
# run that raises every cap raises this with them rather than being bounded by a
# number nobody wrote down.
_DEFAULT_MAX_OUTPUT_TOKENS = 65536

if is_mock_mode():
    # #118: mock mode must start without OPENAI_API_KEY. A real ChatOpenAI is
    # still constructed (so the material graph stays wired to the real llm)
    # with a placeholder key — it is never invoked in mock mode.
    _api_key = SecretStr("mock")
else:
    _api_key = SecretStr(os.getenv("OPENAI_API_KEY", ""))

llm = ChatOpenAI(
    api_key=_api_key,
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    # max_tokens is a chat-completions parameter the client forwards rather than
    # one its signature names, so the call needs the ignore it always needed.
    max_tokens=numeric_setting(  # type: ignore[call-arg]
        (LLM_MAX_OUTPUT_TOKENS_ENV,), _DEFAULT_MAX_OUTPUT_TOKENS, int
    ),
    timeout=numeric_setting(
        (LLM_REQUEST_TIMEOUT_SECONDS_ENV,), _DEFAULT_REQUEST_TIMEOUT_SECONDS, float
    ),
    max_retries=numeric_setting((LLM_MAX_RETRIES_ENV,), _DEFAULT_MAX_RETRIES, int),
)
