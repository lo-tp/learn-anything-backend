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
# Bounded self-repair turns: how many model-driven repair turns a single
# write_slide attempt makes against ``validate_jsx`` before handing off to the
# outer retry routing (MAX_MATERIAL_ATTEMPTS). 0 disables the inner loop.
MAX_SLIDE_SELF_REPAIR_TURNS = max(
    0, int(os.getenv("MAX_SLIDE_SELF_REPAIR_TURNS", "2"))
)
# Safety cap on how many closing braces ``balance_braces`` may append to a
# truncated slide; beyond this the output is garbled, not merely truncated.
MAX_BRACE_APPEND = 10

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


# --- Self-repair tool (validate_jsx) and deterministic repairs ---
#
# ``validate_jsx`` is the tool the ``write_slide`` self-repair loop binds to:
# the model writes JSX, the node validates it here (deterministic repairs
# first, then a sandbox compile), and on failure the reduced error is handed
# back to the model for a bounded repair turn. The sandbox compile is the
# authoritative check — deterministic repairs never replace it.


def _strip_markdown_fences(code: str) -> str:
    """Strip a markdown code fence (```lang ... ```) if present.

    Returns the inner code block if there is a fence pair, else the input
    unchanged. Only the first opening and last closing fence are considered
    (a slide is one block); prose before/after the block is dropped with the
    fences. A leading short language-tag line (e.g. ``tsx``) is removed too.
    """
    start = code.find("```")
    end = code.rfind("```")
    if start == -1 or end <= start:
        return code
    inner = code[start + 3 : end]
    # Drop a leading language-tag line (a short identifier followed by a newline).
    m = re.match(r"^\s*[a-zA-Z0-9_+-]{0,10}\s*\n", inner)
    if m:
        inner = inner[m.end() :]
    return inner.strip()


def _unbalanced_open_braces(code: str) -> int:
    """Net unclosed ``{`` minus ``}`` count, ignoring strings and comments.

    Best-effort (a string-aware scan, so braces inside quotes, template
    literals, ``//`` and ``/* */`` comments are not counted). A positive
    result means the code has unclosed openers (the truncated-output case);
    zero or negative means balanced or excess closers (left alone — deleting
    closers is not a safe repair).
    """
    delta = 0
    i = 0
    n = len(code)
    in_str: str | None = None
    in_line_comment = False
    in_block_comment = False
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_str is not None:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        # Not in a string or comment.
        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if c == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if c in ("'", '"', "`"):
            in_str = c
            i += 1
            continue
        if c == "{":
            delta += 1
        elif c == "}":
            delta -= 1
        i += 1
    return delta


def deterministic_repair(code: str) -> tuple[str, list[str]]:
    """Apply safe, mechanical repairs to raw slide JSX before validation.

    Repairs only ever *add* structure (strip fences, append missing closers);
    they never delete model content, so they cannot break an already-valid
    module. Returns ``(repaired_code, names)`` where ``names`` lists the
    repairs that actually changed the code (empty when nothing needed fixing).
    """
    repairs: list[str] = []
    src = code

    stripped = _strip_markdown_fences(src)
    if stripped != src:
        src = stripped
        repairs.append("strip_markdown_fences")

    src = src.strip()

    delta = _unbalanced_open_braces(src)
    if delta > 0:
        n = min(delta, MAX_BRACE_APPEND)
        src = src + "\n" + "}" * n
        repairs.append("balance_braces")

    return src, repairs


def validate_jsx(code: str) -> dict:
    """Validate a slide's JSX: compile it; if it fails, apply deterministic
    repairs and retry.

    The raw code is compiled *first*, so a valid slide (e.g. one whose text
    contains apostrophes) is returned unchanged and never corrupted by a
    repair. Only code that already failed to compile is touched by the
    additive, best-effort deterministic repairs — which run before the
    failure is reported to the model.

    This is the tool the ``write_slide`` self-repair loop binds to. It returns
    a structured result rather than raising:

    - ``ok`` True: ``compiled_code`` is the sandbox output and ``source`` is
      the (possibly repaired) source that compiled.
    - ``ok`` False: ``error`` is the reduced, LLM-facing compile error.
    - ``repairs`` lists the deterministic repairs applied to reach ``source``.
    """
    compiled, error = _compile_slide(code)
    if compiled is not None:
        return {
            "ok": True,
            "compiled_code": compiled,
            "source": code,
            "error": "",
            "repairs": [],
        }
    # The raw code failed to compile: try the deterministic repairs (additive
    # only, so they cannot break an already-valid module).
    source, repairs = deterministic_repair(code)
    if source != code:
        repaired_compiled, repaired_error = _compile_slide(source)
        if repaired_compiled is not None:
            return {
                "ok": True,
                "compiled_code": repaired_compiled,
                "source": source,
                "error": "",
                "repairs": repairs,
            }
        error = repaired_error
    return {
        "ok": False,
        "compiled_code": None,
        "source": source,
        "error": compile_error_for_feedback(error),
        "repairs": repairs,
    }


# --- Slide failure classes (C1.2) ---
#
# The closed set of classes a failed slide compile attempt maps to. Every
# attempt maps to exactly one; ``other`` is the honest remainder for an error
# that carries no signal (a bare ``500 Internal Server Error``, for example).
# This is the taxonomy ``scripts/classify_slide_failures.py`` reports over
# persisted attempts, and the classes ``compile_slide`` stamps on live
# failures — one taxonomy, so the offline report and the live traces agree.
SLIDE_FAILURE_CLASSES: tuple[str, ...] = (
    "syntax_error",
    "unknown_component",
    "truncated_output",
    "sandbox_timeout",
    "content_spec_invalid",
    "other",
)

# Tokens per class, lower-cased, matched against the reduced error. Classes are
# tested in the order of the checks in ``classify_compile_error``: specific
# signals before generic ones, because an esbuild diagnostic always opens with
# "Build failed" and a sandbox gate rejection can read like a syntax complaint.
_TIMEOUT_TOKENS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "deadline",
    "gateway timeout",
)
# A name the sandbox cannot resolve — at build time (an import it cannot
# satisfy) or at render time (a bare identifier the module never declared).
_UNKNOWN_COMPONENT_TOKENS: tuple[str, ...] = (
    "could not resolve",
    "no matching export",
    "not exported",
    "is not exported",
    "module not found",
    "cannot find module",
    "is not defined",
    "cannot find name",
)
# The sandbox's content gate rejecting what came back *as a slide*: too short
# to be a slide, not a module, default export is not a component, renders
# nothing, or threw while being rendered headlessly. (A gate throw that names an
# unresolvable identifier was already claimed by _UNKNOWN_COMPONENT_TOKENS.)
_CONTENT_SPEC_TOKENS: tuple[str, ...] = (
    "invalid json body",
    "must be a non-empty string",
    "must be at least",
    "code must declare",
    "default export must be a function",
    "renders empty",
    "component failed the gate",
)
# The parser ran out of input: the model's answer was cut off mid-module. Note
# that esbuild's "Unterminated string literal" is deliberately NOT here — the
# persisted occurrences are complete modules with a quoting bug inside a math
# string, not a truncated answer, and they belong with the syntax errors.
_TRUNCATED_TOKENS: tuple[str, ...] = (
    "unexpected eof",
    "unexpected end",
    "end of file",
    "eof",
)
_SYNTAX_TOKENS: tuple[str, ...] = (
    "syntax",
    "expected",
    "parse",
    "unexpected",
    "invalid",
    "missing",
    "unclosed",
    "build failed",
)


def classify_compile_error(error: str) -> str:
    """The failure class for a reduced slide compile error.

    Returns exactly one of ``SLIDE_FAILURE_CLASSES``:

    - ``sandbox_timeout``: the compile never came back (transport/timeout, a
      gateway timeout) — nothing is known about the slide itself.
    - ``unknown_component``: a name the sandbox cannot resolve, at build time
      ("could not resolve", "no matching export") or at render time
      ("``Foo`` is not defined").
    - ``content_spec_invalid``: the content gate rejected what the model
      returned *as a slide* — too short, not a module (no ``export
      default``), default export is not a component, renders empty, or threw
      while rendering. The parser did not reject the code.
    - ``truncated_output``: the parser hit the end of the input: the model's
      answer was cut off mid-module.
    - ``syntax_error``: esbuild rejected the source ("Build failed with 1
      error: ...", "not valid inside a JSX element").
    - ``other``: the error carries no usable signal.

    Best-effort but total: an unrecognised error is ``other``, never a raise.
    Makes slide-skip rates measurable by class; the C1.2 classifier
    (``scripts/classify_slide_failures.py``) reports it over persisted
    attempts.
    """
    e = error.lower()
    if any(t in e for t in _TIMEOUT_TOKENS):
        return "sandbox_timeout"
    if any(t in e for t in _UNKNOWN_COMPONENT_TOKENS):
        return "unknown_component"
    if any(t in e for t in _CONTENT_SPEC_TOKENS):
        return "content_spec_invalid"
    if any(t in e for t in _TRUNCATED_TOKENS):
        return "truncated_output"
    if any(t in e for t in _SYNTAX_TOKENS):
        return "syntax_error"
    return "other"


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
