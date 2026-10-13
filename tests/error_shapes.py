"""Test doubles shaped like the errors this app's LLM backend really answers with.

Same reasoning as ``tool_shapes.py``: #195 is about *recognising* a fault, and a
classifier tested against a convenient fake exception tests the fake. These are
built from ``httpx`` and the openai SDK's own error classes, with the body the
llama.cpp server returns for ``502 structured_output_failed``.
"""

from __future__ import annotations

import json

import httpx2  # the httpx the openai SDK 3.x builds its errors from

# The sentence this backend writes when the model's JSON will not parse, copied
# from the response that ended a deck (#195).
INVALID_ESCAPE = "model did not return valid JSON: Invalid \\escape: line 20 column 48 (char 1595)"

STRUCTURED_OUTPUT_FAILED = {
    "type": "structured_output_failed",
    "code": "structured_output_failed",
    "message": INVALID_ESCAPE,
}


def api_error(
    *, status: int, error: dict[str, object], error_cls: type[BaseException] | None = None
) -> BaseException:
    """An API error with a real response and a real body.

    ``error_cls`` defaults to the langchain wrapper this app actually catches
    (``OpenAIAPIError``, a subclass of ``openai.InternalServerError``); pass
    ``openai.BadRequestError`` or similar to test the same fault at another
    status.
    """
    from langchain_openai.chat_models.base import OpenAIAPIError

    request = httpx2.Request("POST", "http://backend.test/v1/chat/completions")
    response = httpx2.Response(status, request=request, json={"error": error})
    cls = error_cls or OpenAIAPIError
    # The signature that matters here is openai.APIStatusError's, which is what
    # the SDK itself calls and what OpenAIAPIError inherits; pyright resolves
    # ``__init__`` from the model-error base instead, so the real construction
    # needs saying rather than reshaping (verified against a real 502 from the
    # backend — see core/model_answers.py).
    return cls(
        f"Error code: {status} - {json.dumps(error)}",
        response=response,  # pyright: ignore[reportCallIssue,reportArgumentType]
        body=error,  # pyright: ignore[reportCallIssue]
    )


def structured_output_refused(*, status: int = 502) -> BaseException:
    """The refusal this app gets when the model writes an invalid JSON escape."""
    return api_error(status=status, error=dict(STRUCTURED_OUTPUT_FAILED))


def model_unavailable() -> BaseException:
    """A backend that is there and busy: the same *status*, a different fault."""
    return api_error(status=503, error={"message": "upstream model is busy"})
