"""Reading a model's answer when it was asked to answer in JSON (#195).

The measured fact this module exists for
--------------------------------------
On this project's llama.cpp backend (``qwen3.8-flash-next-iq3_xxs``) a call that
asks for structured output is answered with:

.. code-block:: text

    502 {"type": "structured_output_failed",
         "message": "model did not return valid JSON: Invalid \\escape: line 20 column 48"}

The model wrote a backslash — ``\\int``, ``\\frac``, ``\\beta`` — that JSON does
not accept as an escape. The backend compiled a grammar from the schema, the
grammar permitted the backslash, the model wrote it, and the backend then
*validated the completion as JSON* and refused it. Measured 8/8 on the
``plan_slide_contents`` prompt whose input carries LaTeX inherited from
``summarize_step``, at both 8192 and 65536 output tokens, with and without a
reasoning budget: it is the request's shape that fails, not its size.

Why the repair has to live on this side of the wire
--------------------------------------------------
The refusal withholds the completion. There is no text here to fix, so the only
plan that can produce an answer is to ask again without schema enforcement —
where the model's text arrives as *text* (a string in the response, correctly
escaped by the server) and the JSON inside it is ours to read. That is what
:func:`graphs.common.structured_invoke_messages` does; this module is the reading.

Note what that means for the escape rules the backend enforces: on the second
plan nothing constrains the decode, so the model is free to write the same
backslash again, and it usually does — the free-form completions measured here
contained 21-22 invalid escapes. The repair, not the prompt, is what carries
this class.

What it cost, measured on the same input
---------------------------------------
Run through the shipped seam against the same backend, on the same maths step
that reproduced the fault: 4/4 answered where 8/8 had been refused, 27-60
escapes repaired per call, 21-44 backslashes preserved in the stored key points,
and no form feed or backspace among them. The cost is one extra generation on
the class that used to die — ~40s for that stage, against ~30s of identical
retries and then no deck. Skipping those identical retries for this class, and
restoring the control characters the *first* plan already mangled, are the next
measured steps; neither is taken here.

Legal escapes, and the two this app reads as macros
--------------------------------------------------
:func:`repair_json_escapes` doubles every backslash that is *not* part of a
legal JSON escape pair. Legal pairs are consumed as pairs, so a text that was
already correct is returned unchanged: a substitution-based "fix" turns ``\\\\nabla``
into three backslashes and corrupts the strings it was meant to save.

``\\f`` and ``\\b`` are legal JSON escapes and are still doubled, on purpose.
They are what ``\\frac``, ``\\beta``, ``\\binom`` and ``\\bar`` look like when
JSON reads them — a form feed plus ``rac{1}{2}`` — and a real form feed or
backspace is not a thing this app's teaching text contains. Leaving them alone
would be the strictly JSON-correct reading and the wrong one: it stores a
control character where a formula was written, which is the half of #195 that
never raised an error at all.

What the repair cannot know
---------------------------
``\\nu`` is a valid JSON escape, so it is kept as a newline, and a key point
about the Greek letter ν loses its letter. ``\\rho`` and ``\\times`` are the same
case. Deciding whether a backslash-and-letters meant a macro or an escape is a
question about *what the text is for*, and only the prompt knows that: it is why
the notation policy (#195's other half) is a separate decision from this one, and
why a repaired answer says how many escapes it took rather than pretending it
read the model's mind.
"""

from __future__ import annotations

import contextvars
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Literal, get_args, get_origin

from openai import APIStatusError
from pydantic import BaseModel

# The backend's own discriminator for "I could not parse what the model wrote",
# and the sentence it uses when it reports that class without a code. Either one
# is the same fault; neither is a fault worth *not* recovering from.
STRUCTURED_OUTPUT_REFUSED: Final = "structured_output_failed"
JSON_REFUSAL_PHRASE: Final = "model did not return valid JSON"

# How many objects in an answer are tried before the answer is called unreadable.
_MAX_JSON_CANDIDATES = 5

# What JSON accepts after a backslash, and what it accepts but this app reads as
# the start of a macro. See the module docstring.
_LEGAL_JSON_ESCAPES: Final = frozenset('"\\/bfnrtu')
_MACRO_LIKE_ESCAPES: Final = frozenset("fb")


class ModelAnswerUnparseable(RuntimeError):
    """A model's answer could not be made into the shape its stage asked for.

    Named, with the stage in it, because the alternative is an anonymous parse
    error reaching a driver that only knows how to mark a whole session
    ``error``. A stage that cannot answer is a decision the driver gets to make;
    an unparseable blob is not.

    ``raw`` is the text that was read, not the whole completion: it is what
    makes a failure explainable from the log rather than guessable.
    """

    def __init__(self, stage: str, reason: str, raw: str) -> None:
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason
        self.raw = raw


@dataclass(frozen=True)
class Answer:
    """A model's JSON answer, read — and what had to be repaired to read it.

    The count travels with the value on purpose: an answer that needed 21 escapes
    fixed is a different fact from one that needed none, and the stage row is
    where that fact belongs.
    """

    value: Any
    repaired_escapes: int = 0


@dataclass(frozen=True)
class ModelRecovery:
    """One recovered answer, for the stage record that produced it."""

    reason: str
    repaired_escapes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "repaired_escapes": self.repaired_escapes,
        }


# The recoveries attributed to the stage record written next. Drained, not read,
# because the next stage must not be blamed for the last one's repair.
_recoveries: contextvars.ContextVar[list[ModelRecovery] | None] = contextvars.ContextVar(
    "model_answer_recoveries", default=None
)


def record_model_recovery(recovery: ModelRecovery) -> None:
    """Say which stage's answer had to be recovered, for the row written next.

    The row is written by the graph, which is the only place that knows its own
    stage name; this is the part that knows what happened to the answer. Where
    no row is written at all the record simply goes nowhere, and the log remains.
    """
    bucket = _recoveries.get()
    if bucket is None:
        bucket = []
        _recoveries.set(bucket)
    bucket.append(recovery)


def drain_model_recoveries() -> list[dict[str, Any]]:
    """The recoveries since the last drain, as the ``context`` field they become.

    Empty is the normal answer: most stages never need this, and a row that says
    nothing about recovery is a row that needed none.
    """
    bucket = _recoveries.get()
    if not bucket:
        return []
    drained = [recovery.as_dict() for recovery in bucket]
    bucket.clear()
    return drained


def backend_refused_json(exc: BaseException) -> bool:
    """Whether the backend said it could not parse the model's JSON.

    The server's own discriminator, never a status code: 502 is also what a busy
    model answers with, and repeating the request is the right plan for *that*
    one. A shape miss (the answer parsed and was wrong) is not this either — it
    is a prompt problem, and asking again with less enforcement is a guess at it.
    """
    if not isinstance(exc, APIStatusError):
        return False
    return _mentions_refusal(exc.body) or _mentions_refusal(str(exc))


def _mentions_refusal(value: Any) -> bool:
    """Search a response body for the discriminator at any depth.

    The same fault has been reported with the code in ``body["type"]``, nested in
    ``body["error"]``, and only in the message text.
    """
    if isinstance(value, str):
        return STRUCTURED_OUTPUT_REFUSED in value or JSON_REFUSAL_PHRASE in value
    if isinstance(value, Mapping):
        return any(_mentions_refusal(item) for item in value.values())
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return any(_mentions_refusal(item) for item in value)
    return False


def repair_json_escapes(text: str) -> tuple[str, int]:
    """Double every backslash that is not already a legal JSON escape pair.

    Returns the text and how many backslashes were doubled. One pass, legal
    pairs consumed as pairs: a regex that substitutes "a backslash followed by
    letters" would rewrite ``\\n`` in a code snippet, ``\\u00e9`` in a name, and
    its own output on the next pass.

    ``\\f`` and ``\\b`` are doubled although JSON accepts them, for the reason in
    the module docstring: in this app's text they are macros, not control
    characters.
    """
    out: list[str] = []
    repaired = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char != "\\":
            out.append(char)
            index += 1
            continue
        following = text[index + 1] if index + 1 < length else ""
        if following in _LEGAL_JSON_ESCAPES and following not in _MACRO_LIKE_ESCAPES:
            out.append(char)
            out.append(following)
            index += 2
            continue
        out.append("\\\\")
        repaired += 1
        index += 1
    return "".join(out), repaired


def read_json_answer(text: str) -> Answer:
    """Read the JSON a model answered with, repairing the escapes it could not write.

    Accepts the shape models actually produce: fenced, or with a sentence either
    side of it. The object is found by matching braces with strings and escapes
    understood, so a ``{`` inside a key point is not mistaken for structure.

    Raises:
        ValueError: there is no JSON object in the answer — including the
            truncated case, where one starts and never closes.
    """
    candidates = _json_object_candidates(text)
    if not candidates:
        raise ValueError("the model's answer contains no JSON object")

    parse_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        repaired, repaired_count = repair_json_escapes(candidate)
        try:
            return Answer(value=json.loads(repaired), repaired_escapes=repaired_count)
        except json.JSONDecodeError as exc:
            parse_error = exc
    raise ValueError(f"the model's answer is not valid JSON: {parse_error}")


def _json_object_candidates(text: str) -> list[str]:
    """The balanced objects in ``text``, first first, up to a few.

    First-parseable rather than largest: a model that writes "for example
    ``{…}``:" before its real answer still gets its answer read, and a model
    that stops mid-object gets no candidate at all — which is how a truncated
    answer becomes a stated failure instead of a truncated success.
    """
    candidates: list[str] = []
    start = text.find("{")
    while start != -1 and len(candidates) < _MAX_JSON_CANDIDATES:
        end = _matching_brace(text, start)
        if end is not None:
            candidates.append(text[start : end + 1])
            start = text.find("{", end + 1)
        else:
            start = text.find("{", start + 1)
    return candidates


def _matching_brace(text: str, start: int) -> int | None:
    """The index that closes the object opened at ``start``, or None."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def json_shape_spec(schema: type[BaseModel]) -> str:
    """The shape of an *answer*, as an example the model can copy.

    Used when the schema is no longer being enforced by the backend, so it is the
    only thing left holding the contract. It is deliberately not the schema
    dumped out: the one sample of an unconstrained ask on this backend had the
    model answer with the schema itself —
    ``{"type": "object", "additionalProperties": false, …}`` — which parses as
    JSON and matches nothing.

    Field *bounds* (a list's 3-7, a difficulty's 1-5) are not repeated here:
    every prompt that asks for those numbers already states them, and a shape
    spec that starts restating prompt rules is a second place for the two to
    disagree.
    """
    return _object_spec(schema, frozenset())


def _object_spec(model: type[BaseModel], seen: frozenset[type]) -> str:
    """A JSON object with every field of ``model`` filled with an example value."""
    if model in seen:  # a self-referring model: show the field, stop the recursion
        return "{}"
    nested = seen | {model}
    fields = [
        f'"{name}": {_value_spec(field.annotation, nested)}'
        for name, field in model.model_fields.items()
    ]
    return "{" + ", ".join(fields) + "}"


def _value_spec(annotation: Any, seen: frozenset[type]) -> str:
    """An example value for one field, as JSON text."""
    while get_origin(annotation) is not None and get_origin(annotation) is not Literal:
        # Strip Annotated[int, Field(...)] and Optional[...] down to what is asked for.
        if get_origin(annotation) in (list, set, frozenset, tuple) or str(
            get_origin(annotation)
        ).endswith("Sequence"):
            args = [arg for arg in get_args(annotation) if arg is not Ellipsis]
            inner = _value_spec(args[0], seen) if args else '"<value>"'
            return f"[{inner}]"
        if get_origin(annotation) in (dict, Mapping):
            args = [arg for arg in get_args(annotation) if arg is not Ellipsis]
            value = _value_spec(args[1], seen) if len(args) == 2 else "{}"
            return '{"<key>": ' + value + "}"
        args = get_args(annotation)
        annotation = args[0] if args else str
    if get_origin(annotation) is Literal:
        options = " | ".join(str(option) for option in get_args(annotation))
        return f'"<one of: {options}>"'
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _object_spec(annotation, seen)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return f'"<one of: {" | ".join(str(item.value) for item in annotation)}>"'
    if annotation is str:
        return '"<string>"'
    if annotation is int:
        return "0"
    if annotation is float:
        return "0.0"
    if annotation is bool:
        return "false"
    return '"<value>"'
