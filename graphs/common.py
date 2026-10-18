"""Shared helpers for the LangGraph graphs."""

from __future__ import annotations

import logging
import operator
import time
from collections.abc import Callable, Sequence
from typing import Annotated, Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from core.blocking import await_blocking
from core.language import DEFAULT_LANGUAGE, localize_status
from core.model_answers import (
    ModelAnswerUnparseable,
    ModelRecovery,
    backend_refused_json,
    drain_model_recoveries,
    json_shape_spec,
    read_json_answer,
    record_model_recovery,
)
from core.reasoning import require_stage
from core.sampling import apply_configured_sampling

logger = logging.getLogger(__name__)

# How many times a model may reach for tools before it must answer. Each round
# is one model call plus one round trip per requested tool, so this is the
# bound on what a tool-using step can cost: two rounds let the model search,
# look at what came back, and search once more, and no further.
_MAX_TOOL_CALL_ROUNDS = 2

# What a recovered answer is called in a ``graph_stage_timings`` row. #195 is a
# fault that used to end a deck in silence, so the row is where "which stage
# leaned on the second plan, and how much repairing it needed" stops being a
# guess from log lines.
RECOVERED_JSON = "backend_refused_json"


class StageTimingsState(TypedDict, total=False):
    """The state channel every graph shares for per-stage wall timings (#194).

    One entry per stage a run executed, shaped the same way in all four graphs:
    ``{"stage": str, "context": dict, "duration_seconds": float}`` — the stage
    named without its graph prefix, the attribution that makes a row usable (a
    round, a batch, a pass), and the seconds it took.

    The channel accumulates *within* a run because a run has several stages.
    Between runs it is emptied by :func:`services.stage_timings.empty_stage_timings`
    in the state each driver passes, so a resume records what it ran and not what
    an earlier request already had written as rows.

    The material graph's nodes write these entries themselves; Clarify, Probe and
    Plan get them from :func:`timed_node`. Both land in the same
    ``graph_stage_timings`` rows and the same analysis.
    """

    stage_timings: Annotated[list[dict], operator.add]


def recorded_stage_name(qualified_stage: str) -> str:
    """The ``graph_stage_timings.stage`` value for a qualified stage name.

    A stage is named ``clarify.assess_goal`` where a graph builds its nodes and
    where its model call asks for a reasoning budget, and ``assess_goal`` in the
    timings rows — that is what the material graph has always stored there, and
    what ``scripts/slide_analysis.py`` reads, with the graph in its own column.
    This is the one place that maps between the two (#194 review), so a stage has
    one name in the code and the row spelling is derived rather than typed out
    again.

    Whether a stage has a reasoning budget is *not* checked here. That is
    :func:`timed_node`'s question, asked at graph-build time where a stage with
    no budget is a mistake worth stopping; this mapping is also used by the
    slide-writing stages and by ``compile_slide``, which name their own request
    policy in ``graphs/material/sandbox.py`` or call no model at all.
    """
    _, separator, stage = qualified_stage.partition(".")
    if not separator or not stage:
        raise ValueError(f"{qualified_stage!r} is not a qualified stage name")
    return stage


def timed_node[State](
    stage: str,
    node: Callable[[State], dict],
    context: Callable[[State], dict] | None = None,
):
    r"""Wrap a graph node so every execution records its own wall time (#194).

    ``stage`` is the qualified stage name —
    :func:`core.reasoning.stage_name` (``"clarify.assess_goal"``) — the same
    constant the node passes to :func:`core.reasoning.stage_call_kwargs` and to
    the structured call itself. A wrapper is a model call, so it must be a stage
    this app has a reasoning budget for: ``require_stage`` makes that an error at
    graph-build time rather than a row in the timings table that no policy knows
    about.

    The returned node is left un-annotated on purpose: LangGraph's node protocol
    wants a function, and a ``Callable[[State], dict]`` annotation is not one —
    it fails the protocol check for the same code.

    The material graph times its stages inside each node because its driver runs
    the whole graph in one background task and persists the timings between
    interrupts. Clarify, Probe and Plan pause between HTTP requests instead, so
    their timing belongs at the seam rather than in twelve node bodies: the graph
    builders wrap what they add, and each driver persists what is new.

    Entries are shaped exactly like the material graph's ``stage_timings``
    (``{"stage", "context", "duration_seconds"}``, the stage named without its
    graph prefix, as ``graph_stage_timings`` stores it) so both land in the same
    rows and the same analysis.

    Only a node that *does work* is wrapped. An interrupt node's wall time is the
    learner's, spent between requests, and recording it would make the timings
    measure the wrong thing.
    """
    # A wrapper is a model call, so it must be a stage this app has a reasoning
    # budget for: failing here, at graph assembly, is what keeps the timings rows
    # and core/reasoning.py naming the same thing.
    require_stage(stage)
    recorded_stage = recorded_stage_name(stage)

    def timed(state: State) -> dict:
        started = time.monotonic()
        try:
            result = node(state)
        except Exception:
            # A failed stage writes no row, so its recovery (if it made one on
            # the way to failing) is drained here rather than left to be blamed
            # on the next stage that does write one.
            drain_model_recoveries()
            logger.warning(
                "%s: failed after %.1fs", stage, time.monotonic() - started, exc_info=True
            )
            raise
        duration = time.monotonic() - started
        logger.debug("%s: %.1fs", stage, duration)
        entry = {
            "stage": recorded_stage,
            "context": _stage_context(state, context),
            "duration_seconds": duration,
        }
        return {**result, "stage_timings": [entry]}

    return timed


def with_model_recoveries(context: dict) -> dict:
    """A stage row's context, plus any answer this stage had to recover (#195).

    The recoveries are *drained*, not read: they belong to the stage being
    recorded now, and must not be counted again for the next one. A stage that
    needed nothing gains no field at all, which is what keeps these rows readable
    while they are queried for the ones that did.
    """
    recoveries = drain_model_recoveries()
    if not recoveries:
        return context
    return {**context, "model_recoveries": recoveries}


def _stage_context(state: Any, context: Callable[[Any], dict] | None) -> dict:
    """What the caller said this execution was, attributed to the row it describes."""
    return with_model_recoveries(dict(context(state)) if context is not None else {})


# What a recovered answer is called in a ``graph_stage_timings`` row, and why it
# is there: #195 is a fault that used to end a deck silently, and a stage that
# needs this on most calls is a defect worth having a number for.
RECOVERED_JSON = "backend_refused_json"


def structured_invoke[Schema: BaseModel](
    llm: BaseChatModel, schema: type[Schema], system: str, human: str,
    *, stage: str, stage_position: int | None = None, **bind_kwargs: Any,
) -> Schema:
    """Run the LLM with structured output and return a validated schema instance.

    ``bind_kwargs`` are forwarded to :func:`structured_invoke_messages` for
    callers that build the messages inline (e.g. ``reasoning_effort="minimal"``
    on a slide-generation call), and ``stage_position`` with them: which repeat of
    the stage this call is, for a stage that repeats one.
    """
    return structured_invoke_messages(
        llm, schema, [SystemMessage(content=system), HumanMessage(content=human)],
        stage=stage, stage_position=stage_position, **bind_kwargs,
    )


def structured_invoke_messages[Schema: BaseModel](
    llm: BaseChatModel, schema: type[Schema], messages: list[BaseMessage],
    *, stage: str, stage_position: int | None = None, **bind_kwargs: Any,
) -> Schema:
    r"""Run the LLM with structured output over an explicit message list.

    Lets the caller build the messages once and reuse the exact same objects
    for both the call and persistence (e.g. saving the real prompt sent).

    ``bind_kwargs`` (e.g. ``temperature=0.1``) are applied via ``bind`` AFTER
    ``with_structured_output`` — binding before it silently drops the params.

    Whatever the environment configures for this stage is layered on top of them
    here, rather than threaded through every node
    (:func:`core.sampling.apply_configured_sampling`): it sends nothing for a
    parameter nobody set, so the only calls this changes are
    the ones an operator has deliberately retuned, and it is what lets
    ``LLM_TEMPERATURE_PLAN_DESIGN_PLAN`` reach a call without that node knowing
    the knob exists.

    ``stage_position`` is which repeat of this stage the call is, 1-based, and it
    is what makes a setting that names one value per repeat readable:
    ``LLM_TEMPERATURE_MATERIAL_WRITE_SLIDE=0.4,0.2,0.1`` is the slide ladder
    retuned in one line, and only a call that says which attempt it is can be
    given the right rung. Only ``write_slide`` states a position today; every other
    call states nothing, including the ones a graph loop asks again round after
    round (``probe.generate_batch``), which would need their node to carry the
    round the way that node carries its attempt.

    ``stage`` names the call the way this file's caller already names it — the
    qualified constant a pre-slide node passes to
    :func:`core.reasoning.stage_call_kwargs`, the bare name a material node
    already logs and records its stage under. It is not re-validated here: what
    it is for is being able to say which call a recovered, or unreadable, answer
    came from.

    Two plans, in order (#195). The first is the one this app has always made:
    schema enforcement, which is what makes the answer shaped as the schema says.
    When the backend *refuses to parse what the model wrote* — it compiled a
    grammar from the schema, the model wrote ``$\int`` through it, and the
    backend then validated the completion as JSON and rejected it — the refusal
    withholds the completion, so there is nothing here to fix. The second plan
    asks for the same shape with nothing constraining the decode, reads the text
    on this side of the wire (:mod:`core.model_answers`), repairs the escapes JSON
    rejects, and validates the result against the same schema.

    Not everything is recoverable this way, deliberately. A backend that is busy,
    or an answer that parsed and did not fit the schema, is not a parse failure:
    one is served by repeating the request and the other is a prompt problem, and
    rewriting the request to guess at either is how a real failure becomes
    invisible. Those raise unchanged.
    """
    bind_kwargs = apply_configured_sampling(bind_kwargs, stage, stage_position)
    model = llm.with_structured_output(schema)
    if bind_kwargs:
        model = model.bind(**bind_kwargs)
    try:
        return cast(Schema, model.invoke(messages))
    except Exception as exc:
        # Every failure is looked at, not swallowed: only the one class this
        # second plan can actually answer is answered, and everything else is
        # raised to the caller exactly as the model's transport raised it.
        if not backend_refused_json(exc):
            raise
        logger.warning(
            "%s: the backend would not parse the model's JSON (#195); "
            "asking again for the same shape: %s",
            stage,
            exc,
        )
        return cast(
            Schema,
            _same_shape_unconstrained(
                llm, schema, messages, stage=stage, refused=exc, **bind_kwargs
            ),
        )


def _same_shape_unconstrained[Schema: BaseModel](
    llm: BaseChatModel, schema: type[Schema], messages: list[BaseMessage],
    *, stage: str, refused: Exception, **bind_kwargs: Any,
) -> Schema:
    r"""The second plan: the same shape, asked for without enforcing it.

    The stage's own messages are sent unchanged and the shape instruction is
    appended after them, so the language and teaching rules the stage set are
    still the ones in effect. The stage's reasoning budget and output cap are
    bound here too: a recovery is a model call like any other, and a fallback
    that escaped them would be a hole in the policy #194 exists to hold.
    """
    model = llm.bind(**bind_kwargs) if bind_kwargs else llm
    asked = [*messages, HumanMessage(content=_shape_instruction(schema))]
    text = _answer_text(model.invoke(asked))

    try:
        answer = read_json_answer(text)
    except ValueError as unreadable:
        raise ModelAnswerUnparseable(stage, str(unreadable), text) from refused
    try:
        result = schema.model_validate(answer.value)
    except ValidationError as invalid_shape:
        raise ModelAnswerUnparseable(
            stage, f"the answer did not fit {schema.__name__}: {invalid_shape}", text
        ) from invalid_shape

    record_model_recovery(
        ModelRecovery(reason=RECOVERED_JSON, repaired_escapes=answer.repaired_escapes)
    )
    logger.warning(
        "%s: recovered by asking without schema enforcement; %d JSON escape(s) "
        "repaired (#195)",
        stage,
        answer.repaired_escapes,
    )
    return result


def _shape_instruction(schema: type[BaseModel]) -> str:
    """Ask for the shape the backend is no longer enforcing.

    Phrased as a shape to copy, not a schema to describe: the one sample of an
    unconstrained ask on this backend had the model answering with the schema
    itself, which is valid JSON and matches nothing.
    """
    return (
        "Answer with JSON only, in exactly this shape — every field present, "
        "every value filled in, nothing written before or after the JSON. This "
        "is the shape of an answer to copy, not a schema to describe.\n\n"
        f"{json_shape_spec(schema)}"
    )


def _answer_text(response: AIMessage) -> str:
    """The text of a model's reply, whether it came as one string or as blocks."""
    content = response.content
    if isinstance(content, str):
        return content
    parts = [
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    ]
    return "\n".join(part for part in parts if part)


def structured_invoke_with_tools[Schema: BaseModel](
    llm: BaseChatModel, schema: type[Schema], system: str, human: str,
    tools: Sequence[BaseTool], *, stage: str,
    stage_position: int | None = None, **bind_kwargs: Any,
) -> Schema:
    """Run the LLM with structured output, offering it tools first.

    An empty ``tools`` is exactly :func:`structured_invoke` — no ``bind_tools``,
    no extra turn — which is what mock mode and any unconfigured external tool
    produce (#166).

    Otherwise the model is offered the tools and may call them over at most
    :data:`_MAX_TOOL_CALL_ROUNDS` rounds; every result is handed back to it,
    including a failed tool call (a tool that is down must not fail the
    request). The answer is then produced from that whole history through the
    usual structured path, so grounding never changes the output contract.

    ``bind_kwargs`` are bound on **both** halves of the turn (#194): deciding
    whether to search is where a tool-using node does its thinking, so a
    reasoning budget applied only to the answer after the search would leave the
    expensive part unbounded.

    Which node is given tools is decided by the graph that builds it — today
    only ``research_topic`` is (#166). The turn itself is shared because the
    boundary it serves is: an internal tool (slide JSX validation, #165) is
    bound through this same path, in-process.
    """
    # Applied once here, before the first half of the turn: deciding whether to
    # search is a model call like any other, so it runs under the same decoding
    # an operator configured. Applying it again downstream changes nothing —
    # layering the same values twice is layering them once.
    bind_kwargs = apply_configured_sampling(bind_kwargs, stage, stage_position)

    messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=human)]
    if tools:
        by_name = {t.name: t for t in tools}
        model = llm.bind_tools(list(tools), **bind_kwargs)
        for _ in range(_MAX_TOOL_CALL_ROUNDS):
            asked = cast(AIMessage, model.invoke(messages))
            if not asked.tool_calls:
                break
            messages.append(asked)
            for call in asked.tool_calls:
                messages.append(_run_tool_call(by_name, call))
    return structured_invoke_messages(
        llm, schema, messages, stage=stage, stage_position=stage_position,
        **bind_kwargs,
    )


def _run_tool_call(by_name: dict[str, BaseTool], call: ToolCall) -> ToolMessage:
    """Execute one requested tool call and hand its result back to the model.

    A tool that cannot run comes back as an error ``ToolMessage``, not an
    exception: the model sees that the search failed and answers without it. The
    plan graph must still produce a plan when an external service is down (#166).
    """
    tool = by_name.get(call["name"])
    if tool is None:
        return ToolMessage(
            content=f"unknown tool: {call['name']}",
            tool_call_id=call["id"],
            status="error",
        )
    try:
        content = await_blocking(tool.ainvoke(call["args"]))
    except Exception as exc:  # noqa: BLE001 — external tools are optional
        return ToolMessage(
            content=f"tool {call['name']} failed: {exc}",
            tool_call_id=call["id"],
            status="error",
        )
    return ToolMessage(content=content, tool_call_id=call["id"])


def unknown_option(llm: BaseChatModel | None, language: str | None = None) -> str:
    """The localized "I don't know" option appended to question options.

    English fallback on translation failure is built into ``localize_status``.
    """
    return localize_status(llm, language or DEFAULT_LANGUAGE, "I don't know")


def with_unknown_option(
    llm: BaseChatModel | None, language: str | None, options: list[str]
) -> list[str]:
    """Return ``options`` with the "I don't know" option appended last.

    Appending last means it never collides with ``correct_index`` (which always
    points at one of the LLM-generated options), so the learner can never
    score correctly by picking it. If an identical option already exists
    (case-insensitive), no duplicate is appended. The input list is not
    mutated.
    """
    unknown = unknown_option(llm, language)
    if any(o.strip().casefold() == unknown.strip().casefold() for o in options):
        return list(options)
    return [*options, unknown]
