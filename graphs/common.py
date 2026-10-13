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
from pydantic import BaseModel

from core.blocking import await_blocking
from core.language import DEFAULT_LANGUAGE, localize_status
from core.reasoning import require_stage

logger = logging.getLogger(__name__)

# How many times a model may reach for tools before it must answer. Each round
# is one model call plus one round trip per requested tool, so this is the
# bound on what a tool-using step can cost: two rounds let the model search,
# look at what came back, and search once more, and no further.
_MAX_TOOL_CALL_ROUNDS = 2


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
    timings rows — that is what the material graph already stores there, and what
    ``scripts/slide_analysis.py`` reads, with the graph in its own column. This
    is the one place that maps between the two (#194 review).
    """
    return require_stage(qualified_stage).split(".", 1)[1]


def timed_node[State](
    stage: str,
    node: Callable[[State], dict],
    context: Callable[[State], dict] | None = None,
):
    r"""Wrap a graph node so every execution records its own wall time (#194).

    ``stage`` is the qualified stage name —
    :func:`core.reasoning.stage_name` (``"clarify.assess_goal"``) — the same
    constant the node passes to :func:`core.reasoning.stage_call_kwargs`. A
    wrapper is a model call, so it must be a stage this app has a reasoning
    budget for: ``require_stage`` makes that an error at graph-build time rather
    than a row in the timings table that no policy knows about.

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
    recorded_stage = recorded_stage_name(stage)

    def timed(state: State) -> dict:
        started = time.monotonic()
        try:
            result = node(state)
        except Exception:
            logger.warning(
                "%s: failed after %.1fs", stage, time.monotonic() - started, exc_info=True
            )
            raise
        duration = time.monotonic() - started
        logger.debug("%s: %.1fs", stage, duration)
        entry = {
            "stage": recorded_stage,
            "context": context(state) if context is not None else {},
            "duration_seconds": duration,
        }
        return {**result, "stage_timings": [entry]}

    return timed


def structured_invoke[Schema: BaseModel](
    llm: BaseChatModel, schema: type[Schema], system: str, human: str,
    **bind_kwargs: Any,
) -> Schema:
    """Run the LLM with structured output and return a validated schema instance.

    ``bind_kwargs`` are forwarded to :func:`structured_invoke_messages` for
    callers that build the messages inline (e.g. ``reasoning_effort="minimal"``
    on a slide-generation call).
    """
    return structured_invoke_messages(
        llm, schema, [SystemMessage(content=system), HumanMessage(content=human)],
        **bind_kwargs,
    )


def structured_invoke_messages[Schema: BaseModel](
    llm: BaseChatModel, schema: type[Schema], messages: list[BaseMessage],
    **bind_kwargs: Any,
) -> Schema:
    """Run the LLM with structured output over an explicit message list.

    Lets the caller build the messages once and reuse the exact same objects
    for both the call and persistence (e.g. saving the real prompt sent).

    ``bind_kwargs`` (e.g. ``temperature=0.1``) are applied via ``bind`` AFTER
    ``with_structured_output`` — binding before it silently drops the params.
    """
    model = llm.with_structured_output(schema)
    if bind_kwargs:
        model = model.bind(**bind_kwargs)
    return cast(Schema, model.invoke(messages))


def structured_invoke_with_tools[Schema: BaseModel](
    llm: BaseChatModel, schema: type[Schema], system: str, human: str,
    tools: Sequence[BaseTool], **bind_kwargs: Any,
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
        llm, schema, messages, **bind_kwargs
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
