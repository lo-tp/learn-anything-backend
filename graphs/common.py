"""Shared helpers for the LangGraph graphs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

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

# How many times a model may reach for tools before it must answer. Each round
# is one model call plus one round trip per requested tool, so this is the
# bound on what a tool-using step can cost: two rounds let the model search,
# look at what came back, and search once more, and no further.
_MAX_TOOL_CALL_ROUNDS = 2


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
    tools: Sequence[BaseTool],
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

    Which node is given tools is decided by the graph that builds it — today
    only ``research_topic`` is (#166). The turn itself is shared because the
    boundary it serves is: an internal tool (slide JSX validation, #165) is
    bound through this same path, in-process.
    """
    messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=human)]
    if tools:
        by_name = {t.name: t for t in tools}
        model = llm.bind_tools(list(tools))
        for _ in range(_MAX_TOOL_CALL_ROUNDS):
            asked = cast(AIMessage, model.invoke(messages))
            if not asked.tool_calls:
                break
            messages.append(asked)
            for call in asked.tool_calls:
                messages.append(_run_tool_call(by_name, call))
    return structured_invoke_messages(llm, schema, messages)


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
