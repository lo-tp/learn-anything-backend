"""Shared helpers for the LangGraph graphs."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from core.language import DEFAULT_LANGUAGE, localize_status


def structured_invoke[Schema: BaseModel](
    llm: BaseChatModel, schema: type[Schema], system: str, human: str
) -> Schema:
    """Run the LLM with structured output and return a validated schema instance."""
    return structured_invoke_messages(
        llm, schema, [SystemMessage(content=system), HumanMessage(content=human)]
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
