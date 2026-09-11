"""Shared helpers for the LangGraph graphs."""

from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


def structured_invoke[Schema: BaseModel](
    llm: BaseChatModel, schema: type[Schema], system: str, human: str
) -> Schema:
    """Run the LLM with structured output and return a validated schema instance."""
    return cast(
        Schema,
        llm.with_structured_output(schema).invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        ),
    )
