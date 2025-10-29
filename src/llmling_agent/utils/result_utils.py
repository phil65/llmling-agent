"""Response utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from schemez import InlineSchemaDef


if TYPE_CHECKING:
    from llmling_agent.agent import AgentContext


def resolve_response_type(
    type_name: str | InlineSchemaDef,
    context: AgentContext | None,
) -> type[BaseModel]:  # type: ignore
    """Resolve response type from string name to actual type.

    Args:
        type_name: Name of the response type
        context: Agent context containing response definitions

    Returns:
        Resolved Pydantic model type

    Raises:
        ValueError: If type cannot be resolved
    """
    match type_name:
        case str() if context and type_name in context.definition.responses:
            defn = context.definition.responses[type_name]  # from defined responses
            return defn.response_schema.get_schema()
        case InlineSchemaDef():  # Handle inline definition
            return type_name.get_schema()
        case _:
            msg = f"Invalid result type: {type_name}"
            raise ValueError(msg)


def to_type(output_type, context: AgentContext | None = None) -> type[BaseModel | str]:
    match output_type:
        case str():
            return resolve_response_type(output_type, context)
        case InlineSchemaDef():
            return resolve_response_type(output_type, None)
        case None:
            return str
        case type() as model if issubclass(model, BaseModel | str):
            return model
        case _:
            msg = f"Invalid output_type: {type(output_type)}"
            raise TypeError(msg)
