"""Response utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from schemez import InlineSchemaDef


if TYPE_CHECKING:
    from llmling_agent.agent import AgentContext


def to_type(output_type, context: AgentContext | None = None) -> type[BaseModel | str]:
    match output_type:
        case str():
            if context and output_type in context.definition.responses:
                defn = context.definition.responses[output_type]  # from defined responses
                return defn.response_schema.get_schema()
            msg = f"Missing context for response type: {output_type!r}"
            raise ValueError(msg)
        case InlineSchemaDef():
            return output_type.get_schema()
        case None:
            return str
        case type() as model if issubclass(model, BaseModel | str):
            return model
        case _:
            msg = f"Invalid output_type: {type(output_type)}"
            raise TypeError(msg)
