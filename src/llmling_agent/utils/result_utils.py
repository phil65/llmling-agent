"""Response utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from schemez import InlineSchemaDef


if TYPE_CHECKING:
    from llmling_agent_config.output_types import StructuredResponseConfig


def to_type(
    output_type: Any,
    # output_type: str | InlineSchemaDef | type | None,
    responses: dict[str, StructuredResponseConfig] | None = None,
) -> type[BaseModel | str]:
    match output_type:
        case str() if responses and output_type in responses:
            defn = responses[output_type]  # from defined responses
            return defn.response_schema.get_schema()
        case str():
            msg = f"Missing responses dict for response type: {output_type!r}"
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
