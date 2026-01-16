"""Predefined Configuration fields."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from agentpool_config.output_types import StructuredResponseConfig


OutputTypeField = Annotated[
    str | StructuredResponseConfig | None,
    Field(
        default=None,
        examples=["json_response", "code_output"],
        title="Response type",
        description="Optional structured output type for responses. "
        "Can be either a reference to a response defined in manifest.responses, "
        "or an inline StructuredResponseConfig.",
    ),
]
