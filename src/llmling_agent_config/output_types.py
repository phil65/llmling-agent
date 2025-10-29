"""Models for response fields and definitions."""

from __future__ import annotations

from schemez import Schema, SchemaDef


class StructuredResponseConfig(Schema):
    """Base class for response definitions."""

    response_schema: SchemaDef
    """A model describing the response schema. """

    description: str | None = None
    """A description for this response definition."""

    result_tool_name: str = "final_result"
    """The tool name for the Agent tool to create the structured response."""

    result_tool_description: str | None = None
    """The tool description for the Agent tool to create the structured response."""

    output_retries: int | None = None
    """Retry override. How often the Agent should try to validate the response."""
