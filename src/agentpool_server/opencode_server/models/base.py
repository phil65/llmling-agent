"""Base model for all OpenCode API models."""

from pydantic import BaseModel, ConfigDict


class OpenCodeBaseModel(BaseModel):
    """Base model with OpenCode-compatible configuration.

    All OpenCode models should inherit from this to ensure:
    - Fields can be populated by their alias (camelCase) or Python name (snake_case)
    - Serialization uses aliases by default for API compatibility
    """

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,  # Always use camelCase aliases in JSON output
    )
