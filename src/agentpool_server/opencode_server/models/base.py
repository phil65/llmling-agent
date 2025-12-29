"""Base model for all OpenCode API models."""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


def convert(text: str) -> str:
    val = to_camel(text)
    if val.endswith("Id"):
        return val.rstrip("Id") + "ID"
    return val


class OpenCodeBaseModel(BaseModel):
    """Base model with OpenCode-compatible configuration.

    All OpenCode models should inherit from this to ensure:
    - Fields can be populated by their alias (camelCase) or Python name (snake_case)
    - Serialization uses aliases by default for API compatibility
    """

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=convert,
        use_attribute_docstrings=True,
    )
