"""Base class for generated models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


def convert(text: str) -> str:
    if text == "field_meta":
        return "_meta"
    return to_camel(text)


class Schema(BaseModel):
    """Base class for generated models."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=convert,
        use_attribute_docstrings=True,
    )


class AnnotatedObject(Schema):
    """Base class for generated models."""

    field_meta: dict[str, Any] | None = None
    """Extension point for implementations."""


class Request(AnnotatedObject):
    """Base request model."""


class Response(AnnotatedObject):
    """Base request model."""
