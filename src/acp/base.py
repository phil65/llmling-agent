"""Base class for generated models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


def convert(text: str):
    if text == "field_meta":
        return "_meta"
    return to_camel(text)


class Schema(BaseModel):
    """Base class for generated models."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=convert)


class Request(Schema):
    """Base request model."""

    field_meta: Any | None = None
    """Extension point for implementations."""


class Response(Schema):
    """Base request model."""

    field_meta: Any | None = None
    """Extension point for implementations."""
