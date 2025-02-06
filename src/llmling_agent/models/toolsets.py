"""Models for tools."""

from __future__ import annotations

from typing import Annotated, Literal

from llmling import ConfigModel
from llmling.tools.toolsets import ToolSet
from llmling.utils.importing import import_class
from pydantic import Field, field_validator


class BaseToolsetConfig(ConfigModel):
    """Base configuration for toolsets."""

    namespace: str | None = Field(default=None)
    """Optional namespace prefix for tool names"""


class OpenAPIToolsetConfig(BaseToolsetConfig):
    """Configuration for OpenAPI toolsets."""

    type: Literal["openapi"] = Field("openapi", init=False)
    """Discriminator field identifying this as an OpenAPI toolset."""

    spec: str = Field(...)
    """URL or path to the OpenAPI specification document."""

    base_url: str | None = Field(default=None)
    """Optional base URL for API requests, overrides the one in spec."""


class EntryPointToolsetConfig(BaseToolsetConfig):
    """Configuration for entry point toolsets."""

    type: Literal["entry_points"] = Field("entry_points", init=False)
    """Discriminator field identifying this as an entry point toolset."""

    module: str = Field(..., description="Python module path")
    """Python module path to load tools from via entry points."""


class CustomToolsetConfig(BaseToolsetConfig):
    """Configuration for custom toolsets."""

    type: Literal["custom"] = Field("custom", init=False)
    """Discriminator field identifying this as a custom toolset."""

    import_path: str = Field(...)
    """Dotted import path to the custom toolset implementation class."""

    @field_validator("import_path", mode="after")
    @classmethod
    def validate_import_path(cls, v: str) -> str:
        # v is already confirmed to be a str here
        try:
            cls = import_class(v)
            if not issubclass(cls, ToolSet):
                msg = f"{v} must be a ToolSet class"
                raise ValueError(msg)  # noqa: TRY004, TRY301
        except Exception as exc:
            msg = f"Invalid toolset class: {v}"
            raise ValueError(msg) from exc
        return v


# Use discriminated union for toolset types
ToolsetConfig = Annotated[
    OpenAPIToolsetConfig | EntryPointToolsetConfig | CustomToolsetConfig,
    Field(discriminator="type"),
]
