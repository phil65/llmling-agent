"""Models for resource information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field
from schemez import Schema


if TYPE_CHECKING:
    from mcp.types import Resource as MCPResource


@dataclass
class ResourceInfo:
    """Information about an available resource.

    This class provides essential information about a resource that can be loaded.
    Use the resource name with load_resource() to access the actual content.
    """

    name: str
    """Name of the resource, use this with load_resource()"""

    uri: str
    """URI identifying the resource location"""

    description: str | None = None
    """Optional description of the resource's content or purpose"""

    @classmethod
    async def from_mcp_resource(cls, resource: MCPResource) -> Self:
        """Create ResourceInfo from MCP resource."""
        return cls(name=resource.name, uri=str(resource.uri), description=resource.description)


class BaseResourceConfig(Schema):
    """Base configuration for resources."""

    type: str = Field(init=False, title="Resource config type")
    """Type discriminator for resource configs."""

    path: str | None = Field(
        default=None,
        examples=["/data", "documents", "config/templates"],
        title="Path prefix",
    )
    """Optional path prefix within the filesystem."""

    cached: bool = Field(default=False, title="Enable caching")
    """Whether to wrap in caching filesystem."""

    storage_options: dict[str, Any] = Field(
        default_factory=dict,
        examples=[
            {"aws_access_key_id": "AKIAIOSFODNN7EXAMPLE", "aws_secret_access_key": "secret123"},
            {"token": "github_pat_123", "timeout": 30},
            {"username": "user", "password": "pass", "ssl": True},
        ],
        title="Storage options",
    )
    """Protocol-specific storage options."""

    model_config = ConfigDict(frozen=True)


class SourceResourceConfig(BaseResourceConfig):
    """Configuration for a single filesystem source."""

    type: Literal["source"] = Field("source", init=False)
    """Direct filesystem source."""

    uri: str = Field(
        examples=["file:///path/to/docs", "s3://bucket-name/data", "https://api.example.com"],
        title="Resource URI",
    )
    """URI defining the resource location and protocol."""


class UnionResourceConfig(BaseResourceConfig):
    """Configuration for combining multiple resources."""

    type: Literal["union"] = Field("union", init=False)
    """Union of multiple resources."""

    sources: list[ResourceConfig] = Field(title="Resource sources")
    """List of resources to combine."""


# Union type for resource configs
ResourceConfig = Annotated[SourceResourceConfig | UnionResourceConfig, Field(discriminator="type")]
