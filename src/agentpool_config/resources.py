"""Models for resource information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self


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
