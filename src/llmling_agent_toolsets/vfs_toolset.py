"""VFS registry filesystem toolset implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers import StaticResourceProvider


if TYPE_CHECKING:
    from llmling_agent.agents.context import AgentContext


async def vfs_list(  # noqa: D417
    ctx: AgentContext,
    path: str = "",
    *,
    pattern: str = "**/*",
    recursive: bool = True,
    include_dirs: bool = False,
    exclude: list[str] | None = None,
    max_depth: int | None = None,
) -> str:
    """List contents of a resource path.

    Args:
        path: Resource path to query (e.g., "docs", "docs://guides", "data://files")
        pattern: Glob pattern to match files against
        recursive: Whether to search subdirectories
        include_dirs: Whether to include directories in results
        exclude: List of patterns to exclude
        max_depth: Maximum directory depth for recursive search

    Returns:
        Formatted list of matching files and directories
    """
    registry = ctx.definition.vfs_registry

    # If no path given, list all resources
    if not path:
        resources = list(registry)
        return "Available resources:\n" + "\n".join(f"- {name}" for name in resources)

    try:
        files = await registry.query(
            path,
            pattern=pattern,
            recursive=recursive,
            include_dirs=include_dirs,
            exclude=exclude,
            max_depth=max_depth,
        )
        return "\n".join(files) if files else f"No files found in {path}"
    except (OSError, ValueError, KeyError) as e:
        return f"Error listing {path}: {e}"


async def vfs_read(  # noqa: D417
    ctx: AgentContext,
    path: str,
    *,
    encoding: str = "utf-8",
    recursive: bool = True,
    exclude: list[str] | None = None,
    max_depth: int | None = None,
) -> str:
    """Read content from a resource path.

    Args:
        path: Resource path to read (e.g., "docs://file.md", "data://folder")
        encoding: Text encoding for binary content
        recursive: For directories, whether to read recursively
        exclude: For directories, patterns to exclude
        max_depth: For directories, maximum depth to read

    Returns:
        File content or concatenated directory contents
    """
    registry = ctx.definition.vfs_registry

    try:
        return await registry.get_content(
            path,
            encoding=encoding,
            recursive=recursive,
            exclude=exclude,
            max_depth=max_depth,
        )
    except (OSError, ValueError, KeyError) as e:
        return f"Error reading {path}: {e}"


async def vfs_info(ctx: AgentContext) -> str:
    """Get information about all configured resources.

    Returns:
        Formatted information about available resources
    """
    registry = ctx.definition.vfs_registry

    if registry.is_empty:
        return "No resources configured"

    sections = ["## Configured Resources\n"]

    sections.extend(f"- **{name}**: VFS Resource" for name in registry)

    # Add union filesystem info
    try:
        union_fs = registry.get_fs()
        sections.append(f"\n**Union filesystem**: {union_fs.__class__.__name__}")
        sections.append(f"**Total resources**: {len(registry)}")
    except (OSError, AttributeError) as e:
        sections.append(f"\n**Union filesystem error**: {e}")

    return "\n".join(sections)


class VFSTools(StaticResourceProvider):
    """Provider for VFS registry filesystem tools."""

    def __init__(self, name: str = "vfs") -> None:
        tools = [
            self.create_tool(vfs_list, category="search", read_only=True, idempotent=True),
            self.create_tool(vfs_read, category="read", read_only=True, idempotent=True),
            self.create_tool(vfs_info, category="read", read_only=True, idempotent=True),
        ]
        super().__init__(name=name, tools=tools)
