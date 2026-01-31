"""VFS registry filesystem toolset implementation."""

from __future__ import annotations

from upathtools import list_files, read_folder, read_path

from agentpool.agents.context import AgentContext  # noqa: TC001
from agentpool.resource_providers import StaticResourceProvider


async def vfs_list(  # noqa: D417
    ctx: AgentContext,
    path: str = "/",
    *,
    pattern: str = "**/*",
    recursive: bool = True,
    include_dirs: bool = False,
    exclude: list[str] | None = None,
    max_depth: int | None = None,
) -> str:
    """List contents of a filesystem path.

    Lists files from the agent's unified filesystem, which includes:
    - Agent's internal storage (execute_code/, tasks/, etc.)
    - Configured VFS resources

    Args:
        path: Path to query (e.g., "/", "/docs", "/tasks")
        pattern: Glob pattern to match files against
        recursive: Whether to search subdirectories
        include_dirs: Whether to include directories in results
        exclude: List of patterns to exclude
        max_depth: Maximum directory depth for recursive search

    Returns:
        Formatted list of matching files and directories
    """
    fs = ctx.overlay_fs

    try:
        # List root to show available top-level directories
        if path in ("/", ""):
            items = await fs._ls(fs.root_marker, detail=True)
            lines = ["Available paths:"]
            for item in items:
                item_type = "📁" if item.get("type") == "directory" else "📄"
                name = item.get("name", "").strip("/")
                lines.append(f"  {item_type} /{name}")
            return "\n".join(lines)

        # Query specific path
        upath = fs.get_upath(path)
        files = await list_files(
            upath,
            pattern=pattern,
            recursive=recursive,
            include_dirs=include_dirs,
            exclude=exclude,
            max_depth=max_depth,
        )
        return "\n".join(str(f) for f in files) if files else f"No files found in {path}"
    except FileNotFoundError:
        return f"Path not found: {path}"
    except (OSError, ValueError) as e:
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
    """Read content from a filesystem path.

    Reads from the agent's unified filesystem, which includes:
    - Agent's internal storage (execute_code/, tasks/, etc.)
    - Configured VFS resources

    Args:
        path: Path to read (e.g., "/docs/readme.md", "/tasks/task_123/output.md")
        encoding: Text encoding for binary content
        recursive: For directories, whether to read recursively
        exclude: For directories, patterns to exclude
        max_depth: For directories, maximum depth to read

    Returns:
        File content or concatenated directory contents
    """
    fs = ctx.overlay_fs

    try:
        upath = fs.get_upath(path)

        if await fs._isdir(path):
            content_dict = await read_folder(
                upath,
                encoding=encoding,
                recursive=recursive,
                exclude=exclude,
                max_depth=max_depth,
            )
            # Combine all files with headers
            sections = []
            for rel_path, content in sorted(content_dict.items()):
                sections.extend([f"--- {rel_path} ---", content, ""])
            return "\n".join(sections)

        return await read_path(upath, encoding=encoding)
    except FileNotFoundError:
        return f"Path not found: {path}"
    except (OSError, ValueError) as e:
        return f"Error reading {path}: {e}"


async def vfs_info(ctx: AgentContext) -> str:
    """Get information about the agent's unified filesystem.

    Returns:
        Formatted information about available filesystem layers
    """
    fs = ctx.overlay_fs

    sections = ["## Agent Filesystem\n"]
    sections.append("Unified view combining agent storage and configured resources.\n")

    # List layers
    sections.append("### Layers")
    for i, layer in enumerate(fs.layers):
        layer_type = "writable" if i == 0 else "read-only"
        sections.append(f"- Layer {i} ({layer_type}): {type(layer).__name__}")

    # List top-level contents
    sections.append("\n### Contents")
    try:
        items = await fs._ls(fs.root_marker, detail=True)
        for item in items:
            item_type = "dir" if item.get("type") == "directory" else "file"
            name = item.get("name", "").strip("/")
            layer = item.get("layer", "?")
            sections.append(f"- /{name} ({item_type}, layer {layer})")
    except (OSError, FileNotFoundError) as e:
        sections.append(f"Error listing contents: {e}")

    return "\n".join(sections)


class VFSTools(StaticResourceProvider):
    """Provider for unified filesystem tools.

    Provides tools for listing and reading from the agent's overlay filesystem,
    which combines the agent's internal storage with configured VFS resources.
    """

    def __init__(self, name: str = "vfs") -> None:
        super().__init__(name=name, tools=[])
        for tool in [
            self.create_tool(vfs_list, category="search", read_only=True, idempotent=True),
            self.create_tool(vfs_read, category="read", read_only=True, idempotent=True),
            self.create_tool(vfs_info, category="read", read_only=True, idempotent=True),
        ]:
            self.add_tool(tool)
