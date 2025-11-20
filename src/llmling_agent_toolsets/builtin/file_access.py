"""Provider for file access tools."""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse

from upath import UPath
from upathtools import list_files, read_path

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.resource_providers import StaticResourceProvider
from llmling_agent.tools.base import Tool
from llmling_agent.tools.exceptions import ToolError


logger = get_logger(__name__)


async def read_file(  # noqa: D417
    ctx: AgentContext,
    path: str,
    *,
    convert_to_markdown: bool = True,
    encoding: str = "utf-8",
    line: int | None = None,
    limit: int | None = None,
) -> str:
    """Read file content from local or remote path.

    Args:
        path: Path or URL to read
        convert_to_markdown: Whether to convert content to markdown
        encoding: Text encoding to use (default: utf-8)
        line: Optional line number to start reading from (1-based)
        limit: Optional maximum number of lines to read

    Returns:
        File content, optionally converted to markdown
    """
    try:
        # First try to read raw content
        content = await read_path(path, encoding=encoding)

        # Convert to markdown if requested
        if convert_to_markdown and ctx.converter:
            try:
                content = await ctx.converter.convert_file(path)
            except Exception as e:  # noqa: BLE001
                msg = f"Failed to convert to markdown: {e}"
                logger.warning(msg)
                # Continue with raw content

        # Apply line filtering if requested
        if line is not None or limit is not None:
            lines = content.splitlines(keepends=True)
            start_idx = (line - 1) if line is not None else 0
            end_idx = start_idx + limit if limit is not None else len(lines)
            content = "".join(lines[start_idx:end_idx])

    except Exception as e:
        msg = f"Failed to read file {path}: {e}"
        # Emit failure event
        await ctx.events.file_operation("read", path=path, success=False, error=msg)
        raise ToolError(msg) from e
    else:
        # Emit success event
        await ctx.events.file_operation("read", path=path, success=True, size=len(content))
        return content


async def list_directory(  # noqa: D417
    ctx: AgentContext,
    path: str,
    *,
    pattern: str | None = None,
    recursive: bool = True,
    include_dirs: bool = False,
    exclude: list[str] | None = None,
    max_depth: int | None = None,
) -> str:
    """List files / subfolders in a folder.

    Args:
        path: Base directory to read from
        pattern: Glob pattern to match files against (e.g. "**/*.py" for Python files)
        recursive: Whether to search subdirectories
        include_dirs: Whether to include directories in results
        exclude: List of patterns to exclude (uses fnmatch against relative paths)
        max_depth: Maximum directory depth for recursive search

    Returns:
        A list of files / folders.
    """
    pattern = pattern or "**/*"
    try:
        files = await list_files(
            path,
            pattern=pattern,
            include_dirs=include_dirs,
            recursive=recursive,
            exclude=exclude,
            max_depth=max_depth,
        )
        result = "\n".join(str(f) for f in files)
        # Emit success event
        await ctx.events.file_operation("list", path=path, success=True, size=len(files))
    except Exception as e:
        msg = f"Failed to list directory {path}: {e}"
        # Emit failure event
        await ctx.events.file_operation("list", path=path, success=False, error=msg)
        raise ToolError(msg) from e
    else:
        return result


async def download_file(  # noqa: D417
    ctx: AgentContext,
    url: str,
    target_dir: str = "downloads",
    chunk_size: int = 8192,
) -> str:
    """Download a file and return status information.

    Args:
        url: URL to download from
        target_dir: Directory to save the file
        chunk_size: Size of chunks to download

    Returns:
        Status message about the download
    """
    import httpx

    start_time = time.time()
    target_path = UPath(target_dir)
    target_path.mkdir(exist_ok=True)

    filename = UPath(urlparse(url).path).name or "downloaded_file"
    full_path = target_path / filename
    try:
        async with (
            httpx.AsyncClient(verify=False) as client,
            client.stream("GET", url, timeout=30.0) as response,
        ):
            response.raise_for_status()

            total = (
                int(response.headers["Content-Length"])
                if "Content-Length" in response.headers
                else None
            )

            with full_path.open("wb") as f:
                size = 0
                async for chunk in response.aiter_bytes(chunk_size):
                    size += len(chunk)
                    f.write(chunk)

                    if total and (size % (chunk_size * 100) == 0 or size == total):
                        progress = size / total * 100
                        speed_mbps = (size / 1_048_576) / (time.time() - start_time)
                        msg = f"\r{filename}: {progress:.1f}% ({speed_mbps:.1f} MB/s)"
                        await ctx.events.progress(progress, 100, msg)
                        await asyncio.sleep(0)

        duration = time.time() - start_time
        size_mb = size / 1_048_576
        result = f"Downloaded {filename} ({size_mb:.1f}MB) at {size_mb / duration:.1f} MB/s"

        # Emit success event
        await ctx.events.file_operation("read", path=str(full_path), success=True, size=size)

    except httpx.ConnectError as e:
        error_msg = f"Connection error downloading {url}: {e}"
        await ctx.events.file_operation("read", path=url, success=False, error=error_msg)
        return error_msg
    except httpx.TimeoutException:
        error_msg = f"Timeout downloading {url}"
        await ctx.events.file_operation("read", path=url, success=False, error=error_msg)
        return error_msg
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error {e.response.status_code} downloading {url}"
        await ctx.events.file_operation("read", path=url, success=False, error=error_msg)
        return error_msg
    except Exception as e:  # noqa: BLE001
        error_msg = f"Error downloading {url}: {e!s}"
        await ctx.events.file_operation("read", path=url, success=False, error=error_msg)
        return error_msg
    else:
        return result


def create_file_access_tools() -> list[Tool]:
    """Create tools for file and directory access operations."""
    return [
        Tool.from_callable(read_file, source="builtin", category="read"),
        Tool.from_callable(list_directory, source="builtin", category="search"),
        Tool.from_callable(download_file, source="builtin", category="read"),
    ]


class FileAccessTools(StaticResourceProvider):
    """Provider for file access tools."""

    def __init__(self, name: str = "file_access") -> None:
        super().__init__(name=name, tools=create_file_access_tools())


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        tools = FileAccessTools()
        async with tools:
            result = await tools.get_tools()
            print(result)

    asyncio.run(main())
