"""Provider for file access tools."""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse

from llmling import ToolError
from pydantic_ai.tools import RunContext

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


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
    from upathtools import read_path

    if isinstance(ctx, RunContext):
        ctx = ctx.deps

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
        raise ToolError(msg) from e
    else:
        return content


async def list_directory(
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
    from upathtools import list_files

    pattern = pattern or "**/*"
    files = await list_files(
        path,
        pattern=pattern,
        include_dirs=include_dirs,
        recursive=recursive,
        exclude=exclude,
        max_depth=max_depth,
    )
    return "\n".join(str(f) for f in files)


async def download_file(
    context: AgentContext,
    url: str,
    target_dir: str = "downloads",
    chunk_size: int = 8192,
    verify_ssl: bool = False,  # For testing, in prod should be True
) -> str:
    """Download a file and return status information."""
    import httpx
    from upath import UPath

    start_time = time.time()
    target_path = UPath(target_dir)
    target_path.mkdir(exist_ok=True)

    filename = UPath(urlparse(url).path).name or "downloaded_file"
    full_path = target_path / filename
    try:
        async with (
            httpx.AsyncClient(verify=verify_ssl) as client,
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
                        if context.report_progress:
                            await context.report_progress(progress, 100, msg)
                        await asyncio.sleep(0)

        duration = time.time() - start_time
        size_mb = size / 1_048_576

        return f"Downloaded {filename} ({size_mb:.1f}MB) at {size_mb / duration:.1f} MB/s"

    except httpx.ConnectError as e:
        return f"Connection error downloading {url}: {e}"
    except httpx.TimeoutException:
        return f"Timeout downloading {url}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} downloading {url}"
    except Exception as e:  # noqa: BLE001
        return f"Error downloading {url}: {e!s}"


def create_file_access_tools() -> list[Tool]:
    """Create tools for file and directory access operations."""
    return [
        Tool.from_callable(read_file, source="builtin", category="read"),
        Tool.from_callable(list_directory, source="builtin", category="search"),
        Tool.from_callable(download_file, source="builtin", category="read"),
    ]


class FileAccessTools(StaticResourceProvider):
    """Provider for file access tools."""

    def __init__(self, name: str = "file_access"):
        super().__init__(name=name, tools=create_file_access_tools())
