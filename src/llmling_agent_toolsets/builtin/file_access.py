"""Provider for file access tools."""

from __future__ import annotations

import asyncio
import mimetypes
import time
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pydantic_ai import BinaryContent
from upath import UPath
from upathtools import list_files, read_path

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.tools.base import Tool
from llmling_agent.tools.exceptions import ToolError


if TYPE_CHECKING:
    from llmling_agent.prompts.conversion_manager import ConversionManager


logger = get_logger(__name__)

# MIME types that should be treated as text
TEXT_MIME_PREFIXES = ("text/", "application/json", "application/xml", "application/javascript")


def _is_text_mime(mime_type: str | None) -> bool:
    """Check if a MIME type represents text content."""
    if mime_type is None:
        return False
    return any(mime_type.startswith(prefix) for prefix in TEXT_MIME_PREFIXES)


class FileAccessTools(ResourceProvider):
    """Provider for file access tools."""

    def __init__(
        self,
        name: str = "file_access",
        converter: ConversionManager | None = None,
    ) -> None:
        """Initialize file access toolset.

        Args:
            name: Name for this toolset provider
            converter: Optional conversion manager for markdown conversion
        """
        super().__init__(name=name)
        self.converter = converter
        self._tools: list[Tool] | None = None

    async def get_tools(self) -> list[Tool]:
        """Get file access tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            Tool.from_callable(
                self.read_file,
                name_override="read_file",
                description_override=(
                    "Read file natively - returns text for text files, "
                    "binary content for documents/images (for model vision/doc capabilities)"
                ),
                source="builtin",
                category="read",
            ),
            Tool.from_callable(
                self.list_directory,
                name_override="list_directory",
                description_override="List files / subfolders in a folder",
                source="builtin",
                category="search",
            ),
            Tool.from_callable(
                self.download_file,
                name_override="download_file",
                description_override="Download a file and return status information",
                source="builtin",
                category="read",
            ),
        ]

        # Only add read_as_markdown if converter is available
        if self.converter:
            self._tools.append(self.create_tool(self.read_as_markdown, category="read"))
        return self._tools

    async def read_file(  # noqa: D417
        self,
        ctx: AgentContext,
        path: str,
        *,
        encoding: str = "utf-8",
        line: int | None = None,
        limit: int | None = None,
    ) -> str | BinaryContent:
        """Read file natively - text for text files, binary for documents/images.

        Args:
            path: Path or URL to read
            encoding: Text encoding to use for text files (default: utf-8)
            line: Optional line number to start reading from (1-based, text files only)
            limit: Optional maximum number of lines to read (text files only)

        Returns:
            Text content for text files, BinaryContent for binary files
        """
        try:
            mime_type = mimetypes.guess_type(path)[0]

            if _is_text_mime(mime_type):
                content = await read_path(path, encoding=encoding)

                if line is not None or limit is not None:
                    lines = content.splitlines(keepends=True)
                    start_idx = (line - 1) if line is not None else 0
                    end_idx = start_idx + limit if limit is not None else len(lines)
                    content = "".join(lines[start_idx:end_idx])

                await ctx.events.file_operation("read", path=path, success=True, size=len(content))
                return content

            # Binary file - return as BinaryContent for native model handling
            data = await read_path(path, mode="rb")
            await ctx.events.file_operation("read", path=path, success=True, size=len(data))
            return BinaryContent(
                data=data, media_type=mime_type or "application/octet-stream", identifier=path
            )

        except Exception as e:
            msg = f"Failed to read file {path}: {e}"
            await ctx.events.file_operation("read", path=path, success=False, error=msg)
            raise ToolError(msg) from e

    async def read_as_markdown(self, ctx: AgentContext, path: str) -> str:  # noqa: D417
        """Read file and convert to markdown text representation.

        Args:
            path: Path or URL to read

        Returns:
            File content converted to markdown
        """
        assert self.converter is not None, "Converter required for read_as_markdown"

        try:
            content = await self.converter.convert_file(path)
        except Exception as e:
            msg = f"Failed to convert file {path}: {e}"
            await ctx.events.file_operation("read", path=path, success=False, error=msg)
            raise ToolError(msg) from e
        else:
            await ctx.events.file_operation("read", path=path, success=True, size=len(content))
            return content

    async def list_directory(  # noqa: D417
        self,
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
            await ctx.events.file_operation("list", path=path, success=True, size=len(files))
        except Exception as e:
            msg = f"Failed to list directory {path}: {e}"
            await ctx.events.file_operation("list", path=path, success=False, error=msg)
            raise ToolError(msg) from e
        else:
            return result

    async def download_file(  # noqa: D417
        self,
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


if __name__ == "__main__":

    async def main() -> None:
        tools = FileAccessTools()
        async with tools:
            result = await tools.get_tools()
            print(result)

    asyncio.run(main())
