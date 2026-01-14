"""Read file tool implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import BinaryContent

from agentpool.agents.context import AgentContext  # noqa: TC001
from agentpool.log import get_logger
from agentpool.mime_utils import guess_type, is_binary_content, is_binary_mime
from agentpool.tools.base import Tool, ToolResult


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from exxec import ExecutionEnvironment
    from fsspec.asyn import AsyncFileSystem

    from agentpool.prompts.conversion_manager import ConversionManager


logger = get_logger(__name__)


@dataclass
class ReadTool(Tool[ToolResult]):
    """Read files from the filesystem with support for text, binary, and image content.

    A standalone tool for reading files with advanced features:
    - Automatic binary/text detection
    - Image resizing and compression
    - Large file handling with structure maps
    - Line-based partial reads
    - Multi-encoding support

    Use create_read_tool() factory for convenient instantiation with defaults.
    """

    # Tool-specific configuration
    env: ExecutionEnvironment | None = None
    """Execution environment to use. Falls back to agent.env if not set."""

    converter: ConversionManager | None = None
    """Optional converter for binary files. If set and supports the file type, returns markdown."""

    cwd: str | None = None
    """Working directory for resolving relative paths."""

    max_file_size_kb: int = 64
    """Maximum file size in KB for read operations."""

    max_image_size: int | None = 2000
    """Max width/height for images in pixels. Images are auto-resized if larger."""

    max_image_bytes: int | None = None
    """Max file size for images in bytes. Images are compressed if larger."""

    large_file_tokens: int = 12_000
    """Token threshold for switching to structure map for large files."""

    map_max_tokens: int = 2048
    """Maximum tokens for structure map output."""

    _repomap = None  # RepoMap instance (lazy-init)

    def get_callable(
        self,
    ) -> Callable[..., Awaitable[ToolResult]]:
        """Return the read method as the callable."""
        return self._read

    def _get_fs(self, ctx: AgentContext) -> AsyncFileSystem:
        """Get filesystem from env, falling back to agent's env if not set."""
        from fsspec.asyn import AsyncFileSystem
        from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

        # Priority: env.get_fs() > agent.env.get_fs()
        if self.env is not None:
            fs = self.env.get_fs()
            return fs if isinstance(fs, AsyncFileSystem) else AsyncFileSystemWrapper(fs)
        fs = ctx.agent.env.get_fs()
        return fs if isinstance(fs, AsyncFileSystem) else AsyncFileSystemWrapper(fs)

    def _resolve_path(self, path: str, ctx: AgentContext) -> str:
        """Resolve a potentially relative path to an absolute path."""
        # Get cwd: explicit toolset cwd > env.cwd > agent.env.cwd
        cwd: str | None = None
        if self.cwd:
            cwd = self.cwd
        elif self.env and self.env.cwd:
            cwd = self.env.cwd
        elif ctx.agent.env and ctx.agent.env.cwd:
            cwd = ctx.agent.env.cwd

        if cwd and not (path.startswith("/") or (len(path) > 1 and path[1] == ":")):
            return str(Path(cwd) / path)
        return path

    async def _get_file_map(self, path: str, ctx: AgentContext) -> str | None:
        """Get structure map for a large file if language is supported."""
        from agentpool.repomap import RepoMap, is_language_supported

        if not is_language_supported(path):
            return None

        # Lazy init repomap - use file's directory as root
        if self._repomap is None:
            root = str(Path(path).parent)
            fs = self._get_fs(ctx)
            self._repomap = RepoMap(fs, root, max_tokens=self.map_max_tokens)

        return await self._repomap.get_file_map(path, max_tokens=self.map_max_tokens)

    async def _read(  # noqa: PLR0911, PLR0915
        self,
        ctx: AgentContext,
        path: str,
        encoding: str = "utf-8",
        line: int | None = None,
        limit: int | None = None,
    ) -> ToolResult:
        """Read the content of a text file, or use vision capabilities to read images or documents.

        Args:
            ctx: Agent context for event emission and filesystem access
            path: File path to read
            encoding: Text encoding to use for text files (default: utf-8)
            line: Optional line number to start reading from (1-based, text files only)
            limit: Optional maximum number of lines to read (text files only)

        Returns:
            Text content for text files, BinaryContent for binary files (with optional
            dimension note as list when image was resized), or error string
        """
        path = self._resolve_path(path, ctx)
        msg = f"Reading file: {path}"
        from agentpool.agents.events import LocationContentItem

        # Emit progress - use 0 for line if negative (can't resolve until we read file)
        # LocationContentItem/ToolCallLocation require line >= 0 per ACP spec
        display_line = line if (line is not None and line > 0) else 0
        await ctx.events.tool_call_progress(
            title=msg,
            items=[LocationContentItem(path=path, line=display_line)],
        )

        max_file_size = self.max_file_size_kb * 1024

        try:
            mime_type = guess_type(path)
            # Fast path: known binary MIME types (images, audio, video, etc.)
            if is_binary_mime(mime_type):
                # Try converter first if available
                if self.converter is not None:
                    try:
                        content = await self.converter.convert_file(path)
                        await ctx.events.file_operation("read", path=path, success=True)
                    except Exception:  # noqa: BLE001
                        # Converter doesn't support this file type, fall back to binary
                        pass
                    else:
                        # Converter returned markdown
                        lines = content.splitlines()
                        preview = "\n".join(lines[:20])
                        return ToolResult(
                            content=content,
                            metadata={"preview": preview, "truncated": False},
                        )

                # Fall back to native binary handling
                data = await self._get_fs(ctx)._cat_file(path)
                await ctx.events.file_operation("read", path=path, success=True)
                mime = mime_type or "application/octet-stream"
                # Resize images if needed
                if self.max_image_size and mime.startswith("image/"):
                    from agentpool_toolsets.fsspec_toolset.image_utils import (
                        resize_image_if_needed,
                    )

                    data, mime, note = resize_image_if_needed(
                        data, mime, self.max_image_size, self.max_image_bytes
                    )
                    if note:
                        # Return resized image with dimension note for coordinate mapping
                        return ToolResult(
                            content=[
                                note,
                                BinaryContent(data=data, media_type=mime, identifier=path),
                            ],
                            metadata={"preview": "Image read successfully", "truncated": False},
                        )
                return ToolResult(
                    content=[BinaryContent(data=data, media_type=mime, identifier=path)],
                    metadata={"preview": "Binary file read successfully", "truncated": False},
                )

            # Read content and probe for binary (git-style null byte detection)
            data = await self._get_fs(ctx)._cat_file(path)
            if is_binary_content(data):
                # Try converter first if available
                if self.converter is not None:
                    try:
                        content = await self.converter.convert_file(path)
                        await ctx.events.file_operation("read", path=path, success=True)
                    except Exception:  # noqa: BLE001
                        # Converter doesn't support this file type, fall back to binary
                        pass
                    else:
                        # Converter returned markdown
                        lines = content.splitlines()
                        preview = "\n".join(lines[:20])
                        return ToolResult(
                            content=content,
                            metadata={"preview": preview, "truncated": False},
                        )

                # Fall back to native binary handling
                await ctx.events.file_operation("read", path=path, success=True)
                mime = mime_type or "application/octet-stream"
                # Resize images if needed
                if self.max_image_size and mime.startswith("image/"):
                    from agentpool_toolsets.fsspec_toolset.image_utils import (
                        resize_image_if_needed,
                    )

                    data, mime, note = resize_image_if_needed(
                        data, mime, self.max_image_size, self.max_image_bytes
                    )
                    if note:
                        return ToolResult(
                            content=[
                                note,
                                BinaryContent(data=data, media_type=mime, identifier=path),
                            ],
                            metadata={"preview": "Image read successfully", "truncated": False},
                        )
                return ToolResult(
                    content=[BinaryContent(data=data, media_type=mime, identifier=path)],
                    metadata={"preview": "Binary file read successfully", "truncated": False},
                )

            content = data.decode(encoding)

            # Check if file is too large and no targeted read requested
            tokens_approx = len(content) // 4
            if line is None and limit is None and tokens_approx > self.large_file_tokens:
                # Try structure map for supported languages
                map_result = await self._get_file_map(path, ctx)
                if map_result:
                    await ctx.events.file_operation("read", path=path, success=True)
                    content = map_result
                else:
                    # Fallback: head + tail for unsupported languages
                    from agentpool.repomap import truncate_with_notice

                    content = truncate_with_notice(path, content)
                    await ctx.events.file_operation("read", path=path, success=True)
            else:
                # Normal read with optional offset/limit
                from agentpool_toolsets.fsspec_toolset.helpers import truncate_lines

                lines = content.splitlines()
                offset = (line - 1) if line else 0
                result_lines, was_truncated = truncate_lines(lines, offset, limit, max_file_size)
                content = "\n".join(result_lines)
                # Don't pass negative line numbers to events (ACP requires >= 0)
                display_line = line if (line and line > 0) else 0
                await ctx.events.file_operation("read", path=path, success=True, line=display_line)
                if was_truncated:
                    content += f"\n\n[Content truncated at {max_file_size} bytes]"

        except Exception as e:  # noqa: BLE001
            await ctx.events.file_operation("read", path=path, success=False, error=str(e))
            error_msg = f"error: Failed to read file {path}: {e}"
            return ToolResult(
                content=error_msg,
                metadata={"preview": "", "truncated": False},
            )
        else:
            # Emit file content for UI display (formatted at ACP layer)
            from agentpool.agents.events import FileContentItem

            # Use non-negative line for display (negative lines are internal Python convention)
            display_start_line = max(1, line) if line and line > 0 else None
            await ctx.events.tool_call_progress(
                title=f"Read: {path}",
                items=[FileContentItem(content=content, path=path, start_line=display_start_line)],
                replace_content=True,
            )

            # Prepare metadata for OpenCode UI
            lines = content.splitlines()
            preview = "\n".join(lines[:20])
            # Check if content was truncated
            was_truncated = "[Content truncated" in content

            # Return result with metadata
            return ToolResult(
                content=content,
                metadata={"preview": preview, "truncated": was_truncated},
            )
