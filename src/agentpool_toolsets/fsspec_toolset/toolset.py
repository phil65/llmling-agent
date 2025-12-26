"""FSSpec filesystem toolset implementation."""

from __future__ import annotations

import contextlib
from fnmatch import fnmatch
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from exxec.base import ExecutionEnvironment
from pydantic_ai import (
    BinaryContent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,  # noqa: TC002
    TextPart,
    TextPartDelta,
)
from upathtools import is_directory

from agentpool.agents.context import AgentContext  # noqa: TC001
from agentpool.log import get_logger
from agentpool.mime_utils import guess_type, is_binary_content, is_binary_mime
from agentpool.resource_providers import ResourceProvider
from agentpool_toolsets.builtin.file_edit import replace_content
from agentpool_toolsets.builtin.file_edit.fuzzy_matcher import StreamingFuzzyMatcher
from agentpool_toolsets.fsspec_toolset.diagnostics import DiagnosticsManager
from agentpool_toolsets.fsspec_toolset.grep import GrepBackend
from agentpool_toolsets.fsspec_toolset.helpers import (
    format_directory_listing,
    get_changed_line_numbers,
    truncate_lines,
)
from agentpool_toolsets.fsspec_toolset.streaming_diff_parser import (
    NewTextChunk,
    OldTextChunk,
    StreamingDiffParser,
)


if TYPE_CHECKING:
    import fsspec
    from fsspec.asyn import AsyncFileSystem
    from pydantic_ai.messages import ModelResponse

    from agentpool.agents.base_agent import BaseAgent
    from agentpool.common_types import ModelType
    from agentpool.messaging import MessageHistory
    from agentpool.prompts.conversion_manager import ConversionManager
    from agentpool.repomap import RepoMap
    from agentpool.tools.base import Tool


logger = get_logger(__name__)


class FSSpecTools(ResourceProvider):
    """Provider for fsspec filesystem tools.

    NOTE: The ACP execution environment used handles the Terminal events of the protocol,
    the toolset should deal with the ToolCall events for UI display purposes.
    """

    def __init__(
        self,
        source: fsspec.AbstractFileSystem | ExecutionEnvironment | None = None,
        name: str | None = None,
        cwd: str | None = None,
        edit_model: ModelType | None = None,
        converter: ConversionManager | None = None,
        max_file_size_kb: int = 64,
        max_grep_output_kb: int = 64,
        use_subprocess_grep: bool = True,
        enable_diagnostics: bool = False,
        large_file_tokens: int = 12_000,
        map_max_tokens: int = 2048,
    ) -> None:
        """Initialize with an fsspec filesystem or execution environment.

        Args:
            source: Filesystem or execution environment to operate on.
                    If None, falls back to agent.env at runtime.
            name: Name for this toolset provider
            cwd: Optional cwd to resolve relative paths against
            edit_model: Optional edit model for text editing
            converter: Optional conversion manager for markdown conversion
            max_file_size_kb: Maximum file size in KB for read/write operations (default: 64KB)
            max_grep_output_kb: Maximum grep output size in KB (default: 64KB)
            use_subprocess_grep: Use ripgrep/grep subprocess if available (default: True)
            enable_diagnostics: Run LSP CLI diagnostics after file writes (default: False)
            large_file_tokens: Token threshold for switching to structure map (default: 12000)
            map_max_tokens: Maximum tokens for structure map output (default: 2048)
        """
        from fsspec.asyn import AsyncFileSystem
        from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

        if source is None:
            self._fs: AsyncFileSystem | None = None
            self.execution_env: ExecutionEnvironment | None = None
        elif isinstance(source, ExecutionEnvironment):
            self.execution_env = source
            fs = source.get_fs()
            self._fs = fs if isinstance(fs, AsyncFileSystem) else AsyncFileSystemWrapper(fs)
        else:
            self.execution_env = None
            self._fs = (
                source if isinstance(source, AsyncFileSystem) else AsyncFileSystemWrapper(source)
            )
        super().__init__(name=name or f"file_access_{self._fs.protocol if self._fs else 'default'}")
        self.edit_model = edit_model
        self.cwd = cwd
        self.converter = converter
        self.max_file_size = max_file_size_kb * 1024  # Convert KB to bytes
        self.max_grep_output = max_grep_output_kb * 1024  # Convert KB to bytes
        self.use_subprocess_grep = use_subprocess_grep
        self._tools: list[Tool] | None = None
        self._grep_backend: GrepBackend | None = None
        self._enable_diagnostics = enable_diagnostics
        self._diagnostics: DiagnosticsManager | None = None
        self._large_file_tokens = large_file_tokens
        self._map_max_tokens = map_max_tokens
        self._repomap: RepoMap | None = None

    def get_fs(self, agent_ctx: AgentContext) -> AsyncFileSystem:
        """Get filesystem, falling back to agent's env if not set.

        Args:
            agent_ctx: Agent context to get fallback env from
        """
        from fsspec.asyn import AsyncFileSystem
        from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

        if self._fs is not None:
            return self._fs
        fs = agent_ctx.agent.env.get_fs()
        return fs if isinstance(fs, AsyncFileSystem) else AsyncFileSystemWrapper(fs)

    def _get_diagnostics_manager(self, agent_ctx: AgentContext) -> DiagnosticsManager:
        """Get or create the diagnostics manager."""
        if self._diagnostics is None:
            env = self.execution_env or agent_ctx.agent.env
            self._diagnostics = DiagnosticsManager(env if self._enable_diagnostics else None)
        return self._diagnostics

    async def _run_diagnostics(self, agent_ctx: AgentContext, path: str) -> str | None:
        """Run diagnostics on a file if enabled.

        Returns formatted diagnostics string if issues found, None otherwise.
        """
        if not self._enable_diagnostics:
            return None
        mgr = self._get_diagnostics_manager(agent_ctx)
        diagnostics = await mgr.run_for_file(path)
        if diagnostics:
            return mgr.format_diagnostics(diagnostics)
        return None

    async def _get_file_map(self, path: str, agent_ctx: AgentContext) -> str | None:
        """Get structure map for a large file if language is supported.

        Args:
            path: Absolute file path
            agent_ctx: Agent context for filesystem access

        Returns:
            Structure map string or None if language not supported
        """
        from agentpool.repomap import RepoMap, is_language_supported

        if not is_language_supported(path):
            return None

        # Lazy init repomap - use file's directory as root
        if self._repomap is None:
            root = str(Path(path).parent)
            fs = self.get_fs(agent_ctx)
            self._repomap = RepoMap(fs, root, max_tokens=self._map_max_tokens)

        return await self._repomap.get_file_map(path, max_tokens=self._map_max_tokens)

    def _resolve_path(self, path: str, agent_ctx: AgentContext) -> str:
        """Resolve a potentially relative path to an absolute path.

        Gets cwd from self.cwd, execution_env.cwd, or agent.env.cwd.
        If cwd is set and path is relative, resolves relative to cwd.
        Otherwise returns the path as-is.
        """
        # Get cwd: explicit toolset cwd > execution_env.cwd > agent.env.cwd
        cwd: str | None = None
        if self.cwd:
            cwd = self.cwd
        elif self.execution_env and self.execution_env.cwd:
            cwd = self.execution_env.cwd
        elif agent_ctx.agent.env and agent_ctx.agent.env.cwd:
            cwd = agent_ctx.agent.env.cwd

        if cwd and not (path.startswith("/") or (len(path) > 1 and path[1] == ":")):
            return str(Path(cwd) / path)
        return path

    async def get_tools(self) -> list[Tool]:
        """Get filesystem tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            self.create_tool(self.list_directory, category="read", read_only=True, idempotent=True),
            self.create_tool(self.read_file, category="read", read_only=True, idempotent=True),
            self.create_tool(self.grep, category="search", read_only=True, idempotent=True),
            self.create_tool(self.write_file, category="edit"),
            self.create_tool(self.delete_path, category="delete", destructive=True),
            self.create_tool(self.edit_file, category="edit"),
            self.create_tool(self.agentic_edit, category="edit"),
            self.create_tool(self.download_file, category="read", open_world=True),
        ]

        if self.converter:  # Only add read_as_markdown if converter is available
            self._tools.append(
                self.create_tool(
                    self.read_as_markdown, category="read", read_only=True, idempotent=True
                )
            )

        return self._tools

    async def list_directory(  # noqa: D417
        self,
        agent_ctx: AgentContext,
        path: str,
        *,
        pattern: str = "*",
        exclude: list[str] | None = None,
        max_depth: int = 1,
    ) -> str:
        """List files in a directory with filtering support.

        Args:
            path: Base directory to list
            pattern: Glob pattern to match files against. Use "*.py" to match Python
                files in current directory only, or "**/*.py" to match recursively.
                The max_depth parameter limits how deep "**" patterns search.
            exclude: List of patterns to exclude (uses fnmatch against relative paths)
            max_depth: Maximum directory depth to search (default: 1 = current dir only).
                Only affects recursive "**" patterns.

        Returns:
            Markdown-formatted directory listing
        """
        path = self._resolve_path(path, agent_ctx)
        msg = f"Listing directory: {path}"
        await agent_ctx.events.tool_call_start(title=msg, kind="read", locations=[path])

        try:
            fs = self.get_fs(agent_ctx)
            # Check if path exists
            if not await fs._exists(path):
                error_msg = f"Path does not exist: {path}"
                await agent_ctx.events.file_operation(
                    "list", path=path, success=False, error=error_msg
                )
                return f"Error: {error_msg}"

            # Build glob path
            glob_pattern = f"{path.rstrip('/')}/{pattern}"
            paths = await fs._glob(glob_pattern, maxdepth=max_depth, detail=True)

            files: list[dict[str, Any]] = []
            dirs: list[dict[str, Any]] = []

            # Safety check - prevent returning too many items
            total_found = len(paths)
            if total_found > 500:  # noqa: PLR2004
                suggestions = []
                if pattern == "*":
                    suggestions.append("Use a more specific pattern like '*.py', '*.txt', etc.")
                if max_depth > 1:
                    suggestions.append(f"Reduce max_depth from {max_depth} to 1 or 2.")
                if not exclude:
                    suggestions.append("Use exclude parameter to filter out unwanted directories.")

                suggestion_text = " ".join(suggestions) if suggestions else ""
                return f"Error: Too many items ({total_found:,}). {suggestion_text}"

            for file_path, file_info in paths.items():  # pyright: ignore[reportAttributeAccessIssue]
                rel_path = os.path.relpath(str(file_path), path)

                # Skip excluded patterns
                if exclude and any(fnmatch(rel_path, pat) for pat in exclude):
                    continue

                # Use type from glob detail info, falling back to isdir only if needed
                is_dir = await is_directory(fs, file_path, entry_type=file_info.get("type"))  # pyright: ignore[reportArgumentType]

                item_info = {
                    "name": Path(file_path).name,  # pyright: ignore[reportArgumentType]
                    "path": file_path,
                    "relative_path": rel_path,
                    "size": file_info.get("size", 0),
                    "type": "directory" if is_dir else "file",
                }
                if "mtime" in file_info:
                    item_info["modified"] = file_info["mtime"]

                if is_dir:
                    dirs.append(item_info)
                else:
                    files.append(item_info)

            await agent_ctx.events.file_operation("list", path=path, success=True)
            result = format_directory_listing(path, dirs, files, pattern)
            # Emit formatted content for UI display
            from agentpool.agents.events import TextContentItem

            await agent_ctx.events.tool_call_progress(
                title=f"Listed: {path}",
                items=[TextContentItem(text=result)],
                replace_content=True,
            )
        except (OSError, ValueError, FileNotFoundError) as e:
            await agent_ctx.events.file_operation("list", path=path, success=False, error=str(e))
            return f"Error: Could not list directory: {path}. Ensure path is absolute and exists."
        else:
            return result

    async def read_file(  # noqa: D417
        self,
        agent_ctx: AgentContext,
        path: str,
        encoding: str = "utf-8",
        line: int | None = None,
        limit: int | None = None,
    ) -> str | BinaryContent:
        """Read the context of a text file, or use vision capabilites to read images or documents.

        Args:
            path: File path to read
            encoding: Text encoding to use for text files (default: utf-8)
            line: Optional line number to start reading from (1-based, text files only)
            limit: Optional maximum number of lines to read (text files only)

        Returns:
            Text content for text files, BinaryContent for binary files, or dict with error
        """
        path = self._resolve_path(path, agent_ctx)
        msg = f"Reading file: {path}"
        from agentpool.agents.events import LocationContentItem

        await agent_ctx.events.tool_call_progress(
            title=msg,
            items=[LocationContentItem(path=path)],
        )
        try:
            mime_type = guess_type(path)
            # Fast path: known binary MIME types (images, audio, video, etc.)
            if is_binary_mime(mime_type):
                data = await self.get_fs(agent_ctx)._cat_file(path)
                await agent_ctx.events.file_operation("read", path=path, success=True)
                mime = mime_type or "application/octet-stream"
                return BinaryContent(data=data, media_type=mime, identifier=path)
            # Read content and probe for binary (git-style null byte detection)
            data = await self.get_fs(agent_ctx)._cat_file(path)
            if is_binary_content(data):
                # Binary file - return as BinaryContent for native model handling
                await agent_ctx.events.file_operation("read", path=path, success=True)
                mime = mime_type or "application/octet-stream"
                return BinaryContent(data=data, media_type=mime, identifier=path)
            content = data.decode(encoding)

            # Check if file is too large and no targeted read requested
            tokens_approx = len(content) // 4
            if line is None and limit is None and tokens_approx > self._large_file_tokens:
                # Try structure map for supported languages
                map_result = await self._get_file_map(path, agent_ctx)
                if map_result:
                    await agent_ctx.events.file_operation("read", path=path, success=True)
                    content = map_result
                else:
                    # Fallback: head + tail for unsupported languages
                    from agentpool.repomap import truncate_with_notice

                    content = truncate_with_notice(path, content)
                    await agent_ctx.events.file_operation("read", path=path, success=True)
            else:
                # Normal read with optional offset/limit
                lines = content.splitlines()
                offset = (line - 1) if line else 0
                result_lines, was_truncated = truncate_lines(
                    lines, offset, limit, self.max_file_size
                )
                content = "\n".join(result_lines)
                await agent_ctx.events.file_operation("read", path=path, success=True)
                if was_truncated:
                    content += f"\n\n[Content truncated at {self.max_file_size} bytes]"

        except Exception as e:  # noqa: BLE001
            await agent_ctx.events.file_operation("read", path=path, success=False, error=str(e))
            return f"error: Failed to read file {path}: {e}"
        else:
            # Emit file content for UI display (formatted at ACP layer)
            from agentpool.agents.events import FileContentItem

            await agent_ctx.events.tool_call_progress(
                title=f"Read: {path}",
                items=[FileContentItem(content=content, path=path)],
                replace_content=True,
            )
            # Return raw content for agent
            return content

    async def read_as_markdown(self, agent_ctx: AgentContext, path: str) -> str | dict[str, Any]:  # noqa: D417
        """Read file and convert to markdown text representation.

        Args:
            path: Path to read

        Returns:
            File content converted to markdown
        """
        assert self.converter is not None, "Converter required for read_as_markdown"

        path = self._resolve_path(path, agent_ctx)
        msg = f"Reading file as markdown: {path}"
        await agent_ctx.events.tool_call_start(title=msg, kind="read", locations=[path])
        try:
            content = await self.converter.convert_file(path)
            await agent_ctx.events.file_operation("read", path=path, success=True)
            # Emit formatted content for UI display
            from agentpool.agents.events import TextContentItem

            await agent_ctx.events.tool_call_progress(
                title=f"Read as markdown: {path}",
                items=[TextContentItem(text=content)],
                replace_content=True,
            )
        except Exception as e:  # noqa: BLE001
            await agent_ctx.events.file_operation("read", path=path, success=False, error=str(e))
            return f"Error: Failed to convert file {path}: {e}"
        else:
            return content

    async def write_file(  # noqa: D417
        self,
        agent_ctx: AgentContext,
        path: str,
        content: str,
        mode: str = "w",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Write content to a file.

        Args:
            path: File path to write
            content: Content to write
            mode: Write mode ('w' for overwrite, 'a' for append)
            overwrite: Must be True to overwrite existing files (safety check)

        Returns:
            Dictionary with success info or error details
        """
        path = self._resolve_path(path, agent_ctx)
        msg = f"Writing file: {path}"
        await agent_ctx.events.tool_call_start(title=msg, kind="edit", locations=[path])

        content_bytes = len(content.encode("utf-8"))

        try:
            if mode not in ("w", "a"):
                msg = f"Invalid mode '{mode}'. Use 'w' (write) or 'a' (append)"
                await agent_ctx.events.file_operation("write", path=path, success=False, error=msg)
                return {"error": msg}

            # Check size limit
            if content_bytes > self.max_file_size:
                msg = (
                    f"Content size ({content_bytes} bytes) exceeds maximum "
                    f"({self.max_file_size} bytes)"
                )
                await agent_ctx.events.file_operation("write", path=path, success=False, error=msg)
                return {"error": msg}

            # Check if file exists and overwrite protection
            fs = self.get_fs(agent_ctx)
            file_exists = await fs._exists(path)

            if file_exists and mode == "w" and not overwrite:
                msg = (
                    f"File '{path}' already exists. To overwrite it, you must set overwrite=True. "
                    f"This is a safety measure to prevent accidental data loss."
                )
                await agent_ctx.events.file_operation("write", path=path, success=False, error=msg)
                return {"error": msg}

            # Handle append mode: read existing content and prepend it
            if mode == "a" and file_exists:
                try:
                    existing_content = await self._read(agent_ctx, path)
                    if isinstance(existing_content, bytes):
                        existing_content = existing_content.decode("utf-8")
                    content = existing_content + content
                except Exception:  # noqa: BLE001
                    pass  # If we can't read, just write new content

            await self._write(agent_ctx, path, content)

            try:
                info = await fs._info(path)
                size = info.get("size", content_bytes)
            except (OSError, KeyError):
                size = content_bytes

            result: dict[str, Any] = {
                "path": path,
                "size": size,
                "mode": mode,
                "file_existed": file_exists,
                "bytes_written": content_bytes,
            }
            await agent_ctx.events.file_operation("write", path=path, success=True)

            # Run diagnostics if enabled
            if diagnostics_output := await self._run_diagnostics(agent_ctx, path):
                result["diagnostics"] = diagnostics_output
        except Exception as e:  # noqa: BLE001
            await agent_ctx.events.file_operation("write", path=path, success=False, error=str(e))
            return {"error": f"Failed to write file {path}: {e}"}
        else:
            return result

    async def delete_path(  # noqa: D417
        self, agent_ctx: AgentContext, path: str, recursive: bool = False
    ) -> dict[str, Any]:
        """Delete a file or directory.

        Args:
            path: Path to delete
            recursive: Whether to delete directories recursively

        Returns:
            Dictionary with operation result
        """
        path = self._resolve_path(path, agent_ctx)
        msg = f"Deleting path: {path}"
        await agent_ctx.events.tool_call_start(title=msg, kind="delete", locations=[path])
        try:
            # Check if path exists and get its type
            fs = self.get_fs(agent_ctx)
            try:
                info = await fs._info(path)
                path_type = info.get("type", "unknown")
            except FileNotFoundError:
                msg = f"Path does not exist: {path}"
                await agent_ctx.events.file_operation("delete", path=path, success=False, error=msg)
                return {"error": msg}
            except (OSError, ValueError) as e:
                msg = f"Could not check path {path}: {e}"
                await agent_ctx.events.file_operation("delete", path=path, success=False, error=msg)
                return {"error": msg}

            if path_type == "directory":
                if not recursive:
                    try:
                        contents = await fs._ls(path)
                        if contents:  # Check if directory is empty
                            error_msg = (
                                f"Directory {path} is not empty. "
                                f"Use recursive=True to delete non-empty directories"
                            )

                            # Emit failure event
                            await agent_ctx.events.file_operation(
                                "delete", path=path, success=False, error=error_msg
                            )

                            return {"error": error_msg}
                    except (OSError, ValueError):
                        pass  # Continue with deletion attempt

                await fs._rm(path, recursive=recursive)
            else:  # It's a file
                await fs._rm(path)  # or _rm_file?

        except Exception as e:  # noqa: BLE001
            await agent_ctx.events.file_operation("delete", path=path, success=False, error=str(e))
            return {"error": f"Failed to delete {path}: {e}"}
        else:
            result = {"path": path, "deleted": True, "type": path_type, "recursive": recursive}
            await agent_ctx.events.file_operation("delete", path=path, success=True)
            return result

    async def edit_file(  # noqa: D417
        self,
        agent_ctx: AgentContext,
        path: str,
        old_string: str,
        new_string: str,
        description: str,
        replace_all: bool = False,
    ) -> str:
        r"""Edit a file by replacing specific content with smart matching.

        Uses sophisticated matching strategies to handle whitespace, indentation,
        and other variations. Shows the changes as a diff in the UI.

        Args:
            path: File path (absolute or relative to session cwd)
            old_string: Text content to find and replace
            new_string: Text content to replace it with
            description: Human-readable description of what the edit accomplishes
            replace_all: Whether to replace all occurrences (default: False)

        Returns:
            Success message with edit summary
        """
        path = self._resolve_path(path, agent_ctx)
        msg = f"Editing file: {path}"
        await agent_ctx.events.tool_call_start(title=msg, kind="edit", locations=[path])
        if old_string == new_string:
            return "Error: old_string and new_string must be different"

        # Send initial pending notification
        await agent_ctx.events.file_operation("edit", path=path, success=True)

        try:  # Read current file content
            original_content = await self._read(agent_ctx, path)
            if isinstance(original_content, bytes):
                original_content = original_content.decode("utf-8")

            try:  # Apply smart content replacement
                new_content = replace_content(original_content, old_string, new_string, replace_all)
            except ValueError as e:
                error_msg = f"Edit failed: {e}"
                await agent_ctx.events.file_operation(
                    "edit", path=path, success=False, error=error_msg
                )
                return error_msg

            await self._write(agent_ctx, path, new_content)
            success_msg = f"Successfully edited {Path(path).name}: {description}"
            changed_line_numbers = get_changed_line_numbers(original_content, new_content)
            if lines_changed := len(changed_line_numbers):
                success_msg += f" ({lines_changed} lines changed)"

            await agent_ctx.events.file_edit_progress(
                path=path,
                old_text=original_content,
                new_text=new_content,
                status="completed",
            )

            # Run diagnostics if enabled
            if diagnostics_output := await self._run_diagnostics(agent_ctx, path):
                success_msg += f"\n\nDiagnostics:\n{diagnostics_output}"
        except Exception as e:  # noqa: BLE001
            error_msg = f"Error editing file: {e}"
            await agent_ctx.events.file_operation("edit", path=path, success=False, error=error_msg)
            return error_msg
        else:
            return success_msg

    async def grep(  # noqa: D417
        self,
        agent_ctx: AgentContext,
        pattern: str,
        path: str,
        *,
        file_pattern: str = "**/*",
        case_sensitive: bool = False,
        max_matches: int = 100,
        context_lines: int = 0,
    ) -> str:
        """Search file contents for a pattern.

        Args:
            pattern: Regex pattern to search for
            path: Base directory to search in
            file_pattern: Glob pattern to filter files (e.g. "**/*.py")
            case_sensitive: Whether search is case-sensitive
            max_matches: Maximum number of matches to return
            context_lines: Number of context lines before/after match

        Returns:
            Grep results as formatted text
        """
        from agentpool_toolsets.fsspec_toolset.grep import (
            DEFAULT_EXCLUDE_PATTERNS,
            detect_grep_backend,
            grep_with_fsspec,
            grep_with_subprocess,
        )

        resolved_path = self._resolve_path(path, agent_ctx)
        msg = f"Searching for {pattern!r} in {resolved_path}"
        await agent_ctx.events.tool_call_start(title=msg, kind="search", locations=[resolved_path])

        result: dict[str, Any] | None = None
        try:
            # Try subprocess grep if configured and available
            if self.use_subprocess_grep:
                # Get execution environment for running grep command
                env = self.execution_env or agent_ctx.agent.env
                if env is not None:
                    # Detect and cache grep backend
                    if self._grep_backend is None:
                        self._grep_backend = await detect_grep_backend(env)
                    # Only use subprocess if we have a real grep backend
                    if self._grep_backend != GrepBackend.PYTHON:
                        result = await grep_with_subprocess(
                            env=env,
                            pattern=pattern,
                            path=resolved_path,
                            backend=self._grep_backend,
                            case_sensitive=case_sensitive,
                            max_matches=max_matches,
                            max_output_bytes=self.max_grep_output,
                            exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
                            use_gitignore=True,
                            context_lines=context_lines,
                        )

            # Fallback to fsspec grep if subprocess didn't work
            if result is None or "error" in result:
                fs = self.get_fs(agent_ctx)
                result = await grep_with_fsspec(
                    fs=fs,
                    pattern=pattern,
                    path=resolved_path,
                    file_pattern=file_pattern,
                    case_sensitive=case_sensitive,
                    max_matches=max_matches,
                    max_output_bytes=self.max_grep_output,
                    context_lines=context_lines,
                )

            if "error" in result:
                return f"Error: {result['error']}"

            # Format output
            matches = result.get("matches", "")
            match_count = result.get("match_count", 0)
            was_truncated = result.get("was_truncated", False)

            if not matches:
                output = f"No matches found for pattern '{pattern}'"
            else:
                # Format matches based on type (string from subprocess, dict from fsspec)
                if isinstance(matches, dict):
                    # Format fsspec dict results with context support
                    formatted_lines = []
                    for file_path, file_matches in matches.items():
                        for match in file_matches:
                            line_num = match.get("line_number", 0)
                            content = match.get("content", "")
                            # Add context before if present
                            if context_before := match.get("context_before"):
                                for i, ctx_line in enumerate(context_before):
                                    ctx_line_num = line_num - len(context_before) + i
                                    formatted_lines.append(f"{file_path}:{ctx_line_num}-{ctx_line}")
                            # Add the matching line
                            formatted_lines.append(f"{file_path}:{line_num}:{content}")
                            # Add context after if present
                            if context_after := match.get("context_after"):
                                for i, ctx_line in enumerate(context_after):
                                    ctx_line_num = line_num + 1 + i
                                    formatted_lines.append(f"{file_path}:{ctx_line_num}-{ctx_line}")
                            # Add separator between matches with context
                            if context_before or context_after:
                                formatted_lines.append("--")
                    # Remove trailing separator
                    if formatted_lines and formatted_lines[-1] == "--":
                        formatted_lines.pop()
                    matches = "\n".join(formatted_lines)
                output = f"Found {match_count} matches:\n\n{matches}"
                if was_truncated:
                    output += "\n\n[Results truncated]"

            # Emit formatted content for UI display
            from agentpool.agents.events import TextContentItem

            await agent_ctx.events.tool_call_progress(
                title=f"Found {match_count} matches",
                items=[TextContentItem(text=output)],
                replace_content=True,
            )
        except Exception as e:  # noqa: BLE001
            return f"Error: Grep failed: {e}"
        else:
            return output

    async def _read(self, agent_ctx: AgentContext, path: str, encoding: str = "utf-8") -> str:
        # with self.fs.open(path, "r", encoding="utf-8") as f:
        #     return f.read()
        return await self.get_fs(agent_ctx)._cat(path)  # type: ignore[no-any-return]

    async def _write(self, agent_ctx: AgentContext, path: str, content: str | bytes) -> None:
        if isinstance(content, str):
            content = content.encode()
        await self.get_fs(agent_ctx)._pipe_file(path, content)

    async def download_file(  # noqa: D417
        self,
        agent_ctx: AgentContext,
        url: str,
        target_dir: str = "downloads",
        chunk_size: int = 8192,
    ) -> dict[str, Any]:
        """Download a file from URL to the toolset's filesystem.

        Args:
            url: URL to download from
            target_dir: Directory to save the file (relative to cwd if set)
            chunk_size: Size of chunks to download

        Returns:
            Status information about the download
        """
        import asyncio

        import httpx

        start_time = time.time()

        # Resolve target directory
        target_dir = self._resolve_path(target_dir, agent_ctx)

        msg = f"Downloading: {url}"
        await agent_ctx.events.tool_call_start(title=msg, kind="read", locations=[url])

        # Extract filename from URL
        filename = Path(urlparse(url).path).name or "downloaded_file"
        full_path = f"{target_dir.rstrip('/')}/{filename}"

        try:
            fs = self.get_fs(agent_ctx)
            # Ensure target directory exists
            await fs._makedirs(target_dir, exist_ok=True)

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

                # Collect all data
                data = bytearray()
                async for chunk in response.aiter_bytes(chunk_size):
                    data.extend(chunk)
                    size = len(data)

                    if total and (size % (chunk_size * 100) == 0 or size == total):
                        progress = size / total * 100
                        speed_mbps = (size / 1_048_576) / (time.time() - start_time)
                        progress_msg = f"\r{filename}: {progress:.1f}% ({speed_mbps:.1f} MB/s)"
                        await agent_ctx.events.progress(progress, 100, progress_msg)
                        await asyncio.sleep(0)

                # Write to filesystem
                await self._write(agent_ctx, full_path, bytes(data))

            duration = time.time() - start_time
            size_mb = len(data) / 1_048_576

            await agent_ctx.events.file_operation("read", path=full_path, success=True)

            return {
                "path": full_path,
                "filename": filename,
                "size_bytes": len(data),
                "size_mb": round(size_mb, 2),
                "duration_seconds": round(duration, 2),
                "speed_mbps": round(size_mb / duration, 2) if duration > 0 else 0,
            }

        except httpx.ConnectError as e:
            error_msg = f"Connection error downloading {url}: {e}"
            await agent_ctx.events.file_operation("read", path=url, success=False, error=error_msg)
            return {"error": error_msg}
        except httpx.TimeoutException:
            error_msg = f"Timeout downloading {url}"
            await agent_ctx.events.file_operation("read", path=url, success=False, error=error_msg)
            return {"error": error_msg}
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} downloading {url}"
            await agent_ctx.events.file_operation("read", path=url, success=False, error=error_msg)
            return {"error": error_msg}
        except Exception as e:  # noqa: BLE001
            error_msg = f"Error downloading {url}: {e!s}"
            await agent_ctx.events.file_operation("read", path=url, success=False, error=error_msg)
            return {"error": error_msg}

    async def agentic_edit(  # noqa: D417, PLR0915
        self,
        run_ctx: RunContext,
        agent_ctx: AgentContext,
        path: str,
        display_description: str,
        mode: str = "edit",
        matcher: str = "default",
    ) -> str:
        r"""Edit or create a file with streaming support.

        Use this tool for file modifications. Describe what changes you want
        and the tool will apply them progressively as they're generated.

        Args:
            path: File path (absolute or relative to session cwd)
            display_description: What edits to make - be specific about the changes.
                Examples: "Add error handling to the parse function",
                "Rename the 'foo' variable to 'bar' throughout",
                "Add a docstring to the MyClass class"
            mode: How to modify the file:
                - 'edit': Modify specific parts of existing file (default)
                - 'create': Create a new file (fails if file exists)
                - 'overwrite': Replace entire file content
            matcher: Internal matching algorithm - leave as default unless
                you have a reason to change it.

        Returns:
            Success message with edit summary
        """
        from pydantic_ai.messages import CachePoint, ModelRequest

        from agentpool.messaging import ChatMessage, MessageHistory

        path = self._resolve_path(path, agent_ctx)
        title = f"AI editing file: {path}"
        await agent_ctx.events.tool_call_start(title=title, kind="edit", locations=[path])
        await agent_ctx.events.file_operation("edit", path=path, success=True)

        try:
            # Read original content for diff purposes
            if mode == "create":
                original_content = ""
            else:
                original_content = await self._read(agent_ctx, path)
                if isinstance(original_content, bytes):
                    original_content = original_content.decode()

            # Build the edit prompt based on mode
            if mode == "create":
                prompt = _build_create_prompt(path, display_description)
            elif mode == "overwrite":
                prompt = _build_overwrite_prompt(path, display_description, original_content)
            else:
                prompt = _build_edit_prompt(path, display_description, original_content)

            # Get the current agent and its conversation history
            agent = agent_ctx.native_agent

            # Create forked message history from current conversation
            # This preserves full context while isolating the edit's messages
            # We need BOTH:
            # 1. Stored history (previous runs) from agent.conversation
            # 2. Current run messages from run_ctx.messages (not yet stored)
            stored_history = agent.conversation.get_history()

            # Build complete message list
            all_messages: list[ModelRequest | ModelResponse] = []

            # Add stored history from previous runs
            for chat_msg in stored_history:
                all_messages.extend(chat_msg.to_pydantic_ai())

            # Add current run's messages (not yet in stored history)
            # But exclude the last message if it contains the current agentic_edit tool call
            # to avoid the sub-agent seeing "I'm calling agentic_edit" in its context
            from pydantic_ai.messages import ModelResponse, ToolCallPart

            for msg in run_ctx.messages:
                if isinstance(msg, ModelResponse):
                    # Filter out the agentic_edit tool call from the last response
                    filtered_parts = [
                        p
                        for p in msg.parts
                        if not (isinstance(p, ToolCallPart) and p.tool_name == "agentic_edit")
                    ]
                    if filtered_parts:
                        all_messages.append(ModelResponse(parts=filtered_parts))
                else:
                    all_messages.append(msg)

            # Inject CachePoint to cache everything up to this point
            if all_messages:
                cache_request: ModelRequest = ModelRequest(parts=[CachePoint()])  # type: ignore[list-item]
                all_messages.append(cache_request)

                # Wrap in a single ChatMessage for the forked history
                fork_history = MessageHistory(
                    messages=[ChatMessage(messages=all_messages, role="user", content="")]
                )
            else:
                fork_history = MessageHistory()

            # Stream the edit using the same agent but with forked history
            if mode == "edit" and matcher == "zed":
                # TRUE STREAMING with Zed-style DP fuzzy matcher
                new_content = await self._stream_edit_with_matcher(
                    agent, prompt, fork_history, original_content, path, agent_ctx
                )
            elif mode == "edit":
                # TRUE STREAMING with our 9-strategy replace_content (default)
                new_content = await self._stream_edit_with_replace(
                    agent, prompt, fork_history, original_content, path, agent_ctx
                )
            else:
                # CREATE/OVERWRITE: Stream raw content directly
                new_content = await self._stream_raw_content(
                    agent, prompt, fork_history, original_content, path, agent_ctx
                )

            # Write the new content to file
            await self._write(agent_ctx, path, new_content)

            # Build success message
            original_lines = len(original_content.splitlines()) if original_content else 0
            new_lines = len(new_content.splitlines())

            if mode == "create":
                success_msg = f"Successfully created {Path(path).name} ({new_lines} lines)"
            else:
                success_msg = f"Successfully edited {Path(path).name} using AI agent"
                success_msg += f" ({original_lines} → {new_lines} lines)"

            # Send final completion update
            await agent_ctx.events.file_edit_progress(
                path=path,
                old_text=original_content,
                new_text=new_content,
                status="completed",
            )

        except Exception as e:  # noqa: BLE001
            error_msg = f"Error during agentic edit: {e}"
            await agent_ctx.events.file_operation("edit", path=path, success=False, error=error_msg)
            return error_msg
        else:
            return success_msg

    async def _stream_edit_with_matcher(  # noqa: PLR0915
        self,
        agent: BaseAgent,
        prompt: str,
        fork_history: MessageHistory,
        original_content: str,
        path: str,
        agent_ctx: AgentContext,
    ) -> str:
        """TRUE streaming edit using StreamingDiffParser + StreamingFuzzyMatcher.

        Parses diff incrementally, uses DP matcher to find locations as old_text
        streams, and applies new_text edits as they arrive.
        """
        parser = StreamingDiffParser()
        matcher = StreamingFuzzyMatcher(original_content)

        # Track current state
        edited_content = original_content
        pending_old_text: list[str] = []  # Track old text for prefix/suffix calculation
        pending_new_text: list[str] = []
        current_match_range = None

        async for node in agent.run_stream(
            prompt,
            message_history=fork_history,
            store_history=False,
        ):
            match node:
                case (
                    PartStartEvent(part=TextPart(content=chunk))
                    | PartDeltaEvent(delta=TextPartDelta(content_delta=chunk))
                ):
                    # Parse diff chunk and process events
                    for event in parser.push(chunk):
                        if isinstance(event, OldTextChunk):
                            if not event.done:
                                # Track old text for later prefix/suffix calculation
                                pending_old_text.append(event.chunk)
                                # Push to matcher for location resolution
                                match_result = matcher.push(event.chunk, line_hint=event.line_hint)
                                if match_result:
                                    current_match_range = match_result
                            else:
                                # Old text done - finalize location
                                matches = matcher.finish()
                                if matches:
                                    current_match_range = matches[0]
                                # Reset matcher for next hunk
                                matcher = StreamingFuzzyMatcher(edited_content)

                        elif isinstance(event, NewTextChunk):
                            if not event.done:
                                pending_new_text.append(event.chunk)
                            else:
                                # New text done - apply the edit if we have a location
                                if current_match_range and pending_new_text:
                                    new_text = "".join(pending_new_text)
                                    old_text = "".join(pending_old_text)

                                    # The matcher may find a larger range than our old_text
                                    # We need to preserve any prefix/suffix not in old_text
                                    matched_text = edited_content[
                                        current_match_range.start : current_match_range.end
                                    ]

                                    # Find where old_text appears in matched_text
                                    old_text_stripped = old_text.strip("\n")
                                    match_idx = matched_text.find(old_text_stripped)

                                    if match_idx >= 0:
                                        # Preserve prefix (e.g., blank lines before)
                                        prefix = matched_text[:match_idx]
                                        # Preserve suffix after old_text
                                        suffix_start = match_idx + len(old_text_stripped)
                                        suffix = matched_text[suffix_start:]
                                        # Build replacement with preserved prefix/suffix
                                        replacement = prefix + new_text.strip("\n") + suffix
                                    else:
                                        # Fallback: direct replacement
                                        replacement = new_text

                                    # Apply edit to content
                                    edited_content = (
                                        edited_content[: current_match_range.start]
                                        + replacement
                                        + edited_content[current_match_range.end :]
                                    )
                                    # Emit progress update
                                    await agent_ctx.events.file_edit_progress(
                                        path=path,
                                        old_text=original_content,
                                        new_text=edited_content,
                                        status="in_progress",
                                    )

                                # Reset for next hunk
                                pending_old_text = []
                                pending_new_text = []
                                current_match_range = None

        # Process any remaining content
        for event in parser.finish():
            if isinstance(event, OldTextChunk):
                if not event.done:
                    pending_old_text.append(event.chunk)
                    matcher.push(event.chunk, line_hint=event.line_hint)
                else:
                    matches = matcher.finish()
                    if matches:
                        current_match_range = matches[0]
            elif isinstance(event, NewTextChunk):
                if not event.done:
                    pending_new_text.append(event.chunk)
                elif current_match_range and pending_new_text:
                    new_text = "".join(pending_new_text)
                    old_text = "".join(pending_old_text)
                    matched_text = edited_content[
                        current_match_range.start : current_match_range.end
                    ]
                    old_text_stripped = old_text.strip("\n")
                    match_idx = matched_text.find(old_text_stripped)
                    if match_idx >= 0:
                        prefix = matched_text[:match_idx]
                        suffix_start = match_idx + len(old_text_stripped)
                        suffix = matched_text[suffix_start:]
                        replacement = prefix + new_text.strip("\n") + suffix
                    else:
                        replacement = new_text
                    edited_content = (
                        edited_content[: current_match_range.start]
                        + replacement
                        + edited_content[current_match_range.end :]
                    )

        return edited_content

    async def _stream_edit_with_replace(
        self,
        agent: BaseAgent,
        prompt: str,
        fork_history: MessageHistory,
        original_content: str,
        path: str,
        agent_ctx: AgentContext,
    ) -> str:
        """TRUE streaming edit using StreamingDiffParser + replace_content().

        Parses diff incrementally, uses our 9-strategy replace_content() to
        find locations and apply edits as each hunk completes.
        """
        parser = StreamingDiffParser()

        # Track current state
        edited_content = original_content
        pending_old_text: list[str] = []
        pending_new_text: list[str] = []

        async for node in agent.run_stream(
            prompt,
            message_history=fork_history,
            store_history=False,
        ):
            match node:
                case (
                    PartStartEvent(part=TextPart(content=chunk))
                    | PartDeltaEvent(delta=TextPartDelta(content_delta=chunk))
                ):
                    # Parse diff chunk and process events
                    for event in parser.push(chunk):
                        if isinstance(event, OldTextChunk):
                            if not event.done:
                                pending_old_text.append(event.chunk)
                            # When old_text is done, we just wait for new_text

                        elif isinstance(event, NewTextChunk):
                            if not event.done:
                                pending_new_text.append(event.chunk)
                            else:
                                # Hunk complete - apply using replace_content
                                if pending_old_text and pending_new_text:
                                    old_text = "".join(pending_old_text)
                                    new_text = "".join(pending_new_text)

                                    try:
                                        edited_content = replace_content(
                                            edited_content,
                                            old_text,
                                            new_text,
                                            replace_all=False,
                                        )
                                        # Emit progress update
                                        await agent_ctx.events.file_edit_progress(
                                            path=path,
                                            old_text=original_content,
                                            new_text=edited_content,
                                            status="in_progress",
                                        )
                                    except ValueError as e:
                                        # Log but continue - some hunks may fail
                                        logger.warning(
                                            "Streaming hunk failed",
                                            error=str(e),
                                            old_text=old_text[:50],
                                        )

                                # Reset for next hunk
                                pending_old_text = []
                                pending_new_text = []

        # Process any remaining content
        for event in parser.finish():
            if isinstance(event, OldTextChunk) and not event.done:
                pending_old_text.append(event.chunk)
            elif isinstance(event, NewTextChunk):
                if not event.done:
                    pending_new_text.append(event.chunk)
                elif pending_old_text and pending_new_text:
                    old_text = "".join(pending_old_text)
                    new_text = "".join(pending_new_text)
                    with contextlib.suppress(ValueError):
                        edited_content = replace_content(
                            edited_content,
                            old_text,
                            new_text,
                            replace_all=False,
                        )

        return edited_content

    async def _stream_raw_content(
        self,
        agent: BaseAgent,
        prompt: str,
        fork_history: MessageHistory,
        original_content: str,
        path: str,
        agent_ctx: AgentContext,
    ) -> str:
        """Stream raw content for create/overwrite modes.

        Emits progress updates as content streams in.
        """
        streamed_content = ""

        async for node in agent.run_stream(
            prompt,
            message_history=fork_history,
            store_history=False,
        ):
            match node:
                case (
                    PartStartEvent(part=TextPart(content=chunk))
                    | PartDeltaEvent(delta=TextPartDelta(content_delta=chunk))
                ):
                    streamed_content += chunk

                    # Emit periodic progress updates
                    await agent_ctx.events.file_edit_progress(
                        path=path,
                        old_text=original_content,
                        new_text=streamed_content,
                        status="in_progress",
                    )

        return streamed_content


def _build_create_prompt(path: str, description: str) -> str:
    """Build prompt for create mode."""
    return f"""Create a new file at {path} according to this description:

{description}

Output ONLY the complete file content. No explanations, no markdown code blocks, no formatting.
DO NOT use any tools. Just output the file content directly."""


def _build_overwrite_prompt(path: str, description: str, current_content: str) -> str:
    """Build prompt for overwrite mode."""
    return f"""Rewrite the file {path} according to this description:

{description}

<current_file_content>
{current_content}
</current_file_content>

Output ONLY the complete new file content. No explanations, no code blocks, no formatting.
DO NOT use any tools. Just output the file content directly."""


def _build_edit_prompt(path: str, description: str, current_content: str) -> str:
    """Build prompt for diff-based edit mode."""
    return f"""\
You MUST respond with edits in unified diff format. Output your changes inside <diff> tags.

# Diff Format Instructions

Use standard unified diff format with context lines:
- Lines starting with a space are context (unchanged)
- Lines starting with `-` are removed
- Lines starting with `+` are added
- Include 2-3 lines of context before and after each change
- Do NOT include line numbers or @@ headers - locations are matched by context

Example:
<diff>
 def existing_function():
-    old_implementation()
+    new_implementation()
     return result
</diff>

# Rules

- Context lines MUST exactly match the file content (including indentation)
- Include enough context to uniquely identify the location
- For multiple separate changes, include all hunks within a single <diff> block
- Separate hunks with a blank line
- Do not escape quotes or special characters
- Preserve exact indentation from the original file

<file_to_edit path="{path}">
{current_content}
</file_to_edit>

<edit_description>
{description}
</edit_description>

You MUST wrap your response in <diff>...</diff> tags.
DO NOT use any tools. Just output the diff directly."""


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        import fsspec
        from pydantic_ai import RunContext as PyAiContext, RunUsage
        from pydantic_ai.models.test import TestModel

        from agentpool import AgentPool

        fs = fsspec.filesystem("file")
        tools = FSSpecTools(fs, name="local_fs")
        async with AgentPool() as pool:
            agent = await pool.add_agent("test", model="anthropic-max:claude-haiku-4-5")
            agent_ctx = agent.get_context()
            result = await tools.agentic_edit(
                PyAiContext(deps=None, model=TestModel(), usage=RunUsage()),
                agent_ctx,
                path="/home/phil65/dev/oss/agentpool/src/agentpool_toolsets/fsspec_toolset/toolset.py",
                display_description="Append a poem",
            )
            print(result)

    asyncio.run(main())
