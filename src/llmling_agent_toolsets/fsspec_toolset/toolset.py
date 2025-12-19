"""FSSpec filesystem toolset implementation."""

from __future__ import annotations

from fnmatch import fnmatch
import mimetypes
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from exxec.base import ExecutionEnvironment
from pydantic_ai import Agent as PydanticAgent, BinaryContent
from upathtools import is_directory

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent_toolsets.builtin.file_edit import replace_content
from llmling_agent_toolsets.fsspec_toolset.diagnostics import DiagnosticsManager
from llmling_agent_toolsets.fsspec_toolset.grep import GrepBackend
from llmling_agent_toolsets.fsspec_toolset.helpers import (
    apply_structured_edits,
    format_directory_listing,
    get_changed_line_numbers,
    is_binary_content,
    is_definitely_binary_mime,
    truncate_lines,
)


if TYPE_CHECKING:
    import fsspec
    from fsspec.asyn import AsyncFileSystem

    from llmling_agent.common_types import ModelType
    from llmling_agent.prompts.conversion_manager import ConversionManager
    from llmling_agent.repomap import RepoMap
    from llmling_agent.tools.base import Tool


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
        from llmling_agent.repomap import RepoMap, is_language_supported

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

    async def list_directory(
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
            agent_ctx: Agent execution context
            path: Base directory to list
            pattern: Glob pattern to match files against (e.g. "*.py" for Python files)
            exclude: List of patterns to exclude (uses fnmatch against relative paths)
            max_depth: Maximum directory depth to search (default: 1 = current dir only)

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
            from llmling_agent.agents.events import TextContentItem

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

    async def read_file(
        self,
        agent_ctx: AgentContext,
        path: str,
        encoding: str = "utf-8",
        line: int | None = None,
        limit: int | None = None,
    ) -> str | BinaryContent:
        """Read the context of a text file, or use vision capabilites to read images or documents.

        Args:
            agent_ctx: Agent execution context
            path: File path to read
            encoding: Text encoding to use for text files (default: utf-8)
            line: Optional line number to start reading from (1-based, text files only)
            limit: Optional maximum number of lines to read (text files only)

        Returns:
            Text content for text files, BinaryContent for binary files, or dict with error
        """
        path = self._resolve_path(path, agent_ctx)
        msg = f"Reading file: {path}"
        from llmling_agent.agents.events import LocationContentItem

        await agent_ctx.events.tool_call_progress(
            title=msg,
            items=[LocationContentItem(path=path)],
        )
        try:
            mime_type = mimetypes.guess_type(path)[0]
            # Fast path: known binary MIME types (images, audio, video, etc.)
            if is_definitely_binary_mime(mime_type):
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
                    from llmling_agent.repomap import truncate_with_notice

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
            from llmling_agent.agents.events import FileContentItem

            await agent_ctx.events.tool_call_progress(
                title=f"Read: {path}",
                items=[FileContentItem(content=content, path=path)],
                replace_content=True,
            )
            # Return raw content for agent
            return content

    async def read_as_markdown(self, agent_ctx: AgentContext, path: str) -> str | dict[str, Any]:
        """Read file and convert to markdown text representation.

        Args:
            agent_ctx: Agent execution context
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
            from llmling_agent.agents.events import TextContentItem

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

    async def write_file(
        self,
        agent_ctx: AgentContext,
        path: str,
        content: str,
        mode: str = "w",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Write content to a file.

        Args:
            agent_ctx: Agent execution context
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

    async def delete_path(
        self, agent_ctx: AgentContext, path: str, recursive: bool = False
    ) -> dict[str, Any]:
        """Delete a file or directory.

        Args:
            agent_ctx: Agent execution context
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

    async def grep(
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
            agent_ctx: Agent execution context
            pattern: Regex pattern to search for
            path: Base directory to search in
            file_pattern: Glob pattern to filter files (e.g. "**/*.py")
            case_sensitive: Whether search is case-sensitive
            max_matches: Maximum number of matches to return
            context_lines: Number of context lines before/after match

        Returns:
            Grep results as formatted text
        """
        from llmling_agent_toolsets.fsspec_toolset.grep import (
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
                output = f"Found {match_count} matches:\n\n{matches}"
                if was_truncated:
                    output += "\n\n[Results truncated]"

            # Emit formatted content for UI display
            from llmling_agent.agents.events import TextContentItem

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

    async def download_file(
        self,
        agent_ctx: AgentContext,
        url: str,
        target_dir: str = "downloads",
        chunk_size: int = 8192,
    ) -> dict[str, Any]:
        """Download a file from URL to the toolset's filesystem.

        Args:
            agent_ctx: Agent execution context
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
        agent_ctx: AgentContext,
        path: str,
        display_description: str,
        mode: str = "edit",
    ) -> str:
        r"""Edit a file using AI agent with natural language instructions.

        Creates a new agent that processes the file based on the instructions.
        Shows real-time progress and diffs as the agent works.

        Args:
            path: File path (absolute or relative to session cwd)
            display_description: Natural language description of the edits to make
            mode: Edit mode - 'edit', 'create', or 'overwrite' (default: 'edit')

        Returns:
            Success message with edit summary

        Example:
            agentic_edit('src/main.py', 'Add error handling to the main function') ->
            'Successfully edited main.py using AI agent'
        """
        path = self._resolve_path(path, agent_ctx)
        title = f"AI editing file: {path}"
        await agent_ctx.events.tool_call_start(title=title, kind="edit", locations=[path])
        # Send initial pending notification
        await agent_ctx.events.file_operation("edit", path=path, success=True)

        try:
            if mode == "create":  # For create mode, don't read existing file
                original_content = ""
                prompt = _build_create_prompt(path, display_description)
                sys_prompt = "You are a code generator. Create the requested file content."
            elif mode == "overwrite":
                # For overwrite mode, don't read file - agent
                # already read it via system prompt requirement
                original_content = ""  # Will be set later for diff purposes
                prompt = _build_overwrite_prompt(path, display_description)
                sys_prompt = "You are a code editor. Output ONLY the complete new file content."
            else:  # For edit mode, use structured editing approach
                original_content = await self._read(agent_ctx, path)

                # Ensure content is string
                if isinstance(original_content, bytes):
                    original_content = original_content.decode()
                prompt = _build_edit_prompt(path, display_description)
                sys_prompt = (
                    "You are a code editor. Output ONLY structured edits "
                    "using the specified format."
                )
            # Create the editor agent using the same model
            model = self.edit_model or agent_ctx.agent.model_name
            editor_agent = PydanticAgent(model=model, system_prompt=sys_prompt)
            if mode == "edit":
                # For structured editing, get the full response and parse the edits
                edit = await editor_agent.run(prompt)
                new_content = await apply_structured_edits(original_content, edit.output)
            else:
                # For overwrite mode we need to read the current content for diff purposes
                if mode == "overwrite":
                    original_content = await self._read(agent_ctx, path)
                    # Ensure content is string
                    if isinstance(original_content, bytes):
                        original_content = original_content.decode("utf-8")
                # For create/overwrite modes, stream the complete content
                new_content_parts = []
                async with editor_agent.run_stream(prompt) as response:
                    async for chunk in response.stream_text(delta=True):
                        chunk_str = str(chunk)
                        new_content_parts.append(chunk_str)
                        # Build partial content for progress updates
                        partial_content = "".join(new_content_parts)
                        try:  # Send progress update with current diff
                            if len(partial_content.strip()) > 0:
                                # Get line numbers for streaming progress
                                get_changed_line_numbers(original_content, partial_content)
                                await agent_ctx.events.file_edit_progress(
                                    path=path,
                                    old_text=original_content,
                                    new_text=partial_content,
                                    status="in_progress",
                                )
                        except Exception:  # noqa: BLE001
                            pass  # Continue on progress update errors

                new_content = "".join(new_content_parts).strip()

            if not new_content:
                error_msg = "AI agent produced no output"
                await agent_ctx.events.file_operation(
                    "edit", path=path, success=False, error=error_msg
                )
                return error_msg

            # Write the new content to file
            new_content = await self._read(agent_ctx, path)
            original_lines = len(original_content.splitlines()) if original_content else 0
            new_lines = len(new_content.splitlines())

            if mode == "create":
                path = Path(path).name
                success_msg = f"Successfully created {path} ({new_lines} lines)"
            else:
                success_msg = f"Successfully edited {Path(path).name} using AI agent"
                success_msg += f" ({original_lines} → {new_lines} lines)"

            # Get changed line numbers for precise UI highlighting
            get_changed_line_numbers(original_content, new_content)
            # Send final completion update with complete diff and line numbers
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


def _build_create_prompt(path: str, description: str) -> str:
    """Build prompt for create mode."""
    return f"""Create a new file at {path} according to this description:

{description}

Output only the complete file content, no explanations or markdown formatting."""


def _build_overwrite_prompt(path: str, description: str) -> str:
    """Build prompt for overwrite mode."""
    return f"""Rewrite the file {path} according to this description:

{description}

Output only the complete new file content, no explanations or markdown formatting."""


def _build_edit_prompt(path: str, description: str) -> str:
    """Build prompt for structured edit mode."""
    return f"""\
You MUST respond with a series of edits to a file, using the following format:

```
<edits>

<old_text line=10>
OLD TEXT 1 HERE
</old_text>
<new_text>
NEW TEXT 1 HERE
</new_text>

<old_text line=456>
OLD TEXT 2 HERE
</old_text>
<new_text>
NEW TEXT 2 HERE
</new_text>

</edits>
```

# File Editing Instructions

- Use `<old_text>` and `<new_text>` tags to replace content
- `<old_text>` must exactly match existing file content, including indentation
- `<old_text>` must come from the actual file, not an outline
- `<old_text>` cannot be empty
- `line` should be a starting line number for the text to be replaced
- Be minimal with replacements:
- For unique lines, include only those lines
- For non-unique lines, include enough context to identify them
- Do not escape quotes, newlines, or other characters within tags
- For multiple occurrences, repeat the same tag pair for each instance
- Edits are sequential - each assumes previous edits are already applied
- Only edit the specified file
- Always close all tags properly

<file_to_edit>
{path}
</file_to_edit>

<edit_description>
{description}
</edit_description>

Tool calls have been disabled. You MUST start your response with <edits>."""


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        import fsspec

        from llmling_agent import AgentPool

        fs = fsspec.filesystem("file")
        tools = FSSpecTools(fs, name="local_fs")
        async with AgentPool() as pool:
            agent = await pool.add_agent("test", model="openai:gpt-5-nano")
            ctx = agent.context
            result = await tools.list_directory(ctx, path="/")
            print(result)

    asyncio.run(main())
