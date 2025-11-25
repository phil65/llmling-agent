"""FSSpec filesystem toolset implementation."""

from __future__ import annotations

import difflib
import mimetypes
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent as PydanticAgent, BinaryContent, ModelRetry

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.tools.base import Tool
from llmling_agent_toolsets.builtin.file_edit import replace_content


if TYPE_CHECKING:
    import fsspec  # type: ignore[import-untyped]

    from llmling_agent.prompts.conversion_manager import ConversionManager


logger = get_logger(__name__)

# MIME types that should be treated as text
TEXT_MIME_PREFIXES = ("text/", "application/json", "application/xml", "application/javascript")


def _is_text_mime(mime_type: str | None) -> bool:
    """Check if a MIME type represents text content."""
    if mime_type is None:
        return False
    return any(mime_type.startswith(prefix) for prefix in TEXT_MIME_PREFIXES)


class FSSpecTools(ResourceProvider):
    """Provider for fsspec filesystem tools."""

    def __init__(
        self,
        filesystem: fsspec.AbstractFileSystem,
        name: str = "fsspec",
        cwd: str | None = None,
        converter: ConversionManager | None = None,
    ) -> None:
        """Initialize with an fsspec filesystem.

        Args:
            filesystem: The fsspec filesystem instance to operate on
            name: Name for this toolset provider
            cwd: Optional cwd to resolve relative paths against
            converter: Optional conversion manager for markdown conversion
        """
        from fsspec.asyn import AsyncFileSystem  # type: ignore[import-untyped]
        from fsspec.implementations.asyn_wrapper import (  # type: ignore[import-untyped]
            AsyncFileSystemWrapper,
        )

        super().__init__(name=name)
        self.fs = (
            filesystem
            if isinstance(filesystem, AsyncFileSystem)
            else AsyncFileSystemWrapper(filesystem)
        )
        self.cwd = cwd
        self.converter = converter
        self._tools: list[Tool] | None = None

    def _resolve_path(self, path: str) -> str:
        """Resolve a potentially relative path to an absolute path.

        If cwd is set and path is relative, resolves relative to cwd.
        Otherwise returns the path as-is.

        Args:
            path: Path that may be relative or absolute

        Returns:
            Absolute path string
        """
        if self.cwd and not (path.startswith("/") or (len(path) > 1 and path[1] == ":")):
            return str(Path(self.cwd) / path)
        return path

    async def get_tools(self) -> list[Tool]:
        """Get filesystem tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            Tool.from_callable(
                self._list_directory,
                name_override="list_directory",
                description_override="List contents of a directory",
            ),
            Tool.from_callable(
                self._read_file,
                name_override="read_file",
                description_override=(
                    "Read file natively - returns text for text files, "
                    "binary content for documents/images (for model vision/doc capabilities)"
                ),
            ),
            Tool.from_callable(
                self._write_file,
                name_override="write_text_file",
                description_override="Write content to a file",
            ),
            Tool.from_callable(
                self._delete_path,
                name_override="delete_path",
                description_override="Delete a file or directory",
            ),
            Tool.from_callable(
                self.edit_file,
                name_override="edit_file",
                description_override="Edit a file by replacing specific content",
                source="filesystem",
                category="edit",
            ),
            Tool.from_callable(
                self.agentic_edit,
                name_override="agentic_edit",
                description_override="Edit a file using AI agent with natural language instructions",  # noqa: E501
                source="filesystem",
                category="edit",
            ),
        ]

        # Only add read_as_markdown if converter is available
        if self.converter:
            self._tools.append(
                Tool.from_callable(
                    self._read_as_markdown,
                    name_override="read_as_markdown",
                    description_override=(
                        "Read file and convert to markdown text representation. "
                        "Useful for extracting text from PDFs, documents, etc."
                    ),
                    source="filesystem",
                    category="read",
                )
            )

        return self._tools

    async def _list_directory(self, agent_ctx: AgentContext, path: str) -> dict[str, Any]:
        """List contents of a directory.

        Args:
            agent_ctx: Agent execution context
            path: Directory path to list

        Returns:
            Dictionary with directory contents and metadata
        """
        # Emit tool call start event for ACP notifications
        if self.cwd:
            path = self._resolve_path(path)
        await agent_ctx.events.tool_call_start(
            title=f"Listing directory: {path}", kind="read", locations=[path]
        )

        try:
            # Get detailed file information
            entries = await self.fs._ls(path, detail=True)

            files = []
            directories = []

            for entry in entries:
                if isinstance(entry, dict):
                    name = entry.get("name", "")
                    entry_type = entry.get("type", "unknown")
                    size = entry.get("size", 0)

                    item_info = {
                        "name": Path(name).name,
                        "full_path": name,
                        "size": size,
                        "type": entry_type,
                    }

                    # Add modification time if available
                    if "mtime" in entry:
                        item_info["modified"] = entry["mtime"]

                    if entry_type == "directory":
                        directories.append(item_info)
                    else:
                        files.append(item_info)
                else:
                    # Fallback for simple string entries
                    item_info = {
                        "name": Path(str(entry)).name,
                        "full_path": str(entry),
                        "type": "unknown",
                    }
                    files.append(item_info)

            result = {
                "path": path,
                "directories": directories,
                "files": files,
                "total_items": len(directories) + len(files),
            }

            # Emit success event
            await agent_ctx.events.file_operation("list", path=path, success=True)

        except (OSError, ValueError) as e:
            # Emit failure event
            await agent_ctx.events.file_operation("list", path=path, success=False, error=str(e))

            return {"error": f"Failed to list directory {path}: {e}"}
        else:
            return result

    async def _read_file(
        self,
        agent_ctx: AgentContext,
        path: str,
        encoding: str = "utf-8",
        line: int | None = None,
        limit: int | None = None,
    ) -> str | BinaryContent | dict[str, Any]:
        """Read file natively - text for text files, binary for documents/images.

        Args:
            agent_ctx: Agent execution context
            path: File path to read
            encoding: Text encoding to use for text files (default: utf-8)
            line: Optional line number to start reading from (1-based, text files only)
            limit: Optional maximum number of lines to read (text files only)

        Returns:
            Text content for text files, BinaryContent for binary files
        """
        if self.cwd:
            path = self._resolve_path(path)
        await agent_ctx.events.tool_call_start(
            title=f"Reading file: {path}", kind="read", locations=[path]
        )

        try:
            mime_type = mimetypes.guess_type(path)[0]

            if _is_text_mime(mime_type):
                content = await self._read(path, encoding=encoding)
                if isinstance(content, bytes):
                    content = content.decode(encoding)

                # Apply line filtering if specified
                if line is not None or limit is not None:
                    lines = content.splitlines()
                    start_idx = max(0, (line - 1) if line else 0)
                    end_idx = start_idx + limit if limit is not None else len(lines)
                    content = "\n".join(lines[start_idx:end_idx])

                await agent_ctx.events.file_operation(
                    "read", path=path, success=True, size=len(content)
                )
                return content

            # Binary file - return as BinaryContent for native model handling
            data = await self.fs._cat_file(path)
            await agent_ctx.events.file_operation("read", path=path, success=True, size=len(data))
            return BinaryContent(
                data=data, media_type=mime_type or "application/octet-stream", identifier=path
            )

        except (OSError, ValueError) as e:
            await agent_ctx.events.file_operation("read", path=path, success=False, error=str(e))
            return {"error": f"Failed to read file {path}: {e}"}

    async def _read_as_markdown(
        self,
        agent_ctx: AgentContext,
        path: str,
    ) -> str | dict[str, Any]:
        """Read file and convert to markdown text representation.

        Args:
            agent_ctx: Agent execution context
            path: Path to read

        Returns:
            File content converted to markdown
        """
        assert self.converter is not None, "Converter required for read_as_markdown"

        if self.cwd:
            path = self._resolve_path(path)
        await agent_ctx.events.tool_call_start(
            title=f"Reading file as markdown: {path}", kind="read", locations=[path]
        )

        try:
            content = await self.converter.convert_file(path)
            await agent_ctx.events.file_operation(
                "read", path=path, success=True, size=len(content)
            )
            return content
        except Exception as e:
            await agent_ctx.events.file_operation("read", path=path, success=False, error=str(e))
            return {"error": f"Failed to convert file {path}: {e}"}

    async def _write_file(
        self,
        agent_ctx: AgentContext,
        path: str,
        content: str,
        encoding: str = "utf-8",
        mode: str = "w",
    ) -> dict[str, Any]:
        """Write content to a file.

        Args:
            agent_ctx: Agent execution context
            path: File path to write
            content: Content to write
            encoding: Text encoding to use (default: utf-8)
            mode: Write mode ('w' for overwrite, 'a' for append)

        Returns:
            Dictionary with success info or error details
        """
        if self.cwd:
            path = self._resolve_path(path)
        await agent_ctx.events.tool_call_start(
            title=f"Writing file: {path}", kind="edit", locations=[path]
        )
        try:
            # Validate mode
            if mode not in ("w", "a"):
                error_msg = f"Invalid mode '{mode}'. Use 'w' (write) or 'a' (append)"

                # Emit failure event
                await agent_ctx.events.file_operation(
                    "write", path=path, success=False, error=error_msg
                )

                return {"error": error_msg}

            await self._write(path, content)
            # Try to get file size after writing
            try:
                info = await self.fs._info(path)
                size = info.get("size", len(content))
            except (OSError, KeyError):
                size = len(content)

            result = {
                "path": path,
                "bytes_written": len(content.encode(encoding)),
                "size": size,
                "mode": mode,
                "encoding": encoding,
            }

            # Emit success event
            await agent_ctx.events.file_operation("write", path=path, success=True, size=size)
        except (OSError, ValueError) as e:
            # Emit failure event
            await agent_ctx.events.file_operation("write", path=path, success=False, error=str(e))

            return {"error": f"Failed to write file {path}: {e}"}
        else:
            return result

    async def _delete_path(
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
        if self.cwd:
            path = self._resolve_path(path)
        await agent_ctx.events.tool_call_start(
            title=f"Deleting path: {path}", kind="delete", locations=[path]
        )

        try:
            # Check if path exists and get its type
            try:
                info = await self.fs._info(path)
                path_type = info.get("type", "unknown")
            except FileNotFoundError:
                error_msg = f"Path does not exist: {path}"

                # Emit failure event
                await agent_ctx.events.file_operation(
                    "delete", path=path, success=False, error=error_msg
                )

                return {"error": error_msg}
            except (OSError, ValueError) as e:
                error_msg = f"Could not check path {path}: {e}"

                # Emit failure event
                await agent_ctx.events.file_operation(
                    "delete", path=path, success=False, error=error_msg
                )

                return {"error": error_msg}

            if path_type == "directory":
                if not recursive:
                    # Check if directory is empty
                    try:
                        contents = await self.fs._ls(path)
                        if contents:
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

                await self.fs._rm(path, recursive=recursive)
            else:
                # It's a file
                await self.fs._rm(path)  # or _rm_file?

        except (OSError, ValueError) as e:
            # Emit failure event
            await agent_ctx.events.file_operation("delete", path=path, success=False, error=str(e))

            return {"error": f"Failed to delete {path}: {e}"}
        else:
            result = {
                "path": path,
                "deleted": True,
                "type": path_type,
                "recursive": recursive,
            }

            # Emit success event
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
        if self.cwd:
            path = self._resolve_path(path)
        await agent_ctx.events.tool_call_start(
            title=f"Editing file: {path}", kind="edit", locations=[path]
        )

        if old_string == new_string:
            return "Error: old_string and new_string must be different"

        # Send initial pending notification
        await agent_ctx.events.file_operation(
            "edit",
            path=path,
            success=True,  # Initial state
            title=f"Editing file: {path}",
            kind="edit",
            locations=[path],
        )

        try:  # Read current file content
            original_content = await self._read(path)
            if isinstance(original_content, bytes):
                original_content = original_content.decode("utf-8")

            try:  # Apply smart content replacement
                new_content = replace_content(original_content, old_string, new_string, replace_all)
            except ValueError as e:
                error_msg = f"Edit failed: {e}"
                await agent_ctx.events.file_operation(
                    "edit",
                    path=path,
                    success=False,
                    error=error_msg,
                    raw_output=error_msg,
                )
                return error_msg

            await self._write(path, new_content)
            success_msg = f"Successfully edited {Path(path).name}: {description}"
            changed_line_numbers = get_changed_line_numbers(original_content, new_content)
            if lines_changed := len(changed_line_numbers):
                success_msg += f" ({lines_changed} lines changed)"

            await agent_ctx.events.file_edit_progress(
                path=path,
                old_text=original_content,
                new_text=new_content,
                status="completed",
                changed_lines=changed_line_numbers,
            )
        except Exception as e:  # noqa: BLE001
            error_msg = f"Error editing file: {e}"
            await agent_ctx.events.file_operation(
                "edit",
                path=path,
                success=False,
                error=error_msg,
                raw_output=error_msg,
            )
            return error_msg
        else:
            return success_msg

    async def _read(self, path: str, encoding: str = "utf-8") -> str:
        # with self.fs.open(path, "r", encoding="utf-8") as f:
        #     return f.read()
        return await self.fs._cat(path)  # type: ignore[no-any-return]

    async def _write(self, path: str, content: str | bytes) -> None:
        mode = "wt" if isinstance(content, str) else "wb"
        file = await self.fs.open_async(path, mode)
        await file.write(content)
        # with self.fs.open(path, mode="r") as f:
        #     f.write(content)

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
        if self.cwd:
            path = self._resolve_path(path)
        await agent_ctx.events.tool_call_start(
            title=f"AI editing file: {path}", kind="edit", locations=[path]
        )

        # Send initial pending notification
        await agent_ctx.events.file_operation(
            "edit",
            path=path,
            success=True,  # Initial state
            title=f"Editing file: {path}",
            kind="edit",
            locations=[path],
        )

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
                original_content = await self._read(path)

                # Ensure content is string
                if isinstance(original_content, bytes):
                    original_content = original_content.decode("utf-8")
                prompt = _build_edit_prompt(path, display_description)
                sys_prompt = (
                    "You are a code editor. Output ONLY structured edits "
                    "using the specified format."
                )

            # Create the editor agent using the same model
            editor_agent = PydanticAgent(model="openai:gpt-4", system_prompt=sys_prompt)

            if mode == "edit":
                # For structured editing, get the full response and parse the edits
                edit = await editor_agent.run(prompt)
                new_content = await _apply_structured_edits(original_content, edit.output)
            else:
                # For overwrite mode we need to read the current content for diff purposes
                if mode == "overwrite":
                    original_content = await self._read(path)
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
                                progress_line_numbers = get_changed_line_numbers(
                                    original_content, partial_content
                                )
                                await agent_ctx.events.file_edit_progress(
                                    path=path,
                                    old_text=original_content,
                                    new_text=partial_content,
                                    status="in_progress",
                                    changed_lines=progress_line_numbers,
                                )
                        except Exception:  # noqa: BLE001
                            pass  # Continue on progress update errors

                new_content = "".join(new_content_parts).strip()

            if not new_content:
                error_msg = "AI agent produced no output"
                await agent_ctx.events.file_operation(
                    "edit",
                    path=path,
                    success=False,
                    error=error_msg,
                    raw_output=error_msg,
                )
                return error_msg

            # Write the new content to file
            new_content = await self._read(path)
            original_lines = len(original_content.splitlines()) if original_content else 0
            new_lines = len(new_content.splitlines())

            if mode == "create":
                path = Path(path).name
                success_msg = f"Successfully created {path} ({new_lines} lines)"
            else:
                success_msg = f"Successfully edited {Path(path).name} using AI agent"
                success_msg += f" ({original_lines} → {new_lines} lines)"

            # Get changed line numbers for precise UI highlighting
            changed_line_numbers = get_changed_line_numbers(original_content, new_content)

            # Send final completion update with complete diff and line numbers
            await agent_ctx.events.file_edit_progress(
                path=path,
                old_text=original_content,
                new_text=new_content,
                status="completed",
                changed_lines=changed_line_numbers,
            )

        except Exception as e:  # noqa: BLE001
            error_msg = f"Error during agentic edit: {e}"
            await agent_ctx.events.file_operation(
                "edit",
                path=path,
                success=False,
                error=error_msg,
                raw_output=error_msg,
            )
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


async def _apply_structured_edits(original_content: str, edits_response: str) -> str:
    """Apply structured edits from the agent response."""
    # Parse the edits from the response
    edits_match = re.search(r"<edits>(.*?)</edits>", edits_response, re.DOTALL)
    if not edits_match:
        logger.warning("No edits block found in response")
        return original_content

    edits_content = edits_match.group(1)

    # Find all old_text/new_text pairs
    old_text_pattern = r"<old_text[^>]*>(.*?)</old_text>"
    new_text_pattern = r"<new_text>(.*?)</new_text>"

    old_texts = re.findall(old_text_pattern, edits_content, re.DOTALL)
    new_texts = re.findall(new_text_pattern, edits_content, re.DOTALL)

    if len(old_texts) != len(new_texts):
        logger.warning("Mismatch between old_text and new_text blocks")
        return original_content

    # Apply edits sequentially
    content = original_content
    applied_edits = 0

    failed_matches = []
    multiple_matches = []

    for old_text, new_text in zip(old_texts, new_texts, strict=False):
        old_text = old_text.strip()
        new_text = new_text.strip()

        # Check for multiple matches (ambiguity)
        match_count = content.count(old_text)
        if match_count > 1:
            multiple_matches.append(old_text[:50])
        elif match_count == 1:
            content = content.replace(old_text, new_text, 1)
            applied_edits += 1
        else:
            failed_matches.append(old_text[:50])

    # Raise ModelRetry for specific failure cases
    if applied_edits == 0 and len(old_texts) > 0:
        msg = (
            "Some edits were produced but none of them could be applied. "
            "Read the relevant sections of the file again so that "
            "I can perform the requested edits."
        )
        raise ModelRetry(msg)

    if multiple_matches:
        matches_str = ", ".join(multiple_matches)
        msg = (
            f"<old_text> matches multiple positions in the file: {matches_str}... "
            "Read the relevant sections of the file again and extend <old_text> "
            "to be more specific."
        )
        raise ModelRetry(msg)

    logger.info("Applied structured edits", num=applied_edits, total=len(old_texts))
    return content


def get_changed_lines(original_content: str, new_content: str, path: str) -> list[str]:
    old = original_content.splitlines(keepends=True)
    new = new_content.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old, new, fromfile=path, tofile=path, lineterm=""))
    return [line for line in diff if line.startswith(("+", "-"))]


def get_changed_line_numbers(original_content: str, new_content: str) -> list[int]:
    """Extract line numbers where changes occurred for ACP UI highlighting.

    Similar to Claude Code's line tracking for precise change location reporting.
    Returns line numbers in the new content where changes happened.

    Args:
        original_content: Original file content
        new_content: Modified file content

    Returns:
        List of line numbers (1-based) where changes occurred in new content
    """
    old_lines = original_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    # Use SequenceMatcher to find changed blocks
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    changed_line_numbers = set()

    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert", "delete"):
            # For replacements and insertions, mark lines in new content
            # For deletions, mark the position where deletion occurred
            if tag == "delete":
                # Mark the line where deletion occurred (or next line if at end)
                line_num = min(j1 + 1, len(new_lines))
                if line_num > 0:
                    changed_line_numbers.add(line_num)
            else:
                # Mark all affected lines in new content
                for line_num in range(j1 + 1, j2 + 1):  # Convert to 1-based
                    changed_line_numbers.add(line_num)

    return sorted(changed_line_numbers)


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
            result = await tools._list_directory(ctx, path="/")
            print(result)

    asyncio.run(main())
