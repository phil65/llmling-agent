"""Filesystem implementation for ACP (Agent Communication Protocol) sessions."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import shlex
from typing import TYPE_CHECKING, Any

from anyenv import get_os_command_provider
from fsspec.asyn import AsyncFileSystem
from fsspec.spec import AbstractBufferedFile
from upath import UPath

from acp.notifications import ACPNotifications
from acp.requests import ACPRequests


if TYPE_CHECKING:
    from acp.client.protocol import Client


logger = logging.getLogger(__name__)


class ACPPath(UPath):
    """UPath implementation for ACP filesystems."""

    __slots__ = ()


class ACPFile(AbstractBufferedFile):
    """File-like object for ACP filesystem operations."""

    def __init__(self, fs: ACPFileSystem, path: str, mode: str = "rb", **kwargs: Any):
        """Initialize ACP file handle."""
        super().__init__(fs, path, mode, **kwargs)
        self._content: bytes | None = None
        self.forced = False

    def _fetch_range(self, start: int | None, end: int | None) -> bytes:
        """Fetch byte range from file (sync wrapper)."""
        if self._content is None:
            # Run the async operation in the event loop
            self._content = asyncio.run(self.fs._cat_file(self.path))

        if start is None and end is None:
            return self._content
        return self._content[start:end]

    def _upload_chunk(self, final: bool = False) -> bool:
        """Upload buffered data to file (sync wrapper)."""
        if final and self.buffer:
            content = self.buffer.getvalue()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            # Run the async operation in the event loop
            asyncio.run(self.fs._put_file(self.path, content))
        return True


class ACPFileSystem(AsyncFileSystem):
    """Async filesystem for ACP sessions."""

    protocol = "acp"
    sep = "/"

    def __init__(self, client: Client, session_id: str, **kwargs: Any):
        """Initialize ACP filesystem.

        Args:
            client: ACP client for operations
            session_id: Session identifier
            **kwargs: Additional filesystem options
        """
        super().__init__(**kwargs)
        self.client = client
        self.session_id = session_id
        self.requests = ACPRequests(client, session_id)
        self.notifications = ACPNotifications(client, session_id)
        self.command_provider = get_os_command_provider()

    def _make_path(self, path: str) -> UPath:
        """Create a path object from string."""
        return ACPPath(path, **self.storage_options)

    def _parse_command(self, command_str: str) -> tuple[str, list[str]]:
        """Parse a shell command string into command and arguments.

        Args:
            command_str: Shell command string to parse

        Returns:
            Tuple of (command, args_list)
        """
        try:
            parts = shlex.split(command_str)
            if not parts:
                msg = "Empty command string"
                raise ValueError(msg)  # noqa: TRY301
            return parts[0], parts[1:]
        except ValueError as e:
            # Fallback for problematic shell strings
            parts = command_str.split()
            if not parts:
                msg = "Empty command string"
                raise ValueError(msg) from e
            return parts[0], parts[1:]

    async def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> bytes:
        """Read file content via ACP session.

        Args:
            path: File path to read
            start: Start byte position (not supported by ACP)
            end: End byte position (not supported by ACP)
            **kwargs: Additional options

        Returns:
            File content as bytes

        Raises:
            NotImplementedError: If byte range is requested (ACP doesn't support
                partial reads)
        """
        if start is not None or end is not None:
            msg = "ACP filesystem does not support byte range reads"
            raise NotImplementedError(msg)

        try:
            content = await self.requests.read_text_file(path)
            return content.encode("utf-8")
        except Exception as e:
            msg = f"Could not read file {path}: {e}"
            raise FileNotFoundError(msg) from e

    async def _put_file(self, path: str, content: str | bytes, **kwargs: Any) -> None:
        """Write file content via ACP session.

        Args:
            path: File path to write
            content: Content to write (string or bytes)
            **kwargs: Additional options
        """
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        try:
            await self.requests.write_text_file(path, content)
        except Exception as e:
            msg = f"Could not write file {path}: {e}"
            raise OSError(msg) from e

    async def _ls(
        self, path: str = "", detail: bool = True, **kwargs: Any
    ) -> list[dict[str, Any]] | list[str]:
        """List directory contents via terminal command.

        Uses 'ls -la' command through ACP terminal to get directory listings.

        Args:
            path: Directory path to list
            detail: Whether to return detailed file information
            **kwargs: Additional options

        Returns:
            List of file information dictionaries or file names
        """
        # Use OS-specific command to list directory contents
        ls_cmd = self.command_provider.get_command("list_directory").create_command(
            path, detailed=detail
        )

        try:
            command, args = self._parse_command(ls_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=10
            )

            if exit_code != 0:
                msg = f"Error listing directory {path!r}: {output}"
                raise FileNotFoundError(msg)  # noqa: TRY301

            result = self.command_provider.get_command("list_directory").parse_command(
                output, path, detailed=detail
            )

            # Convert DirectoryEntry objects to dict format
            if detail and result:
                converted_result = []
                for item in result:
                    if hasattr(item, "name"):  # DirectoryEntry object
                        converted_result.append({
                            "name": item.name,
                            "path": item.path,
                            "type": item.type,
                            "size": item.size,
                            "timestamp": item.timestamp,
                            "permissions": item.permissions,
                        })
                    else:
                        converted_result.append(item)
                return converted_result

        except Exception as e:
            msg = f"Could not list directory {path}: {e}"
            raise FileNotFoundError(msg) from e
        else:
            return result

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Get file information via stat command.

        Args:
            path: File path to get info for
            **kwargs: Additional options

        Returns:
            File information dictionary
        """
        # Use OS-specific command to get file information
        stat_cmd = self.command_provider.get_command("file_info").create_command(path)

        try:
            command, args = self._parse_command(stat_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=5
            )

            if exit_code != 0:
                msg = f"File not found: {path}"
                raise FileNotFoundError(msg)
            file_info = self.command_provider.get_command("file_info").parse_command(
                output.strip(), path
            )
            return {  # noqa: TRY300
                "name": file_info.name,
                "path": file_info.path,
                "type": file_info.type,
                "size": file_info.size,
                "timestamp": file_info.timestamp,
                "permissions": file_info.permissions,
            }

        except (OSError, ValueError) as e:
            # Fallback: try to get basic info from ls
            try:
                ls_result = await self._ls(str(Path(path).parent), detail=True)
                filename = Path(path).name

                for item in ls_result:
                    if item.get("name") == filename:
                        return {
                            "name": item["name"],
                            "path": path,
                            "type": item["type"],
                            "size": item["size"],
                            "timestamp": item.get("timestamp"),
                            "permissions": item.get("permissions"),
                        }

                msg = f"File not found: {path}"
                raise FileNotFoundError(msg)
            except (OSError, ValueError):
                msg = f"Could not get file info for {path}: {e}"
                raise FileNotFoundError(msg) from e

    async def _exists(self, path: str, **kwargs: Any) -> bool:
        """Check if file exists via test command.

        Args:
            path: File path to check
            **kwargs: Additional options

        Returns:
            True if file exists, False otherwise
        """
        test_cmd = self.command_provider.get_command("exists").create_command(path)

        try:
            command, args = self._parse_command(test_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=5
            )
        except (OSError, ValueError):
            return False
        else:
            return self.command_provider.get_command("exists").parse_command(
                output, exit_code if exit_code is not None else 1
            )

    async def _isdir(self, path: str, **kwargs: Any) -> bool:
        """Check if path is a directory via test command.

        Args:
            path: Path to check
            **kwargs: Additional options

        Returns:
            True if path is a directory, False otherwise
        """
        test_cmd = self.command_provider.get_command("is_directory").create_command(path)

        try:
            command, args = self._parse_command(test_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=5
            )
        except (OSError, ValueError):
            return False
        else:
            return self.command_provider.get_command("is_directory").parse_command(
                output, exit_code if exit_code is not None else 1
            )

    async def _isfile(self, path: str, **kwargs: Any) -> bool:
        """Check if path is a file via test command.

        Args:
            path: Path to check
            **kwargs: Additional options

        Returns:
            True if path is a file, False otherwise
        """
        test_cmd = self.command_provider.get_command("is_file").create_command(path)

        try:
            command, args = self._parse_command(test_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=5
            )
        except (OSError, ValueError):
            return False
        else:
            return self.command_provider.get_command("is_file").parse_command(
                output, exit_code if exit_code is not None else 1
            )

    async def _makedirs(self, path: str, exist_ok: bool = False, **kwargs: Any) -> None:
        """Create directories via mkdir command.

        Args:
            path: Directory path to create
            exist_ok: Don't raise error if directory already exists
            **kwargs: Additional options
        """
        mkdir_cmd = self.command_provider.get_command("create_directory").create_command(
            path, parents=exist_ok
        )

        try:
            command, args = self._parse_command(mkdir_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=5
            )

            success = self.command_provider.get_command("create_directory").parse_command(
                output, exit_code if exit_code is not None else 1
            )
            if not success:
                msg = f"Error creating directory {path}: {output}"
                raise OSError(msg)  # noqa: TRY301
        except Exception as e:
            msg = f"Could not create directory {path}: {e}"
            raise OSError(msg) from e

    async def _rm(self, path: str, recursive: bool = False, **kwargs: Any) -> None:
        """Remove file or directory via rm command.

        Args:
            path: Path to remove
            recursive: Remove directories recursively
            **kwargs: Additional options
        """
        rm_cmd = self.command_provider.get_command("remove_path").create_command(
            path, recursive=recursive
        )

        try:
            command, args = self._parse_command(rm_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=10
            )

            success = self.command_provider.get_command("remove_path").parse_command(
                output, exit_code if exit_code is not None else 1
            )
            if not success:
                msg = f"Error removing {path}: {output}"
                raise OSError(msg)  # noqa: TRY301
        except Exception as e:
            msg = f"Could not remove {path}: {e}"
            raise OSError(msg) from e

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> ACPFile:
        """Open file for reading or writing.

        Args:
            path: File path to open
            mode: File mode ('rb', 'wb', 'ab', 'xb')
            **kwargs: Additional options

        Returns:
            File-like object
        """
        # Convert text modes to binary modes for fsspec compatibility
        if mode == "r":
            mode = "rb"
        elif mode == "w":
            mode = "wb"
        elif mode == "a":
            mode = "ab"
        elif mode == "x":
            mode = "xb"

        return ACPFile(self, path, mode, **kwargs)


# Sync wrapper filesystem for easier integration
class ACPFileSystemSync(ACPFileSystem):
    """Synchronous wrapper around ACPFileSystem."""

    def __init__(
        self,
        client: Client,
        session_id: str,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs: Any,
    ):
        """Initialize sync ACP filesystem.

        Args:
            client: ACP client for operations
            session_id: Session identifier
            loop: Event loop to use for async operations
            **kwargs: Additional filesystem options
        """
        super().__init__(client, session_id, **kwargs)
        self._loop = loop or asyncio.new_event_loop()

    def _run_async(self, coro):
        """Run async coroutine in the event loop."""
        if self._loop.is_running():
            # If loop is already running, we need to use a different approach
            # This is a simplified version - in production, you'd want better handling
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result()
        return self._loop.run_until_complete(coro)

    def ls(
        self, path: str = "", detail: bool = True, **kwargs: Any
    ) -> list[dict[str, Any]] | list[str]:
        """Sync wrapper for ls operation."""
        return self._run_async(self._ls(path, detail, **kwargs))

    def cat(self, path: str, **kwargs: Any) -> bytes:
        """Sync wrapper for cat operation."""
        return self._run_async(self._cat_file(path, **kwargs))

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Sync wrapper for info operation."""
        return self._run_async(self._info(path, **kwargs))

    def exists(self, path: str, **kwargs: Any) -> bool:
        """Sync wrapper for exists operation."""
        return self._run_async(self._exists(path, **kwargs))

    def isdir(self, path: str, **kwargs: Any) -> bool:
        """Sync wrapper for isdir operation."""
        return self._run_async(self._isdir(path, **kwargs))

    def isfile(self, path: str, **kwargs: Any) -> bool:
        """Sync wrapper for isfile operation."""
        return self._run_async(self._isfile(path, **kwargs))

    def makedirs(self, path: str, exist_ok: bool = False, **kwargs: Any) -> None:
        """Sync wrapper for makedirs operation."""
        return self._run_async(self._makedirs(path, exist_ok, **kwargs))

    def rm(self, path: str, recursive: bool = False, **kwargs: Any) -> None:
        """Sync wrapper for rm operation."""
        return self._run_async(self._rm(path, recursive, **kwargs))
