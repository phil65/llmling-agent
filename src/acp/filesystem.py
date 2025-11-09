"""Filesystem implementation for ACP (Agent Communication Protocol) sessions."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import shlex
from typing import TYPE_CHECKING, Any, Literal, overload

from anyenv import get_os_command_provider
from fsspec.asyn import AsyncFileSystem, sync_wrapper
from fsspec.spec import AbstractBufferedFile
from upath import UPath

from acp.acp_requests import ACPRequests
from acp.notifications import ACPNotifications


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

    cat_file = sync_wrapper(_cat_file)  # pyright: ignore[reportAssignmentType]

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

    put_file = sync_wrapper(_put_file)

    @overload
    async def _ls(
        self, path: str = "", detail: Literal[True] = True, **kwargs: Any
    ) -> list[dict[str, Any]]: ...

    @overload
    async def _ls(
        self, path: str = "", detail: Literal[False] = False, **kwargs: Any
    ) -> list[str]: ...

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
        list_cmd = self.command_provider.get_command("list_directory")
        ls_cmd = list_cmd.create_command(path)

        try:
            command, args = self._parse_command(ls_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=10
            )

            if exit_code != 0:
                msg = f"Error listing directory {path!r}: {output}"
                raise FileNotFoundError(msg)  # noqa: TRY301

            result = list_cmd.parse_command(output, path)

            # Convert DirectoryEntry objects to dict format or simple names
            if detail:
                return [
                    {
                        "name": item.name,
                        "path": item.path,
                        "type": item.type,
                        "size": item.size,
                        "timestamp": item.timestamp,
                        "permissions": item.permissions,
                    }
                    for item in result
                ]
            return [item.name for item in result]

        except Exception as e:
            msg = f"Could not list directory {path}: {e}"
            raise FileNotFoundError(msg) from e
        else:
            return result

    ls = sync_wrapper(_ls)

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Get file information via stat command.

        Args:
            path: File path to get info for
            **kwargs: Additional options

        Returns:
            File information dictionary
        """
        # Use OS-specific command to get file information
        info_cmd = self.command_provider.get_command("file_info")
        stat_cmd = info_cmd.create_command(path)

        try:
            command, args = self._parse_command(stat_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=5
            )

            if exit_code != 0:
                msg = f"File not found: {path}"
                raise FileNotFoundError(msg)
            file_info = info_cmd.parse_command(output.strip(), path)
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
                    if item["name"] == filename:
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

    info = sync_wrapper(_info)

    async def _exists(self, path: str, **kwargs: Any) -> bool:
        """Check if file exists via test command.

        Args:
            path: File path to check
            **kwargs: Additional options

        Returns:
            True if file exists, False otherwise
        """
        exists_cmd = self.command_provider.get_command("exists")
        test_cmd = exists_cmd.create_command(path)

        try:
            command, args = self._parse_command(test_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=5
            )
        except (OSError, ValueError):
            return False
        else:
            return exists_cmd.parse_command(
                output, exit_code if exit_code is not None else 1
            )

    exists = sync_wrapper(_exists)  # pyright: ignore[reportAssignmentType]

    async def _isdir(self, path: str, **kwargs: Any) -> bool:
        """Check if path is a directory via test command.

        Args:
            path: Path to check
            **kwargs: Additional options

        Returns:
            True if path is a directory, False otherwise
        """
        isdir_cmd = self.command_provider.get_command("is_directory")
        test_cmd = isdir_cmd.create_command(path)

        try:
            command, args = self._parse_command(test_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=5
            )
        except (OSError, ValueError):
            return False
        else:
            return isdir_cmd.parse_command(
                output, exit_code if exit_code is not None else 1
            )

    isdir = sync_wrapper(_isdir)

    async def _isfile(self, path: str, **kwargs: Any) -> bool:
        """Check if path is a file via test command.

        Args:
            path: Path to check
            **kwargs: Additional options

        Returns:
            True if path is a file, False otherwise
        """
        isfile_cmd = self.command_provider.get_command("is_file")
        test_cmd = isfile_cmd.create_command(path)

        try:
            command, args = self._parse_command(test_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=5
            )
        except (OSError, ValueError):
            return False
        else:
            return isfile_cmd.parse_command(
                output, exit_code if exit_code is not None else 1
            )

    isfile = sync_wrapper(_isfile)

    async def _makedirs(self, path: str, exist_ok: bool = False, **kwargs: Any) -> None:
        """Create directories via mkdir command.

        Args:
            path: Directory path to create
            exist_ok: Don't raise error if directory already exists
            **kwargs: Additional options
        """
        create_cmd = self.command_provider.get_command("create_directory")
        mkdir_cmd = create_cmd.create_command(path, parents=exist_ok)

        try:
            command, args = self._parse_command(mkdir_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=5
            )

            success = create_cmd.parse_command(
                output, exit_code if exit_code is not None else 1
            )
            if not success:
                msg = f"Error creating directory {path}: {output}"
                raise OSError(msg)  # noqa: TRY301
        except Exception as e:
            msg = f"Could not create directory {path}: {e}"
            raise OSError(msg) from e

    makedirs = sync_wrapper(_makedirs)

    async def _rm(self, path: str, recursive: bool = False, **kwargs: Any) -> None:
        """Remove file or directory via rm command.

        Args:
            path: Path to remove
            recursive: Remove directories recursively
            **kwargs: Additional options
        """
        remove_cmd = self.command_provider.get_command("remove_path")
        rm_cmd = remove_cmd.create_command(path, recursive=recursive)

        try:
            command, args = self._parse_command(rm_cmd)
            output, exit_code = await self.requests.run_command(
                command, args=args, timeout_seconds=10
            )

            success = remove_cmd.parse_command(
                output, exit_code if exit_code is not None else 1
            )
            if not success:
                msg = f"Error removing {path}: {output}"
                raise OSError(msg)  # noqa: TRY301
        except Exception as e:
            msg = f"Could not remove {path}: {e}"
            raise OSError(msg) from e

    rm = sync_wrapper(_rm)

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
