"""Filesystem implementation for ACP (Agent Communication Protocol) sessions."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

    def _make_path(self, path: str) -> UPath:
        """Create a path object from string."""
        return ACPPath(path, **self.storage_options)

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
        # Use ls command to list directory contents
        ls_cmd = "ls" if not detail else "ls -la --time-style=+%Y-%m-%d-%H:%M:%S"
        if path:
            ls_cmd += f' "{path}"'

        try:
            output, exit_code = await self.requests.run_command(
                ls_cmd, timeout_seconds=10
            )

            if exit_code != 0:
                msg = f"Error listing directory {path!r}: {output}"
                raise FileNotFoundError(msg)  # noqa: TRY301

            return self._parse_ls_output(output, path, detail)

        except Exception as e:
            msg = f"Could not list directory {path}: {e}"
            raise FileNotFoundError(msg) from e

    def _parse_ls_output(
        self, output: str, base_path: str, detail: bool
    ) -> list[dict[str, Any]] | list[str]:
        """Parse ls command output into file information.

        Args:
            output: Raw ls command output
            base_path: Base directory path
            detail: Whether to return detailed information

        Returns:
            Parsed file information
        """
        lines = output.strip().split("\n")
        if not lines:
            return []

        # Filter out total line and empty lines
        file_lines = [line for line in lines if line and not line.startswith("total ")]

        files: list[Any] = []
        for line in file_lines:
            if not line.strip():
                continue

            if detail:
                # Parse detailed ls -la output
                parts = line.split()
                min_ls_parts = 7
                if len(parts) < min_ls_parts:
                    continue

                permissions = parts[0]
                size = int(parts[4]) if parts[4].isdigit() else 0
                timestamp = parts[5]  # Single timestamp field
                name = " ".join(parts[6:])  # Handle names with spaces

                # Determine file type
                file_type = "directory" if permissions.startswith("d") else "file"
                if permissions.startswith("l"):
                    file_type = "link"

                # Build full path
                full_path = f"{base_path.rstrip('/')}/{name}" if base_path else name

                files.append({
                    "name": name,
                    "path": full_path,
                    "type": file_type,
                    "size": size,
                    "permissions": permissions,
                    "timestamp": timestamp,
                })
            else:
                # Simple name extraction - get filename from detailed output
                parts = line.split()
                if len(parts) >= 7:  # Valid ls -la line  # noqa: PLR2004
                    name = " ".join(parts[6:])  # Name starts at column 6
                    files.append(name)

        return files

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Get file information via stat command.

        Args:
            path: File path to get info for
            **kwargs: Additional options

        Returns:
            File information dictionary
        """
        # Use stat command to get file information
        stat_cmd = f'stat -c "%n|%s|%Y|%A|%F" "{path}"'

        try:
            output, exit_code = await self.requests.run_command(
                stat_cmd, timeout_seconds=5
            )

            if exit_code != 0:
                msg = f"File not found: {path}"
                raise FileNotFoundError(msg)

            return self._parse_stat_output(output.strip(), path)

        except (OSError, ValueError) as e:
            # Fallback: try to get basic info from ls
            try:
                ls_result = await self._ls(str(Path(path).parent), detail=True)
                filename = Path(path).name

                for item in ls_result:
                    if isinstance(item, dict) and item.get("name") == filename:
                        return {
                            "name": item["name"],
                            "path": path,
                            "type": item["type"],
                            "size": item["size"],
                            "permissions": item.get("permissions", ""),
                        }

                msg = f"File not found: {path}"
                raise FileNotFoundError(msg)
            except (OSError, ValueError):
                msg = f"Could not get file info for {path}: {e}"
                raise FileNotFoundError(msg) from e

    def _parse_stat_output(self, output: str, path: str) -> dict[str, Any]:
        """Parse stat command output.

        Args:
            output: Raw stat command output
            path: Original file path

        Returns:
            Parsed file information
        """
        # stat output format: name|size|mtime|permissions|file_type
        parts = output.split("|")
        min_stat_parts = 5
        if len(parts) < min_stat_parts:
            msg = f"Unexpected stat output format: {output}"
            raise ValueError(msg)

        name = Path(path).name
        size = int(parts[1]) if parts[1].isdigit() else 0
        mtime = int(parts[2]) if parts[2].isdigit() else 0
        permissions = parts[3]
        file_type_str = parts[4].lower()

        # Map stat file types to our types
        if "directory" in file_type_str:
            file_type = "directory"
        elif "symbolic link" in file_type_str:
            file_type = "link"
        else:
            file_type = "file"

        return {
            "name": name,
            "path": path,
            "type": file_type,
            "size": size,
            "permissions": permissions,
            "mtime": mtime,
        }

    async def _exists(self, path: str, **kwargs: Any) -> bool:
        """Check if file exists via test command.

        Args:
            path: File path to check
            **kwargs: Additional options

        Returns:
            True if file exists, False otherwise
        """
        test_cmd = f'test -e "{path}"'

        try:
            _, exit_code = await self.requests.run_command(test_cmd, timeout_seconds=5)
        except (OSError, ValueError):
            return False
        else:
            return exit_code == 0

    async def _isdir(self, path: str, **kwargs: Any) -> bool:
        """Check if path is a directory via test command.

        Args:
            path: Path to check
            **kwargs: Additional options

        Returns:
            True if path is a directory, False otherwise
        """
        test_cmd = f'test -d "{path}"'

        try:
            _, exit_code = await self.requests.run_command(test_cmd, timeout_seconds=5)
        except (OSError, ValueError):
            return False
        else:
            return exit_code == 0

    async def _isfile(self, path: str, **kwargs: Any) -> bool:
        """Check if path is a file via test command.

        Args:
            path: Path to check
            **kwargs: Additional options

        Returns:
            True if path is a file, False otherwise
        """
        test_cmd = f'test -f "{path}"'

        try:
            _, exit_code = await self.requests.run_command(test_cmd, timeout_seconds=5)
        except (OSError, ValueError):
            return False
        else:
            return exit_code == 0

    async def _makedirs(self, path: str, exist_ok: bool = False, **kwargs: Any) -> None:
        """Create directories via mkdir command.

        Args:
            path: Directory path to create
            exist_ok: Don't raise error if directory already exists
            **kwargs: Additional options
        """
        mkdir_cmd = f'mkdir -p "{path}"' if exist_ok else f'mkdir "{path}"'

        try:
            _, exit_code = await self.requests.run_command(mkdir_cmd, timeout_seconds=5)
            if exit_code != 0:
                msg = f"Could not create directory: {path}"
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
        rm_cmd = f'rm -rf "{path}"' if recursive else f'rm "{path}"'

        try:
            _, exit_code = await self.requests.run_command(rm_cmd, timeout_seconds=10)
            if exit_code != 0:
                msg = f"Could not remove: {path}"
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
