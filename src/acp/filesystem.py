"""Filesystem implementation for ACP (Agent Communication Protocol) sessions."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path
import shlex
from typing import TYPE_CHECKING, Any, Literal, Required, overload

from anyenv import get_os_command_provider
from fsspec.asyn import sync_wrapper
from fsspec.spec import AbstractBufferedFile
from upathtools.filesystems.base import BaseAsyncFileSystem, BaseUPath, FileInfo

from acp.acp_requests import ACPRequests
from acp.notifications import ACPNotifications


class AcpInfo(FileInfo, total=False):
    """Info dict for ACP filesystem paths."""

    path: str
    islink: bool
    timestamp: str | None
    permissions: str | None
    size: Required[int]


if TYPE_CHECKING:
    from acp.client.protocol import Client


logger = logging.getLogger(__name__)

# MIME types that should be treated as text
TEXT_MIME_PREFIXES = ("text/", "application/json", "application/xml", "application/javascript")


def _is_text_mime(mime_type: str | None) -> bool:
    """Check if a MIME type represents text content."""
    if mime_type is None:
        return True  # Default to text for unknown types
    return any(mime_type.startswith(prefix) for prefix in TEXT_MIME_PREFIXES)


class ACPFile(AbstractBufferedFile):  # type: ignore[misc]
    """File-like object for ACP filesystem operations."""

    def __init__(self, fs: ACPFileSystem, path: str, mode: str = "rb", **kwargs: Any) -> None:
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


class ACPPath(BaseUPath[AcpInfo]):
    """Path for ACP filesystem."""

    __slots__ = ()


class ACPFileSystem(BaseAsyncFileSystem[ACPPath, AcpInfo]):
    """Async filesystem for ACP sessions."""

    protocol = "acp"
    sep = "/"
    upath_cls = ACPPath

    def __init__(
        self,
        client: Client,
        session_id: str,
        *,
        use_cli_find: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize ACP filesystem.

        Args:
            client: ACP client for operations
            session_id: Session identifier
            use_cli_find: Use CLI find command for _find/_glob operations.
                When True (default), uses a single `find` command for recursive
                file discovery, which is much more efficient over the protocol
                barrier than walking the tree with multiple ls calls.
            **kwargs: Additional filesystem options
        """
        super().__init__(**kwargs)
        self.client = client
        self.session_id = session_id
        self.requests = ACPRequests(client, session_id)
        self.notifications = ACPNotifications(client, session_id)
        self.command_provider = get_os_command_provider()
        self.use_cli_find = use_cli_find

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
                raise ValueError("Empty command string")  # noqa: TRY301
            return parts[0], parts[1:]
        except ValueError as e:
            # Fallback for problematic shell strings
            parts = command_str.split()
            if not parts:
                raise ValueError("Empty command string") from e
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

        mime_type = mimetypes.guess_type(path)[0]

        if _is_text_mime(mime_type):
            # Text file - use read_text_file directly
            try:
                content = await self.requests.read_text_file(path)
                return content.encode("utf-8")
            except Exception as e:
                raise FileNotFoundError(f"Could not read file {path}: {e}") from e

        # Binary file - use base64 encoding via terminal command
        try:
            b64_cmd = self.command_provider.get_command("base64_encode")
            cmd_str = b64_cmd.create_command(path)
            cmd, args = self._parse_command(cmd_str)
            output, exit_code = await self.requests.run_command(cmd, args=args, timeout_seconds=30)

            if exit_code != 0:
                msg = f"Could not read binary file {path}: {output}"
                raise FileNotFoundError(msg)  # noqa: TRY301

            return b64_cmd.parse_command(output)
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

    async def _pipe_file(self, path: str, data: bytes, **kwargs: Any) -> None:
        """Write bytes directly to a file path.

        This is the fsspec standard method for writing data to a file.
        Wraps _put_file for compatibility.

        Args:
            path: File path to write
            data: Bytes to write
            **kwargs: Additional options
        """
        await self._put_file(path, data, **kwargs)

    pipe_file = sync_wrapper(_pipe_file)

    @overload
    async def _ls(self, path: str, detail: Literal[True] = ..., **kwargs: Any) -> list[AcpInfo]: ...

    @overload
    async def _ls(self, path: str, detail: Literal[False], **kwargs: Any) -> list[str]: ...

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> list[AcpInfo] | list[str]:
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
            cmd, args = self._parse_command(ls_cmd)
            output, exit_code = await self.requests.run_command(cmd, args=args, timeout_seconds=10)

            if exit_code != 0:
                msg = f"Error listing directory {path!r}: {output}"
                raise FileNotFoundError(msg)  # noqa: TRY301

            result = list_cmd.parse_command(output, path)
            if detail:
                return [
                    AcpInfo(
                        name=item.path,  # fsspec expects full path in 'name'
                        path=item.path,
                        type="file" if item.type == "link" else item.type,
                        size=item.size,
                        islink=item.type == "link",
                        timestamp=item.timestamp,
                        permissions=item.permissions,
                    )
                    for item in result
                ]
            return [item.path for item in result]  # Return full paths for consistency

        except Exception as e:
            msg = f"Could not list directory {path}: {e}"
            raise FileNotFoundError(msg) from e

    ls = sync_wrapper(_ls)

    async def _info(self, path: str, **kwargs: Any) -> AcpInfo:
        """Get file information via stat command.

        Args:
            path: File path to get info for
            **kwargs: Additional options

        Returns:
            File information dictionary
        """
        info_cmd = self.command_provider.get_command("file_info")
        stat_cmd = info_cmd.create_command(path)

        try:
            cmd, args = self._parse_command(stat_cmd)
            output, exit_code = await self.requests.run_command(cmd, args=args, timeout_seconds=5)

            if exit_code != 0:
                raise FileNotFoundError(f"File not found: {path}")
            file_info = info_cmd.parse_command(output.strip(), path)
            return AcpInfo(
                name=file_info.path,  # fsspec expects full path in 'name'
                path=file_info.path,
                type="file" if file_info.type == "link" else file_info.type,
                size=file_info.size,
                islink=file_info.type == "link",
                timestamp=file_info.timestamp,
                permissions=file_info.permissions,
            )

        except (OSError, ValueError) as e:
            # Fallback: try to get basic info from ls
            try:
                ls_result = await self._ls(str(Path(path).parent), detail=True)
                filename = Path(path).name

                for item in ls_result:
                    # After _ls fix, item["name"] is now full path
                    if Path(item["name"]).name == filename:
                        return AcpInfo(
                            name=path,  # fsspec expects full path in 'name'
                            path=path,
                            type=item["type"],
                            size=item["size"],
                            islink=item.get("islink", False),
                            timestamp=item.get("timestamp"),
                            permissions=item.get("permissions"),
                        )

                raise FileNotFoundError(f"File not found: {path}")
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
            cmd, args = self._parse_command(test_cmd)
            output, exit_code = await self.requests.run_command(cmd, args=args, timeout_seconds=5)
        except (OSError, ValueError):
            return False
        else:
            return exists_cmd.parse_command(output, exit_code if exit_code is not None else 1)

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
            cmd, args = self._parse_command(test_cmd)
            output, exit_code = await self.requests.run_command(cmd, args=args, timeout_seconds=5)
        except (OSError, ValueError):
            return False
        else:
            return isdir_cmd.parse_command(output, exit_code if exit_code is not None else 1)

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
            cmd, args = self._parse_command(test_cmd)
            output, exit_code = await self.requests.run_command(cmd, args=args, timeout_seconds=5)
        except (OSError, ValueError):
            return False
        else:
            return isfile_cmd.parse_command(output, exit_code if exit_code is not None else 1)

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
            cmd, args = self._parse_command(mkdir_cmd)
            output, exit_code = await self.requests.run_command(cmd, args=args, timeout_seconds=5)
            success = create_cmd.parse_command(output, exit_code if exit_code is not None else 1)
            if not success:
                msg = f"Error creating directory {path}: {output}"
                raise OSError(msg)  # noqa: TRY301
        except Exception as e:
            msg = f"Could not create directory {path}: {e}"
            raise OSError(msg) from e

    makedirs = sync_wrapper(_makedirs)

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        """Copy a file from path1 to path2.

        Uses CLI cp/copy command for efficiency - single round-trip and
        native binary file support without base64 encoding overhead.

        Args:
            path1: Source file path
            path2: Destination file path
            **kwargs: Additional options
        """
        copy_cmd = self.command_provider.get_command("copy_path")
        cmd_str = copy_cmd.create_command(path1, path2, recursive=False)

        try:
            cmd, args = self._parse_command(cmd_str)
            output, exit_code = await self.requests.run_command(cmd, args=args, timeout_seconds=30)
            success = copy_cmd.parse_command(output, exit_code if exit_code is not None else 1)
            if not success:
                msg = f"Error copying {path1} to {path2}: {output}"
                raise OSError(msg)  # noqa: TRY301
        except Exception as e:
            msg = f"Could not copy {path1} to {path2}: {e}"
            raise OSError(msg) from e

    cp_file = sync_wrapper(_cp_file)

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
            cmd, args = self._parse_command(rm_cmd)
            output, exit_code = await self.requests.run_command(cmd, args=args, timeout_seconds=10)
            success = remove_cmd.parse_command(output, exit_code if exit_code is not None else 1)
            if not success:
                msg = f"Error removing {path}: {output}"
                raise OSError(msg)  # noqa: TRY301
        except Exception as e:
            msg = f"Could not remove {path}: {e}"
            raise OSError(msg) from e

    rm = sync_wrapper(_rm)

    async def _find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        **kwargs: Any,
    ) -> list[str] | dict[str, AcpInfo]:
        """Find files recursively.

        When use_cli_find is enabled, uses a single CLI find command instead of
        walking the directory tree with multiple ls calls. This is much more
        efficient over the protocol barrier.

        Args:
            path: Root path to search from
            maxdepth: Maximum depth to descend (None for unlimited)
            withdirs: Include directories in results
            **kwargs: Additional options (detail=True returns dict with info)

        Returns:
            List of paths, or dict mapping paths to info if detail=True
        """
        if not self.use_cli_find:
            # Fall back to default fsspec implementation (walks tree with _ls)
            return await super()._find(path, maxdepth=maxdepth, withdirs=withdirs, **kwargs)  # type: ignore[no-any-return]

        detail = kwargs.pop("detail", False)
        stripped = self._strip_protocol(path)
        search_path = stripped if isinstance(stripped, str) else stripped[0]

        # Determine file_type filter
        file_type: Literal["file", "directory", "all"] = "all" if withdirs else "file"

        find_cmd = self.command_provider.get_command("find")
        cmd_str = find_cmd.create_command(search_path, maxdepth=maxdepth, file_type=file_type)

        try:
            cmd, args = self._parse_command(cmd_str)
            output, exit_code = await self.requests.run_command(cmd, args=args, timeout_seconds=60)

            if exit_code != 0:
                # If find fails, fall back to default implementation
                logger.warning("CLI find failed, falling back to walk: %s", output)
                return await super()._find(path, maxdepth=maxdepth, withdirs=withdirs, **kwargs)  # type: ignore[no-any-return]

            entries = find_cmd.parse_command(output, search_path)

            if detail:
                # Return dict with basic info from find output
                return {
                    entry.path: AcpInfo(
                        name=entry.path,
                        path=entry.path,
                        type=entry.type,
                        size=entry.size,
                        timestamp=entry.timestamp,
                        permissions=entry.permissions,
                    )
                    for entry in entries
                }
            return [entry.path for entry in entries]

        except Exception as e:  # noqa: BLE001
            logger.warning("CLI find error, falling back to walk: %s", e)
            return await super()._find(path, maxdepth=maxdepth, withdirs=withdirs, **kwargs)  # type: ignore[no-any-return]

    find = sync_wrapper(_find)  # pyright: ignore[reportAssignmentType]

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
        match mode:
            case "r":
                mode = "rb"
            case "w":
                mode = "wb"
            case "a":
                mode = "ab"
            case "x":
                mode = "xb"

        return ACPFile(self, path, mode, **kwargs)
