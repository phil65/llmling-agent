"""FSSpec filesystem toolset implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    import fsspec


class FSSpecTools(ResourceProvider):
    """Provider for fsspec filesystem tools."""

    def __init__(self, filesystem: fsspec.AbstractFileSystem, name: str = "fsspec"):
        """Initialize with an fsspec filesystem.

        Args:
            filesystem: The fsspec filesystem instance to operate on
            name: Name for this toolset provider
        """
        super().__init__(name=name)
        self.fs = filesystem
        self._tools: list[Tool] | None = None

    async def get_tools(self) -> list[Tool]:
        """Get filesystem tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            Tool.from_callable(
                self._list_directory,
                name_override="fs_list",
                description_override="List contents of a directory",
            ),
            Tool.from_callable(
                self._read_file,
                name_override="fs_read",
                description_override="Read contents of a file",
            ),
            Tool.from_callable(
                self._write_file,
                name_override="fs_write",
                description_override="Write content to a file",
            ),
            Tool.from_callable(
                self._delete_path,
                name_override="fs_delete",
                description_override="Delete a file or directory",
            ),
        ]
        return self._tools

    def _list_directory(self, path: str) -> dict[str, Any]:
        """List contents of a directory.

        Args:
            path: Directory path to list

        Returns:
            Dictionary with directory contents and metadata
        """
        try:
            # Get detailed file information
            entries = self.fs.ls(path, detail=True)

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

            return {
                "path": path,
                "directories": directories,
                "files": files,
                "total_items": len(directories) + len(files),
            }

        except (OSError, ValueError) as e:
            return {"error": f"Failed to list directory {path}: {e}"}

    def _read_file(self, path: str, encoding: str = "utf-8") -> dict[str, Any]:
        """Read contents of a file.

        Args:
            path: File path to read
            encoding: Text encoding to use (default: utf-8)

        Returns:
            Dictionary with file contents or error info
        """
        try:
            if encoding == "binary":
                content = self.fs.cat_file(path)
                return {
                    "path": path,
                    "content": content.hex()
                    if isinstance(content, bytes)
                    else str(content),
                    "size": len(content) if content else 0,
                    "encoding": "binary",
                }
            with self.fs.open(path, "r", encoding=encoding) as f:
                content = f.read()

            return {
                "path": path,
                "content": content,
                "size": len(content),
                "encoding": encoding,
            }

        except (OSError, ValueError) as e:
            return {"error": f"Failed to read file {path}: {e}"}

    def _write_file(
        self, path: str, content: str, encoding: str = "utf-8", mode: str = "w"
    ) -> dict[str, Any]:
        """Write content to a file.

        Args:
            path: File path to write to
            content: Content to write
            encoding: Text encoding to use (default: utf-8)
            mode: Write mode ('w' for overwrite, 'a' for append)

        Returns:
            Dictionary with operation result
        """
        try:
            # Validate mode
            if mode not in ("w", "a"):
                return {
                    "error": f"Invalid mode '{mode}'. Use 'w' (write) or 'a' (append)"
                }

            with self.fs.open(path, mode, encoding=encoding) as f:
                f.write(content)

            # Try to get file size after writing
            try:
                info = self.fs.info(path)
                size = info.get("size", len(content))
            except (OSError, KeyError):
                size = len(content)

            return {
                "path": path,
                "bytes_written": len(content.encode(encoding)),
                "size": size,
                "mode": mode,
                "encoding": encoding,
            }

        except (OSError, ValueError) as e:
            return {"error": f"Failed to write file {path}: {e}"}

    def _delete_path(self, path: str, recursive: bool = False) -> dict[str, Any]:
        """Delete a file or directory.

        Args:
            path: Path to delete
            recursive: Whether to delete directories recursively

        Returns:
            Dictionary with operation result
        """
        try:
            # Check if path exists and get its type
            try:
                info = self.fs.info(path)
                path_type = info.get("type", "unknown")
            except FileNotFoundError:
                return {"error": f"Path does not exist: {path}"}
            except (OSError, ValueError) as e:
                return {"error": f"Could not check path {path}: {e}"}

            if path_type == "directory":
                if not recursive:
                    # Check if directory is empty
                    try:
                        contents = self.fs.ls(path)
                        if contents:
                            return {
                                "error": f"Directory {path} is not empty. "
                                f"Use recursive=True to delete non-empty directories"
                            }
                    except (OSError, ValueError):
                        pass  # Continue with deletion attempt

                self.fs.delete(path, recursive=recursive)
            else:
                # It's a file
                self.fs.delete(path)

        except (OSError, ValueError) as e:
            return {"error": f"Failed to delete {path}: {e}"}
        else:
            return {
                "path": path,
                "deleted": True,
                "type": path_type,
                "recursive": recursive,
            }


if __name__ == "__main__":
    import asyncio

    async def main():
        import fsspec

        from llmling_agent import Agent

        # Create a local filesystem for demo
        fs = fsspec.filesystem("file")
        tools = FSSpecTools(fs, name="local_fs")

        agent = Agent(model="gpt-5-nano")
        agent.tools.add_provider(tools)

        result = await agent.run("List the tools available for filesystem operations")
        print(result)

    asyncio.run(main())
