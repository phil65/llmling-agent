from __future__ import annotations

from typing import TYPE_CHECKING, Any

import fsspec
from fsspec import AbstractFileSystem
from upath import UPath
from upathtools import UnionFileSystem

from llmling_agent.log import get_logger
from llmling_agent.utils.baseregistry import BaseRegistry


if TYPE_CHECKING:
    from llmling_agent_config.resources import ResourceConfig

logger = get_logger(__name__)


class VFSRegistry(BaseRegistry[str, AbstractFileSystem]):
    """Registry for virtual filesystems."""

    def register(self, name: str, item: Any, replace: bool = False):
        """Register a new resource."""
        logger.debug("registering resource.", name=name, type=item.__class__.__name__)
        fsspec.register_implementation(name, item.__class__, clobber=True)
        super().register(name, item, replace=replace)

    def _validate_item(self, item: Any) -> AbstractFileSystem:
        if not isinstance(item, AbstractFileSystem):
            msg = f"Expected AbstractFileSystem, got {type(item)}"
            raise self._error_class(msg)
        return item

    def register_from_config(
        self, name: str, config: ResourceConfig
    ) -> AbstractFileSystem:
        """Register a new resource from config."""
        from llmling_agent_config.resources import SourceResourceConfig

        match config:
            case SourceResourceConfig(uri=uri):
                # Extract base path from URI if present
                fs, _url_path = fsspec.url_to_fs(uri, **config.storage_options)
                if config.path:
                    fs = fsspec.filesystem("dir", fs=fs, path=config.path)
                if config.cached:
                    fs = fsspec.filesystem("cached", fs=fs)

                self.register(name, fs)
                return fs
            case _:
                msg = f"Unknown resource config type: {type(config)}"
                raise ValueError(msg)

    def get_fs(self) -> UnionFileSystem:
        """Get unified filesystem view of all resources."""
        filesystems = dict(self.items())
        return UnionFileSystem(filesystems)

    def get_upath(self, resource_name: str | None = None) -> UPath:
        """Get a UPath object for accessing a resource."""
        return UPath("union://")
        # path._fs_cached = self.get_fs()
        # return path

    async def get_content(
        self,
        path: str,
        encoding: str = "utf-8",
        recursive: bool = True,
        exclude: list[str] | None = None,
        max_depth: int | None = None,
    ) -> str:
        """Get content from a resource as text.

        Args:
            path: Path to read, either:
                - resource (whole resource)
                - resource:// (whole resource)
                - resource://file.txt (single file)
                - resource://folder (directory)
            encoding: Text encoding for binary content
            recursive: For directories, whether to read recursively
            exclude: For directories, patterns to exclude
            max_depth: For directories, maximum depth to read

        Returns:
            For files: file content
            For directories: concatenated content of all files
        """
        from upathtools import read_folder, read_path

        if "/" not in path:
            path = f"{path}://"
        _resource, _ = path.split("://", 1)
        fs = self.get_fs()
        is_dir = await fs._isdir(path)

        if is_dir:
            content_dict = await read_folder(
                path,
                mode="rt",
                encoding=encoding,
                recursive=recursive,
                exclude=exclude,
                max_depth=max_depth,
            )
            # Combine all files with headers
            sections = []
            for rel_path, content in sorted(content_dict.items()):
                sections.extend([f"--- {rel_path} ---", content, ""])
            return "\n".join(sections)
        return await read_path(path, encoding=encoding)

    async def query(
        self,
        path: str,
        pattern: str = "**/*",
        *,
        recursive: bool = True,
        include_dirs: bool = False,
        exclude: list[str] | None = None,
        max_depth: int | None = None,
    ) -> list[str]:
        """Query contents of a resource or subfolder.

        Args:
            path: Path to query, either:
                - resource (queries whole resource)
                - resource:// (queries whole resource)
                - resource://subfolder (queries specific folder)
            pattern: Glob pattern to match files against
            recursive: Whether to search subdirectories
            include_dirs: Whether to include directories in results
            exclude: List of patterns to exclude
            max_depth: Maximum directory depth for recursive search

        Example:
            # Query whole resource (all equivalent)
            files = await registry.query("docs")
            files = await registry.query("docs://")

            # Query specific subfolder
            files = await registry.query("docs://guides", pattern="*.md")
        """
        from upathtools import list_files

        if "/" not in path:
            # Simple resource name - add protocol
            path = f"{path}://"
        resource, _ = path.split("://", 1)
        if resource not in self:
            msg = f"Resource not found: {resource}"
            raise ValueError(msg)

        files = await list_files(
            path,
            pattern=pattern,
            recursive=recursive,
            include_dirs=include_dirs,
            exclude=exclude,
            max_depth=max_depth,
        )
        return [str(p) for p in files]


if __name__ == "__main__":
    registry = VFSRegistry()
    p = registry.get_upath()
    print(p, type(p))
