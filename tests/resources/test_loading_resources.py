from __future__ import annotations

import fsspec
import pytest
from upathtools import UnionFileSystem
import yamling

from llmling_agent.models import AgentsManifest


fsspec.register_implementation("union", UnionFileSystem, clobber=True)

MANIFEST_CONFIG = """
resources:
    # Full config with storage options
    docs:
        fs_type: uri
        uri: "memory://"
        storage_options:
            auto_mkdir: true
    # String shorthand
    data: "memory://"
"""

GITHUB_CONFIG = """
resources:
    # String shorthand for github
    mknodes_docs: "github://phil65:mknodes@main"
"""


async def test_vfs_registry():
    """Test VFS registry and unified filesystem access."""
    # Setup
    manifest = AgentsManifest.model_validate(yamling.load_yaml(MANIFEST_CONFIG))
    fs = manifest.vfs_registry.get_fs()

    # Test root listing shows protocols
    root_listing = await fs._ls("/", detail=False)
    assert len(root_listing) == 2  # noqa: PLR2004
    assert {"docs://", "data://"} == set(root_listing)

    # Test write and read operations
    test_content = b"docs content"
    await fs._pipe_file("docs://test.txt", test_content)
    assert await fs._cat_file("docs://test.txt") == test_content

    # Test directory listing
    docs_listing = await fs._ls("docs://", detail=False)
    assert "docs://test.txt" in docs_listing

    # Test file info
    info = await fs._info("docs://test.txt")
    assert info["type"] == "file"
    assert info["name"] == "docs://test.txt"
    assert info["size"] == len(test_content)


async def test_resource_path():
    """Test UPath-based resource access."""
    manifest = AgentsManifest.model_validate(yamling.load_yaml(MANIFEST_CONFIG))
    # TODO: which behviour do we wnt here?
    path = manifest.vfs_registry.get_upath("docs")
    print(list(path.iterdir()))
    assert str(path) == "docs"
    assert path.fs == manifest.vfs_registry.get_fs()


# async def test_github_path():
#     """Test UPath-based resource access."""
#     manifest = AgentsManifest.model_validate(yamling.load_yaml(GITHUB_CONFIG))
#     # TODO: which behviour do we wnt here?
#     path = manifest.vfs_registry.get_upath("mknodes_docs")
#     print(list(path.iterdir()))
#     assert str(path) == "mknodes_dsocs"

#     # Test write/read
#     test_content = "test content"
#     test_file = path / "test.txt"
#     test_file.write_text(test_content)
#     assert test_file.read_text(encoding="utf-8") == test_content

#     # Test exists
#     assert test_file.exists()
#     assert not (path / "nonexistent.txt").exists()

#     # Test glob
#     (path / "dir" / "file1.txt").write_text("content 1")
#     (path / "dir" / "file2.txt").write_text("content 2")
#     files = [str(p) for p in path.glob("dir/*.txt")]
#     assert len(files) == 2
#     assert "docs://dir/file1.txt" in files
#     assert "docs://dir/file2.txt" in files


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
