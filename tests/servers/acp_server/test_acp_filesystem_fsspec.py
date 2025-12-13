"""Test ACP filesystem fsspec operations with headless client.

This test validates that the ACP filesystem can be used as a real fsspec
filesystem to perform operations on the client's filesystem using the
headless client with real operations.
"""

from pathlib import Path
import tempfile

import pytest

from acp.client.implementations import HeadlessACPClient
from acp.filesystem import ACPFileSystem


async def test_acp_filesystem_fsspec_operations():
    """Test ACP filesystem with real fsspec operations using headless client."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)

        # Create test files
        (temp_path / "file1.txt").write_text("Content of file 1")
        (temp_path / "file2.md").write_text("# File 2")

        subdir = temp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested content")

        # Create headless client with real operations
        client = HeadlessACPClient(
            working_dir=temp_path,
            allow_file_operations=True,
            auto_grant_permissions=True,
        )

        try:
            session_id = "test_session"
            fs = ACPFileSystem(client=client, session_id=session_id)

            # Test reading files
            content = await fs._cat_file(str(temp_path / "file1.txt"))
            assert content == b"Content of file 1"

            # Test writing files
            new_file = temp_path / "written.txt"
            await fs._put_file(str(new_file), "New content")
            assert new_file.exists()
            assert new_file.read_text("utf-8") == "New content"

            # Test directory listing
            files = await fs._ls(str(temp_path), detail=True)
            # fsspec convention: 'name' contains full path, extract basename for comparison
            file_names = [Path(f["name"]).name for f in files]

            assert "file1.txt" in file_names
            assert "file2.md" in file_names
            assert "subdir" in file_names
            assert "written.txt" in file_names

            # Test file existence
            assert await fs._exists(str(temp_path / "file1.txt")) is True
            assert await fs._exists(str(temp_path / "nonexistent.txt")) is False

            # Test isfile/isdir
            assert await fs._isfile(str(temp_path / "file1.txt")) is True
            assert await fs._isdir(str(temp_path / "subdir")) is True
            assert await fs._isfile(str(temp_path / "subdir")) is False

            # Test file info
            info = await fs._info(str(temp_path / "file1.txt"))
            # fsspec convention: 'name' contains full path
            assert Path(info["name"]).name == "file1.txt"
            assert info["type"] == "file"
            assert info["size"] == 17  # Length of "Content of file 1"  # noqa: PLR2004

        finally:
            await client.cleanup()


#
# async def test_acp_filesystem_with_nested_directories():
#     """Test ACP filesystem operations with nested directory structures."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         temp_path = Path(tmpdir)

#         # Create nested structure
#         (temp_path / "root.txt").write_text("Root file")

#         level1 = temp_path / "level1"
#         level1.mkdir()
#         (level1 / "l1_file.txt").write_text("Level 1 file")

#         level2 = level1 / "level2"
#         level2.mkdir()
#         (level2 / "l2_file.txt").write_text("Level 2 file")

#         client = HeadlessACPClient(
#             working_dir=temp_path,
#             allow_file_operations=True,
#             auto_grant_permissions=True,
#         )

#         try:
#             fs = ACPFileSystem(client=client, session_id="test_session")

#             # Test reading from nested directories
#             content = await fs._cat_file(str(level2 / "l2_file.txt"))
#             assert content == b"Level 2 file"

#             # Test listing nested directories
#             level1_files = await fs._ls(str(level1), detail=True)
#             level1_names = [f["name"] for f in level1_files]
#             assert "l1_file.txt" in level1_names
#             assert "level2" in level1_names

#             # Test creating files in nested dirs
#             new_nested_file = level2 / "created.txt"
#             await fs._put_file(str(new_nested_file), "Created in nested dir")
#             assert new_nested_file.exists()
#             assert new_nested_file.read_text("utf-8") == "Created in nested dir"

#             print("✅ Nested directory operations working!")

#         finally:
#             await client.cleanup()


#
# async def test_acp_filesystem_error_conditions():
#     """Test ACP filesystem error handling."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         temp_path = Path(tmpdir)

#         client = HeadlessACPClient(
#             working_dir=temp_path,
#             allow_file_operations=True,
#             auto_grant_permissions=True,
#         )

#         try:
#             fs = ACPFileSystem(client=client, session_id="test_session")

#             # Test reading non-existent file
#             with pytest.raises(FileNotFoundError):
#                 await fs._cat_file(str(temp_path / "nonexistent.txt"))

#             # Test file existence for non-existent file
#             assert await fs._exists(str(temp_path / "nonexistent.txt")) is False

#             # Test listing non-existent directory
#             with pytest.raises(FileNotFoundError):
#                 await fs._ls(str(temp_path / "nonexistent_dir"))

#             print("✅ Error handling working correctly!")

#         finally:
#             await client.cleanup()


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
