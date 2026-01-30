"""Tests for the file editor tool's file I/O integration."""

import os
from pathlib import Path
import tempfile

import pytest

from agentpool_toolsets.builtin.file_edit.file_edit import edit_file_tool


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("hello world\nfoo bar\n")
        temp_path = f.name

    yield temp_path

    if Path(temp_path).exists():
        Path(temp_path).unlink()


class TestEditFileToolIO:
    """Test edit_file_tool file I/O handling."""

    async def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            await edit_file_tool(
                file_path="/nonexistent/file.py", old_string="old", new_string="new"
            )

    async def test_directory_path_error(self, tmp_path):
        with pytest.raises(ValueError, match="directory, not a file"):
            await edit_file_tool(file_path=str(tmp_path), old_string="old", new_string="new")

    async def test_empty_file_handling(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            result = await edit_file_tool(
                file_path=temp_path, old_string="", new_string="new content"
            )

            assert result["success"] is True
            assert Path(temp_path).read_text("utf-8") == "new content"
        finally:
            Path(temp_path).unlink()

    async def test_unicode_handling(self):
        content = "# Test with émojis 🐍 and ñoño"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write(content)
            temp_path = f.name

        try:
            result = await edit_file_tool(
                file_path=temp_path, old_string="émojis 🐍", new_string="Unicode 🎉"
            )

            assert result["success"] is True
            new_content = Path(temp_path).read_text(encoding="utf-8")
            assert "Unicode 🎉" in new_content
        finally:
            Path(temp_path).unlink()

    async def test_relative_path_handling(self, temp_file):
        relative_path = Path(temp_file).name
        original_cwd = Path.cwd()

        try:
            os.chdir(Path(temp_file).parent)
            result = await edit_file_tool(
                file_path=relative_path, old_string="hello", new_string="greet"
            )

            assert result["success"] is True
            assert "greet" in Path(relative_path).read_text("utf-8")
        finally:
            os.chdir(original_cwd)

    async def test_result_contains_diff(self, temp_file):
        result = await edit_file_tool(file_path=temp_file, old_string="hello", new_string="goodbye")

        assert result["success"] is True
        assert result["diff"]  # Has diff output
        assert result["file_path"] == temp_file
        assert result["lines_changed"] >= 1
