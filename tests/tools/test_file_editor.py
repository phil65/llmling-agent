"""Tests for the sophisticated file editor tool."""

import os
from pathlib import Path
import tempfile

import pytest

from llmling_agent_toolsets.builtin.file_edit.file_edit import (
    _block_anchor_replacer,
    _context_aware_replacer,
    _escape_normalized_replacer,
    _indentation_flexible_replacer,
    _levenshtein_distance,
    _line_trimmed_replacer,
    _multi_occurrence_replacer,
    _simple_replacer,
    _whitespace_normalized_replacer,
    edit_file_tool,
    replace_content,
)


class TestReplacers:
    """Test individual replacer strategies."""

    def test_simple_replacer(self):
        content = "Hello world, this is a test"
        matches = list(_simple_replacer(content, "world"))
        assert matches == ["world"]

        matches = list(_simple_replacer(content, "missing"))
        assert matches == []

    def test_line_trimmed_replacer(self):
        content = """  line 1
    line 2
line 3"""
        find = """line 1
  line 2
line 3"""

        matches = list(_line_trimmed_replacer(content, find))
        assert len(matches) == 1
        assert "line 1" in matches[0]

    def test_whitespace_normalized_replacer(self):
        content = "This   has    multiple     spaces"
        find = "This has multiple spaces"

        matches = list(_whitespace_normalized_replacer(content, find))
        assert len(matches) == 1

    def test_indentation_flexible_replacer(self):
        content = """    def function():
        print("hello")
        return True"""

        find = """def function():
    print("hello")
    return True"""

        matches = list(_indentation_flexible_replacer(content, find))
        assert len(matches) == 1

    def test_escape_normalized_replacer(self):
        content = 'print("Hello\\nWorld")'
        find = 'print("Hello\nWorld")'

        matches = list(_escape_normalized_replacer(content, find))
        assert len(matches) == 1

    def test_multi_occurrence_replacer(self):
        content = "test test test"
        matches = list(_multi_occurrence_replacer(content, "test"))
        assert len(matches) == 3  # noqa: PLR2004


class TestLevenshteinDistance:
    """Test Levenshtein distance implementation."""

    def test_empty_strings(self):
        assert _levenshtein_distance("", "") == 0
        assert _levenshtein_distance("abc", "") == 3  # noqa: PLR2004
        assert _levenshtein_distance("", "abc") == 3  # noqa: PLR2004

    def test_identical_strings(self):
        assert _levenshtein_distance("hello", "hello") == 0

    def test_single_operations(self):
        assert _levenshtein_distance("cat", "bat") == 1  # substitution
        assert _levenshtein_distance("cat", "cats") == 1  # insertion
        assert _levenshtein_distance("cats", "cat") == 1  # deletion

    def test_complex_cases(self):
        assert _levenshtein_distance("kitten", "sitting") == 3  # noqa: PLR2004


class TestReplaceContent:
    """Test the main replace_content function."""

    def test_simple_replacement(self):
        content = "Hello world"
        result = replace_content(content, "world", "Python")
        assert result == "Hello Python"

    def test_multiple_occurrences_error(self):
        content = "test test test"
        with pytest.raises(ValueError, match="multiple times"):
            replace_content(content, "test", "replace")

    def test_replace_all(self):
        content = "test test test"
        result = replace_content(content, "test", "replace", replace_all=True)
        assert result == "replace replace replace"

    def test_not_found_error(self):
        content = "Hello world"
        with pytest.raises(ValueError, match="not found"):
            replace_content(content, "missing", "replacement")

    def test_same_strings_error(self):
        with pytest.raises(ValueError, match="must be different"):
            replace_content("content", "same", "same")

    def test_multiline_replacement(self):
        content = """def old_function():
    print("old")
    return False

def other_function():
    pass"""

        old_string = """def old_function():
    print("old")
    return False"""

        new_string = """def new_function():
    print("new")
    return True"""

        result = replace_content(content, old_string, new_string)
        assert "def new_function():" in result
        assert "def other_function():" in result

    def test_whitespace_flexibility(self):
        content = "if   condition   :\n    do_something()"
        old_string = "if condition:\n    do_something()"
        new_string = "if new_condition:\n    do_something_else()"

        result = replace_content(content, old_string, new_string)
        assert "new_condition" in result


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
        f.write("""def hello():
    print("Hello, World!")
    return True

def calculate(a, b):
    result = a + b
    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    hello()
    calculate(5, 3)
""")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if Path(temp_path).exists():
        Path(temp_path).unlink()


class TestEditFileTool:
    """Test the main edit_file_tool function."""

    async def test_simple_edit(self, temp_file):
        result = await edit_file_tool(
            file_path=temp_file,
            old_string='print("Hello, World!")',
            new_string='print("Hello, Python!")',
        )

        assert result["success"] is True
        assert "Hello, Python!" in Path(temp_file).read_text()
        assert result["diff"]
        assert result["lines_changed"] >= 1

    async def test_multiline_edit(self, temp_file):
        old_function = """def calculate(a, b):
    result = a + b
    print(f"Result: {result}")
    return result"""

        new_function = """def calculate(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    result = a + b
    print(f"Sum: {result}")
    return result"""

        result = await edit_file_tool(
            file_path=temp_file, old_string=old_function, new_string=new_function
        )

        assert result["success"] is True
        content = Path(temp_file).read_text()
        assert "TypeError" in content
        assert "Sum:" in content

    async def test_replace_all(self, temp_file):
        result = await edit_file_tool(
            file_path=temp_file, old_string="result", new_string="total", replace_all=True
        )

        assert result["success"] is True
        content = Path(temp_file).read_text()
        assert "result" not in content
        assert "total" in content

    async def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            await edit_file_tool(
                file_path="/nonexistent/file.py", old_string="old", new_string="new"
            )

    async def test_directory_path_error(self, tmp_path):
        with pytest.raises(ValueError, match="directory, not a file"):
            await edit_file_tool(file_path=str(tmp_path), old_string="old", new_string="new")

    async def test_same_strings_error(self, temp_file):
        with pytest.raises(ValueError, match="must be different"):
            await edit_file_tool(file_path=temp_file, old_string="same", new_string="same")

    async def test_string_not_found(self, temp_file):
        with pytest.raises(ValueError, match="not found"):
            await edit_file_tool(
                file_path=temp_file,
                old_string="nonexistent string",
                new_string="replacement",
            )

    async def test_empty_file_handling(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            result = await edit_file_tool(
                file_path=temp_path, old_string="", new_string="new content"
            )

            assert result["success"] is True
            assert Path(temp_path).read_text() == "new content"
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
        # Test with relative path
        relative_path = Path(temp_file).name

        # Change to the directory containing the file
        original_cwd = Path.cwd()
        try:
            os.chdir(Path(temp_file).parent)

            result = await edit_file_tool(
                file_path=relative_path, old_string="hello()", new_string="greet()"
            )

            assert result["success"] is True
            assert "greet()" in Path(relative_path).read_text()
        finally:
            os.chdir(original_cwd)


class TestBlockAnchorReplacer:
    """Test the sophisticated block anchor replacer."""

    def test_simple_block_match(self):
        content = """def function():
    print("hello")
    return True

def other():
    pass"""

        find = """def function():
    print("hello")
    return True"""

        matches = list(_block_anchor_replacer(content, find))
        assert len(matches) == 1
        assert "def function():" in matches[0]

    def test_no_match_insufficient_lines(self):
        content = "line1\nline2"
        find = "line1\nline2"  # Only 2 lines, needs at least 3

        matches = list(_block_anchor_replacer(content, find))
        assert len(matches) == 0

    def test_similarity_scoring(self):
        content = """def function():
    print("hello world")
    x = 1
    return True"""

        find = """def function():
    print("hello python")  # Different middle line
    return True"""

        matches = list(_block_anchor_replacer(content, find))
        # Should still match due to anchor lines and reasonable similarity
        assert len(matches) == 1


class TestContextAwareReplacer:
    """Test the context-aware replacer."""

    def test_context_match(self):
        content = """# Start marker
some code here
more code
# End marker

other content"""

        find = """# Start marker
some code here
more code
# End marker"""

        matches = list(_context_aware_replacer(content, find))
        assert len(matches) == 1
        assert "Start marker" in matches[0]
        assert "End marker" in matches[0]

    def test_insufficient_context(self):
        content = "line1\nline2"
        find = "line1\nline2"

        matches = list(_context_aware_replacer(content, find))
        assert len(matches) == 0  # Need at least 3 lines
