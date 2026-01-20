"""Tests for Claude Code SDK to OpenCode metadata conversion."""

from __future__ import annotations

from agentpool.agents.claude_code_agent.converters import (
    convert_tool_result_to_opencode_metadata,
)


class TestConvertToolResultToOpencodeMetadata:
    """Tests for convert_tool_result_to_opencode_metadata function."""

    def test_write_tool_result(self) -> None:
        """Test conversion of Write tool result."""
        sdk_result = {
            "type": "create",
            "filePath": "/tmp/test/hello.py",
            "content": "def hello():\n    print('Hello')\n",
            "structuredPatch": [],
            "originalFile": None,
        }

        metadata = convert_tool_result_to_opencode_metadata("Write", sdk_result)

        assert metadata is not None
        assert metadata["filePath"] == "/tmp/test/hello.py"
        assert metadata["content"] == "def hello():\n    print('Hello')\n"

    def test_edit_tool_result(self) -> None:
        """Test conversion of Edit tool result."""
        sdk_result = {
            "filePath": "/tmp/test/hello.py",
            "oldString": "def hello():\n    print('Hello')",
            "newString": 'def hello():\n    """Say hello."""\n    print(\'Hello\')',
            "originalFile": "def hello():\n    print('Hello')\n",
            "structuredPatch": [
                {
                    "oldStart": 1,
                    "oldLines": 2,
                    "newStart": 1,
                    "newLines": 3,
                    "lines": [
                        " def hello():",
                        '+    """Say hello."""',
                        "     print('Hello')",
                    ],
                }
            ],
            "userModified": False,
            "replaceAll": False,
        }

        metadata = convert_tool_result_to_opencode_metadata("Edit", sdk_result)

        assert metadata is not None
        assert "diff" in metadata
        assert "filediff" in metadata

        filediff = metadata["filediff"]
        assert filediff["file"] == "/tmp/test/hello.py"
        assert filediff["before"] == "def hello():\n    print('Hello')\n"
        assert '"""Say hello."""' in filediff["after"]
        assert filediff["additions"] == 1
        assert filediff["deletions"] == 0

    def test_read_tool_result(self) -> None:
        """Test conversion of Read tool result."""
        sdk_result = {
            "type": "text",
            "file": {
                "filePath": "/tmp/test/hello.py",
                "content": "def hello():\n    print('Hello')\n",
                "numLines": 2,
                "startLine": 1,
                "totalLines": 2,
            },
        }

        metadata = convert_tool_result_to_opencode_metadata("Read", sdk_result)

        assert metadata is not None
        assert metadata["filePath"] == "/tmp/test/hello.py"
        assert metadata["content"] == "def hello():\n    print('Hello')\n"
        assert metadata["numLines"] == 2
        assert metadata["startLine"] == 1
        assert metadata["totalLines"] == 2

    def test_bash_tool_result_success(self) -> None:
        """Test conversion of Bash tool success result."""
        sdk_result = {
            "stdout": "Hello from bash",
            "stderr": "",
            "interrupted": False,
            "isImage": False,
        }
        tool_input = {
            "command": "echo 'Hello from bash'",
            "description": "Print greeting",
        }

        metadata = convert_tool_result_to_opencode_metadata("Bash", sdk_result, tool_input)

        assert metadata is not None
        assert metadata["output"] == "Hello from bash"
        assert metadata["exit"] == 0
        assert metadata["description"] == "Print greeting"

    def test_bash_tool_result_with_stderr(self) -> None:
        """Test Bash result with both stdout and stderr."""
        sdk_result = {
            "stdout": "output line",
            "stderr": "warning: something",
            "interrupted": False,
            "isImage": False,
        }
        tool_input = {"command": "some_command"}

        metadata = convert_tool_result_to_opencode_metadata("Bash", sdk_result, tool_input)

        assert metadata is not None
        assert "output line" in metadata["output"]
        assert "warning: something" in metadata["output"]
        assert metadata["description"] == "some_command"

    def test_bash_tool_result_interrupted(self) -> None:
        """Test Bash result when command was interrupted."""
        sdk_result = {
            "stdout": "partial output",
            "stderr": "",
            "interrupted": True,
            "isImage": False,
        }

        metadata = convert_tool_result_to_opencode_metadata("Bash", sdk_result)

        assert metadata is not None
        assert metadata["exit"] is None  # Interrupted commands have no clean exit

    def test_bash_tool_result_error(self) -> None:
        """Test that Bash error results (strings) return None."""
        sdk_result = "Error: Exit code 1\ncat: /nonexistent: No such file"

        metadata = convert_tool_result_to_opencode_metadata("Bash", sdk_result)

        assert metadata is None

    def test_none_result(self) -> None:
        """Test that None tool_use_result returns None."""
        metadata = convert_tool_result_to_opencode_metadata("Write", None)

        assert metadata is None

    def test_unknown_tool(self) -> None:
        """Test that unknown tools return None."""
        sdk_result = {"some": "data"}

        metadata = convert_tool_result_to_opencode_metadata("UnknownTool", sdk_result)

        assert metadata is None

    def test_case_insensitive_tool_name(self) -> None:
        """Test that tool name matching is case-insensitive."""
        sdk_result = {
            "type": "create",
            "filePath": "/tmp/test.py",
            "content": "# test",
            "structuredPatch": [],
            "originalFile": None,
        }

        # All these should work
        assert convert_tool_result_to_opencode_metadata("write", sdk_result) is not None
        assert convert_tool_result_to_opencode_metadata("WRITE", sdk_result) is not None
        assert convert_tool_result_to_opencode_metadata("Write", sdk_result) is not None

    def test_edit_with_missing_original_file(self) -> None:
        """Test Edit conversion when originalFile is None."""
        sdk_result = {
            "filePath": "/tmp/test.py",
            "oldString": "old",
            "newString": "new",
            "originalFile": None,  # Can happen in some edge cases
            "structuredPatch": [],
            "userModified": False,
            "replaceAll": False,
        }

        metadata = convert_tool_result_to_opencode_metadata("Edit", sdk_result)

        assert metadata is not None
        assert metadata["filediff"]["before"] == ""
        # after is empty string when we can't compute it without originalFile
        assert metadata["filediff"]["after"] == ""

    def test_write_with_missing_fields(self) -> None:
        """Test Write conversion with missing fields returns None."""
        # Missing filePath
        sdk_result = {"content": "test"}
        metadata = convert_tool_result_to_opencode_metadata("Write", sdk_result)
        assert metadata is None

        # Missing content
        sdk_result = {"filePath": "/tmp/test.py"}
        metadata = convert_tool_result_to_opencode_metadata("Write", sdk_result)
        assert metadata is None

    def test_read_with_missing_file_field(self) -> None:
        """Test Read conversion with missing file field returns None."""
        sdk_result = {"type": "text"}  # Missing "file" object

        metadata = convert_tool_result_to_opencode_metadata("Read", sdk_result)

        assert metadata is None


class TestTodoWriteConversion:
    """Tests for TodoWrite tool result conversion."""

    def test_todowrite_basic(self) -> None:
        """Test basic TodoWrite conversion."""
        sdk_result = {
            "oldTodos": [],
            "newTodos": [
                {
                    "content": "Fix critical bug",
                    "status": "pending",
                    "activeForm": "Fixing critical bug",
                },
                {
                    "content": "Review code",
                    "status": "in_progress",
                    "activeForm": "Reviewing code",
                },
            ],
        }

        metadata = convert_tool_result_to_opencode_metadata("TodoWrite", sdk_result)

        assert metadata is not None
        assert "todos" in metadata
        assert len(metadata["todos"]) == 2

        # Check first todo
        todo1 = metadata["todos"][0]
        assert todo1["content"] == "Fix critical bug"
        assert todo1["status"] == "pending"
        assert "id" in todo1
        assert "priority" in todo1

        # Check second todo
        todo2 = metadata["todos"][1]
        assert todo2["content"] == "Review code"
        assert todo2["status"] == "in_progress"

    def test_todowrite_priority_keywords(self) -> None:
        """Test that priority is inferred from keywords."""
        sdk_result = {
            "oldTodos": [],
            "newTodos": [
                {"content": "Critical security fix", "status": "pending"},
                {"content": "Nice to have feature", "status": "pending"},
                {"content": "Regular task", "status": "pending"},
            ],
        }

        metadata = convert_tool_result_to_opencode_metadata("TodoWrite", sdk_result)

        assert metadata is not None
        todos = metadata["todos"]

        # "Critical" keyword -> high
        assert todos[0]["priority"] == "high"
        # "Nice to have" keyword -> low
        assert todos[1]["priority"] == "low"
        # No keyword, first position -> high (position-based)
        # Actually third position out of 3, so it would be low
        # But "Regular task" has no keywords, so position-based: index 2/2 = 1.0 > 0.67 -> low
        assert todos[2]["priority"] == "low"

    def test_todowrite_empty_todos(self) -> None:
        """Test TodoWrite with empty newTodos returns None."""
        sdk_result = {
            "oldTodos": [{"content": "old", "status": "completed"}],
            "newTodos": [],
        }

        metadata = convert_tool_result_to_opencode_metadata("TodoWrite", sdk_result)

        assert metadata is None

    def test_todowrite_case_insensitive(self) -> None:
        """Test that tool name matching is case-insensitive."""
        sdk_result = {
            "oldTodos": [],
            "newTodos": [{"content": "Task", "status": "pending"}],
        }

        # All these should work
        assert convert_tool_result_to_opencode_metadata("todowrite", sdk_result) is not None
        assert convert_tool_result_to_opencode_metadata("TODOWRITE", sdk_result) is not None
        assert convert_tool_result_to_opencode_metadata("TodoWrite", sdk_result) is not None


class TestStructuredPatchToDiff:
    """Tests for structured patch to unified diff conversion."""

    def test_structured_patch_conversion(self) -> None:
        """Test that structuredPatch is converted to proper unified diff."""
        sdk_result = {
            "filePath": "/tmp/test.py",
            "oldString": "old",
            "newString": "new",
            "originalFile": "line1\nold\nline3\n",
            "structuredPatch": [
                {
                    "oldStart": 2,
                    "oldLines": 1,
                    "newStart": 2,
                    "newLines": 1,
                    "lines": ["-old", "+new"],
                }
            ],
            "userModified": False,
            "replaceAll": False,
        }

        metadata = convert_tool_result_to_opencode_metadata("Edit", sdk_result)

        assert metadata is not None
        diff = metadata["diff"]

        # Should contain unified diff header
        assert "--- a/test.py" in diff
        assert "+++ b/test.py" in diff
        # Should contain the changes
        assert "-old" in diff
        assert "+new" in diff
