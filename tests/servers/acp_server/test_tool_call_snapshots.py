"""Snapshot tests for tool call JSON-RPC messages using full ACPSession flow.

These tests use the ToolCallTestHarness to capture the exact wire format of
JSON-RPC notifications for regression testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from exxec.models import ExecutionResult
import pytest
from syrupy.extensions.json import JSONSnapshotExtension

from agentpool_config.toolsets import ExecutionEnvironmentToolsetConfig, FSSpecToolsetConfig

from .tool_call_harness import ToolCallTestHarness


if TYPE_CHECKING:
    from syrupy import SnapshotAssertion


@pytest.fixture
def snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Use JSON serialization for cleaner snapshots."""
    return snapshot.use_extension(JSONSnapshotExtension)


@pytest.fixture
def harness() -> ToolCallTestHarness:
    """Create a fresh test harness for each test."""
    return ToolCallTestHarness()


class TestReadFileSnapshots:
    """Snapshot tests for read_file tool."""

    async def test_read_file_basic(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test basic file read produces expected notifications."""
        await harness.mock_env.set_file_content("/test/hello.txt", "Hello, World!")

        messages = await harness.execute_tool(
            tool_name="read_file",
            tool_args={"path": "/test/hello.txt"},
            toolsets=[FSSpecToolsetConfig()],
        )

        assert messages == snapshot

    async def test_read_file_with_line_range(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test file read with line/limit produces expected notifications."""
        content = "\n".join(f"Line {i}" for i in range(1, 11))
        await harness.mock_env.set_file_content("/test/lines.txt", content)

        messages = await harness.execute_tool(
            tool_name="read_file",
            tool_args={"path": "/test/lines.txt", "line": 3, "limit": 2},
            toolsets=[FSSpecToolsetConfig()],
        )

        assert messages == snapshot


class TestWriteFileSnapshots:
    """Snapshot tests for write_file tool."""

    async def test_write_file_new(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test writing a new file produces expected notifications."""
        messages = await harness.execute_tool(
            tool_name="write_file",
            tool_args={"path": "/test/new_file.txt", "content": "New content here"},
            toolsets=[FSSpecToolsetConfig()],
        )

        assert messages == snapshot

    async def test_write_file_overwrite(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test overwriting existing file produces expected notifications."""
        await harness.mock_env.set_file_content("/test/existing.txt", "Old content")

        messages = await harness.execute_tool(
            tool_name="write_file",
            tool_args={
                "path": "/test/existing.txt",
                "content": "Updated content",
                "overwrite": True,
            },
            toolsets=[FSSpecToolsetConfig()],
        )

        assert messages == snapshot


class TestExecuteCodeSnapshots:
    """Snapshot tests for execute_code tool."""

    async def test_execute_code_simple(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test simple code execution produces expected notifications."""
        harness.mock_env._code_results["print('hello')"] = ExecutionResult(
            result=None, duration=0.01, success=True, stdout="hello\n", exit_code=0
        )

        messages = await harness.execute_tool(
            tool_name="execute_code",
            tool_args={"code": "print('hello')"},
            toolsets=[ExecutionEnvironmentToolsetConfig()],
        )

        assert messages == snapshot

    async def test_execute_code_with_error(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test code execution with error produces expected notifications."""
        harness.mock_env._code_results["raise ValueError('test error')"] = ExecutionResult(
            result=None,
            duration=0.01,
            success=False,
            stderr="ValueError: test error\n",
            exit_code=1,
            error="ValueError: test error",
            error_type="ValueError",
        )

        messages = await harness.execute_tool(
            tool_name="execute_code",
            tool_args={"code": "raise ValueError('test error')"},
            toolsets=[ExecutionEnvironmentToolsetConfig()],
        )

        assert messages == snapshot

    async def test_execute_code_multiline(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test multiline code execution produces expected notifications."""
        code = "x = 1\ny = 2\nprint(x + y)"
        harness.mock_env._code_results[code] = ExecutionResult(
            result=None, duration=0.01, success=True, stdout="3\n", exit_code=0
        )

        messages = await harness.execute_tool(
            tool_name="execute_code",
            tool_args={"code": code},
            toolsets=[ExecutionEnvironmentToolsetConfig()],
        )

        assert messages == snapshot


class TestExecuteCommandSnapshots:
    """Snapshot tests for execute_command tool."""

    async def test_execute_command_simple(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test simple command execution produces expected notifications."""
        harness.mock_env._command_results["echo hello"] = ExecutionResult(
            result=None, duration=0.01, success=True, stdout="hello\n", exit_code=0
        )

        messages = await harness.execute_tool(
            tool_name="execute_command",
            tool_args={"command": "echo hello"},
            toolsets=[ExecutionEnvironmentToolsetConfig()],
        )

        assert messages == snapshot

    async def test_execute_command_with_stderr(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test command with stderr produces expected notifications."""
        harness.mock_env._command_results["ls /nonexistent"] = ExecutionResult(
            result=None,
            duration=0.01,
            success=False,
            stderr="ls: cannot access '/nonexistent': No such file or directory\n",
            exit_code=2,
            error="Command failed",
            error_type="CommandError",
        )

        messages = await harness.execute_tool(
            tool_name="execute_command",
            tool_args={"command": "ls /nonexistent"},
            toolsets=[ExecutionEnvironmentToolsetConfig()],
        )

        assert messages == snapshot

    async def test_execute_command_with_output_limit(
        self,
        harness: ToolCallTestHarness,
        snapshot: SnapshotAssertion,
    ) -> None:
        """Test command with output limit produces expected notifications."""
        long_output = "line\n" * 100
        harness.mock_env._command_results["cat bigfile"] = ExecutionResult(
            result=None, duration=0.01, success=True, stdout=long_output, exit_code=0
        )

        messages = await harness.execute_tool(
            tool_name="execute_command",
            tool_args={"command": "cat bigfile", "output_limit": 50},
            toolsets=[ExecutionEnvironmentToolsetConfig()],
        )

        assert messages == snapshot


if __name__ == "__main__":
    pytest.main(["-v", __file__])
