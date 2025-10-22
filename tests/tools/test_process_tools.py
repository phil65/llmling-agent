"""Tests for process management tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llmling_agent.agent.context import AgentContext
from llmling_agent.agent.process_manager import ProcessManager, ProcessOutput
from llmling_agent_toolsets.builtin import process_management


@pytest.fixture
def mock_context():
    """Create a mock AgentContext for testing."""
    context = MagicMock(spec=AgentContext)
    context.process_manager = MagicMock(spec=ProcessManager)
    return context


@pytest.fixture
def mock_process_output():
    """Create a mock ProcessOutput for testing."""
    return ProcessOutput(
        stdout="Hello World\n",
        stderr="",
        combined="Hello World\n",
        truncated=False,
        exit_code=0,
        signal=None,
    )


class TestStartProcess:
    """Tests for start_process tool."""

    async def test_start_process_success(self, mock_context):
        """Test successful process start."""
        mock_context.process_manager.start_process.return_value = "proc_123"

        result = await process_management.start_process(
            mock_context,
            command="echo",
            args=["hello"],
            cwd="/tmp",
            env={"TEST": "value"},
            output_limit=1024,
        )

        assert result == "Started process: proc_123"
        mock_context.process_manager.start_process.assert_called_once_with(
            command="echo",
            args=["hello"],
            cwd="/tmp",
            env={"TEST": "value"},
            output_limit=1024,
        )

    async def test_start_process_minimal_args(self, mock_context):
        """Test starting process with minimal arguments."""
        mock_context.process_manager.start_process.return_value = "proc_456"

        result = await process_management.start_process(mock_context, command="ls")

        assert result == "Started process: proc_456"
        mock_context.process_manager.start_process.assert_called_once_with(
            command="ls",
            args=None,
            cwd=None,
            env=None,
            output_limit=None,
        )

    async def test_start_process_failure(self, mock_context):
        """Test handling process start failure."""
        mock_context.process_manager.start_process.side_effect = OSError(
            "Command not found"
        )

        result = await process_management.start_process(mock_context, command="badcmd")

        assert "Failed to start process" in result
        assert "Command not found" in result


class TestGetProcessOutput:
    """Tests for get_process_output tool."""

    async def test_get_output_success(self, mock_context, mock_process_output):
        """Test successful output retrieval."""
        mock_context.process_manager.get_output.return_value = mock_process_output

        result = await process_management.get_process_output(mock_context, "proc_123")

        assert "Process proc_123:" in result
        assert "STDOUT:\nHello World" in result
        assert "Exit code: 0" in result
        mock_context.process_manager.get_output.assert_called_once_with("proc_123")

    async def test_get_output_with_stderr(self, mock_context):
        """Test output retrieval with stderr content."""
        output = ProcessOutput(
            stdout="Success\n",
            stderr="Warning: something\n",
            combined="Success\nWarning: something\n",
            truncated=False,
            exit_code=1,
            signal=None,
        )
        mock_context.process_manager.get_output.return_value = output

        result = await process_management.get_process_output(mock_context, "proc_123")

        assert "STDOUT:\nSuccess" in result
        assert "STDERR:\nWarning: something" in result
        assert "Exit code: 1" in result

    async def test_get_output_truncated(self, mock_context):
        """Test output retrieval with truncated output."""
        output = ProcessOutput(
            stdout="Partial output...",
            stderr="",
            combined="Partial output...",
            truncated=True,
            exit_code=None,
            signal=None,
        )
        mock_context.process_manager.get_output.return_value = output

        result = await process_management.get_process_output(mock_context, "proc_123")

        assert "Note: Output was truncated due to size limits" in result

    async def test_get_output_not_found(self, mock_context):
        """Test handling non-existent process."""
        mock_context.process_manager.get_output.side_effect = ValueError(
            "Process not found"
        )

        result = await process_management.get_process_output(mock_context, "nonexistent")

        assert result == "Process not found"

    async def test_get_output_error(self, mock_context):
        """Test handling unexpected errors."""
        mock_context.process_manager.get_output.side_effect = Exception(
            "Unexpected error"
        )

        result = await process_management.get_process_output(mock_context, "proc_123")

        assert "Error getting process output" in result
        assert "Unexpected error" in result


class TestWaitForProcess:
    """Tests for wait_for_process tool."""

    async def test_wait_success(self, mock_context, mock_process_output):
        """Test successful process wait."""
        mock_context.process_manager.wait_for_exit.return_value = 0
        mock_context.process_manager.get_output.return_value = mock_process_output

        result = await process_management.wait_for_process(mock_context, "proc_123")

        assert "Process proc_123 completed with exit code 0" in result
        assert "STDOUT:\nHello World" in result
        mock_context.process_manager.wait_for_exit.assert_called_once_with("proc_123")
        mock_context.process_manager.get_output.assert_called_once_with("proc_123")

    async def test_wait_failure_exit_code(self, mock_context):
        """Test waiting for process with non-zero exit code."""
        mock_context.process_manager.wait_for_exit.return_value = 1
        output = ProcessOutput(
            stdout="",
            stderr="Error occurred\n",
            combined="Error occurred\n",
            truncated=False,
            exit_code=1,
            signal=None,
        )
        mock_context.process_manager.get_output.return_value = output

        result = await process_management.wait_for_process(mock_context, "proc_123")

        assert "completed with exit code 1" in result
        assert "STDERR:\nError occurred" in result

    async def test_wait_not_found(self, mock_context):
        """Test waiting for non-existent process."""
        mock_context.process_manager.wait_for_exit.side_effect = ValueError(
            "Process not found"
        )

        result = await process_management.wait_for_process(mock_context, "nonexistent")

        assert result == "Process not found"


class TestKillProcess:
    """Tests for kill_process tool."""

    async def test_kill_success(self, mock_context):
        """Test successful process termination."""
        mock_context.process_manager.kill_process.return_value = None

        result = await process_management.kill_process(mock_context, "proc_123")

        assert result == "Process proc_123 has been terminated"
        mock_context.process_manager.kill_process.assert_called_once_with("proc_123")

    async def test_kill_not_found(self, mock_context):
        """Test killing non-existent process."""
        mock_context.process_manager.kill_process.side_effect = ValueError(
            "Process not found"
        )

        result = await process_management.kill_process(mock_context, "nonexistent")

        assert result == "Process not found"

    async def test_kill_error(self, mock_context):
        """Test handling kill errors."""
        mock_context.process_manager.kill_process.side_effect = Exception("Kill failed")

        result = await process_management.kill_process(mock_context, "proc_123")

        assert "Error killing process" in result
        assert "Kill failed" in result


class TestReleaseProcess:
    """Tests for release_process tool."""

    async def test_release_success(self, mock_context):
        """Test successful process resource release."""
        mock_context.process_manager.release_process.return_value = None

        result = await process_management.release_process(mock_context, "proc_123")

        assert result == "Process proc_123 resources have been released"
        mock_context.process_manager.release_process.assert_called_once_with("proc_123")

    async def test_release_not_found(self, mock_context):
        """Test releasing non-existent process."""
        mock_context.process_manager.release_process.side_effect = ValueError(
            "Process not found"
        )

        result = await process_management.release_process(mock_context, "nonexistent")

        assert result == "Process not found"

    async def test_release_error(self, mock_context):
        """Test handling release errors."""
        mock_context.process_manager.release_process.side_effect = Exception(
            "Release failed"
        )

        result = await process_management.release_process(mock_context, "proc_123")

        assert "Error releasing process" in result
        assert "Release failed" in result


class TestListProcesses:
    """Tests for list_processes tool."""

    async def test_list_no_processes(self, mock_context):
        """Test listing when no processes are active."""
        mock_context.process_manager.list_processes.return_value = []

        result = await process_management.list_processes(mock_context)

        assert result == "No active processes"

    async def test_list_with_processes(self, mock_context):
        """Test listing active processes."""
        mock_context.process_manager.list_processes.return_value = [
            "proc_123",
            "proc_456",
        ]
        mock_context.process_manager.get_process_info.side_effect = [
            {
                "command": "echo",
                "args": ["hello"],
                "is_running": True,
                "exit_code": None,
            },
            {
                "command": "sleep",
                "args": ["60"],
                "is_running": False,
                "exit_code": 0,
            },
        ]

        result = await process_management.list_processes(mock_context)

        assert "Active processes:" in result
        assert "proc_123: echo hello [running]" in result
        assert "proc_456: sleep 60 [exited (0)]" in result

    async def test_list_with_info_error(self, mock_context):
        """Test listing when process info retrieval fails."""
        mock_context.process_manager.list_processes.return_value = ["proc_123"]
        mock_context.process_manager.get_process_info.side_effect = Exception(
            "Info error"
        )

        result = await process_management.list_processes(mock_context)

        assert "proc_123: Error getting info - Info error" in result

    async def test_list_general_error(self, mock_context):
        """Test handling general list errors."""
        mock_context.process_manager.list_processes.side_effect = Exception("List failed")

        result = await process_management.list_processes(mock_context)

        assert "Error listing processes" in result
        assert "List failed" in result


class TestPydanticAICompatibility:
    """Tests for PydanticAI RunContext compatibility."""

    async def test_runcontext_handling(self):
        """Test that tools handle PydanticAI RunContext correctly."""
        from pydantic_ai.tools import RunContext

        # Create mock RunContext
        mock_deps = MagicMock()
        mock_deps.process_manager = MagicMock()

        # Mock the async method properly
        async def mock_start_process(*args, **kwargs):
            return "proc_123"

        mock_deps.process_manager.start_process = mock_start_process

        mock_run_context = MagicMock(spec=RunContext)
        mock_run_context.deps = mock_deps

        result = await process_management.start_process(mock_run_context, command="echo")

        assert result == "Started process: proc_123"
