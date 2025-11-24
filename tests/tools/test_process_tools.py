"""End-to-end tests for process management tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from anyenv.process_manager import ProcessOutput
import pytest

from llmling_agent_toolsets.builtin.process_management import ProcessManagementTools


@pytest.fixture
def mock_process_manager():
    """Create a mock ProcessManager for testing."""
    return MagicMock()


@pytest.fixture
def tools(mock_process_manager):
    """Create ProcessManagementTools instance with mocked process manager."""
    return ProcessManagementTools(process_manager=mock_process_manager)


class TestToolsInitialization:
    """Test tools initialization and configuration."""

    async def test_tools_can_be_retrieved(self, tools):
        """Test that tools can be retrieved from the provider."""
        tool_list = await tools.get_tools()
        assert len(tool_list) == 6  # noqa: PLR2004
        assert all(hasattr(tool, "name") for tool in tool_list)
        assert all(hasattr(tool, "description") for tool in tool_list)

    def test_custom_process_manager_is_used(self, mock_process_manager):
        """Test that custom process manager is properly set."""
        tools = ProcessManagementTools(process_manager=mock_process_manager)
        assert tools.process_manager is mock_process_manager

    def test_default_process_manager_is_created(self):
        """Test that ProcessManagementTools creates default ProcessManager."""
        tools = ProcessManagementTools()
        assert tools.process_manager is not None


class TestProcessLifecycleEndToEnd:
    """Test complete process lifecycle scenarios."""

    async def test_start_and_wait_workflow(self, tools):
        """Test realistic workflow: start process, get output, wait for completion."""
        tools.process_manager.start_process = AsyncMock(return_value="proc_123")
        tools.process_manager.wait_for_exit = AsyncMock(return_value=0)
        tools.process_manager.get_output = AsyncMock(
            return_value=ProcessOutput(
                stdout="Process completed\n",
                stderr="",
                combined="Process completed\n",
                truncated=False,
                exit_code=0,
                signal=None,
            )
        )

        # Start process
        start_result = await tools.start_process(command="echo", args=["hello"])
        assert "proc_123" in start_result

        # Wait for completion
        wait_result = await tools.wait_for_process("proc_123")
        assert "completed with exit code 0" in wait_result
        assert "Process completed" in wait_result

    async def test_process_failure_workflow(self, tools):
        """Test workflow with process failure."""
        tools.process_manager.start_process = AsyncMock(return_value="proc_456")
        tools.process_manager.wait_for_exit = AsyncMock(return_value=1)
        tools.process_manager.get_output = AsyncMock(
            return_value=ProcessOutput(
                stdout="",
                stderr="Command failed\n",
                combined="Command failed\n",
                truncated=False,
                exit_code=1,
                signal=None,
            )
        )

        start_result = await tools.start_process(command="false")
        assert "proc_456" in start_result

        wait_result = await tools.wait_for_process("proc_456")
        assert "exit code 1" in wait_result
        assert "Command failed" in wait_result

    async def test_start_list_kill_workflow(self, tools):
        """Test workflow: start, list, and kill processes."""
        tools.process_manager.start_process = AsyncMock(return_value="proc_111")
        tools.process_manager.list_processes = AsyncMock(return_value=["proc_111"])
        tools.process_manager.get_process_info = AsyncMock(
            return_value={
                "command": "sleep",
                "args": ["60"],
                "is_running": True,
                "exit_code": None,
            }
        )
        tools.process_manager.kill_process = AsyncMock(return_value=None)

        # Start
        start_result = await tools.start_process(command="sleep", args=["60"])
        assert "proc_111" in start_result

        # List
        list_result = await tools.list_processes()
        assert "Active processes:" in list_result
        assert "proc_111" in list_result
        assert "sleep 60" in list_result
        assert "[running]" in list_result

        # Kill
        kill_result = await tools.kill_process("proc_111")
        assert "terminated" in kill_result


class TestErrorHandling:
    """Test error handling in realistic scenarios."""

    async def test_start_process_command_not_found(self, tools):
        """Test handling when command doesn't exist."""
        tools.process_manager.start_process = AsyncMock(
            side_effect=FileNotFoundError("Command not found")
        )

        result = await tools.start_process(command="nonexistent_cmd")
        assert "Failed to start process" in result
        assert "Command not found" in result

    async def test_wait_for_nonexistent_process(self, tools):
        """Test waiting for process that doesn't exist."""
        tools.process_manager.wait_for_exit = AsyncMock(side_effect=ValueError("Process not found"))

        result = await tools.wait_for_process("invalid_pid")
        assert "Process not found" in result

    async def test_get_output_for_killed_process(self, tools):
        """Test getting output from killed process."""
        tools.process_manager.get_output = AsyncMock(side_effect=RuntimeError("Process was killed"))

        result = await tools.get_process_output("killed_proc")
        assert "Error getting process output" in result
        assert "Process was killed" in result

    async def test_list_processes_system_error(self, tools):
        """Test listing processes when system error occurs."""
        tools.process_manager.list_processes = AsyncMock(
            side_effect=PermissionError("Access denied")
        )

        result = await tools.list_processes()
        assert "Error listing processes" in result
        assert "Access denied" in result

    async def test_list_processes_with_partial_info_failure(self, tools):
        """Test listing when retrieving info for one process fails."""
        tools.process_manager.list_processes = AsyncMock(return_value=["proc_1", "proc_2"])
        tools.process_manager.get_process_info = AsyncMock(
            side_effect=[
                {
                    "command": "sleep",
                    "args": ["60"],
                    "is_running": True,
                    "exit_code": None,
                },
                RuntimeError("Can't get info for proc_2"),
            ]
        )

        result = await tools.list_processes()
        assert "proc_1" in result
        assert "sleep 60" in result
        assert "proc_2" in result
        assert "Error getting info" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_large_truncated_output(self, tools):
        """Test handling of truncated output from large process."""
        tools.process_manager.get_output = AsyncMock(
            return_value=ProcessOutput(
                stdout="[TRUNCATED - 10000 bytes]",
                stderr="",
                combined="[TRUNCATED - 10000 bytes]",
                truncated=True,
                exit_code=None,
                signal=None,
            )
        )

        result = await tools.get_process_output("big_proc")
        assert "STDOUT:" in result
        assert "[TRUNCATED - 10000 bytes]" in result
        assert "Output was truncated" in result

    async def test_process_with_both_stdout_and_stderr(self, tools):
        """Test process output with both stdout and stderr."""
        tools.process_manager.get_output = AsyncMock(
            return_value=ProcessOutput(
                stdout="Normal output\n",
                stderr="Warning: something\n",
                combined="Normal output\nWarning: something\n",
                truncated=False,
                exit_code=0,
                signal=None,
            )
        )

        result = await tools.get_process_output("mixed_proc")
        assert "STDOUT:\nNormal output" in result
        assert "STDERR:\nWarning: something" in result
        assert "Exit code: 0" in result

    async def test_no_active_processes(self, tools):
        """Test listing when no processes are running."""
        tools.process_manager.list_processes = AsyncMock(return_value=[])

        result = await tools.list_processes()
        assert result == "No active processes"

    async def test_process_killed_by_signal(self, tools):
        """Test process terminated by signal."""
        tools.process_manager.wait_for_exit = AsyncMock(return_value=-15)
        tools.process_manager.get_output = AsyncMock(
            return_value=ProcessOutput(
                stdout="",
                stderr="Killed\n",
                combined="Killed\n",
                truncated=False,
                exit_code=-15,
                signal="15",
            )
        )

        result = await tools.wait_for_process("killed_proc")
        assert "completed with exit code -15" in result
        assert "Killed" in result
