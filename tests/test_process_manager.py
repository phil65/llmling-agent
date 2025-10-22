"""Tests for process management functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmling_agent.agent.process_manager import (
    ProcessManager,
    ProcessOutput,
    RunningProcess,
)


@pytest.fixture
def process_manager():
    """Create a ProcessManager instance for testing."""
    return ProcessManager()


@pytest.fixture
def mock_process():
    """Create a mock subprocess.Process for testing."""
    process = MagicMock()
    process.returncode = None
    process.stdout = AsyncMock()
    process.stderr = AsyncMock()
    process.wait = AsyncMock(return_value=0)
    process.terminate = MagicMock()
    process.kill = MagicMock()
    return process


async def test_process_manager_initialization(process_manager):
    """Test ProcessManager initializes correctly."""
    assert isinstance(process_manager._processes, dict)
    assert isinstance(process_manager._output_tasks, dict)
    assert len(process_manager._processes) == 0


async def test_start_process_success(process_manager, mock_process):
    """Test successfully starting a process."""
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        process_id = await process_manager.start_process("echo", ["hello"])

        assert process_id.startswith("proc_")
        assert process_id in process_manager._processes
        assert process_id in process_manager._output_tasks

        running_proc = process_manager._processes[process_id]
        assert running_proc.command == "echo"
        assert running_proc.args == ["hello"]
        assert running_proc.process == mock_process


async def test_start_process_with_options(process_manager, mock_process):
    """Test starting a process with environment and working directory."""
    with patch(
        "asyncio.create_subprocess_exec", return_value=mock_process
    ) as mock_create:
        await process_manager.start_process(
            "test_cmd",
            args=["arg1", "arg2"],
            cwd="/tmp",
            env={"TEST_VAR": "test_value"},
            output_limit=1024,
        )

        # Verify subprocess was called with correct parameters
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[0] == ("test_cmd", "arg1", "arg2")
        assert call_args[1]["cwd"].name == "tmp"
        assert "TEST_VAR" in call_args[1]["env"]


async def test_start_process_failure(process_manager):
    """Test handling process creation failure."""
    with (
        patch("asyncio.create_subprocess_exec", side_effect=OSError("Command not found")),
        pytest.raises(OSError, match="Failed to start process"),
    ):
        await process_manager.start_process("nonexistent_command")


async def test_get_output_success(process_manager, mock_process):
    """Test getting process output."""
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        process_id = await process_manager.start_process("echo", ["hello"])

        # Simulate some output
        running_proc = process_manager._processes[process_id]
        running_proc.add_output(stdout="Hello\n", stderr="")

        output = await process_manager.get_output(process_id)
        assert isinstance(output, ProcessOutput)
        assert output.stdout == "Hello\n"
        assert output.stderr == ""
        assert output.combined == "Hello\n"


async def test_get_output_nonexistent_process(process_manager):
    """Test getting output for non-existent process."""
    with pytest.raises(ValueError, match="Process nonexistent not found"):
        await process_manager.get_output("nonexistent")


async def test_wait_for_exit(process_manager, mock_process):
    """Test waiting for process completion."""
    mock_process.wait.return_value = 42

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        process_id = await process_manager.start_process("test_cmd")

        exit_code = await process_manager.wait_for_exit(process_id)
        assert exit_code == 42  # noqa: PLR2004
        mock_process.wait.assert_called_once()


async def test_kill_process(process_manager, mock_process):
    """Test killing a running process."""
    mock_process.returncode = None  # Still running

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        process_id = await process_manager.start_process("test_cmd")

        await process_manager.kill_process(process_id)

        mock_process.terminate.assert_called_once()


async def test_kill_nonexistent_process(process_manager):
    """Test killing non-existent process."""
    with pytest.raises(ValueError, match="Process nonexistent not found"):
        await process_manager.kill_process("nonexistent")


async def test_release_process(process_manager, mock_process):
    """Test releasing process resources."""
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        process_id = await process_manager.start_process("test_cmd")

        # Verify process is tracked
        assert process_id in process_manager._processes
        assert process_id in process_manager._output_tasks

        await process_manager.release_process(process_id)

        # Verify process is removed
        assert process_id not in process_manager._processes
        assert process_id not in process_manager._output_tasks


async def test_list_processes(process_manager, mock_process):
    """Test listing active processes."""
    assert process_manager.list_processes() == []

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        process_id1 = await process_manager.start_process("cmd1")
        process_id2 = await process_manager.start_process("cmd2")

        processes = process_manager.list_processes()
        assert len(processes) == 2  # noqa: PLR2004
        assert process_id1 in processes
        assert process_id2 in processes


async def test_get_process_info(process_manager, mock_process):
    """Test getting process information."""
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        process_id = await process_manager.start_process("test_cmd", ["arg1"])

        info = await process_manager.get_process_info(process_id)

        assert info["process_id"] == process_id
        assert info["command"] == "test_cmd"
        assert info["args"] == ["arg1"]
        assert "created_at" in info
        assert "is_running" in info


async def test_cleanup(process_manager, mock_process):
    """Test cleaning up all processes."""
    mock_process.returncode = None  # Still running

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        await process_manager.start_process("cmd1")
        await process_manager.start_process("cmd2")

        # Verify processes exist
        assert len(process_manager._processes) == 2  # noqa: PLR2004

        await process_manager.cleanup()

        # Verify all processes are cleaned up
        assert len(process_manager._processes) == 0
        assert len(process_manager._output_tasks) == 0


async def test_output_truncation():
    """Test output truncation when limit is exceeded."""
    output_limit = 100
    running_proc = RunningProcess(
        process_id="test",
        command="test",
        args=[],
        cwd=None,
        env={},
        process=MagicMock(),
        output_limit=output_limit,
    )

    # Add output that exceeds limit
    large_output = "x" * 150
    running_proc.add_output(stdout=large_output)

    output = running_proc.get_output()
    assert output.truncated
    assert len(output.stdout.encode()) < output_limit


async def test_builtin_toolset_tools():
    """Test that process management tools are available in builtin toolset."""
    from llmling_agent_toolsets.builtin import ProcessManagementTools

    provider = ProcessManagementTools()
    tools = await provider.get_tools()
    tool_names = [tool.name for tool in tools]

    expected_tools = [
        "start_process",
        "get_process_output",
        "wait_for_process",
        "kill_process",
        "release_process",
        "list_processes",
    ]

    for tool_name in expected_tools:
        assert tool_name in tool_names


class TestRunningProcess:
    """Tests for RunningProcess class."""

    def test_add_output(self):
        """Test adding output to process."""
        mock_process = MagicMock()
        proc = RunningProcess(
            process_id="test",
            command="test",
            args=[],
            cwd=None,
            env={},
            process=mock_process,
        )

        proc.add_output(stdout="hello", stderr="error")

        output = proc.get_output()
        assert output.stdout == "hello"
        assert output.stderr == "error"
        assert output.combined == "helloerror"

    async def test_is_running(self):
        """Test checking if process is running."""
        mock_process = MagicMock()
        mock_process.returncode = None

        proc = RunningProcess(
            process_id="test",
            command="test",
            args=[],
            cwd=None,
            env={},
            process=mock_process,
        )

        assert await proc.is_running()

        mock_process.returncode = 0
        assert not await proc.is_running()


class TestProcessOutput:
    """Tests for ProcessOutput class."""

    def test_process_output_creation(self):
        """Test ProcessOutput creation."""
        output = ProcessOutput(
            stdout="hello",
            stderr="error",
            combined="helloerror",
            truncated=True,
            exit_code=0,
            signal=None,
        )

        assert output.stdout == "hello"
        assert output.stderr == "error"
        assert output.combined == "helloerror"
        assert output.truncated is True
        assert output.exit_code == 0
        assert output.signal is None
