"""End-to-end tests for execution environment tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock

from anyenv.code_execution.events import (
    OutputEvent,
    ProcessCompletedEvent,
    ProcessErrorEvent,
    ProcessStartedEvent,
)
from anyenv.process_manager import ProcessOutput
import pytest

from llmling_agent.agent.context import AgentContext
from llmling_agent.agent.event_emitter import AgentEventEmitter
from llmling_agent.models.agents import AgentConfig
from llmling_agent_toolsets.builtin.execution_environment import ExecutionEnvironmentTools


def create_mock_agent_context() -> AgentContext:
    """Create a mock AgentContext for testing."""
    context = Mock(spec=AgentContext)
    context.node_name = "test_agent"
    context.events = Mock(spec=AgentEventEmitter)
    context.events.process_started = AsyncMock()
    context.events.process_output = AsyncMock()
    context.events.process_exit = AsyncMock()
    context.events.process_killed = AsyncMock()
    context.events.process_released = AsyncMock()
    context.config = Mock(spec=AgentConfig)
    context.tool_call_id = "test_call_123"
    context.tool_input = {"command": "echo", "args": ["hello"]}
    return context


@pytest.fixture
def agent_ctx():
    """Create mock agent context."""
    return create_mock_agent_context()


@pytest.fixture
def mock_env():
    """Create a mock ExecutionEnvironment for testing."""
    env = MagicMock()
    env.process_manager = MagicMock()
    return env


@pytest.fixture
def tools(mock_env):
    """Create ExecutionEnvironmentTools instance with mocked environment."""
    return ExecutionEnvironmentTools(env=mock_env)


class TestToolsInitialization:
    """Test tools initialization and configuration."""

    def test_custom_env_is_used(self, mock_env):
        """Test that custom environment is properly set."""
        tools = ExecutionEnvironmentTools(env=mock_env)
        assert tools.env is mock_env

    def test_default_env_is_created(self):
        """Test that ExecutionEnvironmentTools creates default LocalExecutionEnvironment."""
        tools = ExecutionEnvironmentTools()
        assert tools.env is not None


class TestCodeExecution:
    """Test code execution tools."""

    async def test_execute_code_success(self, tools, agent_ctx):
        """Test successful code execution."""

        async def mock_stream_code(code):
            yield ProcessStartedEvent(command=f"execute({len(code)} chars)")
            yield OutputEvent(data="42\n", stream="stdout")
            yield ProcessCompletedEvent(exit_code=0, duration=0.1)

        tools.env.stream_code = mock_stream_code
        result = await tools.execute_code(agent_ctx, "print(42)")
        assert result["success"] is True
        assert "42" in result["output"]
        assert result["exit_code"] == 0
        agent_ctx.events.process_started.assert_called()
        agent_ctx.events.process_exit.assert_called()

    async def test_execute_code_failure(self, tools, agent_ctx):
        """Test code execution failure."""

        async def mock_stream_code(code):
            yield ProcessStartedEvent(command=f"execute({len(code)} chars)")
            yield ProcessErrorEvent(
                error="NameError: name 'x' is not defined",
                error_type="NameError",
                exit_code=1,
            )

        tools.env.stream_code = mock_stream_code
        result = await tools.execute_code(agent_ctx, "print(x)")
        assert result["success"] is False
        assert "NameError" in result["error"]
        agent_ctx.events.process_exit.assert_called()

    async def test_execute_code_exception(self, tools, agent_ctx):
        """Test code execution with exception."""

        async def mock_stream_code(code):
            raise RuntimeError("Execution failed")
            yield  # Make it a generator

        tools.env.stream_code = mock_stream_code
        result = await tools.execute_code(agent_ctx, "bad code")
        assert result["success"] is False
        assert "Execution failed" in result["error"]

    async def test_execute_command_success(self, tools, agent_ctx):
        """Test successful command execution."""

        async def mock_stream_command(command):
            yield ProcessStartedEvent(command=command)
            yield OutputEvent(data="hello world\n", stream="stdout")
            yield ProcessCompletedEvent(exit_code=0, duration=0.2)

        tools.env.stream_command = mock_stream_command
        result = await tools.execute_command(agent_ctx, "echo hello world")
        assert result["success"] is True
        assert result["stdout"] == "hello world\n"
        assert result["exit_code"] == 0
        agent_ctx.events.process_started.assert_called()
        agent_ctx.events.process_exit.assert_called()

    async def test_execute_command_with_output_limit(self, tools, agent_ctx):
        """Test command execution with output truncation."""
        long_output = "x" * 1000

        async def mock_stream_command(command):
            yield ProcessStartedEvent(command=command)
            yield OutputEvent(data=long_output, stream="stdout")
            yield ProcessCompletedEvent(exit_code=0, duration=0.1)

        tools.env.stream_command = mock_stream_command
        result = await tools.execute_command(agent_ctx, "echo", output_limit=100)
        assert result["success"] is True
        assert "[truncated]" in result["stdout"]
        assert result["truncated"] is True


class TestProcessLifecycle:
    """Test process lifecycle scenarios."""

    async def test_start_process_success(self, tools, agent_ctx):
        """Test successful process start."""
        tools.env.process_manager.start_process = AsyncMock(return_value="proc_123")
        result = await tools.start_process(
            agent_ctx,
            command="echo",
            args=["hello", "world"],
            cwd="/tmp",
        )
        assert result["process_id"] == "proc_123"
        assert result["command"] == "echo"
        assert result["args"] == ["hello", "world"]
        assert result["status"] == "started"
        agent_ctx.events.process_started.assert_called_once()
        call_args = agent_ctx.events.process_started.call_args[1]
        assert call_args["success"] is True

    async def test_start_process_failure(self, tools, agent_ctx):
        """Test process start failure."""
        tools.env.process_manager.start_process = AsyncMock(
            side_effect=FileNotFoundError("Command not found")
        )
        result = await tools.start_process(agent_ctx, command="nonexistent")
        assert "error" in result
        assert "Command not found" in result["error"]
        agent_ctx.events.process_started.assert_called_once()
        call_args = agent_ctx.events.process_started.call_args[1]
        assert call_args["success"] is False

    async def test_get_process_output_running(self, tools, agent_ctx):
        """Test getting output from running process."""
        tools.env.process_manager.get_output = AsyncMock(
            return_value=ProcessOutput(
                stdout="output line 1\noutput line 2\n",
                stderr="",
                combined="output line 1\noutput line 2\n",
                exit_code=None,
            )
        )
        result = await tools.get_process_output(agent_ctx, "proc_123")
        assert result["process_id"] == "proc_123"
        assert result["status"] == "running"
        assert "output line 1" in result["stdout"]
        agent_ctx.events.process_output.assert_called_once()

    async def test_get_process_output_completed(self, tools, agent_ctx):
        """Test getting output from completed process."""
        tools.env.process_manager.get_output = AsyncMock(
            return_value=ProcessOutput(
                stdout="done\n",
                stderr="",
                combined="done\n",
                exit_code=0,
            )
        )
        result = await tools.get_process_output(agent_ctx, "proc_123")
        assert result["status"] == "completed"
        assert result["exit_code"] == 0

    async def test_wait_for_process_success(self, tools, agent_ctx):
        """Test waiting for process completion."""
        tools.env.process_manager.wait_for_exit = AsyncMock(return_value=0)
        tools.env.process_manager.get_output = AsyncMock(
            return_value=ProcessOutput(
                stdout="Process completed\n",
                stderr="",
                combined="Process completed\n",
                exit_code=0,
            )
        )
        result = await tools.wait_for_process(agent_ctx, "proc_123")
        assert result["process_id"] == "proc_123"
        assert result["exit_code"] == 0
        assert result["status"] == "completed"
        assert "Process completed" in result["stdout"]
        agent_ctx.events.process_exit.assert_called_once()

    async def test_wait_for_process_failure(self, tools, agent_ctx):
        """Test waiting for failed process."""
        tools.env.process_manager.wait_for_exit = AsyncMock(return_value=1)
        tools.env.process_manager.get_output = AsyncMock(
            return_value=ProcessOutput(
                stdout="",
                stderr="Command failed\n",
                combined="Command failed\n",
                exit_code=1,
            )
        )
        result = await tools.wait_for_process(agent_ctx, "proc_456")
        assert result["exit_code"] == 1
        assert "Command failed" in result["stderr"]

    async def test_kill_process_success(self, tools, agent_ctx):
        """Test killing a process."""
        tools.env.process_manager.kill_process = AsyncMock()
        result = await tools.kill_process(agent_ctx, "proc_123")
        assert result["process_id"] == "proc_123"
        assert result["status"] == "killed"
        agent_ctx.events.process_killed.assert_called_once()
        call_args = agent_ctx.events.process_killed.call_args[1]
        assert call_args["success"] is True

    async def test_kill_process_not_found(self, tools, agent_ctx):
        """Test killing nonexistent process."""
        tools.env.process_manager.kill_process = AsyncMock(
            side_effect=ValueError("Process not found")
        )

        result = await tools.kill_process(agent_ctx, "invalid")
        assert "error" in result
        agent_ctx.events.process_killed.assert_called_once()
        call_args = agent_ctx.events.process_killed.call_args[1]
        assert call_args["success"] is False

    async def test_release_process_success(self, tools, agent_ctx):
        """Test releasing process resources."""
        tools.env.process_manager.release_process = AsyncMock()
        result = await tools.release_process(agent_ctx, "proc_123")
        assert result["process_id"] == "proc_123"
        assert result["status"] == "released"
        agent_ctx.events.process_released.assert_called_once()

    async def test_list_processes_empty(self, tools, agent_ctx):
        """Test listing when no processes running."""
        tools.env.process_manager.list_processes = AsyncMock(return_value=[])
        result = await tools.list_processes(agent_ctx)
        assert result["processes"] == []
        assert result["count"] == 0

    async def test_list_processes_with_results(self, tools, agent_ctx):
        """Test listing active processes."""
        tools.env.process_manager.list_processes = AsyncMock(return_value=["proc_1", "proc_2"])
        tools.env.process_manager.get_process_info = AsyncMock(
            side_effect=[
                {
                    "command": "sleep",
                    "args": ["60"],
                    "is_running": True,
                    "exit_code": None,
                    "cwd": "/tmp",
                    "created_at": "2024-01-01T00:00:00",
                },
                {
                    "command": "echo",
                    "args": ["done"],
                    "is_running": False,
                    "exit_code": 0,
                    "cwd": "/home",
                    "created_at": "2024-01-01T00:01:00",
                },
            ]
        )

        result = await tools.list_processes(agent_ctx)
        assert result["count"] == 2  # noqa: PLR2004
        assert len(result["processes"]) == 2  # noqa: PLR2004
        assert result["processes"][0]["command"] == "sleep"
        assert result["processes"][0]["is_running"] is True
        assert result["processes"][1]["command"] == "echo"
        assert result["processes"][1]["is_running"] is False

    async def test_list_processes_partial_failure(self, tools, agent_ctx):
        """Test listing when some process info fails."""
        tools.env.process_manager.list_processes = AsyncMock(return_value=["proc_1", "proc_2"])
        tools.env.process_manager.get_process_info = AsyncMock(
            side_effect=[
                {"command": "sleep", "args": [], "is_running": True},
                RuntimeError("Can't get info"),
            ]
        )
        result = await tools.list_processes(agent_ctx)
        assert result["count"] == 2  # noqa: PLR2004
        assert result["processes"][0]["command"] == "sleep"
        assert "error" in result["processes"][1]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_truncated_output(self, tools, agent_ctx):
        """Test handling of truncated output."""
        tools.env.process_manager.get_output = AsyncMock(
            return_value=ProcessOutput(
                stdout="partial output...",
                stderr="",
                combined="partial output...",
                truncated=True,
                exit_code=None,
            )
        )
        result = await tools.get_process_output(agent_ctx, "big_proc")
        assert result["truncated"] is True

    async def test_process_with_signal(self, tools, agent_ctx):
        """Test process terminated by signal."""
        tools.env.process_manager.wait_for_exit = AsyncMock(return_value=-15)
        tools.env.process_manager.get_output = AsyncMock(
            return_value=ProcessOutput(
                stdout="",
                stderr="Killed\n",
                combined="Killed\n",
                exit_code=-15,
                signal="15",
            )
        )
        result = await tools.wait_for_process(agent_ctx, "killed_proc")
        assert result["exit_code"] == -15  # noqa: PLR2004
