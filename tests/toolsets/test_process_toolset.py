"""Tests for injectable ProcessTools toolset."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

from anyenv.process_manager import ProcessOutput
import pytest

from llmling_agent import AgentContext
from llmling_agent.agent.event_emitter import AgentEventEmitter
from llmling_agent.models.agents import AgentConfig
from llmling_agent_toolsets.process_toolset import ProcessTools


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
def mock_process_manager():
    """Create a mock process manager."""
    manager = AsyncMock()
    manager.start_process = AsyncMock(return_value="proc_123")
    manager.get_output = AsyncMock(
        return_value=ProcessOutput(
            stdout="Hello World",
            stderr="",
            combined="Hello World",
            truncated=False,
            exit_code=None,
            signal=None,
        )
    )
    manager.wait_for_exit = AsyncMock(return_value=0)
    manager.kill_process = AsyncMock()
    manager.release_process = AsyncMock()
    manager.list_processes = AsyncMock(return_value=["proc_123", "proc_456"])
    manager.get_process_info = AsyncMock(
        return_value={
            "command": "echo",
            "args": ["hello"],
            "cwd": "/tmp",
            "is_running": True,
            "exit_code": None,
            "created_at": "2023-01-01T00:00:00",
        }
    )
    return manager


@pytest.fixture
def process_tools(mock_process_manager):
    """Create ProcessTools instance with mock manager."""
    return ProcessTools(mock_process_manager)


@pytest.fixture
def agent_ctx():
    """Create mock agent context."""
    return create_mock_agent_context()


async def test_start_process_success(process_tools: ProcessTools, agent_ctx: AgentContext):
    """Test successful process start."""
    tools = await process_tools.get_tools()
    start_tool = next(tool for tool in tools if tool.name == "start_process")

    result = await start_tool.execute(
        agent_ctx=agent_ctx,
        command="echo",
        args=["hello", "world"],
        cwd="/tmp",
    )

    # Verify result
    assert isinstance(result, dict)
    assert result["process_id"] == "proc_123"
    assert result["command"] == "echo"
    assert result["args"] == ["hello", "world"]
    assert result["status"] == "started"

    # Verify process manager was called
    process_tools.process_manager.start_process.assert_called_once_with(
        command="echo",
        args=["hello", "world"],
        cwd="/tmp",
        env=None,
        output_limit=None,
    )

    # Verify event was emitted
    agent_ctx.events.process_started.assert_called_once()
    call_args = agent_ctx.events.process_started.call_args[1]
    assert call_args["process_id"] == "proc_123"
    assert call_args["command"] == "echo"
    assert call_args["success"] is True


async def test_start_process_failure(process_tools: ProcessTools, agent_ctx: AgentContext):
    """Test process start failure."""
    process_tools.process_manager.start_process.side_effect = OSError("Permission denied")

    tools = await process_tools.get_tools()
    start_tool = next(tool for tool in tools if tool.name == "start_process")

    result = await start_tool.execute(
        agent_ctx=agent_ctx,
        command="echo",
        args=["hello"],
    )

    # Verify error result
    assert isinstance(result, dict)
    assert "error" in result
    assert "Permission denied" in result["error"]

    # Verify failure event was emitted
    agent_ctx.events.process_started.assert_called_once()
    call_args = agent_ctx.events.process_started.call_args[1]
    assert call_args["success"] is False
    assert "Permission denied" in call_args["error"]


async def test_get_process_output_success(process_tools: ProcessTools, agent_ctx: AgentContext):
    """Test getting process output."""
    tools = await process_tools.get_tools()
    output_tool = next(tool for tool in tools if tool.name == "get_process_output")

    result = await output_tool.execute(
        agent_ctx=agent_ctx,
        process_id="proc_123",
    )

    # Verify result
    assert isinstance(result, dict)
    assert result["process_id"] == "proc_123"
    assert result["stdout"] == "Hello World"
    assert result["combined"] == "Hello World"
    assert result["status"] == "running"

    # Verify process manager was called
    process_tools.process_manager.get_output.assert_called_once_with("proc_123")

    # Verify event was emitted
    agent_ctx.events.process_output.assert_called_once()


async def test_get_process_output_with_exit_code(
    process_tools: ProcessTools, agent_ctx: AgentContext
):
    """Test getting output from completed process."""
    process_tools.process_manager.get_output.return_value = ProcessOutput(
        stdout="Done",
        stderr="",
        combined="Done",
        truncated=False,
        exit_code=0,
        signal=None,
    )

    tools = await process_tools.get_tools()
    output_tool = next(tool for tool in tools if tool.name == "get_process_output")

    result = await output_tool.execute(
        agent_ctx=agent_ctx,
        process_id="proc_123",
    )

    # Verify result shows completed status
    assert result["status"] == "completed"
    assert result["exit_code"] == 0


async def test_wait_for_process_success(process_tools: ProcessTools, agent_ctx: AgentContext):
    """Test waiting for process completion."""
    process_tools.process_manager.get_output.return_value = ProcessOutput(
        stdout="Final output",
        stderr="",
        combined="Final output",
        truncated=False,
        exit_code=0,
        signal=None,
    )

    tools = await process_tools.get_tools()
    wait_tool = next(tool for tool in tools if tool.name == "wait_for_process")

    result = await wait_tool.execute(
        agent_ctx=agent_ctx,
        process_id="proc_123",
    )

    # Verify result
    assert isinstance(result, dict)
    assert result["process_id"] == "proc_123"
    assert result["exit_code"] == 0
    assert result["status"] == "completed"
    assert result["combined"] == "Final output"

    # Verify process manager was called
    process_tools.process_manager.wait_for_exit.assert_called_once_with("proc_123")
    process_tools.process_manager.get_output.assert_called_once_with("proc_123")

    # Verify event was emitted
    agent_ctx.events.process_exit.assert_called_once()
    call_args = agent_ctx.events.process_exit.call_args[1]
    assert call_args["process_id"] == "proc_123"
    assert call_args["exit_code"] == 0


async def test_kill_process_success(process_tools: ProcessTools, agent_ctx: AgentContext):
    """Test killing a process."""
    tools = await process_tools.get_tools()
    kill_tool = next(tool for tool in tools if tool.name == "kill_process")

    result = await kill_tool.execute(
        agent_ctx=agent_ctx,
        process_id="proc_123",
    )

    # Verify result
    assert isinstance(result, dict)
    assert result["process_id"] == "proc_123"
    assert result["status"] == "killed"

    # Verify process manager was called
    process_tools.process_manager.kill_process.assert_called_once_with("proc_123")

    # Verify event was emitted
    agent_ctx.events.process_killed.assert_called_once()
    call_args = agent_ctx.events.process_killed.call_args[1]
    assert call_args["process_id"] == "proc_123"
    assert call_args["success"] is True


async def test_kill_process_failure(process_tools: ProcessTools, agent_ctx: AgentContext):
    """Test killing a process that fails."""
    process_tools.process_manager.kill_process.side_effect = ValueError("Process not found")

    tools = await process_tools.get_tools()
    kill_tool = next(tool for tool in tools if tool.name == "kill_process")

    result = await kill_tool.execute(
        agent_ctx=agent_ctx,
        process_id="proc_999",
    )

    # Verify error result
    assert isinstance(result, dict)
    assert "error" in result
    assert "Process not found" in result["error"]

    # Verify failure event was emitted
    agent_ctx.events.process_killed.assert_called_once()
    call_args = agent_ctx.events.process_killed.call_args[1]
    assert call_args["success"] is False


async def test_release_process_success(process_tools: ProcessTools, agent_ctx: AgentContext):
    """Test releasing process resources."""
    tools = await process_tools.get_tools()
    release_tool = next(tool for tool in tools if tool.name == "release_process")

    result = await release_tool.execute(
        agent_ctx=agent_ctx,
        process_id="proc_123",
    )

    # Verify result
    assert isinstance(result, dict)
    assert result["process_id"] == "proc_123"
    assert result["status"] == "released"

    # Verify process manager was called
    process_tools.process_manager.release_process.assert_called_once_with("proc_123")

    # Verify event was emitted
    agent_ctx.events.process_released.assert_called_once()


async def test_list_processes_success(process_tools: ProcessTools, agent_ctx: AgentContext):
    """Test listing processes."""
    tools = await process_tools.get_tools()
    list_tool = next(tool for tool in tools if tool.name == "list_processes")

    result = await list_tool.execute(agent_ctx=agent_ctx)

    # Verify result
    assert isinstance(result, dict)
    assert "processes" in result
    assert result["count"] == 2  # noqa: PLR2004
    assert len(result["processes"]) == 2  # noqa: PLR2004

    # Verify process manager was called
    process_tools.process_manager.list_processes.assert_called_once()
    assert process_tools.process_manager.get_process_info.call_count == 2  # noqa: PLR2004


async def test_list_processes_empty(process_tools: ProcessTools, agent_ctx: AgentContext):
    """Test listing processes when none exist."""
    process_tools.process_manager.list_processes.return_value = []

    tools = await process_tools.get_tools()
    list_tool = next(tool for tool in tools if tool.name == "list_processes")

    result = await list_tool.execute(agent_ctx=agent_ctx)

    # Verify result
    assert isinstance(result, dict)
    assert result["processes"] == []
    assert result["count"] == 0
    assert "No active processes" in result["message"]


async def test_generic_process_toolset_events(process_tools: ProcessTools, agent_ctx: AgentContext):
    """Test that process toolset emits clean, generic events."""
    tools = await process_tools.get_tools()
    start_tool = next(tool for tool in tools if tool.name == "start_process")

    await start_tool.execute(
        agent_ctx=agent_ctx,
        command="echo",
        args=["test"],
    )

    # Verify event does NOT include transport-specific fields like terminal_id
    agent_ctx.events.process_started.assert_called_once()
    call_args = agent_ctx.events.process_started.call_args[1]
    assert "terminal_id" not in call_args  # Should be generic
    assert call_args["process_id"] == "proc_123"
    assert call_args["command"] == "echo"
    assert call_args["success"] is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
