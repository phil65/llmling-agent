"""Test integration of ExecutionEnvironmentTools with MockExecutionEnvironment."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

from anyenv.code_execution import MockExecutionEnvironment
from anyenv.code_execution.models import ExecutionResult
from anyenv.process_manager.models import ProcessOutput
import pytest

from llmling_agent import AgentContext
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
def mock_ctx() -> AgentContext:
    """Create a fresh mock context for each test."""
    return create_mock_agent_context()


@pytest.fixture
def mock_env() -> MockExecutionEnvironment:
    """Create mock execution environment with predefined responses."""
    return MockExecutionEnvironment(
        command_results={
            "echo hello world": ExecutionResult(
                result=None,
                duration=0.01,
                success=True,
                stdout="hello world\n",
                exit_code=0,
            ),
        },
        process_outputs={
            "echo": ProcessOutput(
                stdout="hello world\n",
                stderr="",
                combined="hello world\n",
                exit_code=0,
            ),
            "sleep": ProcessOutput(
                stdout="",
                stderr="",
                combined="",
                exit_code=0,
            ),
        },
    )


@pytest.fixture
def execution_tools(mock_env: MockExecutionEnvironment) -> ExecutionEnvironmentTools:
    """Create execution environment tools with mock environment."""
    return ExecutionEnvironmentTools(env=mock_env, name="test_execution_tools")


async def test_start_process(
    execution_tools: ExecutionEnvironmentTools,
    mock_ctx: AgentContext,
):
    """Test starting a process through execution tools."""
    tools = await execution_tools.get_tools()
    start_tool = next(tool for tool in tools if tool.name == "start_process")

    result = await start_tool.execute(
        agent_ctx=mock_ctx,
        command="echo",
        args=["hello", "world"],
        cwd="/tmp",
    )

    assert isinstance(result, dict)
    assert result["process_id"].startswith("mock_")
    assert result["command"] == "echo"
    assert result["args"] == ["hello", "world"]
    assert result["status"] == "started"

    mock_ctx.events.process_started.assert_called_once()
    call_args = mock_ctx.events.process_started.call_args[1]
    assert call_args["command"] == "echo"
    assert call_args["success"] is True


async def test_get_process_output(
    execution_tools: ExecutionEnvironmentTools,
    mock_env: MockExecutionEnvironment,
    mock_ctx: AgentContext,
):
    """Test getting process output."""
    # Start a process first
    process_id = await mock_env.process_manager.start_process("echo", ["hello", "world"])

    tools = await execution_tools.get_tools()
    output_tool = next(tool for tool in tools if tool.name == "get_process_output")

    result = await output_tool.execute(
        agent_ctx=mock_ctx,
        process_id=process_id,
    )

    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["stdout"] == "hello world\n"
    assert result["combined"] == "hello world\n"

    mock_ctx.events.process_output.assert_called_once()


async def test_kill_process(
    execution_tools: ExecutionEnvironmentTools,
    mock_env: MockExecutionEnvironment,
    mock_ctx: AgentContext,
):
    """Test killing a process."""
    # Start a process first
    process_id = await mock_env.process_manager.start_process("sleep", ["10"])

    tools = await execution_tools.get_tools()
    kill_tool = next(tool for tool in tools if tool.name == "kill_process")

    result = await kill_tool.execute(
        agent_ctx=mock_ctx,
        process_id=process_id,
    )

    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["status"] == "killed"

    mock_ctx.events.process_killed.assert_called_once()
    call_args = mock_ctx.events.process_killed.call_args[1]
    assert call_args["process_id"] == process_id
    assert call_args["success"] is True


async def test_wait_for_process(
    execution_tools: ExecutionEnvironmentTools,
    mock_env: MockExecutionEnvironment,
    mock_ctx: AgentContext,
):
    """Test waiting for process completion."""
    # Start a process first
    process_id = await mock_env.process_manager.start_process("echo", ["test"])

    tools = await execution_tools.get_tools()
    wait_tool = next(tool for tool in tools if tool.name == "wait_for_process")

    result = await wait_tool.execute(
        agent_ctx=mock_ctx,
        process_id=process_id,
    )

    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["exit_code"] == 0
    assert result["status"] == "completed"

    mock_ctx.events.process_exit.assert_called_once()


async def test_release_process(
    execution_tools: ExecutionEnvironmentTools,
    mock_env: MockExecutionEnvironment,
    mock_ctx: AgentContext,
):
    """Test releasing process resources."""
    # Start a process first
    process_id = await mock_env.process_manager.start_process("echo", ["test"])

    tools = await execution_tools.get_tools()
    release_tool = next(tool for tool in tools if tool.name == "release_process")

    result = await release_tool.execute(
        agent_ctx=mock_ctx,
        process_id=process_id,
    )

    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["status"] == "released"

    # Verify process is no longer tracked
    processes = await mock_env.process_manager.list_processes()
    assert process_id not in processes

    mock_ctx.events.process_released.assert_called_once()


async def test_list_processes(
    execution_tools: ExecutionEnvironmentTools,
    mock_env: MockExecutionEnvironment,
    mock_ctx: AgentContext,
):
    """Test listing all processes."""
    # Start some processes
    pid1 = await mock_env.process_manager.start_process("echo", ["1"])
    pid2 = await mock_env.process_manager.start_process("echo", ["2"])

    tools = await execution_tools.get_tools()
    list_tool = next(tool for tool in tools if tool.name == "list_processes")

    result = await list_tool.execute(agent_ctx=mock_ctx)

    assert isinstance(result, dict)
    assert "processes" in result
    process_ids = [p["process_id"] for p in result["processes"]]
    assert pid1 in process_ids
    assert pid2 in process_ids


async def test_execute_command(
    execution_tools: ExecutionEnvironmentTools,
    mock_ctx: AgentContext,
):
    """Test executing a command directly."""
    tools = await execution_tools.get_tools()
    cmd_tool = next(tool for tool in tools if tool.name == "execute_command")

    result = await cmd_tool.execute(
        agent_ctx=mock_ctx,
        command="echo hello world",
    )

    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["stdout"] == "hello world\n"
    assert result["exit_code"] == 0


async def test_process_not_found(
    execution_tools: ExecutionEnvironmentTools,
    mock_ctx: AgentContext,
):
    """Test error handling for non-existent process."""
    tools = await execution_tools.get_tools()
    output_tool = next(tool for tool in tools if tool.name == "get_process_output")

    result = await output_tool.execute(
        agent_ctx=mock_ctx,
        process_id="nonexistent_process",
    )

    assert isinstance(result, dict)
    assert "error" in result
    assert "not found" in result["error"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
