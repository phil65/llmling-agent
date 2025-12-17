"""Test integration of ExecutionEnvironmentTools with MockExecutionEnvironment."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from anyenv.process_manager.models import ProcessOutput
from exxec import MockExecutionEnvironment
from exxec.models import ExecutionResult
import pytest

from llmling_agent import Agent, AgentContext
from llmling_agent.agents.events import ToolCallProgressEvent
from llmling_agent.models.agents import AgentConfig
from llmling_agent.models.manifest import AgentsManifest
from llmling_agent_toolsets.builtin.execution_environment import ExecutionEnvironmentTools


if TYPE_CHECKING:
    from llmling_agent.agents.events import RichAgentStreamEvent


def drain_event_queue(agent: Agent) -> list[RichAgentStreamEvent]:
    """Drain all events from the agent's event queue."""
    events: list[RichAgentStreamEvent] = []
    while not agent._event_queue.empty():
        try:
            events.append(agent._event_queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return events


def get_progress_events(agent: Agent) -> list[ToolCallProgressEvent]:
    """Get all ToolCallProgressEvent from the agent's queue."""
    events = drain_event_queue(agent)
    return [e for e in events if isinstance(e, ToolCallProgressEvent)]


@pytest.fixture
def test_agent() -> Agent[None]:
    """Create a minimal agent for testing event emission."""
    return Agent(name="test_agent")


@pytest.fixture
def agent_ctx(test_agent: Agent[None]) -> AgentContext:
    """Create a real AgentContext for testing."""
    return AgentContext(
        node=test_agent,
        config=AgentConfig(name="test_agent"),
        definition=AgentsManifest(),
        tool_call_id="test_call_123",
        tool_name="test_tool",
        tool_input={"command": "echo", "args": ["hello"]},
    )


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
    agent_ctx: AgentContext,
    test_agent: Agent[None],
):
    """Test starting a process through execution tools."""
    tools = await execution_tools.get_tools()
    start_tool = next(tool for tool in tools if tool.name == "start_process")

    result = await start_tool.execute(
        agent_ctx=agent_ctx,
        command="echo",
        args=["hello", "world"],
        cwd="/tmp",
    )

    assert isinstance(result, dict)
    assert result["process_id"].startswith("mock_")
    assert result["args"] == ["hello", "world"]
    assert result["status"] == "started"

    # Check event was emitted to the queue
    events = get_progress_events(test_agent)
    assert len(events) == 1
    assert events[0].title is not None
    assert "Running: echo" in events[0].title


async def test_get_process_output(
    execution_tools: ExecutionEnvironmentTools,
    mock_env: MockExecutionEnvironment,
    agent_ctx: AgentContext,
    test_agent: Agent[None],
):
    """Test getting process output."""
    # Start a process first
    process_id = await mock_env.process_manager.start_process("echo", ["hello", "world"])

    tools = await execution_tools.get_tools()
    output_tool = next(tool for tool in tools if tool.name == "get_process_output")

    result = await output_tool.execute(
        agent_ctx=agent_ctx,
        process_id=process_id,
    )

    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["stdout"] == "hello world\n"
    assert result["combined"] == "hello world\n"

    # Check event was emitted (title contains output)
    events = get_progress_events(test_agent)
    assert len(events) == 1
    assert events[0].title is not None
    assert "hello world" in events[0].title


async def test_kill_process(
    execution_tools: ExecutionEnvironmentTools,
    mock_env: MockExecutionEnvironment,
    agent_ctx: AgentContext,
    test_agent: Agent[None],
):
    """Test killing a process."""
    # Start a process first
    process_id = await mock_env.process_manager.start_process("sleep", ["10"])

    tools = await execution_tools.get_tools()
    kill_tool = next(tool for tool in tools if tool.name == "kill_process")

    result = await kill_tool.execute(
        agent_ctx=agent_ctx,
        process_id=process_id,
    )

    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["status"] == "killed"

    # Check event was emitted
    events = get_progress_events(test_agent)
    assert len(events) == 1
    assert events[0].title is not None
    assert "Killed process" in events[0].title
    assert process_id in events[0].title


async def test_wait_for_process(
    execution_tools: ExecutionEnvironmentTools,
    mock_env: MockExecutionEnvironment,
    agent_ctx: AgentContext,
    test_agent: Agent[None],
):
    """Test waiting for process completion."""
    # Start a process first
    process_id = await mock_env.process_manager.start_process("echo", ["test"])

    tools = await execution_tools.get_tools()
    wait_tool = next(tool for tool in tools if tool.name == "wait_for_process")

    result = await wait_tool.execute(
        agent_ctx=agent_ctx,
        process_id=process_id,
    )

    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["exit_code"] == 0
    assert result["status"] == "completed"

    # Check event was emitted
    events = get_progress_events(test_agent)
    assert len(events) == 1
    assert events[0].title is not None
    assert "Process exited" in events[0].title
    assert "exit 0" in events[0].title


async def test_release_process(
    execution_tools: ExecutionEnvironmentTools,
    mock_env: MockExecutionEnvironment,
    agent_ctx: AgentContext,
    test_agent: Agent[None],
):
    """Test releasing process resources."""
    # Start a process first
    process_id = await mock_env.process_manager.start_process("echo", ["test"])

    tools = await execution_tools.get_tools()
    release_tool = next(tool for tool in tools if tool.name == "release_process")

    result = await release_tool.execute(
        agent_ctx=agent_ctx,
        process_id=process_id,
    )

    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["status"] == "released"

    # Verify process is no longer tracked
    processes = await mock_env.process_manager.list_processes()
    assert process_id not in processes

    # Check event was emitted
    events = get_progress_events(test_agent)
    assert len(events) == 1
    assert events[0].title is not None
    assert "Released process" in events[0].title
    assert process_id in events[0].title


async def test_list_processes(
    execution_tools: ExecutionEnvironmentTools,
    mock_env: MockExecutionEnvironment,
    agent_ctx: AgentContext,
):
    """Test listing all processes."""
    # Start some processes
    pid1 = await mock_env.process_manager.start_process("echo", ["1"])
    pid2 = await mock_env.process_manager.start_process("echo", ["2"])

    tools = await execution_tools.get_tools()
    list_tool = next(tool for tool in tools if tool.name == "list_processes")

    result = await list_tool.execute(agent_ctx=agent_ctx)

    assert isinstance(result, dict)
    assert "processes" in result
    process_ids = [p["process_id"] for p in result["processes"]]
    assert pid1 in process_ids
    assert pid2 in process_ids


async def test_execute_command(
    execution_tools: ExecutionEnvironmentTools,
    agent_ctx: AgentContext,
    test_agent: Agent[None],
):
    """Test executing a command directly."""
    tools = await execution_tools.get_tools()
    cmd_tool = next(tool for tool in tools if tool.name == "execute_command")

    result = await cmd_tool.execute(
        agent_ctx=agent_ctx,
        command="echo hello world",
    )

    assert isinstance(result, dict)
    assert result["stdout"] == "hello world\n"
    assert result["exit_code"] == 0

    # Check events were emitted (start + output + exit)
    events = get_progress_events(test_agent)
    assert len(events) >= 1  # At least process start event


async def test_process_not_found(
    execution_tools: ExecutionEnvironmentTools,
    agent_ctx: AgentContext,
):
    """Test error handling for non-existent process."""
    tools = await execution_tools.get_tools()
    output_tool = next(tool for tool in tools if tool.name == "get_process_output")

    result = await output_tool.execute(
        agent_ctx=agent_ctx,
        process_id="nonexistent_process",
    )

    assert isinstance(result, dict)
    assert "error" in result
    assert "not found" in result["error"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
