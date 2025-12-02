"""Test integration of ExecutionEnvironmentTools with ACP session."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from acp import ClientCapabilities
from llmling_agent import AgentContext, AgentPool
from llmling_agent.agent.event_emitter import AgentEventEmitter
from llmling_agent.models.agents import AgentConfig
from llmling_agent_server.acp_server.acp_agent import LLMlingACPAgent
from llmling_agent_server.acp_server.session import ACPSession
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


CTX = create_mock_agent_context()


@pytest.fixture
def mock_connection():
    """Create a mock ACP connection."""
    return Mock()


@pytest.fixture
def mock_agent_pool() -> AgentPool:
    """Create a mock agent pool."""
    pool = Mock(spec=AgentPool)
    pool.agents = {}
    return pool


@pytest.fixture
async def acp_agent(mock_connection, mock_agent_pool: AgentPool):
    """Create ACP agent."""
    mock_agent = Mock()
    mock_tools = {}

    def register_tool(tool):
        mock_tools[tool.name] = tool

    mock_agent.tools = Mock()
    mock_agent.tools.register_tool = register_tool
    mock_agent_pool.agents = {"test_agent": mock_agent}  # pyright: ignore[reportAttributeAccessIssue]

    return LLMlingACPAgent(
        connection=mock_connection,
        agent_pool=mock_agent_pool,
        terminal_access=True,
    )


@pytest.fixture
async def session(acp_agent: LLMlingACPAgent, mock_connection):
    """Create test session."""
    capabilities = ClientCapabilities(terminal=True)

    return ACPSession(
        session_id="test_session",
        agent_pool=acp_agent.agent_pool,
        current_agent_name="test_agent",
        cwd="/test",
        client=mock_connection,
        acp_agent=acp_agent,
        client_capabilities=capabilities,
    )


@pytest.fixture
async def execution_tools(session: ACPSession):
    """Create execution environment tools provider for testing."""
    from anyenv.code_execution.acp_provider import ACPExecutionEnvironment

    env = ACPExecutionEnvironment(fs=session.fs, requests=session.requests)
    return ExecutionEnvironmentTools(env=env, name="test_execution_tools")


async def test_start_process_with_acp_session(
    execution_tools: ExecutionEnvironmentTools,
    session: ACPSession,
):
    """Test starting process with ACP session."""
    mock_response = Mock()
    mock_response.terminal_id = "term_123"

    # Mock the requests object inside the execution environment directly
    mock_requests = Mock()
    mock_requests.create_terminal = AsyncMock(return_value=mock_response)
    execution_tools.env._requests = mock_requests  # pyright: ignore[reportAttributeAccessIssue]
    # Reset process manager so it picks up new requests
    execution_tools.env._process_manager = None

    tools = await execution_tools.get_tools()
    start_tool = next(tool for tool in tools if tool.name == "start_process")

    result = await start_tool.execute(
        agent_ctx=CTX,
        command="echo",
        args=["hello", "world"],
        cwd="/tmp",
    )

    assert isinstance(result, dict)
    assert result["process_id"] == "term_123"
    assert result["command"] == "echo"
    assert result["args"] == ["hello", "world"]
    assert result["status"] == "started"

    mock_requests.create_terminal.assert_called_once()
    call_args = mock_requests.create_terminal.call_args[1]
    assert call_args["command"] == "echo"
    assert call_args["args"] == ["hello", "world"]
    assert call_args["cwd"] == "/tmp"

    CTX.events.process_started.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]
    call_args = CTX.events.process_started.call_args[1]  # pyright: ignore[reportAttributeAccessIssue]
    assert "terminal_id" not in call_args
    assert call_args["process_id"] == "term_123"
    assert call_args["command"] == "echo"
    assert call_args["success"] is True


async def test_get_process_output_with_acp_session(
    execution_tools: ExecutionEnvironmentTools,
    session: ACPSession,
):
    """Test getting process output through ACP."""
    mock_terminal_response = Mock()
    mock_terminal_response.terminal_id = "term_123"

    # Mock the requests object inside the execution environment directly
    mock_requests = Mock()
    mock_requests.create_terminal = AsyncMock(return_value=mock_terminal_response)
    execution_tools.env._requests = mock_requests  # pyright: ignore[reportAttributeAccessIssue]
    # Reset process manager so it picks up new requests
    execution_tools.env._process_manager = None

    await execution_tools.env.process_manager.start_process("echo", ["test"])
    process_id = next(iter(execution_tools.env.process_manager._processes.keys()))  # pyright: ignore[reportAttributeAccessIssue]

    mock_output_response = Mock()
    mock_output_response.output = "test output\n"
    mock_output_response.truncated = False
    mock_requests.terminal_output = AsyncMock(return_value=mock_output_response)

    tools = await execution_tools.get_tools()
    output_tool = next(tool for tool in tools if tool.name == "get_process_output")

    result = await output_tool.execute(
        agent_ctx=CTX,
        process_id=process_id,
    )

    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["stdout"] == "test output\n"
    assert result["combined"] == "test output\n"
    assert result["status"] == "running"

    mock_requests.terminal_output.assert_called_once_with("term_123")
    CTX.events.process_output.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]


async def test_kill_process_with_acp_session(
    execution_tools: ExecutionEnvironmentTools,
    session: ACPSession,
):
    """Test killing process through ACP."""
    mock_terminal_response = Mock()
    mock_terminal_response.terminal_id = "term_123"

    # Mock the requests object inside the execution environment directly
    mock_requests = Mock()
    mock_requests.create_terminal = AsyncMock(return_value=mock_terminal_response)
    mock_requests.kill_terminal = AsyncMock()
    execution_tools.env._requests = mock_requests  # pyright: ignore[reportAttributeAccessIssue]
    # Reset process manager so it picks up new requests
    execution_tools.env._process_manager = None

    await execution_tools.env.process_manager.start_process("sleep", ["10"])
    process_id = next(iter(execution_tools.env.process_manager._processes.keys()))  # pyright: ignore[reportAttributeAccessIssue]

    tools = await execution_tools.get_tools()
    kill_tool = next(tool for tool in tools if tool.name == "kill_process")

    result = await kill_tool.execute(
        agent_ctx=CTX,
        process_id=process_id,
    )

    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["status"] == "killed"

    mock_requests.kill_terminal.assert_called_once_with("term_123")

    CTX.events.process_killed.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]
    call_args = CTX.events.process_killed.call_args[1]  # pyright: ignore[reportAttributeAccessIssue]
    assert "terminal_id" not in call_args
    assert call_args["process_id"] == process_id
    assert call_args["success"] is True


async def test_execution_tools_use_acp_environment(session: ACPSession):
    """Test that execution tools use ACP execution environment."""
    from anyenv.code_execution.acp_provider import ACPExecutionEnvironment

    provider = session._acp_provider
    assert provider is not None, "ACP provider not initialized"

    execution_provider = None
    for p in provider.providers:
        if isinstance(p, ExecutionEnvironmentTools):
            execution_provider = p
            break

    assert execution_provider is not None, "ExecutionEnvironmentTools provider not found"
    assert isinstance(execution_provider.env, ACPExecutionEnvironment)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
