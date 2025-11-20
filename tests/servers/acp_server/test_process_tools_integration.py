"""Test integration of ProcessTools with ACP session."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from acp import ClientCapabilities
from llmling_agent import AgentContext, AgentPool
from llmling_agent.agent.event_emitter import AgentEventEmitter
from llmling_agent.models.agents import AgentConfig
from llmling_agent_server.acp_server.acp_agent import LLMlingACPAgent
from llmling_agent_server.acp_server.session import ACPSession
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
    # Create mock agent
    mock_agent = Mock()
    mock_tools = {}

    def register_tool(tool):
        mock_tools[tool.name] = tool

    mock_agent.tools = Mock()
    mock_agent.tools.register_tool = register_tool
    mock_agent_pool.agents = {"test_agent": mock_agent}

    # Create ACP agent (terminal access enabled for process tools)
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
async def process_tools(session: ACPSession):
    """Create process tools provider for testing."""
    return ProcessTools(session.process_manager, name="test_process_tools")


async def test_start_process_with_acp_session(
    process_tools: ProcessTools,
    session: ACPSession,
):
    """Test starting process with ACP session."""
    # Mock the ACP terminal creation response
    mock_response = Mock()
    mock_response.terminal_id = "term_123"
    session.requests = Mock()
    session.requests.create_terminal = AsyncMock(return_value=mock_response)

    # Get start_process tool
    tools = await process_tools.get_tools()
    start_tool = next(tool for tool in tools if tool.name == "start_process")

    result = await start_tool.execute(
        agent_ctx=CTX,
        command="echo",
        args=["hello", "world"],
        cwd="/tmp",
    )

    # Verify result contains terminal ID as process ID
    assert isinstance(result, dict)
    assert result["process_id"] == "term_123"  # terminal_id is now used directly
    assert result["command"] == "echo"
    assert result["args"] == ["hello", "world"]
    assert result["status"] == "started"

    # Verify ACP terminal creation was called
    session.requests.create_terminal.assert_called_once()
    call_args = session.requests.create_terminal.call_args[1]
    assert call_args["command"] == "echo"
    assert call_args["args"] == ["hello", "world"]
    assert call_args["cwd"] == "/tmp"

    # Verify process is tracked
    assert len(session.process_manager._processes) == 1
    process = next(iter(session.process_manager._processes.values()))
    assert process.terminal_id == "term_123"
    assert process.command == "echo"

    # Verify generic event was emitted (session will add ACP context)
    CTX.events.process_started.assert_called_once()
    call_args = CTX.events.process_started.call_args[1]
    assert "terminal_id" not in call_args  # Generic event shouldn't have ACP fields
    assert call_args["process_id"] == "term_123"  # terminal_id is now used directly
    assert call_args["command"] == "echo"
    assert call_args["success"] is True


async def test_get_process_output_with_acp_session(
    process_tools: ProcessTools,
    session: ACPSession,
):
    """Test getting process output through ACP."""
    # Set up a tracked process
    mock_terminal_response = Mock()
    mock_terminal_response.terminal_id = "term_123"
    session.requests = Mock()
    session.requests.create_terminal = AsyncMock(return_value=mock_terminal_response)

    # Start a process first
    await session.process_manager.start_process("echo", ["test"])
    process_id = next(iter(session.process_manager._processes.keys()))

    # Mock terminal output response
    mock_output_response = Mock()
    mock_output_response.output = "test output\n"
    mock_output_response.truncated = False
    session.requests.terminal_output = AsyncMock(return_value=mock_output_response)

    # Get process output
    tools = await process_tools.get_tools()
    output_tool = next(tool for tool in tools if tool.name == "get_process_output")

    result = await output_tool.execute(
        agent_ctx=CTX,
        process_id=process_id,
    )

    # Verify result
    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["stdout"] == "test output\n"
    assert result["combined"] == "test output\n"
    assert result["status"] == "running"

    # Verify ACP terminal output was called
    session.requests.terminal_output.assert_called_once_with("term_123")

    # Verify event was emitted
    CTX.events.process_output.assert_called_once()


async def test_kill_process_with_acp_session(
    process_tools: ProcessTools,
    session: ACPSession,
):
    """Test killing process through ACP."""
    # Set up a tracked process
    mock_terminal_response = Mock()
    mock_terminal_response.terminal_id = "term_123"
    session.requests = Mock()
    session.requests.create_terminal = AsyncMock(return_value=mock_terminal_response)
    session.requests.kill_terminal = AsyncMock()

    # Start a process first
    await session.process_manager.start_process("sleep", ["10"])
    process_id = next(iter(session.process_manager._processes.keys()))

    # Kill the process
    tools = await process_tools.get_tools()
    kill_tool = next(tool for tool in tools if tool.name == "kill_process")

    result = await kill_tool.execute(
        agent_ctx=CTX,
        process_id=process_id,
    )

    # Verify result
    assert isinstance(result, dict)
    assert result["process_id"] == process_id
    assert result["status"] == "killed"

    # Verify ACP kill terminal was called
    session.requests.kill_terminal.assert_called_once_with("term_123")

    # Verify generic event was emitted (session will add ACP context)
    CTX.events.process_killed.assert_called_once()
    call_args = CTX.events.process_killed.call_args[1]
    assert "terminal_id" not in call_args  # Generic event shouldn't have ACP fields
    assert call_args["process_id"] == process_id
    assert call_args["success"] is True


async def test_process_tools_availability_in_acp_provider(session: ACPSession):
    """Test that process tools are available through ACP provider."""
    # Get the ACP provider
    provider = session._acp_provider
    assert provider is not None, "ACP provider not initialized"

    # Get all tools
    tools = await provider.get_tools()
    tool_names = [tool.name for tool in tools]

    # Verify process tools are available
    expected_process_tools = [
        "start_process",
        "get_process_output",
        "wait_for_process",
        "kill_process",
        "release_process",
        "list_processes",
    ]

    for tool_name in expected_process_tools:
        assert tool_name in tool_names, f"Process tool '{tool_name}' not found in ACP provider"


async def test_process_tools_have_acp_process_manager(session: ACPSession):
    """Test that process tools use ACP process manager."""
    provider = session._acp_provider
    assert provider is not None, "ACP provider not initialized"

    # Find the ProcessTools provider
    process_provider = None
    for p in provider.providers:
        if hasattr(p, "process_manager"):
            process_provider = p
            break

    assert process_provider is not None, "ProcessTools provider not found"

    # Verify it uses the ACP process manager
    assert process_provider.process_manager is session.process_manager
    assert hasattr(process_provider.process_manager, "_processes")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
