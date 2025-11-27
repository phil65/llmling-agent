"""Tests for client-side filesystem tools that make ACP requests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest

from acp import AgentSideConnection, ClientCapabilities, FileSystemCapability, InitializeRequest
from llmling_agent import AgentContext, AgentPool
from llmling_agent.agent.event_emitter import AgentEventEmitter
from llmling_agent.models.agents import AgentConfig
from llmling_agent_server.acp_server.acp_agent import LLMlingACPAgent
from llmling_agent_toolsets.fsspec_toolset import FSSpecTools


if TYPE_CHECKING:
    from llmling_agent_server.acp_server.session import ACPSession


def create_mock_agent_context() -> AgentContext:
    """Create a mock AgentContext for testing."""
    context = Mock(spec=AgentContext)
    context.node_name = "test_agent"
    context.events = Mock(spec=AgentEventEmitter)
    context.events.file_operation = AsyncMock()
    context.config = Mock(spec=AgentConfig)
    return context


CTX = create_mock_agent_context()


@pytest.fixture
def mock_connection():
    """Create a mock ACP connection."""
    return Mock(spec=AgentSideConnection)


@pytest.fixture
def mock_agent_pool() -> AgentPool:
    """Create a mock agent pool."""
    pool = Mock(spec=AgentPool)
    pool.agents = {}
    return pool


@pytest.fixture
async def acp_agent(mock_connection: AgentSideConnection, mock_agent_pool: AgentPool):
    """Create ACP agent with filesystem support."""
    # Create mock agent
    mock_agent = Mock()
    mock_tools = {}

    def register_tool(tool):
        mock_tools[tool.name] = tool

    mock_agent.tools = Mock()
    mock_agent.tools.register_tool = register_tool
    mock_agent_pool.agents = {"test_agent": mock_agent}  # pyright: ignore[reportAttributeAccessIssue]

    # Create ACP agent (filesystem tools are always registered)
    agent = LLMlingACPAgent(
        connection=mock_connection,
        agent_pool=mock_agent_pool,
        terminal_access=False,  # Disable terminal tools for cleaner testing
    )

    # Initialize with filesystem capabilities
    fs_cap = FileSystemCapability(read_text_file=True, write_text_file=True)
    client_cap = ClientCapabilities(fs=fs_cap, terminal=False)
    request = InitializeRequest(protocol_version=1, client_capabilities=client_cap)
    await agent.initialize(request)
    return agent


@pytest.fixture
async def session(acp_agent: LLMlingACPAgent, mock_connection: AgentSideConnection):
    """Create test session."""
    from acp.schema import ClientCapabilities, FileSystemCapability
    from llmling_agent_server.acp_server.session import ACPSession

    fs_cap = FileSystemCapability(read_text_file=True, write_text_file=True)
    capabilities = ClientCapabilities(fs=fs_cap, terminal=False)

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
async def fs_provider(session: ACPSession):
    """Create filesystem capability provider for testing."""
    # Mock the ACP filesystem operations
    session.fs._cat_file = AsyncMock()
    session.fs._put_file = AsyncMock()
    session.fs._info = AsyncMock()
    return FSSpecTools(filesystem=session.fs)


async def test_read_file_success(fs_provider: FSSpecTools):
    """Test successful file reading."""
    # Mock filesystem read operation
    fs_provider.fs._cat_file = AsyncMock(return_value=b"Hello, World!\nThis is a test file.\n")

    # Get read_file tool from provider
    tools = await fs_provider.get_tools()
    read_tool = next(tool for tool in tools if tool.name == "read_file")
    result = await read_tool.execute(agent_ctx=CTX, path="/home/user/test.txt")

    # read_file returns content directly as string for text files
    assert isinstance(result, str)
    assert "Hello, World!" in result
    assert "This is a test file." in result

    # Verify filesystem call was made
    fs_provider.fs._cat_file.assert_called_once_with("/home/user/test.txt")


async def test_read_file_with_line_and_limit(fs_provider: FSSpecTools):
    """Test file reading with line and limit parameters."""
    # Mock filesystem read operation
    fs_provider.fs._cat_file = AsyncMock(return_value=b"Line 1\nLine 2\nLine 3\nLine 4\nLine 5")

    # Get read_file tool from provider
    tools = await fs_provider.get_tools()
    read_tool = next(tool for tool in tools if tool.name == "read_file")
    result = await read_tool.execute(
        agent_ctx=CTX,
        path="/home/user/large_file.txt",
        line=2,
        limit=2,
    )

    # Verify result - should contain lines 2 and 3 (line parameter is 1-based)
    assert isinstance(result, str)
    content_lines = result.split("\n")
    assert "Line 2" in content_lines
    assert "Line 3" in content_lines
    assert len(content_lines) == 2  # noqa: PLR2004

    # Verify filesystem call was made
    fs_provider.fs._cat_file.assert_called_once_with("/home/user/large_file.txt")


async def test_read_file_error(fs_provider: FSSpecTools):
    """Test file reading error handling."""
    # Mock filesystem read error
    fs_provider.fs._cat_file = AsyncMock(side_effect=FileNotFoundError("File not found"))

    # Get read_file tool from provider
    tools = await fs_provider.get_tools()
    read_tool = next(tool for tool in tools if tool.name == "read_file")

    result = await read_tool.execute(agent_ctx=CTX, path="/home/user/nonexistent.txt")

    # Verify error handling - FSSpec tools return error dict on failure
    assert isinstance(result, dict)
    assert "error" in result
    assert "File not found" in result["error"]


async def test_write_text_file_success(fs_provider: FSSpecTools):
    """Test successful file writing."""
    # Mock filesystem write operations
    mock_file = AsyncMock()
    mock_file.write = AsyncMock()
    fs_provider.fs.open_async = AsyncMock(return_value=mock_file)
    fs_provider.fs._info = AsyncMock(return_value={"size": 42})

    # Get write_text_file tool from provider
    tools = await fs_provider.get_tools()
    write_tool = next(tool for tool in tools if tool.name == "write_text_file")
    result = await write_tool.execute(
        agent_ctx=CTX,
        path="/home/user/output.txt",
        content="Hello, World!\nThis is written content.\n",
    )

    # Verify result format
    assert isinstance(result, dict)
    assert result["path"] == "/home/user/output.txt"
    assert "bytes_written" in result
    assert "size" in result

    # Verify filesystem calls were made
    fs_provider.fs.open_async.assert_called_once_with("/home/user/output.txt", "wt")
    mock_file.write.assert_called_once_with("Hello, World!\nThis is written content.\n")


async def test_write_text_file_json(fs_provider: FSSpecTools):
    """Test writing JSON content."""
    # Mock filesystem write operations
    mock_file = AsyncMock()
    mock_file.write = AsyncMock()
    fs_provider.fs.open_async = AsyncMock(return_value=mock_file)
    fs_provider.fs._info = AsyncMock(return_value={"size": 42})

    json_str = '{\n  "debug": true,\n  "version": "1.0.0"\n}'

    # Get write_text_file tool from provider
    tools = await fs_provider.get_tools()
    write_tool = next(tool for tool in tools if tool.name == "write_text_file")
    result = await write_tool.execute(
        agent_ctx=CTX,
        path="/home/user/config.json",
        content=json_str,
    )

    # Verify result format
    assert isinstance(result, dict)
    assert result["path"] == "/home/user/config.json"

    # Verify content was written correctly
    fs_provider.fs.open_async.assert_called_once_with("/home/user/config.json", "wt")
    mock_file.write.assert_called_once_with(json_str)


async def test_write_text_file_error(fs_provider: FSSpecTools):
    """Test file writing error handling."""
    # Mock filesystem write error
    fs_provider.fs.open_async = AsyncMock(side_effect=PermissionError("Permission denied"))

    # Get write_text_file tool from provider
    tools = await fs_provider.get_tools()
    write_tool = next(tool for tool in tools if tool.name == "write_text_file")
    result = await write_tool.execute(
        agent_ctx=CTX,
        path="/root/protected.txt",
        content="This should fail",
    )

    # Verify error handling
    assert isinstance(result, dict)
    assert "error" in result
    assert "Permission denied" in result["error"]


async def test_read_empty_file(fs_provider: FSSpecTools):
    """Test reading an empty file."""
    # Mock empty filesystem read
    fs_provider.fs._cat_file = AsyncMock(return_value=b"")

    # Get read_file tool from provider
    tools = await fs_provider.get_tools()
    read_tool = next(tool for tool in tools if tool.name == "read_file")
    result = await read_tool.execute(
        agent_ctx=CTX,
        path="/home/user/empty.txt",
    )

    # Verify empty content is handled correctly
    assert isinstance(result, str)
    assert result == ""


async def test_write_empty_file(fs_provider: FSSpecTools):
    """Test writing empty content to a file."""
    # Mock filesystem write operations
    mock_file = AsyncMock()
    mock_file.write = AsyncMock()
    fs_provider.fs.open_async = AsyncMock(return_value=mock_file)
    fs_provider.fs._info = AsyncMock(return_value={"size": 0})

    # Get write_text_file tool from provider
    tools = await fs_provider.get_tools()
    write_tool = next(tool for tool in tools if tool.name == "write_text_file")
    result = await write_tool.execute(
        agent_ctx=CTX,
        path="/home/user/empty_output.txt",
        content="",
    )

    # Verify result
    assert isinstance(result, dict)
    assert result["path"] == "/home/user/empty_output.txt"
    assert result["bytes_written"] == 0

    # Verify empty content was written
    fs_provider.fs.open_async.assert_called_once_with("/home/user/empty_output.txt", "wt")
    mock_file.write.assert_called_once_with("")


async def test_read_file_with_unicode(fs_provider: FSSpecTools):
    """Test reading file with unicode content."""
    unicode_content = "Hello 世界! 🌍\nThis has émojis and spëcial chars: café"

    # Mock filesystem read with unicode
    fs_provider.fs._cat_file = AsyncMock(return_value=unicode_content.encode("utf-8"))

    # Get read_file tool from provider
    tools = await fs_provider.get_tools()
    read_tool = next(tool for tool in tools if tool.name == "read_file")
    result = await read_tool.execute(agent_ctx=CTX, path="/home/user/unicode.txt")

    # Verify unicode content is preserved
    assert isinstance(result, str)
    assert "世界" in result
    assert "🌍" in result
    assert "émojis" in result
    assert "café" in result


async def test_write_file_with_unicode(fs_provider: FSSpecTools):
    """Test writing file with unicode content."""
    unicode_content = "Testing unicode: 日本語, русский, العربية 🎉"

    # Mock filesystem write operations
    mock_file = AsyncMock()
    mock_file.write = AsyncMock()
    fs_provider.fs.open_async = AsyncMock(return_value=mock_file)
    fs_provider.fs._info = AsyncMock(return_value={"size": len(unicode_content)})

    # Get write_text_file tool from provider
    tools = await fs_provider.get_tools()
    write_tool = next(tool for tool in tools if tool.name == "write_text_file")
    result = await write_tool.execute(
        agent_ctx=CTX,
        path="/home/user/unicode_output.txt",
        content=unicode_content,
    )

    # Verify result
    assert isinstance(result, dict)
    assert result["path"] == "/home/user/unicode_output.txt"

    # Verify unicode content was written correctly
    fs_provider.fs.open_async.assert_called_once_with("/home/user/unicode_output.txt", "wt")
    mock_file.write.assert_called_once_with(unicode_content)


async def test_file_operations_with_provider_session(fs_provider):
    """Test that file operations work with the filesystem provider."""
    # Mock filesystem operations
    fs_provider.fs._cat_file = AsyncMock(return_value=b"session content")
    mock_file = AsyncMock()
    mock_file.write = AsyncMock()
    fs_provider.fs.open_async = AsyncMock(return_value=mock_file)
    fs_provider.fs._info = AsyncMock(return_value={"size": 12})

    # Get tools from provider
    tools = await fs_provider.get_tools()
    read_tool = next(tool for tool in tools if tool.name == "read_file")
    await read_tool.execute(agent_ctx=CTX, path="/home/user/test.txt")

    write_tool = next(tool for tool in tools if tool.name == "write_text_file")
    await write_tool.execute(
        agent_ctx=CTX,
        path="/home/user/test.txt",
        content="test content",
    )

    # Verify filesystem calls were made
    fs_provider.fs._cat_file.assert_called_with("/home/user/test.txt")
    fs_provider.fs.open_async.assert_called_with("/home/user/test.txt", "wt")
    mock_file.write.assert_called_with("test content")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
