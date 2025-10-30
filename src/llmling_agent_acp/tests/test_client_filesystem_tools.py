"""Tests for client-side filesystem tools that make ACP requests."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

from pydantic_ai import RunContext, RunUsage
from pydantic_ai.models.test import TestModel
import pytest

from acp import AgentSideConnection
from acp.schema import (
    ClientCapabilities,
    FileSystemCapability,
    InitializeRequest,
    ReadTextFileResponse,
    WriteTextFileResponse,
)
from llmling_agent import AgentPool
from llmling_agent_acp.acp_agent import LLMlingACPAgent
from llmling_agent_acp.acp_tools import ACPFileSystemProvider


CTX = RunContext(tool_call_id="test", deps=None, model=TestModel(), usage=RunUsage())


class TestClientFilesystemTools:
    """Test client-side filesystem tools that request file operations from ACP client."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock ACP connection."""
        return Mock(spec=AgentSideConnection)

    @pytest.fixture
    def mock_agent_pool(self):
        """Create a mock agent pool."""
        pool = Mock(spec=AgentPool)
        pool.agents = {}
        return pool

    @pytest.fixture
    async def acp_agent(self, mock_connection, mock_agent_pool):
        """Create ACP agent with filesystem support."""
        # Create mock agent
        mock_agent = Mock()
        mock_tools = {}

        def register_tool(tool):
            mock_tools[tool.name] = tool

        mock_agent.tools = Mock()
        mock_agent.tools.register_tool = register_tool
        mock_agent_pool.agents = {"test_agent": mock_agent}

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
        await agent.handle_request(request)
        return agent

    @pytest.fixture
    async def session(self, acp_agent: LLMlingACPAgent, mock_connection):
        """Create test session."""
        from acp.schema import ClientCapabilities, FileSystemCapability
        from llmling_agent_acp.session import ACPSession

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
    async def fs_provider(self, session):
        """Create filesystem capability provider for testing."""
        return ACPFileSystemProvider(session=session)

    async def test_filesystem_tools_registered(
        self,
        fs_provider: ACPFileSystemProvider,
    ):
        """Test that filesystem tools are registered."""
        tools = await fs_provider.get_tools()
        tool_names = {tool.name for tool in tools}
        expected_tools = {"read_text_file", "write_text_file"}
        assert expected_tools.issubset(tool_names)
        # Check that filesystem tools have correct source
        fs_tools = [tool for tool in tools if tool.name in expected_tools]
        for tool in fs_tools:
            assert tool.source == "filesystem"

    async def test_read_text_file_success(
        self,
        acp_agent: LLMlingACPAgent,
        fs_provider: ACPFileSystemProvider,
    ):
        """Test successful file reading."""
        # Mock read file response
        response = ReadTextFileResponse(content="Hello, World!\nThis is a test file.\n")
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=response
        )

        # Get read_text_file tool from provider
        tools = await fs_provider.get_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_text_file")
        result = await read_tool.execute(ctx=CTX, path="/home/user/test.txt")

        # Verify result
        assert "Hello, World!" in result
        assert "This is a test file." in result

        # Verify ACP call was made
        acp_agent.connection.read_text_file.assert_called_once()
        call_args = acp_agent.connection.read_text_file.call_args[0][0]
        assert call_args.path == "/home/user/test.txt"
        assert call_args.session_id == "test_session"

    async def test_read_text_file_with_line_and_limit(
        self,
        acp_agent: LLMlingACPAgent,
        fs_provider: ACPFileSystemProvider,
    ):
        """Test file reading with line and limit parameters."""
        # Mock read file response
        response = ReadTextFileResponse(content="Line 10\nLine 11\nLine 12\n")
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=response
        )

        # Get read_text_file tool from provider
        tools = await fs_provider.get_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_text_file")
        result = await read_tool.execute(
            ctx=CTX,
            path="/home/user/large_file.txt",
            line=10,
            limit=3,
        )

        # Verify result
        assert "Line 10" in result
        assert "Line 11" in result
        assert "Line 12" in result

        # Verify ACP call parameters
        call_args = acp_agent.connection.read_text_file.call_args[0][0]
        assert call_args.path == "/home/user/large_file.txt"
        assert call_args.line == 10  # noqa: PLR2004
        assert call_args.limit == 3  # noqa: PLR2004

    async def test_read_text_file_error(
        self,
        acp_agent: LLMlingACPAgent,
        fs_provider: ACPFileSystemProvider,
    ):
        """Test file reading error handling."""
        # Mock read file error
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            side_effect=FileNotFoundError("File not found")
        )

        # Get read_text_file tool from provider
        tools = await fs_provider.get_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_text_file")

        result = await read_tool.execute(ctx=CTX, path="/home/user/nonexistent.txt")
        # Verify error handling
        assert result.startswith("Error reading file:")
        assert "File not found" in result

    async def test_write_text_file_success(
        self,
        acp_agent: LLMlingACPAgent,
        fs_provider: ACPFileSystemProvider,
    ):
        """Test successful file writing."""
        # Mock write file response
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=WriteTextFileResponse()
        )

        # Get write_text_file tool from provider
        tools = await fs_provider.get_tools()
        write_tool = next(tool for tool in tools if tool.name == "write_text_file")
        result = await write_tool.execute(
            ctx=CTX,
            path="/home/user/output.txt",
            content="Hello, World!\nThis is written content.\n",
        )

        # Verify result
        assert "Successfully wrote file" in result
        assert "/home/user/output.txt" in result

        # Verify ACP call was made
        acp_agent.connection.write_text_file.assert_called_once()
        call_args = acp_agent.connection.write_text_file.call_args[0][0]
        assert call_args.path == "/home/user/output.txt"
        assert call_args.content == "Hello, World!\nThis is written content.\n"
        assert call_args.session_id == "test_session"

    async def test_write_text_file_json(
        self,
        acp_agent: LLMlingACPAgent,
        fs_provider: ACPFileSystemProvider,
    ):
        """Test writing JSON content."""
        # Mock write file response
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=WriteTextFileResponse()
        )

        json_str = '{\n  "debug": true,\n  "version": "1.0.0"\n}'

        # Get write_text_file tool from provider
        tools = await fs_provider.get_tools()
        write_tool = next(tool for tool in tools if tool.name == "write_text_file")
        result = await write_tool.execute(
            ctx=CTX,
            path="/home/user/config.json",
            content=json_str,
        )

        # Verify result
        assert "Successfully wrote file" in result
        assert "/home/user/config.json" in result

        # Verify content was written correctly
        call_args = acp_agent.connection.write_text_file.call_args[0][0]
        assert call_args.content == json_str

    async def test_write_text_file_error(
        self, acp_agent: LLMlingACPAgent, fs_provider: ACPFileSystemProvider
    ):
        """Test file writing error handling."""
        # Mock write file error
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            side_effect=PermissionError("Permission denied")
        )

        # Get write_text_file tool from provider
        tools = await fs_provider.get_tools()
        write_tool = next(tool for tool in tools if tool.name == "write_text_file")
        result = await write_tool.execute(
            ctx=CTX,
            path="/root/protected.txt",
            content="This should fail",
        )

        # Verify error handling
        assert result.startswith("Error writing file:")
        assert "Permission denied" in result

    async def test_read_empty_file(
        self, acp_agent: LLMlingACPAgent, fs_provider: ACPFileSystemProvider
    ):
        """Test reading an empty file."""
        # Mock empty file response
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=ReadTextFileResponse(content="")
        )

        # Get read_text_file tool from provider
        tools = await fs_provider.get_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_text_file")
        result = await read_tool.execute(
            ctx=CTX,
            path="/home/user/empty.txt",
        )

        # Verify empty content is handled correctly
        assert result == ""

    async def test_write_empty_file(
        self, acp_agent: LLMlingACPAgent, fs_provider: ACPFileSystemProvider
    ):
        """Test writing empty content to a file."""
        # Mock write file response
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=WriteTextFileResponse()
        )

        # Get write_text_file tool from provider
        tools = await fs_provider.get_tools()
        write_tool = next(tool for tool in tools if tool.name == "write_text_file")
        result = await write_tool.execute(
            ctx=CTX,
            path="/home/user/empty_output.txt",
            content="",
        )

        # Verify result
        assert "Successfully wrote file" in result

        # Verify empty content was written
        call_args = acp_agent.connection.write_text_file.call_args[0][0]
        assert call_args.content == ""

    async def test_read_file_with_unicode(
        self, acp_agent: LLMlingACPAgent, fs_provider: ACPFileSystemProvider
    ):
        """Test reading file with unicode content."""
        unicode_content = "Hello 世界! 🌍\nThis has émojis and spëcial chars: café"

        # Mock read file response with unicode
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=ReadTextFileResponse(content=unicode_content)
        )

        # Get read_text_file tool from provider
        tools = await fs_provider.get_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_text_file")
        result = await read_tool.execute(ctx=CTX, path="/home/user/unicode.txt")

        # Verify unicode content is preserved
        assert "世界" in result
        assert "🌍" in result
        assert "émojis" in result
        assert "café" in result

    async def test_write_file_with_unicode(
        self, acp_agent: LLMlingACPAgent, fs_provider: ACPFileSystemProvider
    ):
        """Test writing file with unicode content."""
        unicode_content = "Testing unicode: 日本語, русский, العربية 🎉"

        # Mock write file response
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=WriteTextFileResponse()
        )

        # Get write_text_file tool from provider
        tools = await fs_provider.get_tools()
        write_tool = next(tool for tool in tools if tool.name == "write_text_file")
        result = await write_tool.execute(
            ctx=CTX,
            path="/home/user/unicode_output.txt",
            content=unicode_content,
        )

        # Verify result
        assert "Successfully wrote file" in result

        # Verify unicode content was written correctly
        call_args = acp_agent.connection.write_text_file.call_args[0][0]
        assert call_args.content == unicode_content

    async def test_file_operations_with_provider_session(
        self, acp_agent: LLMlingACPAgent, fs_provider
    ):
        """Test that file operations use the provider's session ID."""
        # Mock responses
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=ReadTextFileResponse(content="session content")
        )
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=WriteTextFileResponse()
        )

        # Get tools from provider
        tools = await fs_provider.get_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_text_file")
        await read_tool.execute(ctx=CTX, path="/home/user/test.txt")

        write_tool = next(tool for tool in tools if tool.name == "write_text_file")
        await write_tool.execute(
            ctx=CTX,
            path="/home/user/test.txt",
            content="test content",
        )

        # Verify both operations use the provider's session ID
        read_call_args = acp_agent.connection.read_text_file.call_args[0][0]
        assert read_call_args.session_id == "test_session"

        write_call_args = acp_agent.connection.write_text_file.call_args[0][0]
        assert write_call_args.session_id == "test_session"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
