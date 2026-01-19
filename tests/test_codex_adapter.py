"""Tests for Codex adapter."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from codex_adapter import (
    CodexClient,
    CodexEvent,
    HttpMcpServer,
    StdioMcpServer,
)
from codex_adapter.client import _mcp_config_to_toml_inline
from codex_adapter.exceptions import CodexProcessError, CodexRequestError


@pytest.fixture
def mock_process():
    """Mock subprocess.Popen for testing."""
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.flush = MagicMock()
    process.stdout = MagicMock()
    process.poll.return_value = None
    return process


@pytest.mark.asyncio
async def test_codex_client_initialization():
    """Test CodexClient can be instantiated."""
    client = CodexClient()
    assert client._codex_command == "codex"
    assert client._profile is None
    assert client._process is None


@pytest.mark.asyncio
async def test_codex_client_with_profile():
    """Test CodexClient with custom profile."""
    client = CodexClient(codex_command="/usr/local/bin/codex", profile="test-profile")
    assert client._codex_command == "/usr/local/bin/codex"
    assert client._profile == "test-profile"


def test_codex_event_from_notification():
    """Test CodexEvent creation from notification."""
    from codex_adapter.models import AgentMessageDeltaData

    event = CodexEvent.from_notification(
        "item/agentMessage/delta",
        {"delta": "Hello", "itemId": "123", "threadId": "t1", "turnId": "u1"},
    )

    assert event.event_type == "item/agentMessage/delta"
    assert isinstance(event.data, AgentMessageDeltaData)
    assert event.data.delta == "Hello"  # API uses "delta" not "text"
    assert event.data.item_id == "123"  # snake_case auto-converted from camelCase


def test_codex_event_is_delta():
    """Test event type checking."""
    delta_event = CodexEvent.from_notification(
        "item/agentMessage/delta",
        {"delta": "", "itemId": "1", "threadId": "t1", "turnId": "u1"},
    )
    # TurnCompletedData has turn object, not flat fields
    completed_event = CodexEvent.from_notification(
        "turn/completed",
        {
            "threadId": "t1",
            "turn": {
                "id": "u1",
                "status": "completed",
                "items": [],
            },
        },
    )

    assert delta_event.is_delta() is True
    assert delta_event.is_completed() is False
    assert completed_event.is_delta() is False
    assert completed_event.is_completed() is True


def test_codex_event_get_text_delta():
    """Test extracting text from different event types."""
    # Agent message delta
    event1 = CodexEvent.from_notification(
        "item/agentMessage/delta",
        {"delta": "Hello", "itemId": "1", "threadId": "t1", "turnId": "u1"},
    )
    assert event1.get_text_delta() == "Hello"

    # Command output delta - uses 'delta' field, not 'output'!
    event2 = CodexEvent.from_notification(
        "item/commandExecution/outputDelta",
        {"delta": "test output", "itemId": "1", "threadId": "t1", "turnId": "u1"},
    )
    assert event2.get_text_delta() == "test output"

    # Non-delta event
    event3 = CodexEvent.from_notification(
        "turn/completed",
        {
            "threadId": "t1",
            "turn": {
                "id": "u1",
                "status": "completed",
                "items": [],
            },
        },
    )
    assert event3.get_text_delta() == ""


@pytest.mark.asyncio
async def test_process_message_response():
    """Test processing JSON-RPC response."""
    client = CodexClient()

    # Setup pending request
    future = asyncio.Future()
    client._pending_requests[1] = future

    # Process success response
    await client._process_message({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"threadId": "thread-123"},
    })

    assert future.done()
    assert future.result() == {"threadId": "thread-123"}


@pytest.mark.asyncio
async def test_process_message_error_response():
    """Test processing JSON-RPC error response."""
    client = CodexClient()

    # Setup pending request
    future = asyncio.Future()
    client._pending_requests[1] = future

    # Process error response
    await client._process_message({
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32602,
            "message": "Invalid params",
            "data": {"details": "model not found"},
        },
    })

    assert future.done()
    with pytest.raises(CodexRequestError) as exc_info:
        future.result()

    assert exc_info.value.code == -32602
    assert exc_info.value.message == "Invalid params"


@pytest.mark.asyncio
async def test_process_message_notification():
    """Test processing JSON-RPC notification."""
    from codex_adapter.models import AgentMessageDeltaData

    client = CodexClient()

    # Process notification
    await client._process_message({
        "jsonrpc": "2.0",
        "method": "item/agentMessage/delta",
        "params": {"delta": "Hello", "itemId": "123", "threadId": "t1", "turnId": "u1"},
    })

    # Check event was queued
    event = await asyncio.wait_for(client._event_queue.get(), timeout=1.0)
    assert event.event_type == "item/agentMessage/delta"
    assert isinstance(event.data, AgentMessageDeltaData)
    assert event.data.delta == "Hello"


@pytest.mark.asyncio
async def test_client_not_started_error():
    """Test error when sending request before starting."""
    client = CodexClient()

    with pytest.raises(CodexProcessError, match="Not connected"):
        await client._send_request("thread/start", {})


def test_codex_event_typed_data():
    """Test typed event data access with proper types."""
    from codex_adapter.models import AgentMessageDeltaData

    # Create an agent message delta event
    event = CodexEvent.from_notification(
        "item/agentMessage/delta",
        {
            "threadId": "thread-123",
            "turnId": "turn-456",
            "itemId": "item-789",
            "delta": "Hello world",
        },
    )

    # Data is automatically validated as AgentMessageDeltaData
    assert isinstance(event.data, AgentMessageDeltaData)

    # Common fields work with snake_case
    assert event.data.thread_id == "thread-123"
    assert event.data.turn_id == "turn-456"
    assert event.data.item_id == "item-789"

    # Event-specific fields are properly typed
    assert event.data.delta == "Hello world"


def test_codex_event_common_fields():
    """Test that common event fields are accessible on all event types."""
    from codex_adapter.models import ThreadStartedData

    # V1 ThreadStarted has nested thread object
    event = CodexEvent.from_notification(
        "thread/started",
        {
            "thread": {
                "id": "thread-abc",
                "preview": "",
                "modelProvider": "openai",
                "createdAt": 0,
                "path": "",
                "cwd": "",
                "cliVersion": "",
                "source": "appServer",
            }
        },
    )

    assert isinstance(event.data, ThreadStartedData)
    assert event.data.thread.id == "thread-abc"


def test_mcp_config_stdio_to_toml():
    """Test converting StdioMcpServer to TOML inline format."""
    config = StdioMcpServer(
        command="npx",
        args=["-y", "@openai/codex-shell-tool-mcp"],
    )
    result = _mcp_config_to_toml_inline("bash", config)
    expected = 'mcp_servers.bash={command = "npx", args = ["-y", "@openai/codex-shell-tool-mcp"]}'
    assert result == expected


def test_mcp_config_http_to_toml():
    """Test converting HttpMcpServer to TOML inline format."""
    config = HttpMcpServer(
        url="http://localhost:8000/mcp",
        bearer_token_env_var="MY_TOKEN",
    )
    result = _mcp_config_to_toml_inline("tools", config)
    expected = (
        'mcp_servers.tools={url = "http://localhost:8000/mcp", bearer_token_env_var = "MY_TOKEN"}'
    )
    assert result == expected


def test_mcp_config_http_with_headers():
    """Test HttpMcpServer with custom headers."""
    config = HttpMcpServer(
        url="http://localhost:8000/mcp",
        http_headers={"X-Custom": "value", "X-Another": "test"},
    )
    result = _mcp_config_to_toml_inline("tools", config)
    # Note: dict iteration order is preserved in Python 3.7+
    assert "mcp_servers.tools={" in result
    assert 'url = "http://localhost:8000/mcp"' in result
    assert 'http_headers = {X-Custom = "value", X-Another = "test"}' in result


def test_mcp_config_with_env():
    """Test StdioMcpServer with environment variables."""
    config = StdioMcpServer(
        command="my-server",
        args=["--port", "8080"],
        env={"API_KEY": "secret123", "MODE": "production"},
    )
    result = _mcp_config_to_toml_inline("custom", config)
    assert "mcp_servers.custom={" in result
    assert 'command = "my-server"' in result
    assert 'args = ["--port", "8080"]' in result
    assert 'env = {API_KEY = "secret123", MODE = "production"}' in result


def test_codex_client_with_mcp_servers():
    """Test CodexClient initialization with MCP servers."""
    mcp_servers = {
        "tools": HttpMcpServer(url="http://localhost:8000/mcp"),
        "bash": StdioMcpServer(command="npx", args=["-y", "@openai/codex-shell-tool-mcp"]),
    }

    client = CodexClient(mcp_servers=mcp_servers)
    assert client._mcp_servers == mcp_servers
    assert "tools" in client._mcp_servers
    assert "bash" in client._mcp_servers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
