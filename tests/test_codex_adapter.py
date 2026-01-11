"""Tests for Codex adapter."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from codex_adapter import CodexClient, CodexEvent, CodexThread
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
    event = CodexEvent.from_notification(
        "item/agentMessage/delta",
        {"text": "Hello", "itemId": "123"},
    )

    assert event.event_type == "item/agentMessage/delta"
    assert event.data.text == "Hello"  # Attribute access to extra fields
    assert event.data.item_id == "123"  # snake_case auto-converted from camelCase


def test_codex_event_is_delta():
    """Test event type checking."""
    delta_event = CodexEvent.from_notification("item/agentMessage/delta", {})
    completed_event = CodexEvent.from_notification("turn/completed", {})

    assert delta_event.is_delta() is True
    assert delta_event.is_completed() is False
    assert completed_event.is_delta() is False
    assert completed_event.is_completed() is True


def test_codex_event_get_text_delta():
    """Test extracting text from different event types."""
    # Agent message delta
    event1 = CodexEvent.from_notification(
        "item/agentMessage/delta",
        {"text": "Hello"},
    )
    assert event1.get_text_delta() == "Hello"

    # Command output delta
    event2 = CodexEvent.from_notification(
        "item/commandExecution/outputDelta",
        {"output": "test output"},
    )
    assert event2.get_text_delta() == "test output"

    # Other event
    event3 = CodexEvent.from_notification("turn/completed", {})
    assert event3.get_text_delta() == ""


def test_codex_thread():
    """Test CodexThread dataclass."""
    thread = CodexThread(
        id="thread-123",
        preview="Test conversation",
        model_provider="openai",
        created_at=1234567890,
    )

    assert thread.id == "thread-123"
    assert thread.preview == "Test conversation"
    assert thread.model_provider == "openai"


def test_codex_request_error():
    """Test CodexRequestError exception."""
    error = CodexRequestError(
        code=-32000,
        message="Invalid params",
        data={"field": "model"},
    )

    assert error.code == -32000
    assert error.message == "Invalid params"
    assert error.data == {"field": "model"}
    assert "[-32000]" in str(error)
    assert "Invalid params" in str(error)


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
    client = CodexClient()

    # Process notification
    await client._process_message({
        "jsonrpc": "2.0",
        "method": "item/agentMessage/delta",
        "params": {"text": "Hello", "itemId": "123"},
    })

    # Check event was queued
    event = await asyncio.wait_for(client._event_queue.get(), timeout=1.0)
    assert event.event_type == "item/agentMessage/delta"
    assert event.data.text == "Hello"


@pytest.mark.asyncio
async def test_client_not_started_error():
    """Test error when sending request before starting."""
    client = CodexClient()

    with pytest.raises(CodexProcessError, match="Not connected"):
        await client._send_request("thread/start", {})


def test_codex_event_typed_data():
    """Test typed event data access with BaseEventData."""
    from codex_adapter.models import BaseEventData

    # Create an agent message delta event
    event = CodexEvent.from_notification(
        "item/agentMessage/delta",
        {
            "threadId": "thread-123",
            "turnId": "turn-456",
            "itemId": "item-789",
            "text": "Hello world",
        },
    )

    # Data is automatically validated as BaseEventData
    assert isinstance(event.data, BaseEventData)

    # Common fields work with snake_case
    assert event.data.thread_id == "thread-123"
    assert event.data.turn_id == "turn-456"
    assert event.data.item_id == "item-789"

    # Extra fields accessible as attributes (via __getattr__)
    assert event.data.text == "Hello world"


def test_codex_event_common_fields():
    """Test that common event fields are always accessible."""
    from codex_adapter.models import BaseEventData

    # Event with only some common fields
    event = CodexEvent.from_notification(
        "thread/started",
        {
            "threadId": "thread-abc",
            # No turnId or itemId
        },
    )

    assert isinstance(event.data, BaseEventData)
    assert event.data.thread_id == "thread-abc"
    assert event.data.turn_id is None  # Optional field
    assert event.data.item_id is None  # Optional field


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
