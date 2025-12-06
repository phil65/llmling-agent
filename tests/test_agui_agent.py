"""Tests for AG-UI remote agent."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from ag_ui.core import (
    TextMessageContentEvent,
    TextMessageStartEvent,
    ToolCallStartEvent,
)
import httpx
import pytest

from llmling_agent.agent.agui_agent import AGUIAgent, AGUISessionState
from llmling_agent.agent.agui_converters import (
    agui_to_native_event,
    extract_text_from_event,
    is_text_event,
)
from llmling_agent.messaging import ChatMessage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
def mock_sse_response():
    """Create mock SSE response."""

    def _create_response(events: list[str]) -> AsyncMock:
        """Create mock response with SSE events."""
        response = AsyncMock(spec=httpx.Response)
        response.raise_for_status = MagicMock()

        async def aiter_text() -> AsyncIterator[str]:
            for event in events:
                yield event

        response.aiter_text = aiter_text
        return response

    return _create_response


@pytest.mark.asyncio
async def test_agui_agent_initialization():
    """Test AGUIAgent initialization."""
    agent = AGUIAgent(
        endpoint="http://localhost:8000/run",
        name="test-agent",
    )

    assert agent.endpoint == "http://localhost:8000/run"
    assert agent.name == "test-agent"
    assert agent.conversation_id is not None
    assert agent._client is None
    assert agent._state is None


@pytest.mark.asyncio
async def test_agui_agent_context_manager():
    """Test AGUIAgent context manager."""
    async with AGUIAgent(
        endpoint="http://localhost:8000/run",
        name="test-agent",
    ) as agent:
        assert agent._client is not None
        assert agent._state is not None
        assert isinstance(agent._state, AGUISessionState)

    # After exit, should be cleaned up
    assert agent._client is None
    assert agent._state is None


@pytest.mark.asyncio
async def test_agui_agent_run_stream(mock_sse_response):
    """Test AGUIAgent streaming execution."""
    # Create mock SSE events
    events = [
        'data: {"type":"TEXT_MESSAGE_START","messageId":"msg1"}\n\n',
        'data: {"type":"TEXT_MESSAGE_CONTENT","messageId":"msg1","delta":"Hello"}\n\n',
        'data: {"type":"TEXT_MESSAGE_CONTENT","messageId":"msg1","delta":" world"}\n\n',
        'data: {"type":"TEXT_MESSAGE_END","messageId":"msg1"}\n\n',
    ]

    mock_response = mock_sse_response(events)

    with patch("llmling_agent.agent.agui_agent.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # stream() should return an async context manager
        mock_stream_cm = MagicMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(return_value=mock_stream_cm)

        async with AGUIAgent(
            endpoint="http://localhost:8000/run",
            name="test-agent",
        ) as agent:
            collected_events = [event async for event in agent.run_stream("Test prompt")]

            # Should have text deltas and final message
            assert len(collected_events) > 0
            assert agent._state.text_chunks == ["Hello", " world"]  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_agui_agent_run(mock_sse_response):
    """Test AGUIAgent non-streaming execution."""
    events = [
        'data: {"type":"TEXT_MESSAGE_START","messageId":"msg1"}\n\n',
        'data: {"type":"TEXT_MESSAGE_CONTENT","messageId":"msg1","delta":"Result"}\n\n',
        'data: {"type":"TEXT_MESSAGE_END","messageId":"msg1"}\n\n',
    ]

    mock_response = mock_sse_response(events)

    with patch("llmling_agent.agent.agui_agent.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # stream() should return an async context manager
        mock_stream_cm = MagicMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(return_value=mock_stream_cm)

        async with AGUIAgent(
            endpoint="http://localhost:8000/run",
            name="test-agent",
        ) as agent:
            result = await agent.run("Test prompt")

            assert isinstance(result, ChatMessage)
            assert result.content == "Result"
            assert result.role == "assistant"
            assert result.name == "test-agent"


def test_extract_text_from_event():
    """Test text extraction from AG-UI events."""
    # Text content event
    event1 = TextMessageContentEvent(message_id="msg1", delta="Hello")
    assert extract_text_from_event(event1) == "Hello"

    # Text start event (no text)
    event2 = TextMessageStartEvent(message_id="msg2")
    assert extract_text_from_event(event2) is None

    # Tool call event (no text)
    event3 = ToolCallStartEvent(
        tool_call_id="call1",
        tool_call_name="test_tool",
    )
    assert extract_text_from_event(event3) is None


def test_is_text_event():
    """Test text event detection."""
    event1 = TextMessageContentEvent(message_id="msg1", delta="Hello")
    assert is_text_event(event1) is True

    event2 = TextMessageStartEvent(message_id="msg2")
    assert is_text_event(event2) is False


def test_agui_to_native_event_text_content():
    """Test conversion of text content events."""
    event = TextMessageContentEvent(message_id="msg1", delta="Test content")
    native = agui_to_native_event(event)

    assert native is not None
    from pydantic_ai import PartDeltaEvent

    assert isinstance(native, PartDeltaEvent)


def test_agui_to_native_event_tool_call():
    """Test conversion of tool call events."""
    event = ToolCallStartEvent(
        tool_call_id="call1",
        tool_call_name="test_tool",
    )
    native = agui_to_native_event(event)

    assert native is not None
    from llmling_agent.agent.events import ToolCallStartEvent as NativeToolCallStart

    assert isinstance(native, NativeToolCallStart)
    assert native.tool_call_id == "call1"
    assert native.tool_name == "test_tool"


@pytest.mark.asyncio
async def test_agui_agent_to_tool():
    """Test converting AGUIAgent to a tool."""
    events = [
        'data: {"type":"TEXT_MESSAGE_CONTENT","messageId":"msg1","delta":"Answer"}\n\n',
    ]

    with patch("llmling_agent.agent.agui_agent.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()

        async def aiter_text():
            for event in events:
                yield event

        mock_response.aiter_text = aiter_text

        # stream() should return an async context manager
        mock_stream_cm = MagicMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(return_value=mock_stream_cm)

        async with AGUIAgent(
            endpoint="http://localhost:8000/run",
            name="test-agent",
        ) as agent:
            tool = agent.to_tool("Test tool description")

            assert callable(tool)
            assert tool.__name__ == "test-agent"
            assert "Test tool description" in str(tool.__doc__)

            result = await tool("Test question")
            assert result == "Answer"


@pytest.mark.asyncio
async def test_agui_agent_get_stats():
    """Test getting agent statistics."""
    from llmling_agent.talk.stats import MessageStats

    async with AGUIAgent(
        endpoint="http://localhost:8000/run",
        name="test-agent",
    ) as agent:
        stats = await agent.get_stats()

        assert isinstance(stats, MessageStats)
        assert stats.message_count == 0
        assert isinstance(stats.messages, list)


@pytest.mark.asyncio
async def test_agui_agent_error_handling(mock_sse_response):
    """Test error handling in AGUIAgent."""
    with patch("llmling_agent.agent.agui_agent.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Simulate HTTP error
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=MagicMock(),
            response=MagicMock(),
        )

        # stream() should return an async context manager
        mock_stream_cm = MagicMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(return_value=mock_stream_cm)

        async with AGUIAgent(
            endpoint="http://localhost:8000/run",
            name="test-agent",
        ) as agent:
            with pytest.raises(httpx.HTTPStatusError):
                async for _ in agent.run_stream("Test"):
                    pass
            assert agent._state
            assert agent._state.error is not None
