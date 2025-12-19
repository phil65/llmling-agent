"""Tests for builtin event handlers."""

from __future__ import annotations

from io import StringIO
import sys

from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent, PartStartEvent
from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart
import pytest

from llmling_agent.agents.events import (
    RunErrorEvent,
    StreamCompleteEvent,
    ToolCallStartEvent,
    detailed_print_handler,
    simple_print_handler,
)


@pytest.fixture
def mock_run_context():
    """Create a mock RunContext for testing."""

    class MockRunContext:
        def __init__(self):
            self.deps = None
            self.tool_name = "test_tool"
            self.tool_call_id = "test_call_123"

    return MockRunContext()


@pytest.mark.asyncio
async def test_simple_print_handler_text(mock_run_context):
    """Test simple handler prints text content."""
    output = StringIO()
    sys.stderr = output

    try:
        event = PartStartEvent(part=TextPart(content="Hello world"), index=0)
        await simple_print_handler(mock_run_context, event)

        assert output.getvalue() == "Hello world"
    finally:
        sys.stderr = sys.__stderr__


@pytest.mark.asyncio
async def test_simple_print_handler_tool_call(mock_run_context):
    """Test simple handler prints tool call."""
    output = StringIO()
    sys.stderr = output

    try:
        part = ToolCallPart(tool_name="test_tool", args={"param": "value"}, tool_call_id="call_123")
        event = FunctionToolCallEvent(part=part)
        await simple_print_handler(mock_run_context, event)

        assert "🔧 test_tool" in output.getvalue()
    finally:
        sys.stderr = sys.__stderr__


@pytest.mark.asyncio
async def test_simple_print_handler_error(mock_run_context):
    """Test simple handler prints errors."""
    output = StringIO()
    sys.stderr = output

    try:
        event = RunErrorEvent(message="Something went wrong", code="ERR001")
        await simple_print_handler(mock_run_context, event)

        result = output.getvalue()
        assert "❌" in result
        assert "Something went wrong" in result
    finally:
        sys.stderr = sys.__stderr__


@pytest.mark.asyncio
async def test_detailed_print_handler_tool_call(mock_run_context):
    """Test detailed handler prints tool call with inputs."""
    output = StringIO()
    sys.stderr = output

    try:
        event = ToolCallStartEvent(
            tool_call_id="call_123", tool_name="test_tool", title="Testing something"
        )
        await detailed_print_handler(mock_run_context, event)

        result = output.getvalue()
        assert "🔧 test_tool" in result
        assert "Testing something" in result
        assert "call_123"[:8] in result
    finally:
        sys.stderr = sys.__stderr__


@pytest.mark.asyncio
async def test_detailed_print_handler_tool_result(mock_run_context):
    """Test detailed handler prints tool results."""
    output = StringIO()
    sys.stderr = output

    try:
        result_part = ToolReturnPart(
            tool_name="test_tool", content="Operation successful", tool_call_id="call_123"
        )
        event = FunctionToolResultEvent(result=result_part)
        await detailed_print_handler(mock_run_context, event)

        result = output.getvalue()
        assert "✅" in result
        assert "test_tool" in result
        assert "Operation successful" in result
    finally:
        sys.stderr = sys.__stderr__


@pytest.mark.asyncio
async def test_detailed_print_handler_truncates_long_output(mock_run_context):
    """Test detailed handler truncates long outputs."""
    output = StringIO()
    sys.stderr = output

    try:
        long_content = "x" * 200
        result_part = ToolReturnPart(
            tool_name="test_tool", content=long_content, tool_call_id="call_123"
        )
        event = FunctionToolResultEvent(result=result_part)
        await detailed_print_handler(mock_run_context, event)

        result = output.getvalue()
        assert "..." in result
        assert len(result) < len(long_content)
    finally:
        sys.stderr = sys.__stderr__


@pytest.mark.asyncio
async def test_stream_complete_adds_newline(mock_run_context):
    """Test both handlers add final newline on stream complete."""
    from llmling_agent.messaging import ChatMessage

    for handler in [simple_print_handler, detailed_print_handler]:
        output = StringIO()
        sys.stderr = output

        try:
            message = ChatMessage(content="test", role="user")
            event = StreamCompleteEvent(message=message)
            await handler(mock_run_context, event)

            assert output.getvalue() == "\n"
        finally:
            sys.stderr = sys.__stderr__
