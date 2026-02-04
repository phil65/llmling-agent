"""Tests for Codex adapter."""

from __future__ import annotations

import asyncio

from pydantic import ValidationError
import pytest

from codex_adapter import CodexClient, HttpMcpServer, StdioMcpServer
from codex_adapter.client import _mcp_config_to_toml_inline
from codex_adapter.events import (
    AgentMessageDeltaEvent,
    get_text_delta,
    is_completed_event,
    is_delta_event,
    parse_codex_event,
)
from codex_adapter.exceptions import CodexProcessError, CodexRequestError


def test_parse_codex_event_camel_to_snake():
    """Test camelCase JSON is converted to snake_case fields."""
    event = parse_codex_event(
        "item/agentMessage/delta",
        {"delta": "Hello", "itemId": "123", "threadId": "t1", "turnId": "u1"},
    )
    assert isinstance(event, AgentMessageDeltaEvent)
    assert event.data.item_id == "123"  # camelCase -> snake_case
    assert event.data.thread_id == "t1"


def test_parse_codex_event_unknown_type_raises():
    """Unknown event types raise ValidationError (strict mode)."""
    with pytest.raises(ValidationError):
        parse_codex_event("unknown/event/type", {"threadId": "t1"})


def test_parse_codex_event_legacy_v1_returns_none():
    """Legacy codex/event/* methods are filtered out."""
    assert parse_codex_event("codex/event/task_started", {"id": "0"}) is None


def test_event_helper_functions():
    """Test is_delta_event, is_completed_event, get_text_delta."""
    delta = parse_codex_event(
        "item/agentMessage/delta",
        {"delta": "text", "itemId": "1", "threadId": "t", "turnId": "u"},
    )
    completed = parse_codex_event(
        "turn/completed",
        {"threadId": "t", "turn": {"id": "u", "status": "completed", "items": []}},
    )
    assert delta
    assert completed

    assert is_delta_event(delta) is True
    assert is_completed_event(delta) is False
    assert get_text_delta(delta) == "text"

    assert is_delta_event(completed) is False
    assert is_completed_event(completed) is True
    assert get_text_delta(completed) == ""


@pytest.mark.asyncio
async def test_process_message_routes_response_to_future():
    """JSON-RPC responses are routed to pending request futures."""
    client = CodexClient()
    future: asyncio.Future[dict] = asyncio.Future()
    client._pending_requests[1] = future

    await client._process_message({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"threadId": "thread-123"},
    })

    assert future.result() == {"threadId": "thread-123"}


@pytest.mark.asyncio
async def test_process_message_error_raises():
    """JSON-RPC error responses set exception on future."""
    client = CodexClient()
    future: asyncio.Future[dict] = asyncio.Future()
    client._pending_requests[1] = future

    await client._process_message({
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32602, "message": "Invalid params"},
    })

    with pytest.raises(CodexRequestError) as exc:
        future.result()
    assert exc.value.code == -32602


@pytest.mark.asyncio
async def test_process_message_notification_queued():
    """JSON-RPC notifications are parsed and queued."""
    client = CodexClient()

    await client._process_message({
        "jsonrpc": "2.0",
        "method": "item/agentMessage/delta",
        "params": {"delta": "Hello", "itemId": "1", "threadId": "t", "turnId": "u"},
    })

    event = await asyncio.wait_for(client._event_queue.get(), timeout=1.0)
    assert isinstance(event, AgentMessageDeltaEvent)
    assert event.data.delta == "Hello"


@pytest.mark.asyncio
async def test_send_request_not_connected_raises():
    """Sending request before connecting raises CodexProcessError."""
    client = CodexClient()
    with pytest.raises(CodexProcessError, match="Not connected"):
        await client._send_request("thread/start")


def test_mcp_config_to_toml_stdio():
    """StdioMcpServer serializes to TOML inline format."""
    config = StdioMcpServer(command="npx", args=["-y", "pkg"])
    result = _mcp_config_to_toml_inline("bash", config)
    assert result == 'mcp_servers.bash={command = "npx", args = ["-y", "pkg"]}'


def test_mcp_config_to_toml_http():
    """HttpMcpServer serializes to TOML inline format."""
    config = HttpMcpServer(url="http://localhost:8000", bearer_token_env_var="TOKEN")
    result = _mcp_config_to_toml_inline("api", config)
    assert 'url = "http://localhost:8000"' in result
    assert 'bearer_token_env_var = "TOKEN"' in result
