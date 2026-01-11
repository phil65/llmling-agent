"""Codex app-server Python adapter.

Provides programmatic control over Codex via the app-server JSON-RPC protocol.

Example:
    async with CodexClient() as client:
        thread = await client.thread_start(cwd="/path/to/project")
        async for event in client.turn_stream(thread.id, "Help me refactor"):
            if event.event_type == "item/agentMessage/delta":
                print(event.data.get("text", ""), end="", flush=True)
"""

from codex_adapter.client import CodexClient
from codex_adapter.events import CodexEvent, EventType
from codex_adapter.exceptions import CodexError, CodexProcessError, CodexRequestError
from codex_adapter.codex_types import CodexThread, CodexTurn

__all__ = [
    "CodexClient",
    "CodexError",
    "CodexEvent",
    "CodexProcessError",
    "CodexRequestError",
    "CodexThread",
    "CodexTurn",
    "EventType",
]
