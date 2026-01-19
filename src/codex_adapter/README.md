# Codex Adapter

Python adapter for the [Codex](https://github.com/openai/codex) app-server JSON-RPC protocol.

Provides programmatic control over Codex conversations with streaming event support.

## Features

- **Async/await API** - Built on asyncio for efficient I/O
- **Streaming events** - Real-time access to agent messages, reasoning, and tool execution
- **Structured responses** - Type-safe JSON output with Pydantic models (PydanticAI-like)
- **Thread management** - Create, resume, fork, archive, and rollback conversation threads
- **Per-turn overrides** - Change model, reasoning effort, or approval policy per turn
- **Type-safe** - Full type hints with strict mypy compliance
- **Comprehensive API** - All 28 event types, 12 API methods fully supported
- **Concurrent turns** - Per-turn event queues prevent cross-contamination

## Installation

```bash
# Requires Codex CLI to be installed
# See: https://github.com/openai/codex
```

## Quick Start

```python
import asyncio
from codex_adapter import CodexClient

async def main():
    async with CodexClient() as client:
        # Start a new conversation thread
        thread = await client.thread_start(
            cwd="/path/to/project",
            model="gpt-5-codex",
            effort="high",
        )
        
        # Send a message and stream responses
        async for event in client.turn_stream(thread.id, "Help me refactor this code"):
            # Print agent messages as they stream
            if event.event_type == "item/agentMessage/delta":
                print(event.get_text_delta(), end="", flush=True)
            
            # Handle turn completion
            elif event.event_type == "turn/completed":
                print("\n[Turn completed]")
                break

asyncio.run(main())
```

## Structured Responses (PydanticAI-like)

Get type-safe structured output with a single method call:

```python
from pydantic import BaseModel
from codex_adapter import CodexClient

class FileList(BaseModel):
    files: list[str]
    total: int
    summary: str

async with CodexClient() as client:
    thread = await client.thread_start(cwd=".")

    # One line - automatic schema generation and parsing!
    result = await client.turn_stream_structured(
        thread.id,
        "List Python files in current directory",
        FileList,  # Pass Pydantic type
    )

    # Fully typed result with IDE autocomplete!
    print(f"Found {result.total} files: {result.files}")
```

**See [`STRUCTURED_RESPONSES.md`](./STRUCTURED_RESPONSES.md) for complete documentation.**

## API Overview

### CodexClient

Main client class for interacting with Codex app-server.

```python
async with CodexClient(
    codex_command="codex",  # Path to codex binary
    profile=None,            # Optional profile name
) as client:
    ...
```

### Methods

#### `thread_start(**kwargs) -> ThreadResponse`

Start a new conversation thread.

**Parameters:**
- `cwd` (str, optional): Working directory for the thread
- `model` (str, optional): Model to use (e.g., "gpt-5-codex")
- `effort` (str, optional): Reasoning effort ("low", "medium", "high")

**Returns:** `ThreadResponse` with `thread` containing thread data (access thread ID via `response.thread.id`)

#### `turn_stream(thread_id, user_input, **kwargs) -> AsyncIterator[CodexEvent]`

Start a turn and stream events in real-time.

**Parameters:**
- `thread_id` (str): Thread ID from `thread_start()`
- `user_input` (str | list[dict]): User message or list of items
- `model` (str, optional): Override model for this turn
- `effort` (str, optional): Override reasoning effort
- `approval_policy` (str, optional): "always", "never", or "auto"
- `output_schema` (dict | type, optional): JSON Schema dict or Pydantic type

**Yields:** `CodexEvent` instances

#### `turn_stream_structured(thread_id, user_input, result_type, **kwargs) -> ResultType`

Start a turn with structured output and return the parsed result. PydanticAI-like convenience method.

**Parameters:**
- `thread_id` (str): Thread ID from `thread_start()`
- `user_input` (str | list[dict]): User message or list of items
- `result_type` (type[ResultType]): Pydantic model class for expected result
- `model` (str, optional): Override model for this turn
- `effort` (str, optional): Override reasoning effort
- `approval_policy` (str, optional): "always", "never", or "auto"

**Returns:** Parsed Pydantic model instance

#### Thread Operations

- `thread_resume(thread_id)` - Resume existing thread
- `thread_fork(thread_id)` - Fork thread into new conversation
- `thread_list(**kwargs)` - List stored threads with pagination
- `thread_loaded_list()` - List thread IDs currently in memory
- `thread_archive(thread_id)` - Archive a thread
- `thread_rollback(thread_id, turns)` - Rollback N turns

#### Other Operations

- `turn_interrupt(thread_id, turn_id)` - Interrupt running turn
- `skills_list(**kwargs)` - List available skills
- `model_list()` - List available models
- `command_exec(command, **kwargs)` - Execute command without thread

### Events

All events have the following attributes:

- `event_type` (str): Notification method (e.g., "item/agentMessage/delta")
- `data` (dict): Event-specific payload
- `raw` (dict): Full JSON-RPC notification

**Common event types:**

| Event Type | Description |
|------------|-------------|
| `item/agentMessage/delta` | Streaming agent message text |
| `item/agentMessage/completed` | Agent message finished |
| `item/reasoning/delta` | Streaming reasoning content |
| `item/commandExecution/outputDelta` | Streaming tool output |
| `turn/completed` | Turn finished successfully |
| `turn/error` | Turn encountered an error |
| `approval/requested` | Tool requires approval |

**Helper methods:**

```python
event.is_delta()         # True for streaming events
event.is_completed()     # True for completion events
event.is_error()         # True for error events
event.get_text_delta()   # Extract text from message/output deltas
```

## Example: Multi-turn Conversation

```python
async def chat_session():
    async with CodexClient() as client:
        thread = await client.thread_start(cwd=".")
        
        messages = [
            "What files are in this directory?",
            "Show me the main.py file",
            "Add a docstring to the main function",
        ]
        
        for message in messages:
            print(f"\n> {message}")
            
            async for event in client.turn_stream(thread.id, message):
                if event.event_type == "item/agentMessage/delta":
                    print(event.get_text_delta(), end="", flush=True)
                elif event.event_type == "turn/completed":
                    print()
                    break
```

## Example: Change Model Mid-Conversation

```python
async def switch_models():
    async with CodexClient() as client:
        thread = await client.thread_start(model="gpt-5-codex")
        
        # First turn with default model
        async for event in client.turn_stream(thread.id, "Analyze this code"):
            ...
        
        # Second turn with different model (sticky override)
        async for event in client.turn_stream(
            thread.id,
            "Now refactor it",
            model="claude-opus-4",
            effort="high",
        ):
            ...
```

## Architecture

```
CodexClient
├── Spawns: `codex app-server` (stdio subprocess)
├── Protocol: JSON-RPC 2.0 over newline-delimited JSON
├── Reader task: Async loop processing stdout
├── Per-turn queues: Events routed by threadId:turnId
└── Request tracking: Futures for request/response pairing
```

The client maintains a persistent connection to the app-server subprocess and handles:
- Request/response pairing (by numeric ID)
- Notification routing (per-turn event queues)
- Process lifecycle management
- Error handling and cleanup
- Concurrent turn isolation

## Implementation Notes

- **Event streaming**: Events are routed to per-turn queues based on `threadId:turnId`
- **Concurrency**: Multiple turns can stream concurrently without cross-contamination
- **Type safety**: Full mypy strict compliance with generic types for structured responses
- **Error handling**: JSON-RPC errors raise `CodexRequestError`, process errors raise `CodexProcessError`
- **Cleanup**: Context manager ensures subprocess termination and resource cleanup

## Implemented Features

- ✅ Thread management (create, resume, fork, archive, rollback, list)
- ✅ Turn streaming with 28 event types
- ✅ Per-turn overrides (model, effort, approval policy)
- ✅ Structured responses (Pydantic types + dict schemas)
- ✅ PydanticAI-like API (`turn_stream_structured`)
- ✅ Skills and model listing
- ✅ Command execution
- ✅ Concurrent turn support (per-turn event queues)
- ✅ Type-safe with full mypy compliance
- ⚠️ Approval handling (events exposed but no handler callbacks yet)

## Future Enhancements

1. **Approval handlers**: Register callbacks for `approval/requested` events
2. **Retry logic**: Automatic reconnection on subprocess failure
3. **Logging**: Structured logging with event filtering
4. **Context builders**: Helper methods for complex input construction

## See Also

- [Codex app-server docs](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)
- [ACP connection](../acp/connection.py) - JSON-RPC infrastructure used by this adapter
