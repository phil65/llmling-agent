# Codex Adapter

Python adapter for the [Codex](https://github.com/openai/codex) app-server JSON-RPC protocol.

## Quick Start

```python
import asyncio
from codex_adapter import CodexClient
from codex_adapter.events import AgentMessageDeltaEvent, TurnCompletedEvent, get_text_delta

async def main():
    async with CodexClient() as client:
        thread = await client.thread_start(cwd="/path/to/project")
        
        async for event in client.turn_stream(thread.id, "Help me refactor this code"):
            match event:
                case AgentMessageDeltaEvent():
                    print(get_text_delta(event), end="", flush=True)
                case TurnCompletedEvent():
                    break

asyncio.run(main())
```

## Structured Responses

```python
from pydantic import BaseModel

class FileList(BaseModel):
    files: list[str]
    total: int

async with CodexClient() as client:
    thread = await client.thread_start(cwd=".")
    result = await client.turn_stream_structured(
        thread.id,
        "List Python files",
        FileList,
    )
    print(result.files)  # Typed result
```

## Events

Events are a discriminated union. Use pattern matching or helper functions:

```python
from codex_adapter.events import (
    AgentMessageDeltaEvent,
    CommandExecutionOutputDeltaEvent,
    TurnCompletedEvent,
    TurnErrorEvent,
    get_text_delta,
    is_delta_event,
)

async for event in client.turn_stream(thread_id, message):
    match event:
        case AgentMessageDeltaEvent() | CommandExecutionOutputDeltaEvent():
            print(get_text_delta(event), end="")
        case TurnCompletedEvent():
            break
        case TurnErrorEvent(data=data):
            print(f"Error: {data.error}")
```

## See Also

- [Codex app-server docs](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)
