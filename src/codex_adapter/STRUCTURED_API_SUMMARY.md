# Structured Response API - Summary

## What We Built

The Codex adapter now has a comprehensive, type-safe API for structured responses with both high-level convenience methods and low-level streaming control.

## API Overview

### 1. High-Level API: `turn_stream_structured()` (PydanticAI-like)

**Single method call** that handles everything:
- ✅ Automatic JSON Schema generation from Pydantic types
- ✅ Automatic response parsing and validation
- ✅ Full type safety and IDE autocomplete
- ✅ Clean, simple API

```python
result = await client.turn_stream_structured(
    thread.id,
    "List Python files in current directory",
    FileList,  # Pydantic type
)
# result is fully typed - IDE autocomplete works!
print(f"Found {result.total} files")
```

**Signature:**
```python
async def turn_stream_structured(
    self,
    thread_id: str,
    user_input: str | list[dict[str, Any]],
    result_type: type[ResultType],
    *,
    model: str | None = None,
    effort: str | None = None,
    approval_policy: str | None = None,
) -> ResultType:
```

### 2. Low-Level API: `turn_stream()`

**Full control** over event streaming:
- ✅ Accepts both dict schemas and Pydantic types
- ✅ Stream events individually
- ✅ Process events as they arrive
- ✅ Handle errors inline

```python
async for event in client.turn_stream(
    thread.id,
    "List Python files",
    output_schema=FileList,  # Or a dict schema
):
    if event.event_type == "item/agentMessage/delta":
        print(event.get_text_delta(), end="", flush=True)
```

**Signature:**
```python
async def turn_stream(
    self,
    thread_id: str,
    user_input: str | list[dict[str, Any]],
    *,
    model: str | None = None,
    effort: str | None = None,
    approval_policy: str | None = None,
    output_schema: dict[str, Any] | type[Any] | None = None,
) -> AsyncIterator[CodexEvent]:
```

## Type Safety Features

### TypeVar for Generic Types

```python
ResultType = TypeVar("ResultType", bound=BaseModel)
```

This ensures `turn_stream_structured()` returns the exact type you pass in:

```python
class FileList(BaseModel):
    total: int

# Type checker knows result is FileList, not just BaseModel
result: FileList = await client.turn_stream_structured(
    thread.id,
    prompt,
    FileList,
)
```

### Schema Generation with TypeAdapter

The adapter uses Pydantic's `TypeAdapter` to extract JSON Schema from types:

```python
if isinstance(output_schema, dict):
    params["outputSchema"] = output_schema
else:
    # Auto-generate schema from Pydantic type
    adapter = TypeAdapter(output_schema)
    params["outputSchema"] = adapter.json_schema()
```

## Design Rationale

### Why Two Methods?

**`turn_stream_structured()`** is optimized for the common case:
- You know the expected output type
- You want a typed result
- You don't need to process individual events

**`turn_stream()`** is for advanced use cases:
- Processing events as they arrive (progress indicators, etc.)
- Using raw dict schemas
- Custom error handling per event
- Collecting multiple pieces of data from different event types

### Why Not Accept Dict in `turn_stream_structured()`?

If we accepted `dict[str, Any]` schemas in `turn_stream_structured()`, the return type would have to be `dict[str, Any]`, which defeats the purpose of the typed API. The design is:

- Want typed result? → Use `turn_stream_structured()` with Pydantic type
- Want dict schema? → Use `turn_stream()` and parse manually

This keeps the APIs focused and type-safe.

## Examples

### Simple Example (Recommended)

```python
from codex_adapter import CodexClient
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

    print(f"Found {result.total} files")
```

### Advanced Example (Manual Streaming)

```python
from codex_adapter import CodexClient

schema = {
    "type": "object",
    "properties": {
        "files": {"type": "array", "items": {"type": "string"}},
        "total": {"type": "integer"},
    },
}

async with CodexClient() as client:
    thread = await client.thread_start(cwd=".")

    response_text = ""
    async for event in client.turn_stream(
        thread.id,
        "List Python files",
        output_schema=schema,  # Dict schema
    ):
        if event.event_type == "item/agentMessage/delta":
            delta = event.get_text_delta()
            response_text += delta
            print(delta, end="", flush=True)  # Show progress

        elif event.event_type == "turn/completed":
            break

    # Parse manually
    import json
    result = json.loads(response_text)
    print(f"\n\nFound {result['total']} files")
```

## Files

- **`client.py`**: Core implementation with both methods
- **`example_simple_structured.py`**: Simple PydanticAI-like examples
- **`example_typed_structured.py`**: Detailed Pydantic type examples
- **`example_structured.py`**: Dict schema examples
- **`STRUCTURED_RESPONSES.md`**: Complete documentation
- **`STRUCTURED_API_SUMMARY.md`**: This summary

## Benefits

✅ **Type Safety**: Full mypy/pyright support
✅ **IDE Support**: Autocomplete for all fields
✅ **Validation**: Automatic Pydantic validation
✅ **Flexibility**: Choose between simple or advanced API
✅ **PydanticAI-like**: Familiar developer experience
✅ **No Boilerplate**: Auto-generate schemas, auto-parse results
