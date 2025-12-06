# AG-UI Remote Agent Implementation

## Overview

The AG-UI remote agent implementation enables llmling-agent to connect to and use remote agents that implement the [AG-UI (Agent-User Interaction) protocol](https://docs.ag-ui.com). This provides a standardized way to integrate distributed agent services into the llmling-agent ecosystem.

## Architecture

### Components

1. **`AGUIAgent`** - Main client class that connects to AG-UI endpoints
2. **`agui_converters.py`** - Event conversion between AG-UI and native formats
3. **`AGUISessionState`** - State tracking for active sessions

### Design Principles

- **Protocol Adherence**: Full compliance with AG-UI protocol specification
- **Native Integration**: Seamless conversion to native llmling-agent events
- **Streaming Support**: Real-time event processing via Server-Sent Events
- **Type Safety**: Full Pydantic validation throughout

## Protocol Details

### Communication Flow

```
Client (AGUIAgent)                    Server (AG-UI Service)
      |                                      |
      |-- POST /agent/run ------------------>|
      |   (RunAgentInput JSON)               |
      |                                      |
      |<-- SSE Stream ----------------------|
      |   data: {TEXT_MESSAGE_START}         |
      |   data: {TEXT_MESSAGE_CONTENT}       |
      |   data: {TOOL_CALL_START}            |
      |   data: {TEXT_MESSAGE_END}           |
      |                                      |
```

### Request Format

The client sends `RunAgentInput` with camelCase fields:

```json
{
  "threadId": "thread-123",
  "runId": "run-456",
  "state": {},
  "messages": [
    {
      "id": "msg-1",
      "role": "user",
      "content": "Hello, world!"
    }
  ],
  "tools": [],
  "context": [],
  "forwardedProps": {}
}
```

### Response Format

Server responds with SSE stream:

```
data: {"type":"TEXT_MESSAGE_START","messageId":"msg-1","role":"assistant"}

data: {"type":"TEXT_MESSAGE_CONTENT","messageId":"msg-1","delta":"Hello"}

data: {"type":"TEXT_MESSAGE_END","messageId":"msg-1"}
```

## Event Conversion

### Mapping Table

| AG-UI Event Type | Native Event Type | Conversion Logic |
|-----------------|-------------------|------------------|
| `TEXT_MESSAGE_CONTENT` | `PartDeltaEvent` with `TextPartDelta` | Direct delta mapping |
| `TEXT_MESSAGE_CHUNK` | `PartDeltaEvent` with `TextPartDelta` | Direct delta mapping |
| `THINKING_TEXT_MESSAGE_CONTENT` | `PartDeltaEvent` with `ThinkingPartDelta` | Maps to thinking content |
| `TOOL_CALL_START` | `ToolCallStartEvent` | Creates tool call with metadata |
| `TOOL_CALL_CHUNK` | `ToolCallStartEvent` or `ToolCallProgressEvent` | New call or progress update |
| `TOOL_CALL_ARGS` | `ToolCallProgressEvent` | In-progress status |
| `TOOL_CALL_RESULT` | `ToolCallProgressEvent` | Completed with content |
| `TOOL_CALL_END` | `ToolCallProgressEvent` | Completed status |
| `TEXT_MESSAGE_START/END` | None | Metadata tracking only |
| `THINKING_START/END` | None | Metadata tracking only |
| `ACTIVITY_SNAPSHOT/DELTA` | None | Reserved for future use |

### Converter Implementation

The converter uses pattern matching for type-safe event conversion:

```python
def agui_to_native_event(event: Event) -> RichAgentStreamEvent[Any] | None:
    """Convert AG-UI event to native streaming event."""
    match event:
        case TextMessageContentEvent(delta=delta):
            return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=delta))
        
        case ToolCallStartEvent(tool_call_id=id, tool_call_name=name):
            return NativeToolCallStartEvent(
                tool_call_id=id,
                tool_name=name,
                title=name,
                kind="other",
                content=[],
                locations=[],
                raw_input={},
            )
        # ... more patterns
```

## State Management

### Session State

Each `AGUIAgent` maintains session state:

```python
@dataclass
class AGUISessionState:
    thread_id: str                    # Conversation thread
    run_id: str | None                # Current run ID
    text_chunks: list[str]            # Accumulated text
    thought_chunks: list[str]         # Accumulated thoughts
    tool_calls: dict[str, dict]       # Active tool calls
    is_complete: bool                 # Run completion status
    error: str | None                 # Error message if failed
```

### Lifecycle

1. **Initialization**: Create client and session on context entry
2. **Execution**: Send request, stream events, accumulate state
3. **Completion**: Build final message from accumulated state
4. **Cleanup**: Close client on context exit

## Usage Patterns

### Basic Usage

```python
from llmling_agent.agent import AGUIAgent

async with AGUIAgent(
    endpoint="http://localhost:8000/agent/run",
    name="remote-agent"
) as agent:
    result = await agent.run("What is 2+2?")
    print(result.content)
```

### Streaming

```python
async for event in agent.run_stream("Tell me a story"):
    if isinstance(event, PartDeltaEvent):
        print(event.delta.content_delta, end="", flush=True)
```

### Tool Conversion

```python
calculator_tool = agent.to_tool("Calculator agent")
result = await calculator_tool("What is 157 * 89?")
```

## Error Handling

### HTTP Errors

```python
try:
    result = await agent.run("Test")
except httpx.HTTPError as e:
    logger.error("Network error", error=str(e))
```

### Protocol Errors

Invalid SSE events are logged but don't stop streaming:

```python
try:
    event = event_adapter.validate_json(json_str)
    yield event
except Exception as e:
    self.log.warning("Failed to parse event", error=str(e))
```

## Comparison with ACP Agent

| Feature | AGUIAgent | ACPAgent |
|---------|-----------|----------|
| **Protocol** | AG-UI (HTTP/SSE) | ACP (JSON-RPC) |
| **Transport** | HTTP | Process stdio |
| **Deployment** | Remote web service | Local subprocess |
| **Streaming** | Native SSE | Session notifications |
| **Authentication** | HTTP headers | Process permissions |
| **State** | Stateless HTTP | Stateful session |
| **Use Cases** | Microservices, distributed systems | Local tools, isolated environments |

## Testing Strategy

### Unit Tests

- Event conversion correctness
- Text extraction from events
- Session state management

### Integration Tests

- Mock HTTP client for SSE streaming
- Request/response format validation
- Error handling scenarios
- Context manager lifecycle

### Test Fixtures

```python
@pytest.fixture
def mock_sse_response():
    """Create mock SSE response with events."""
    def _create_response(events: list[str]) -> AsyncMock:
        response = AsyncMock(spec=httpx.Response)
        response.aiter_text = lambda: (e for e in events)
        return response
    return _create_response
```

## Future Enhancements

### Potential Additions

1. **WebSocket Support**: For bidirectional communication
2. **Activity Events**: Map to plan updates or progress tracking
3. **State Persistence**: Session state recovery across restarts
4. **Connection Pool**: Reuse connections for multiple requests
5. **Retry Logic**: Automatic retry with exponential backoff
6. **Metrics**: Detailed performance and usage tracking

### Protocol Extensions

- **Binary Content**: Support for images, files in messages
- **Multimodal Input**: Handle complex input types from AG-UI
- **Custom Events**: Application-specific event types
- **State Synchronization**: Bidirectional state updates

## Dependencies

- **`httpx`**: HTTP client with async/await support
- **`ag-ui-protocol`**: Official Python SDK for AG-UI types
- **`pydantic`**: Type validation and serialization
- **`pydantic-ai`**: Native event types

## References

- [AG-UI Protocol Specification](https://docs.ag-ui.com)
- [AG-UI Python SDK](https://github.com/tawkit/ag-ui-protocol)
- [ACP Agent Implementation](../advanced/acp_agent.md)
- [llmling-agent Architecture](../../key_concepts.md)