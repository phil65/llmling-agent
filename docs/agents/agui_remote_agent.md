# AG-UI Remote Agent

The `AGUIAgent` is a `MessageNode` that connects to remote agents implementing the [AG-UI protocol](https://docs.ag-ui.com). This enables distributed agent architectures where agents run as separate services, while maintaining full compatibility with llmling-agent's messaging system, teams, and connections.

## Overview

AG-UI (Agent-User Interaction) is a protocol for real-time communication between AI agents and clients using Server-Sent Events (SSE). The `AGUIAgent` class inherits from `MessageNode`, providing full integration with llmling-agent's multi-agent system including connections, teams, and event routing.

## Basic Usage

```python
from llmling_agent.agent import AGUIAgent

async with AGUIAgent(
    endpoint="http://localhost:8000/agent/run",
    name="remote-assistant"
) as agent:
    # Non-streaming execution
    result = await agent.run("What is the capital of France?")
    print(result.content)
```

## Streaming Responses

The AG-UI protocol supports streaming, allowing real-time event processing:

```python
from pydantic_ai import PartDeltaEvent
from pydantic_ai.messages import TextPartDelta

async with AGUIAgent(
    endpoint="http://localhost:8000/agent/run",
    name="streaming-agent"
) as agent:
    async for event in agent.run_stream("Tell me a story"):
        # Process text deltas as they arrive
        if isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta):
                print(event.delta.content_delta, end="", flush=True)
```

## Configuration Options

### Constructor Parameters

- **`endpoint`** (str): HTTP endpoint for the AG-UI agent
- **`name`** (str): Agent identifier (default: `"agui-agent"`)
- **`description`** (str | None): Agent description
- **`display_name`** (str | None): Human-readable display name
- **`timeout`** (float): Request timeout in seconds (default: 60.0)
- **`headers`** (dict[str, str] | None): Additional HTTP headers
- **`mcp_servers`** (Sequence | None): MCP servers to connect
- **`agent_pool`** (AgentPool | None): Agent pool for multi-agent coordination
- **`enable_logging`** (bool): Whether to enable database logging (default: True)
- **`event_configs`** (Sequence | None): Event trigger configurations

### Example with Custom Configuration

```python
async with AGUIAgent(
    endpoint="https://api.example.com/agent",
    name="production-agent",
    description="Remote production agent",
    timeout=120.0,
    headers={
        "X-API-Key": "your-api-key",
        "X-User-ID": "user-123"
    },
    agent_pool=pool,
    enable_logging=True
) as agent:
    result = await agent.run("Process this request")
```

## Converting to Tools

AG-UI agents can be converted into callable tools for use by other agents:

```python
async with AGUIAgent(
    endpoint="http://localhost:8000/calculator",
    name="calculator"
) as agent:
    # Convert to tool
    calculator_tool = agent.to_tool(
        description="A calculator that solves math problems"
    )
    
    # Use in another agent
    from llmling_agent import Agent
    
    main_agent = Agent(
        name="main",
        model="openai:gpt-4",
        tools=[calculator_tool]
    )
```

## Multi-Agent Integration

Since `AGUIAgent` inherits from `MessageNode`, it works seamlessly with llmling-agent's multi-agent features:

### Agent Connections

```python
from llmling_agent import Agent
from llmling_agent.agent import AGUIAgent

# Create local and remote agents
local_agent = Agent(name="local", model="openai:gpt-4")
remote_agent = AGUIAgent(
    endpoint="http://localhost:8000/agent",
    name="remote"
)

# Connect them
async with local_agent, remote_agent:
    local_agent >> remote_agent  # Pipe results to remote agent
```

### Agent Teams

```python
from llmling_agent.delegation import Team

# Create a team with local and remote agents
team = Team(
    name="hybrid-team",
    members=[local_agent, remote_agent]
)

async with team:
    result = await team.run("Collaborate on this task")
```

### Agent Pools

```python
from llmling_agent.delegation import AgentPool

pool = AgentPool()
pool.register("local", local_agent)
pool.register("remote", remote_agent)

async with pool:
    # Use agents from the pool
    result = await pool.get("remote").run("Remote task")
```

## Event Conversion

The `AGUIAgent` automatically converts AG-UI protocol events to native llmling-agent events:

| AG-UI Event | Native Event |
|-------------|--------------|
| `TEXT_MESSAGE_CONTENT` | `PartDeltaEvent` with `TextPartDelta` |
| `TEXT_MESSAGE_CHUNK` | `PartDeltaEvent` with `TextPartDelta` |
| `THINKING_TEXT_MESSAGE_CONTENT` | `PartDeltaEvent` with `ThinkingPartDelta` |
| `TOOL_CALL_START` | `ToolCallStartEvent` |
| `TOOL_CALL_RESULT` | `ToolCallProgressEvent` (completed) |
| `TOOL_CALL_END` | `ToolCallProgressEvent` (completed) |

## Error Handling

The agent provides comprehensive error handling for network and protocol issues:

```python
import httpx

async with AGUIAgent(
    endpoint="http://localhost:8000/agent",
    name="error-handling-example"
) as agent:
    try:
        result = await agent.run("Test query")
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

## Statistics and Monitoring

Track agent usage with built-in statistics:

```python
async with AGUIAgent(
    endpoint="http://localhost:8000/agent",
    name="monitored-agent"
) as agent:
    await agent.run("First query")
    await agent.run("Second query")
    
    stats = await agent.get_stats()
    print(f"Message count: {stats.message_count}")
    print(f"Total tokens: {stats.token_count}")
    print(f"Total cost: ${stats.total_cost}")
```

## Comparison with ACP Agent

Both `AGUIAgent` and `ACPAgent` enable remote agent execution, but use different protocols:

| Feature | AGUIAgent | ACPAgent |
|---------|-----------|----------|
| **Base Class** | MessageNode | MessageNode |
| **Protocol** | AG-UI (HTTP/SSE) | ACP (JSON-RPC/stdio) |
| **Transport** | HTTP | Process stdio |
| **Deployment** | Remote web service | Local subprocess |
| **Use Case** | Distributed systems | Local processes |
| **Authentication** | HTTP headers | Process permissions |
| **Multi-agent** | Full support | Full support |

## AG-UI Protocol Details

The AG-UI protocol uses:

- **Server-Sent Events (SSE)** for streaming responses
- **JSON serialization** with camelCase field names
- **Structured events** for text, thinking, and tool calls
- **HTTP POST** for initiating runs

### Request Format

```json
{
  "threadId": "thread-123",
  "runId": "run-456",
  "state": {},
  "messages": [
    {
      "id": "msg-1",
      "role": "user",
      "content": "Hello"
    }
  ],
  "tools": [],
  "context": [],
  "forwardedProps": {}
}
```

### Response Format (SSE)

```
data: {"type":"TEXT_MESSAGE_START","messageId":"msg-1"}

data: {"type":"TEXT_MESSAGE_CONTENT","messageId":"msg-1","delta":"Hello"}

data: {"type":"TEXT_MESSAGE_END","messageId":"msg-1"}
```

## Best Practices

1. **Use Context Managers**: Always use `async with` to ensure proper cleanup
2. **Set Appropriate Timeouts**: Configure timeouts based on expected response times
3. **Handle Network Errors**: Implement retry logic for production use
4. **Monitor Statistics**: Track usage for debugging and optimization
5. **Secure Endpoints**: Use HTTPS and authentication in production
6. **Use Agent Pools**: Register in pools for multi-agent coordination
7. **Connect Agents**: Leverage MessageNode connections for workflows

## Example: Production Setup

```python
import asyncio
from llmling_agent.agent import AGUIAgent

async def query_remote_agent(prompt: str) -> str:
    """Query remote agent with retry logic."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            async with AGUIAgent(
                endpoint="https://api.example.com/agent/run",
                name="production-agent",
                timeout=60.0,
                headers={
                    "Authorization": f"Bearer {get_api_token()}",
                    "X-Request-ID": generate_request_id()
                }
            ) as agent:
                result = await agent.run(prompt)
                return result.content
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    raise RuntimeError("Max retries exceeded")

def get_api_token() -> str:
    """Get API token from secure storage."""
    # Implementation depends on your security setup
    return "your-secure-token"

def generate_request_id() -> str:
    """Generate unique request ID for tracing."""
    from uuid import uuid4
    return str(uuid4())
```

## See Also

- [MessageNode Base Class](../concepts/messagenode.md) - Base interface for all agents
- [ACP Remote Agent](../advanced/acp_agent.md) - Process-based remote agents
- [Agent Teams](team.md) - Multi-agent coordination
- [Agent Connections](../concepts/connections.md) - Message routing between nodes
- [Tool Manager](tool_manager.md) - Tool integration
- [AG-UI Protocol Docs](https://docs.ag-ui.com) - Official protocol documentation