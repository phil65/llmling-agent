# Codex Adapter API Reference

Complete API reference for the Codex app-server Python adapter.

## Table of Contents

- [Client Initialization](#client-initialization)
- [Thread Operations](#thread-operations)
- [Turn Operations](#turn-operations)
- [Skills & Models](#skills--models)
- [Command Execution](#command-execution)
- [Event Types](#event-types)
- [Type Safety](#type-safety)

## Client Initialization

### CodexClient

```python
from codex_adapter import CodexClient

async with CodexClient(
    codex_command="codex",  # Path to codex binary
    profile=None,            # Optional profile name
) as client:
    # Use client methods
    pass
```

## Thread Operations

### thread_start

Start a new conversation thread.

```python
response = await client.thread_start(
    cwd="/path/to/project",  # Optional: working directory
    model="gpt-5.1-codex",   # Optional: model to use
    effort="high",            # Optional: reasoning effort ("low", "medium", "high", "xhigh")
)
# Returns: ThreadResponse with response.thread containing thread data
thread_id = response.thread.id
```

### thread_resume

Resume an existing thread by ID.

```python
response = await client.thread_resume(thread_id="019bab...")
# Returns: ThreadResponse with thread data and conversation history (response.thread.turns)
```

### thread_fork

Fork an existing thread into a new thread with copied history.

```python
response = await client.thread_fork(thread_id="019bab...")
# Returns: ThreadResponse containing the new forked thread
```

### thread_list

List stored threads with pagination support.

```python
result = await client.thread_list(
    cursor=None,                        # Pagination cursor from previous response
    limit=25,                            # Max threads to return
    model_providers=["openai"],          # Optional: filter by providers
)
# Returns: {"data": [...], "nextCursor": str | None}
```

### thread_loaded_list

List thread IDs currently loaded in memory.

```python
loaded_ids = await client.thread_loaded_list()
# Returns: ["019bab...", "019bac..."]
```

### thread_archive

Archive a thread (move to archived directory).

```python
await client.thread_archive(thread_id="019bab...")
```

### thread_rollback

Rollback the last N turns from a thread.

```python
result = await client.thread_rollback(
    thread_id="019bab...",
    turns=2,  # Number of turns to rollback
)
# Returns: Updated thread object with turns populated
```

## Turn Operations

### turn_stream

Start a turn and stream events in real-time.

```python
async for event in client.turn_stream(
    thread_id="019bab...",
    user_input="Help me refactor this code",  # str or list[dict]
    model="claude-opus-4",      # Optional: override model
    effort="high",              # Optional: override effort
    approval_policy="never",    # Optional: "always", "never", "unlessTrusted"
):
    if event.event_type == "item/agentMessage/delta":
        print(event.get_text_delta(), end="", flush=True)
    elif event.event_type == "turn/completed":
        break
```

**Input formats:**
```python
# Simple text
user_input = "Help me debug"

# Multiple items
user_input = [
    {"type": "text", "text": "Analyze this diff"},
    {"type": "localImage", "path": "/path/to/screenshot.png"},
]
```

### turn_interrupt

Interrupt a running turn.

```python
await client.turn_interrupt(
    thread_id="019bab...",
    turn_id="0",
)
```

## Skills & Models

### skills_list

List available skills.

```python
skills = await client.skills_list(
    cwd="/path/to/project",  # Optional: scope to directory
    force_reload=False,       # Optional: force cache reload
)
# Returns: [{"name": "skill-creator", "description": "...", ...}, ...]
```

### model_list

List available models with reasoning effort options.

```python
models = await client.model_list()
# Returns: [
#   {
#     "id": "gpt-5.1-codex-max",
#     "model": "gpt-5.1-codex-max",
#     "displayName": "gpt-5.1-codex-max",
#     "description": "...",
#     "supportedReasoningEfforts": [...],
#     "defaultReasoningEffort": "medium",
#     "isDefault": true
#   },
#   ...
# ]
```

## Command Execution

### command_exec

Execute a command without creating a thread/turn.

```python
result = await client.command_exec(
    command=["ls", "-la"],
    cwd="/path/to/dir",        # Optional
    sandbox_policy={            # Optional
        "type": "workspaceWrite",
        "networkAccess": "enabled",
    },
    timeout_ms=10000,           # Optional
)
# Returns: {"exitCode": 0, "stdout": "...", "stderr": ""}
```

## Event Types

### Complete Event Type List

All 28 event types from app-server:

**Error Events:**
- `error` - Turn errors

**Thread Lifecycle:**
- `thread/started` - Thread created
- `thread/tokenUsage/updated` - Token usage stats
- `thread/compacted` - History compacted

**Turn Lifecycle:**
- `turn/started` - Turn begins
- `turn/completed` - Turn finished
- `turn/diff/updated` - Unified diff update
- `turn/plan/updated` - Agent plan update

**Item Lifecycle:**
- `item/started` - Item begins
- `item/completed` - Item finished
- `rawResponseItem/completed` - Raw response item

**Agent Message Deltas:**
- `item/agentMessage/delta` - Streaming agent text

**Reasoning Deltas:**
- `item/reasoning/summaryTextDelta` - Reasoning summary
- `item/reasoning/summaryPartAdded` - New summary section
- `item/reasoning/textDelta` - Raw reasoning text

**Command Execution:**
- `item/commandExecution/outputDelta` - Command output streaming
- `item/commandExecution/terminalInteraction` - Terminal interaction

**File Changes:**
- `item/fileChange/outputDelta` - File change output

**MCP Tool Calls:**
- `item/mcpToolCall/progress` - MCP tool progress

**MCP OAuth:**
- `mcpServer/oauthLogin/completed` - OAuth flow finished

**Account/Auth:**
- `account/updated` - Auth state changed
- `account/rateLimits/updated` - Rate limits changed
- `account/login/completed` - Login finished
- `authStatusChange` - Legacy auth change
- `loginChatGptComplete` - Legacy ChatGPT login

**System:**
- `sessionConfigured` - Session configured
- `deprecationNotice` - Feature deprecated
- `windows/worldWritableWarning` - Security warning

### CodexEvent API

```python
event = CodexEvent.from_notification(method, params)

# Properties
event.event_type  # str: e.g., "item/agentMessage/delta"
event.data        # dict[str, Any]: event payload
event.raw         # dict[str, Any]: full notification

# Helper methods
event.is_delta()          # True for streaming events
event.is_completed()      # True for completion events
event.is_error()          # True for error events
event.get_text_delta()    # Extract text from deltas
```

## Type Safety

### Data Models

```python
from codex_adapter import ThreadResponse, ThreadData
from codex_adapter.codex_types import CodexTurn, CodexItem

# ThreadResponse contains ThreadData
# Access thread data via response.thread after thread_start/resume/fork
response: ThreadResponse
thread_data: ThreadData = response.thread
print(thread_data.id)       # Thread ID
print(thread_data.preview)  # Conversation preview
print(thread_data.turns)    # List of Turn objects with conversation history

# CodexTurn
turn = CodexTurn(
    id="0",
    thread_id="019bab...",
    status="inProgress",
    items=[],
    error=None,
    usage=None,
)

# CodexItem
item = CodexItem(
    id="call_abc",
    type="commandExecution",
    content="...",
    status="completed",
    metadata={},
)
```

### Exception Handling

```python
from codex_adapter.exceptions import CodexError, CodexProcessError, CodexRequestError

try:
    await client.thread_start()
except CodexProcessError as e:
    # Subprocess/connection errors
    print(f"Process error: {e}")
except CodexRequestError as e:
    # JSON-RPC error responses
    print(f"Request error [{e.code}]: {e.message}")
    print(f"Details: {e.data}")
except CodexError as e:
    # Base exception
    print(f"Codex error: {e}")
```

## Advanced Usage

### Pagination

```python
# Iterate through all threads
cursor = None
all_threads = []

while True:
    result = await client.thread_list(cursor=cursor, limit=100)
    all_threads.extend(result["data"])

    cursor = result.get("nextCursor")
    if not cursor:
        break
```

### Event Filtering

```python
async for event in client.turn_stream(thread_id, "analyze code"):
    # Filter by event type
    if event.event_type.startswith("item/agentMessage"):
        # Handle agent messages
        if event.is_delta():
            print(event.get_text_delta(), end="")
        elif event.is_completed():
            print("\n[Message completed]")

    elif event.event_type.startswith("item/commandExecution"):
        # Handle command execution
        if event.is_delta():
            print(f"[Command output]: {event.get_text_delta()}")

    elif event.event_type == "turn/completed":
        # Turn finished
        usage = event.data.get("turn", {}).get("usage", {})
        print(f"Token usage: {usage}")
        break
```

### Multi-turn Conversation

```python
async with CodexClient() as client:
    thread = await client.thread_start(cwd=".")

    messages = [
        "What's in this directory?",
        "Show me the main.py file",
        "Add type hints to all functions",
    ]

    for msg in messages:
        print(f"\n> {msg}")
        async for event in client.turn_stream(thread.id, msg):
            if event.event_type == "item/agentMessage/delta":
                print(event.get_text_delta(), end="", flush=True)
            elif event.event_type == "turn/completed":
                print("\n")
                break
```

### Forking Conversations

```python
# Create base conversation
thread = await client.thread_start()
async for event in client.turn_stream(thread.id, "Setup a Python project"):
    if event.event_type == "turn/completed":
        break

# Fork to try different approaches
fork1 = await client.thread_fork(thread.id)
fork2 = await client.thread_fork(thread.id)

# Different continuations
async for event in client.turn_stream(fork1.id, "Use pytest for testing"):
    ...

async for event in client.turn_stream(fork2.id, "Use unittest instead"):
    ...
```

## Response Formats

### thread/start, thread/resume, thread/fork

```json
{
  "thread": {
    "id": "019bab...",
    "preview": "Conversation about...",
    "modelProvider": "openai",
    "createdAt": 1768110000,
    "path": "/home/user/.codex/sessions/...",
    "cwd": "/path/to/project",
    "gitInfo": {...},
    "turns": []
  },
  "model": "gpt-5.1-codex-mini",
  "modelProvider": "openai",
  "cwd": "/path/to/project",
  "approvalPolicy": "on-request",
  "sandbox": {...},
  "reasoningEffort": "high"
}
```

### turn/start

```json
{
  "turn": {
    "id": "0",
    "items": [],
    "status": "inProgress",
    "error": null
  }
}
```

### turn/completed (event)

```json
{
  "threadId": "019bab...",
  "turn": {
    "id": "0",
    "items": [...],
    "status": "completed",
    "error": null,
    "usage": {
      "totalTokens": 1234,
      "inputTokens": 1000,
      "outputTokens": 234
    }
  }
}
```

## See Also

- [README.md](README.md) - Quick start guide
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Implementation notes
- [Codex app-server docs](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)
