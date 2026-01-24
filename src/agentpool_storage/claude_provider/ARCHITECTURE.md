# Claude Code Storage Architecture

This document explains how Claude Code persists conversation data, sessions, and agent state to the filesystem.

## Directory Structure

```
~/.claude/
├── projects/                          # Per-project conversation storage
│   ├── {encoded-project-path}/       # One directory per project
│   │   ├── {sessionId}.jsonl         # Main session conversation log
│   │   ├── agent-{shortId}.jsonl     # Sub-agent conversation logs
│   │   └── ...                       # Multiple sessions per project
│   └── ...
├── todos/                             # Per-agent todo/plan persistence
│   ├── {sessionId}-agent-{sessionId}.json      # Main agent todos
│   ├── {sessionId}-agent-{agentId}.json        # Sub-agent todos
│   └── ...
├── session-env/                       # Session environment state
│   └── {sessionId}/                   # Environment per session (often empty)
├── plans/                             # Standalone plan documents
│   └── {name}.md                      # Named plans (legacy?)
├── file-history/                      # File edit history tracking
├── history.jsonl                      # Global command history
└── settings.json                      # User settings
```

## Core Concepts

### 1. Projects

**Project Directory**: `~/.claude/projects/{encoded-project-path}/`

- The project path is URL-encoded to create a safe directory name
- Example: `/home/user/myproject` → `-home-user-myproject`
- All sessions for a given working directory are stored together
- Multiple sessions can exist per project (different conversations over time)

### 2. Sessions

**Session ID**: UUID (e.g., `e8d973c7-481e-43d9-809d-9a4880e8cebc`)

A session represents a single conversation thread with Claude. Each session:

- Has a unique UUID identifier
- Is tied to a specific project/working directory
- Contains a sequence of messages (user, assistant, tool calls, etc.)
- Has one **main agent** and zero or more **sub-agents**
- Persists to a JSONL file: `{sessionId}.jsonl`

**Session File Format**: JSONL (JSON Lines)
- Each line is a separate JSON object representing one entry
- Entries are appended chronologically
- Enables efficient streaming writes and incremental reads

### 3. Agents

#### Main Agent
- **Agent ID** = **Session ID** (same UUID)
- The primary Claude instance handling the conversation
- Session file: `{sessionId}.jsonl`
- Todo file: `{sessionId}-agent-{sessionId}.json`

#### Sub-Agents (Delegated Tasks)
- **Agent ID**: Short hex identifier (e.g., `a753668` - 7 characters)
- Created when tasks are delegated via the Task tool
- Run independently with their own context
- Session file: `agent-{shortId}.jsonl`
- Todo file: `{sessionId}-agent-{shortId}.json`
- Still belong to parent session (share sessionId in metadata)

**Key Pattern**: All agents in a session share the parent `sessionId`:
```
Session: e8d973c7-481e-43d9-809d-9a4880e8cebc
├── Main Agent: e8d973c7-481e-43d9-809d-9a4880e8cebc
│   └── Todo: e8d973c7-481e-43d9-809d-9a4880e8cebc-agent-e8d973c7-481e-43d9-809d-9a4880e8cebc.json
└── Sub-Agent: a753668
    └── Todo: e8d973c7-481e-43d9-809d-9a4880e8cebc-agent-a753668.json
```

### 4. Todo/Plan Storage

**Location**: `~/.claude/todos/`

**Naming**: `{sessionId}-agent-{agentId}.json`

**Format**: JSON array of todo entries
```json
[
  {
    "content": "Task description",
    "status": "pending" | "in_progress" | "completed",
    "priority": "high" | "medium" | "low",
    "activeForm": "Current action description (optional)"
  }
]
```

**Persistence Pattern**:
- Todos are scoped to a specific agent within a session
- Main agent's todos use redundant naming: `{sessionId}-agent-{sessionId}.json`
- Sub-agent todos use: `{sessionId}-agent-{shortId}.json`
- All todos for a session share the session ID prefix
- Survives across conversation resumes

## Entry Types

Claude Code uses different entry types in JSONL session files:

### User Entry
```json
{
  "type": "user",
  "sessionId": "uuid",
  "uuid": "message-uuid",
  "parentUuid": "parent-message-uuid",
  "timestamp": "ISO-8601",
  "message": {
    "role": "user",
    "content": "text or array of content blocks"
  }
}
```

### Assistant Entry
```json
{
  "type": "assistant",
  "sessionId": "uuid",
  "uuid": "message-uuid",
  "parentUuid": "parent-message-uuid",
  "timestamp": "ISO-8601",
  "message": {
    "role": "assistant",
    "content": [
      {"type": "text", "text": "..."},
      {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
    ]
  },
  "requestId": "req_...",
  "model": "claude-..."
}
```

### Queue Operation Entry
```json
{
  "type": "queue-operation",
  "operation": "enqueue" | "dequeue",
  "sessionId": "uuid",
  "timestamp": "ISO-8601"
}
```

Marks session lifecycle events (start/end of processing).

### System Entry
```json
{
  "type": "system",
  "sessionId": "uuid",
  "timestamp": "ISO-8601",
  "message": {
    "role": "system",
    "content": "System message text"
  }
}
```

### Tool Result Entry
Embedded within user entries as tool_result content blocks:
```json
{
  "type": "user",
  "message": {
    "role": "user",
    "content": [
      {
        "type": "tool_result",
        "tool_use_id": "toolu_...",
        "content": "result text or structured data"
      }
    ]
  },
  "toolUseResult": ["..."]  // Duplicate for UI
}
```

### Summary Entry
```json
{
  "type": "summary",
  "sessionId": "uuid",
  "timestamp": "ISO-8601",
  "content": {
    "summary": "Condensed conversation summary",
    "messageCount": 10,
    "summarizedUntil": "message-uuid"
  }
}
```

Used for conversation compaction/context management.

### File History Entry
```json
{
  "type": "file-history",
  "sessionId": "uuid",
  "timestamp": "ISO-8601",
  "path": "/absolute/path/to/file",
  "operation": "edit" | "create" | "delete",
  "snippet": "Preview of changes..."
}
```

Tracks file operations for undo/history features.

## Message Flow & Ancestry

### Parent-Child Relationships

Every message (except the first) has a `parentUuid` that references the previous message:

```
User Message 1 (uuid: a)
  └─> Assistant Reply (uuid: b, parentUuid: a)
       └─> Tool Results (uuid: c, parentUuid: b)
            └─> Assistant Response (uuid: d, parentUuid: c)
```

This creates a **linked list** of messages forming the conversation thread.

### Branching (Sidechains)

Field: `isSidechain: boolean`

When a conversation is forked or edited:
- Original messages remain unchanged
- New branch starts with `isSidechain: true`
- Allows exploring alternative conversation paths
- Main chain: `isSidechain: false` or omitted

### Conversation Reconstruction

To read a conversation:
1. Parse all entries from `{sessionId}.jsonl`
2. Filter by `type` (user, assistant, etc.)
3. Follow the parent chain from latest message backward
4. Stop at session start or desired depth
5. Reverse to get chronological order

## Storage Provider Implementation

The `ClaudeStorageProvider` class bridges between:
- **Claude's storage format** (JSONL entries)
- **Agentpool's abstractions** (ChatMessage, ConversationStats, etc.)

### Key Responsibilities

1. **Path Management**
   - Encode/decode project paths to safe directory names
   - Ensure project directories exist

2. **Session Management**
   - List all sessions for a project
   - Read session entries (with optional filtering)
   - Write new entries atomically

3. **Message Conversion**
   - JSONL Entry → `ChatMessage` (for agentpool)
   - `ChatMessage` → JSONL Entry (for persistence)
   - Handle tool calls, results, and metadata

4. **Conversation Queries**
   - Get message count, token usage stats
   - Retrieve specific messages by UUID
   - Trace message ancestry
   - Fork conversations at specific points

5. **Filtering & Compaction**
   - Filter by message type, date range
   - Skip tool calls or system messages
   - Enable conversation summarization

## Data Consistency

### Atomic Writes
- Each entry is appended as a complete line
- JSONL format allows crash-safe appends
- No partial writes visible to readers

### Concurrent Access
- Multiple sessions can write to different files safely
- Same session should have single writer (main process)
- Readers can stream entries while writing continues

### Cleanup
- Empty session files may accumulate
- No automatic garbage collection (user-managed)
- `reset()` method can delete project history

## Integration Points

### With Agentpool
- `ClaudeStorageProvider` implements `StorageProvider` protocol
- Converts between storage formats and domain models
- Enables conversation persistence for agents

### With TodoTracker
- Separate persistence in `~/.claude/todos/`
- Not integrated with main conversation storage
- Must be loaded/saved independently
- File naming links todos to session + agent

### With File History
- Separate tracking in `~/.claude/file-history/`
- Records file operations during conversation
- Enables undo/diff features
- Cross-referenced by session ID

## Usage Patterns

### Starting a New Session
```python
provider = ClaudeStorageProvider(base_dir="~/.claude")
session_id = str(uuid.uuid4())

# Log initial user message
await provider.log_message(
    project_path="/path/to/project",
    message=ChatMessage(role="user", content="Hello"),
    session_id=session_id,
)
```

### Resuming a Session
```python
# Get existing sessions
conversations = await provider.get_sessions(
    project_path="/path/to/project"
)

# Load latest session
messages = await provider.get_session_messages(
    project_path="/path/to/project",
    session_id=conversations[0]["sessionId"],
    limit=50  # Last 50 messages
)
```

### Creating a Sub-Agent
```python
# Sub-agent gets own short ID
agent_id = "a" + secrets.token_hex(3)  # e.g., "a753668"

# But shares parent sessionId
await provider.log_message(
    project_path="/path/to/project",
    message=agent_message,
    session_id=parent_session_id,  # Same as parent!
    agent_id=agent_id,  # Different agent ID
)

# Creates: agent-a753668.jsonl in project dir
# Creates: {sessionId}-agent-a753668.json in todos/
```

### Saving Todos
```python
# Not part of ClaudeStorageProvider!
# Separate JSON file management:

todo_path = (
    f"~/.claude/todos/{session_id}-agent-{agent_id}.json"
)
with open(todo_path, "w") as f:
    json.dump(todos, f)
```

## Design Rationale

### Why JSONL?
- **Streamable**: Can read/write incrementally
- **Crash-safe**: Each line is atomic
- **Human-readable**: Easy debugging
- **Flexible schema**: Each entry can evolve independently

### Why Separate Agent Files?
- **Isolation**: Sub-agents run independently
- **Parallelism**: Multiple agents can write concurrently
- **Clarity**: Easy to see which agent did what
- **Performance**: Don't need to parse entire session for sub-agent context

### Why Redundant Session ID in Main Agent Todos?
- **Consistency**: All todos follow `{sessionId}-agent-{agentId}` pattern
- **Simplicity**: Single naming rule, no special cases
- **Glob-friendly**: Easy to find all todos for a session: `{sessionId}-agent-*.json`

### Why Separate Todos from Conversation Log?
- **Orthogonal concerns**: Todos are working memory, not conversation history
- **Update frequency**: Todos change frequently, conversations append-only
- **Size**: Todos stay small, conversations grow large
- **Format**: Todos are mutable array, conversations are immutable log

## Future Considerations

### Scalability
- Large projects accumulate many session files
- May need session archival/cleanup strategies
- Consider session indexing for faster queries

### Consistency
- No transactional guarantees across files
- Todo updates not synchronized with conversation log
- Sub-agent todos could become orphaned

### Migration
- Current format lacks version markers
- Schema evolution requires careful handling
- Consider adding format version to entries

### Compression
- JSONL files can grow large
- Consider gzip compression for archived sessions
- Balance between size and read performance

---

**Related Files:**
- Implementation: [`claude_provider.py`](./claude_provider.py)
- Models: [`models.py`](./models.py)
- Base Protocol: [`../agentpool/storage/storage_provider.py`](../agentpool/storage/storage_provider.py)
