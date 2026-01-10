# OpenCode Storage Architecture

This document explains how OpenCode persists conversation data to the filesystem.

## Directory Structure

```
~/.local/share/opencode/
├── storage/
│   ├── message/                       # Message metadata by session
│   │   └── {sessionID}/
│   │       └── {messageID}.json      # Message metadata
│   ├── part/                          # Message content parts
│   │   └── {messageID}/
│   │       └── {partID}.json         # Content blocks (text, tool_use, etc.)
│   ├── session/                       # Session metadata by project
│   │   ├── global/                    # Sessions not tied to a project
│   │   │   └── {sessionID}.json      # Session metadata
│   │   └── {projectID}/               # Project-specific sessions
│   │       └── {sessionID}.json      # Session metadata
│   ├── session_diff/                  # Session diffs (unused?)
│   ├── session_share/                 # Shared sessions (unused?)
│   └── project/                       # Project metadata
├── snapshot/                          # Project snapshots
├── log/                               # Application logs
└── auth.json                          # Authentication credentials
```

## Core Concepts

### 1. Storage Model: Normalized Database on Filesystem

OpenCode uses a **normalized relational model** stored as JSON files:
- **Sessions** → Metadata about conversations
- **Messages** → Message-level metadata (role, time, agent)
- **Parts** → Content blocks within messages (text, tool_use, tool_result)

This is fundamentally different from Claude Code's append-only JSONL approach.

### 2. ID Format

All IDs use a custom format with prefixes:
- **Session**: `ses_{random}` (e.g., `ses_4afbda00cffeVl5YERm4op7JEG`)
- **Message**: `msg_{random}` (e.g., `msg_b50425ff7001vgFiMUNFzLtCda`)
- **Part**: `prt_{random}` (e.g., `prt_b50425ff7002egcP330jM5wQcU`)
- **Project**: SHA1 hash of directory path (e.g., `486ce75a8fddd4372018ab816ac62d8004dc52fd`)

The random portion appears to be a timestamp-based identifier.

### 3. Projects

**Project ID**: SHA1 hash of the absolute directory path

Example:
```bash
echo -n "/home/phil65/dev/oss/agentpool" | sha1sum
# → 486ce75a8fddd4372018ab816ac62d8004dc52fd
```

**Special Project**: `global`
- Used for sessions not tied to a specific directory
- Sessions created from home directory or no working directory

**Storage**:
- Session files: `storage/session/{projectID}/{sessionID}.json`
- Global sessions: `storage/session/global/{sessionID}.json`

### 4. Sessions

**Session File**: `storage/session/{projectID}/{sessionID}.json`

```json
{
  "id": "ses_4afbda00cffeVl5YERm4op7JEG",
  "version": "1.0.193",
  "projectID": "486ce75a8fddd4372018ab816ac62d8004dc52fd",
  "directory": "/home/phil65/dev/oss/agentpool",
  "title": "Claude capabilities overview",
  "time": {
    "created": 1766578085876,
    "updated": 1766578154946
  },
  "summary": {
    "additions": 0,
    "deletions": 0,
    "files": 0
  }
}
```

**Fields**:
- `id`: Session identifier
- `version`: OpenCode version that created the session
- `projectID`: SHA1 hash of directory or "global"
- `directory`: Absolute path to working directory
- `title`: Auto-generated summary of session topic
- `time.created`: Unix timestamp (milliseconds)
- `time.updated`: Unix timestamp (milliseconds)
- `summary`: File change statistics

### 5. Messages

**Message File**: `storage/message/{sessionID}/{messageID}.json`

```json
{
  "id": "msg_b50425ff7001vgFiMUNFzLtCda",
  "sessionID": "ses_4afbda00cffeVl5YERm4op7JEG",
  "role": "user",
  "time": {
    "created": 1766578085884
  },
  "summary": {
    "title": "Exploring available tools",
    "diffs": []
  },
  "agent": "build",
  "model": {
    "providerID": "anthropic",
    "modelID": "claude-opus-4-5-20251101"
  }
}
```

**Fields**:
- `id`: Message identifier
- `sessionID`: Parent session
- `role`: "user" | "assistant"
- `time.created`: Unix timestamp (milliseconds)
- `summary.title`: Auto-generated message summary
- `summary.diffs`: File changes in this message
- `agent`: Agent name (e.g., "build", custom agent names)
- `model`: Model configuration (for assistant messages)
  - `providerID`: "anthropic", "openai", etc.
  - `modelID`: Full model identifier

### 6. Parts (Message Content)

**Part File**: `storage/part/{messageID}/{partID}.json`

Parts represent the actual content blocks within a message.

#### Text Part
```json
{
  "id": "prt_b50425ff7002egcP330jM5wQcU",
  "sessionID": "ses_4afbda00cffeVl5YERm4op7JEG",
  "messageID": "msg_b50425ff7001vgFiMUNFzLtCda",
  "type": "text",
  "text": "what tools do you have?"
}
```

#### Tool Use Part
```json
{
  "id": "prt_...",
  "sessionID": "ses_...",
  "messageID": "msg_...",
  "type": "tool_use",
  "name": "read_file",
  "input": {
    "path": "src/main.py"
  }
}
```

#### Tool Result Part
```json
{
  "id": "prt_...",
  "sessionID": "ses_...",
  "messageID": "msg_...",
  "type": "tool_result",
  "tool_use_id": "toolu_...",
  "content": "file contents here..."
}
```

**Part Types**:
- `text`: Text content
- `tool_use`: Tool invocation
- `tool_result`: Tool execution result
- `thinking`: Claude's thinking process (extended thinking)
- `image`: Image content (base64 or URL)

### 7. Message Flow

Unlike Claude Code's linked list, OpenCode doesn't store parent-child relationships explicitly in the storage layer. The conversation flow is determined by:

1. **Message order**: Files in `storage/message/{sessionID}/` directory
2. **Timestamp**: `time.created` field determines chronological order
3. **No parent references**: Must reconstruct flow from timestamps

To read a conversation:
```python
# 1. List all message files in session directory
messages = list_files(f"storage/message/{session_id}/")

# 2. Read each message
for msg_file in messages:
    msg = load_json(msg_file)
    parts = load_parts(f"storage/part/{msg['id']}/")
    # Combine message + parts
```

## Key Differences from Claude Code

| Aspect | OpenCode | Claude Code |
|--------|----------|-------------|
| **Format** | Normalized JSON files | Append-only JSONL |
| **Structure** | Relational (sessions → messages → parts) | Linear log with parent refs |
| **Message Flow** | Timestamp-based ordering | Explicit parent-child links |
| **Updates** | Files can be updated in place | Append-only, immutable |
| **Branches** | Not supported | Sidechains with `isSidechain` flag |
| **Projects** | SHA1 hash of directory | URL-encoded path |
| **IDs** | Custom prefixed format | UUIDs or short hex |
| **Content** | Separated into parts | Inline in message entry |

## Storage Provider Implementation

### Key Responsibilities

1. **Path Management**
   - Hash directory paths to project IDs
   - Organize sessions by project
   - Handle "global" project for unscoped sessions

2. **Message Reconstruction**
   - Load message metadata from `message/` directory
   - Load content parts from `part/` directory
   - Combine into unified message representation

3. **Conversation Queries**
   - List sessions for a project
   - Get message count and statistics
   - Retrieve messages in chronological order

4. **Format Conversion**
   - OpenCode format → `ChatMessage` (for agentpool)
   - `ChatMessage` → OpenCode format (for persistence)

### Challenges

1. **No Parent Links**
   - Cannot trace message ancestry efficiently
   - Must rely on timestamps for ordering
   - Forking/branching not supported

2. **Scattered Data**
   - Each message requires multiple file reads
   - Parts are in separate directories
   - No atomic transactions across files

3. **No Versioning**
   - Files can be updated in place
   - No history of edits
   - No way to detect concurrent modifications

## Data Consistency

### File Organization
- Messages grouped by session in directories
- Parts grouped by message in directories
- Sessions grouped by project in directories

### Atomic Operations
- Individual JSON file writes are atomic
- No atomicity across multiple files
- No transaction support

### Concurrent Access
- No locking mechanism
- Last write wins on conflicts
- Reading while writing may see partial state

## Integration Points

### With Agentpool
- Implements `StorageProvider` protocol
- Converts between OpenCode format and domain models
- Enables conversation persistence

### Missing Features
- **Todos/Plans**: No built-in todo tracking
- **File History**: Separate from message storage
- **Branching**: No conversation forking support
- **Ancestry**: No parent-child relationships

## Usage Patterns

### Reading a Session
```python
provider = OpenCodeStorageProvider()

# 1. Get session metadata
session_file = f"~/.local/share/opencode/storage/session/{project_id}/{session_id}.json"
session = load_json(session_file)

# 2. List messages in session
message_dir = f"~/.local/share/opencode/storage/message/{session_id}/"
message_files = list_files(message_dir)

# 3. Load each message + parts
messages = []
for msg_file in sorted(message_files):  # Sort by timestamp in ID
    msg = load_json(msg_file)
    
    # Load parts
    part_dir = f"~/.local/share/opencode/storage/part/{msg['id']}/"
    parts = [load_json(p) for p in list_files(part_dir)]
    
    messages.append(combine(msg, parts))
```

### Writing a Message
```python
# 1. Create message metadata
message = {
    "id": f"msg_{generate_id()}",
    "sessionID": session_id,
    "role": "user",
    "time": {"created": time_ms()},
    "summary": {"title": "...", "diffs": []},
    "agent": "default",
}
write_json(f"storage/message/{session_id}/{message['id']}.json", message)

# 2. Create parts
for part in content_parts:
    part_data = {
        "id": f"prt_{generate_id()}",
        "sessionID": session_id,
        "messageID": message['id'],
        "type": part['type'],
        **part['data']
    }
    write_json(f"storage/part/{message['id']}/{part_data['id']}.json", part_data)
```

## Design Rationale

### Why Normalized Structure?
- **Flexibility**: Can update individual components
- **Modularity**: Parts can be processed independently
- **Extensibility**: Easy to add new part types

### Why Separate Parts?
- **Streaming**: Can load message metadata without content
- **Lazy Loading**: Only load parts when needed
- **Type Safety**: Each part type has specific schema

### Why SHA1 for Project ID?
- **Deterministic**: Same path always gives same ID
- **Collision-resistant**: Very unlikely hash collisions
- **Path-independent**: ID doesn't reveal directory structure

### Drawbacks
- **Performance**: Many small files, lots of I/O
- **Consistency**: No atomic multi-file operations
- **Complexity**: More complex than append-only log
- **No History**: Can't track conversation evolution

## Future Considerations

### Performance
- Consider SQLite for better query performance
- Index sessions by project and timestamp
- Cache frequently accessed metadata

### Consistency
- Implement write-ahead logging
- Add transaction support
- Version individual files

### Features
- Add parent-child message links
- Support conversation branching
- Track message edit history
- Implement proper locking

---

**Related Files:**
- Implementation: [`provider.py`](./provider.py)
- Base Protocol: [`../base.py`](../base.py)
