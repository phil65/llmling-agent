---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/session.py
---

# Session Configuration

Session configuration allows loading previous conversations and managing agent state. Sessions can be identified by name or configured using detailed query parameters.

## Basic Configuration
Simple session by name:
```yaml
agents:
  assistant:
    session: "previous_chat"  # Simple session ID
```

You can also configure memory settings:
```yaml
agents:
  assistant:
    session:
      enable: true            # Enable history tracking
      max_tokens: 4000        # Maximum tokens to keep in context
      max_messages: 50        # Maximum messages to keep in context
      provider: "sql"         # Override default storage provider
      session:                # Previous session query
        name: "previous_chat"
```

## Detailed Query Configuration
Complex filtering of previous conversations:
```yaml
agents:
  assistant:
    session:
      name: "coding_session"           # Session identifier
      agents: ["analyzer", "planner"]  # Filter by agent names
      since: "2h"                      # Time periods: 1h, 2d, 1w, etc.
      until: "1h"                      # Up until time period
      contains: "python"               # Filter by content
      roles: ["user", "assistant"]     # Filter by message roles
      limit: 100                       # Max messages to load
      include_forwarded: true          # Include forwarded messages
```

## Time Period Examples
The `since` and `until` fields support various formats:
```yaml
agents:
  assistant:
    session:
      # Supported formats
      since: "1h"    # 1 hour ago
      since: "2d"    # 2 days ago
      since: "1w"    # 1 week ago
      since: "30m"   # 30 minutes ago
      since: "1.5h"  # 1.5 hours ago
```

## Role Filtering
Filter messages by their roles:
```yaml
agents:
  assistant:
    session:
      roles:
        - "user"       # User messages
        - "assistant"  # Assistant responses
        - "system"     # System messages
```

## Usage Examples

### Continue Previous Chat
```yaml
agents:
  assistant:
    session: "last_chat"  # Simple continuation
```

### Load Recent History
```yaml
agents:
  assistant:
    session:
      since: "24h"    # Last 24 hours
      limit: 50       # Max 50 messages
```

### Load Specific Topic
```yaml
agents:
  assistant:
    session:
      contains: "project X"  # Messages about Project X
      agents: ["planner"]    # From planner agent
      since: "1w"           # From last week
```

### Team History
```yaml
agents:
  coordinator:
    session:
      agents: ["researcher", "writer", "reviewer"]
      roles: ["assistant"]  # Only assistant messages
      since: "1d"          # From last day
```
