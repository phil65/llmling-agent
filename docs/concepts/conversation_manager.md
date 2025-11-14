# Conversation Manager

## Overview

The ConversationManager handles message history and context for agents. It provides:

- Message history storage and retrieval
- Conversation context management
- Session recovery
- Token counting and limiting
- Message filtering

## Core Functionality

### Managing History
```python
# Access conversation manager
conversation = agent.conversation

# Get message history
messages = conversation.get_history()

# Get specific messages
recent = conversation[-5:]  # Last 5 messages
agent_msgs = conversation["other_agent"]  # Messages from specific agent

# Clear history
conversation.clear()

# Set history
conversation.set_history(new_messages)
```

### Context Management
```python
# Add context
conversation.add_context_message(
    content="Important background info",
    source="documentation",
    metadata={"type": "background"}
)

# Load context from sources
await conversation.load_context_source("docs/api.md")
await conversation.add_context_from_prompt(system_prompt)

# Format history for context
history_text = await conversation.format_history(
    max_tokens=1000,
    include_system=False,
    num_messages=5
)
```

### Token Management
```python
# Get token counts
total = conversation.get_history_tokens()

# Format with token limit
context = await conversation.format_history(
    max_tokens=2000,
    num_messages=None  # All messages within token limit
)
```

## Session Management

Sessions allow conversation recovery and continuation:

```python
# Create agent with session
agent = pool.get_agent(
    "assistant",
    session="previous_chat"  # Session ID
)

# Or with query
agent = pool.get_agent(
    "assistant",
    session=SessionQuery(
        name="previous_chat",
        since="1h",
        roles={"assistant", "user"}
    )
)
```

## YAML Configuration

Session configuration is part of the agent definition:

```yaml
agents:
  my_agent:
    model: openai:gpt-5
    description: "Support assistant"

    # Session configuration
    session:
      name: support_chat        # Session identifier
      since: 1h                # Time period to load
      until: 5m                # Up to this time ago
      agents: [support, user]   # Only these agents
      roles: [user, assistant] # Only these roles
      contains: "error"        # Text search
      limit: 50                # Max messages
      include_forwarded: true  # Include forwarded messages
```

You can also provide just the session ID:
```yaml
agents:
  my_agent:
    session: previous_chat  # Simple form with just ID
```

When using `Agent.__init__()`:
```python
# With session query
async with Agent(
    ...,
    session=SessionQuery(
        name="support_chat",
        since="1h"
    )
) as agent:
    ...

# With simple session ID
async with Agent(
    ...,
    session="previous_chat"
) as agent:
    ...
```

## Loading History

### Time-Based Loading
```python
# Load recent history
conversation.load_history_from_database(
    since="1h",    # Last hour
    until="5m",    # Up to 5 minutes ago
    limit=50       # Max 50 messages
)

# Specific timeframe
conversation.load_history_from_database(
    since=datetime(2023, 1, 1),
    until=datetime(2023, 1, 2)
)
```

### Filtered Loading
```python
# Filter by content
history = conversation.filter_messages(
    SessionQuery(
        contains="error",
        roles={"user", "assistant"}
    )
)

# Filter by agents
history = conversation.filter_messages(
    SessionQuery(
        agents={"support_bot", "user"},
        limit=10
    )
)
```

## Events

The ConversationManager emits events for history changes:

```python
# History cleared event
@conversation.history_cleared.connect
def on_clear(event: ConversationManager.HistoryCleared):
    print(f"History cleared for session {event.session_id}")
```
