---
title: Message history
description: Message history and management
icon: material/history
---

## Overview

The MessageHistory handles message history and context for agents. It provides:

- Message history storage and retrieval
- Conversation context management
- Session recovery
- Token counting and limiting
- Message filtering

## Core Functionality

### Managing History

```python
# Access Message history
history = agent.conversation
# Get message history
messages = history.get_history()
# Get specific messages
recent = history[-5:]  # Last 5 messages
agent_msgs = history["other_agent"]  # Messages from specific agent
# Clear history
history.clear()
# Set history
history.set_history(new_messages)
```

### Context Management

```python
# Add context
history.add_context_message(
    content="Important background info",
    source="documentation",
    metadata={"type": "background"}
)

# Load context from sources
await history.load_context_source("docs/api.md")
await history.add_context_from_prompt(system_prompt)

# Format history for context
history_text = await history.format_history(
    max_tokens=1000,
    include_system=False,
    num_messages=5
)
```

### Token Management

```python
# Get token counts
total = history.get_history_tokens()
# Format with token limit
context = await history.format_history(
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
    session=SessionQuery(name="support_chat", since="1h")
) as agent:
    ...

# With simple session ID
async with Agent(..., session="previous_chat") as agent:
    ...
```
