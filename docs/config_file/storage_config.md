---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/storage.py
---

# Storage Configuration

The storage configuration defines how agent interactions, messages, and tool usage are logged. It's defined at the root level of the manifest.

## Basic Structure

```yaml
storage:
  # List of storage providers
  providers:
    - type: "sql"  # SQL database (default)
      url: "sqlite:///history.db"
      pool_size: 5
      auto_migration: true
      agents: ["planner", "executor"]  # Only log these agents
      log_messages: true
      log_conversations: true
      log_commands: true
      log_context: true

  # Global agent filtering
  agents: ["planner", "executor", "analyzer"]  # Global agent filter
  filter_mode: "and"  # How to combine filters: "and" | "override"

  # Default provider for history queries
  default_provider: "sql"

  # Global logging flags (apply to all providers)
  log_messages: true      # Log all messages
  log_conversations: true # Log conversation metadata
  log_commands: true     # Log command executions
  log_context: true      # Log context additions
```

## Agent Filtering

You can filter which agents get logged at both global and provider levels:

```yaml
storage:
  # Global filter - affects all providers
  agents: ["planner", "executor", "analyzer"]

  # How filters are combined:
  filter_mode: "and"     # Both filters must allow the agent
  # filter_mode: "override" # Provider filter overrides global if set

  providers:
    - type: "sql"
      url: "sqlite:///history.db"
      agents: ["planner", "executor"]  # Provider-specific filter
```

With `filter_mode: "and"`:

- An agent must be allowed by both global AND provider filters
- If either filter is None, only the other filter applies
- If both are None, all agents are logged

With `filter_mode: "override"`:

- Provider filter takes precedence if set
- Falls back to global filter if provider filter is None
- If both are None, all agents are logged

## Available Providers

### SQL Storage (Default)
Uses SQLModel for database storage. Supports SQLite, PostgreSQL, etc.

```yaml
storage:
  providers:
    - type: "sql"
      url: "sqlite:///history.db"  # Database URL
      pool_size: 5  # Connection pool size
      auto_migration: true  # Automatically add missing columns
      agents: ["planner", "executor"]  # Agent filter
      # Logging flags (override global)
      log_messages: true
      log_conversations: true
      log_commands: true
      log_context: true
```

### Text Log Templates

The text log provider supports customizable Jinja2 templates. You can either use predefined formats or provide your own template file.

```yaml
storage:
  providers:
    - type: "text_file"
      path: "logs/chat.log"
      # Use predefined template:
      format: "chronological"  # or "conversations"
      # Or use custom template:
      template: "templates/custom.j2"
```

#### Available Templates

Two predefined formats are available:

1. `chronological`: All events in chronological order
2. `conversations`: Grouped by conversation with commands at the end

#### Custom Templates

You can create custom templates with access to these variables:

##### Entry Types and Fields
Each entry has a `type` field and type-specific data:

```python
# Message entry
{
    "type": "message",
    "timestamp": datetime,
    "conversation_id": str,
    "content": str,
    "role": str,
    "name": str | None,
    "model": str | None,
    "cost_info": {
        "token_usage": {
            "total": int,
            "prompt": int,
            "completion": int
        },
        "total_cost": float
    },
    "response_time": float | None,
    "forwarded_from": list[str] | None
}

# Conversation start entry
{
    "type": "conversation_start",
    "timestamp": datetime,
    "conversation_id": str,
    "agent_name": str
}

# Tool call entry
{
    "type": "tool_call",
    "timestamp": datetime,
    "conversation_id": str,
    "message_id": str,
    "tool_name": str,
    "args": dict[str, Any],
    "result": Any
}

# Command entry
{
    "type": "command",
    "timestamp": datetime,
    "agent_name": str,
    "session_id": str,
    "command": str,
    "context_type": str | None,
    "metadata": dict[str, Any] | None
}
```
Refer to the source for the details, an in-depth explanation will follow.

### File Storage
Stores data in structured files (JSON, YAML, etc.).

```yaml
storage:
  providers:
    - type: "file"
      path: "data/history.json"  # Storage file path
      format: "auto"  # "auto" | "json" | "yaml" | "toml" | "ini"
      encoding: "utf-8"  # File encoding
      agents: ["planner"]  # Agent filter
      # Logging flags
      log_messages: true
      log_conversations: true
      log_commands: true
      log_context: true
```

### Memory Storage
In-memory storage for testing.

```yaml
storage:
  providers:
    - type: "memory"  # No additional configuration needed
      agents: null  # Log all agents
      # Logging flags
      log_messages: true
      log_conversations: true
      log_commands: true
      log_context: true
```

## Multiple Providers

You can use multiple providers simultaneously, each with their own filtering:

```yaml
storage:
  # Global settings
  agents: ["planner", "executor", "analyzer"]
  filter_mode: "and"
  log_messages: true
  log_conversations: true
  log_commands: true
  log_context: true
  default_provider: "sql"

  providers:
    - type: "sql"  # Primary storage in database
      url: "sqlite:///history.db"
      agents: ["planner", "executor"]  # Only execution pipeline

    - type: "text_file"  # Analysis logs
      path: "logs/analysis.log"
      agents: ["analyzer"]  # Only analysis pipeline
      format: "chronological"

    - type: "file"  # Complete history
      path: "data/history.json"
      agents: null  # Log everything

```

## Provider Selection

When loading history, providers are selected in this order:

1. Explicitly specified preferred provider
2. Default provider if configured in `default_provider`
3. First capable provider in the list
4. Raises error if no capable provider found

## Logging Flags

All logging flags can be set both globally and per provider:

- `log_messages`: Log all messages exchanged
- `log_conversations`: Log conversation metadata
- `log_commands`: Log command executions
- `log_context`: Log context additions and changes

Provider flags are combined with global flags using AND logic:

```yaml
storage:
  log_messages: true     # Global setting
  providers:
    - type: "sql"
      log_messages: true # Must also be true here to log messages
```

## Notes

- Individual provider settings are combined with global settings
- SQL provider is recommended for production use
- Memory provider is useful for testing
- Text logs are good for debugging and monitoring
- File storage is suitable for simple deployments
- Agent filtering allows splitting logs across different providers
