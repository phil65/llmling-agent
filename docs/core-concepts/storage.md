---
title: Storage System
description: Storage and persistence system
icon: material/database
---

# Storage System

The storage system in LLMling Agent provides flexible logging and history management through multiple storage providers. It allows tracking conversations, tool usage, and commands while supporting different storage backends and filtering options.

## Overview

The storage system serves multiple purposes:

- Recording conversation history for context and analysis
- Logging tool usage and command execution
- Enabling conversation recovery and continuation
- Supporting debugging and monitoring
- Allowing to pretty-print and output conversations in a customizable way

Multiple providers can be active simultaneously, each handling different aspects of storage:

- Some providers focus on full history tracking
- Others specialize in output logging or monitoring
- Providers can be filtered to handle specific agents

Settings can be configured at two levels:

- Global settings affecting all providers
- Provider-specific settings for fine-grained control

## Storage Providers

### SQL Storage

The SQL provider offers full history tracking with database persistence:

```yaml
storage:
  providers:
    - type: sql
      url: sqlite:///history.db  # Default location
      pool_size: 5              # Connection pool size
      auto_migration: true      # Add missing columns automatically
```

Features:

- Complete conversation history
- Efficient querying and filtering
- Support for SQLite and other SQL databases
- Automatic schema management
- Default provider if none specified

### File Storage

File-based storage using YAML or JSON formats:

```yaml
storage:
  providers:
    - type: file
      path: history.yml        # Or .json
      format: auto            # Or yaml/json
      encoding: utf-8
```

Features:

- Human-readable storage format
- Easy version control integration
- Flexible file organization
- Support for multiple file formats

### Text Log Provider

Simple text-based logging with customizable formats:

```yaml
storage:
  providers:
    - type: text_file
      path: agent_logs.txt
      format: chronological     # Or conversations
      template: custom_template.txt  # Optional
```

Features:

- Simple text file output
- Customizable log formats using jinja2-templating
- Streaming-friendly
- Output-only (no history queries)

### Memory Storage

In-memory storage for testing and development:

```yaml
storage:
  providers:
    - type: memory
```

Provides temporary storage without persistence, ideal for testing and development.

## Provider Capabilities

History Providers (SQL, File):

- Full conversation history
- Message querying and filtering
- Session recovery
- Statistics and analytics

Output Providers (Text Log):

- Message logging
- Format customization
- Real-time monitoring
- No query capabilities

Storable Content:

- Messages (user/assistant/system)
- Conversation metadata
- Tool call details
- Command execution logs
- Context additions

## Configuration

Global Settings:

```yaml
storage:
  # Global filters
  agents: [agent1, agent2]     # Only log these agents
  log_messages: true           # Enable message logging
  log_conversations: true      # Enable conversation tracking

  log_commands: true          # Enable command logging
  log_context: true           # Enable context logging

  # Provider handling
  filter_mode: and            # How to combine filters (and/override)
  default_provider: sql       # Provider for history queries
```

Provider Filtering:

- Global agent filters
- Provider-specific filters
- Combination modes:
  - `and`: Both global and provider filters must match
  - `override`: Provider filter takes precedence

Multiple Providers Example:

```yaml
storage:
  providers:
    - type: sql               # Main history storage
      url: sqlite:///history.db

    - type: text_file         # Additional logging
      path: logs/output.txt
      agents: [assistant]     # Only log specific agent
```

## Usage Examples

Basic Setup:

```yaml
storage:
  providers:
    - type: sql              # Single provider
```

Multiple Providers:

```yaml
storage:
  providers:
    - type: sql             # Main storage
      url: postgresql://...

    - type: text_file      # Debug logging
      path: debug.log

    - type: file           # Backup
      path: backup.yml
```

History Queries:

```python
# Get recent messages
messages = await storage.filter_messages(
    SessionQuery(
        name="session_1",
        since="1h",
        roles={"assistant"}
    )
)

# Get statistics
stats = await storage.get_conversation_stats(
    filters=StatsFilters(
        cutoff=datetime.now(UTC) - timedelta(days=1),
        group_by="agent"
    )
)
```

Custom Filtering:

```yaml
storage:
  # Global filter
  agents: [assistant, researcher]

  providers:
    - type: sql
      agents: [assistant]     # Only assistant

    - type: text_file
      agents: [researcher]    # Only researcher
```
