---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/workers.py
title: Worker Configuration
description: Worker agent setup and management
icon: material/account-group
---

# Worker Configuration

Workers are agents that are registered as tools with a parent agent, allowing for hierarchical agent structures. They can be configured using either detailed configuration or shorthand syntax.

## Basic Configuration
Simple worker by name:
```yaml
agents:
  senior_dev:
    workers:
      - "code_reviewer"     # Simple reference to another node
      - "bug_analyzer"      # Each becomes available as a tool
```

## Detailed Configuration
Full worker configuration with all options:
```yaml
agents:
  senior_dev:
    workers:
      - type: agent
        name: "code_reviewer"
        reset_history_on_run: true    # Fresh conversation each time
        pass_message_history: false   # Don't share parent's history

      - type: agent
        name: "bug_analyzer"
        reset_history_on_run: false   # Maintain conversation between runs
        pass_message_history: true    # See parent's conversation
```

## Configuration Options

### `reset_history_on_run`
- `true` (default): Start fresh conversation for each invocation
- `false`: Maintain conversation context between runs

### `pass_message_history`
- `true`: Worker sees parent's conversation history
- `false` (default): Worker only sees current request


## Usage Examples

### Independent Workers
Workers that operate independently:
```yaml
agents:
  lead_dev:
    workers:
      - type: agent
        name: "linter"
        reset_history_on_run: true     # Fresh start each time
        pass_message_history: false    # Independent operation
```

### Context-Aware Workers
Workers that share context with parent:
```yaml
agents:
  architect:
    workers:
      - type: agent
        name: "code_reviewer"
        reset_history_on_run: false    # Remember previous reviews
        pass_message_history: true     # See full discussion
```

### Mixed Team
Different workers with different settings:
```yaml
agents:
  team_lead:
    workers:
      # Independent linting
      - type: agent
        name: "linter"
        reset_history_on_run: true

      # Contextual code review
      - type: agent
        name: "reviewer"
        reset_history_on_run: false
        pass_message_history: true

      # Simple reference to existing agent
      - "formatter"
```
