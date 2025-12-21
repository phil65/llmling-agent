---
title: Agent Management Toolset
description: Create and manage agents dynamically
icon: material/account-cog
---

# Agent Management Toolset

The Agent Management toolset allows agents to dynamically create and manage other agents at runtime.

## Basic Usage

```yaml
agents:
  orchestrator:
    toolsets:
      - type: agent_management
```

## Available Tools

| Tool | Description |
|------|-------------|
| `create_worker_agent` | Create a new worker agent |
| `add_agent` | Add an agent to the pool |
| `add_team` | Create a new team |
| `connect_nodes` | Connect agents in the pool |

## Tool Selection

Limit exposed tools:

```yaml
toolsets:
  - type: agent_management
    tools:
      - create_worker_agent
      - add_agent
```

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.AgentManagementToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///

## Use Cases

- **Dynamic scaling**: Create agents based on workload
- **Specialized workers**: Spawn task-specific agents
- **Orchestration**: Build agent networks at runtime
