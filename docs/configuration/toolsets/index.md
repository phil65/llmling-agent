---
title: Toolsets
description: Toolset configuration overview
icon: material/toolbox
---

# Toolsets

Toolsets are collections of tools that can be dynamically loaded and assigned to agents. They provide a modular way to extend agent capabilities.

## Overview

Toolsets are configured in your manifest under the agent's `toolsets` field:

```yaml
agents:
  my_agent:
    toolsets:
      - type: file_access
        fs: "file:///workspace"
      - type: search
        provider: tavily
```

## Available Toolsets

### Filesystem & Resources

| Toolset | Description |
|---------|-------------|
| [File Access](./file-access.md) | Read, write, edit files on any fsspec-compatible filesystem |
| [VFS](./vfs.md) | Access resources defined in manifest's `resources` section |

### External Integrations

| Toolset | Description |
|---------|-------------|
| [OpenAPI](./openapi.md) | Generate tools from OpenAPI/Swagger specifications |
| [Entry Points](./entry-points.md) | Load tools from Python entry points |
| [Composio](./composio.md) | Integration with Composio tool platform |
| [Search](./search.md) | Web and news search capabilities |
| [Notifications](./notifications.md) | Send notifications via various channels |

### Agent & Workflow

| Toolset | Description |
|---------|-------------|
| [Agent Management](./agent-management.md) | Create and manage agents dynamically |
| [Subagent](./subagent.md) | Delegate tasks to other agents |
| [Workers](./workers.md) | Manage worker agents |

### Code & Execution

| Toolset | Description |
|---------|-------------|
| [Execution](./execution.md) | Execute code and commands |
| [Code](./code.md) | Code analysis and manipulation tools |
| [Code Mode](./code-mode.md) | Wrap toolsets for code-based interaction |
| [Remote Code Mode](./remote-code-mode.md) | Remote code-based interaction |

### Memory & History

| Toolset | Description |
|---------|-------------|
| [History](./history.md) | Access conversation history |
| [Semantic Memory](./semantic-memory.md) | Vector-based semantic memory |

### Utility

| Toolset | Description |
|---------|-------------|
| [Tool Management](./tool-management.md) | Enable/disable tools at runtime |
| [User Interaction](./user-interaction.md) | Interact with users |
| [Skills](./skills.md) | Load and execute skills |
| [Integrations](./integrations.md) | Integration utilities |
| [Config Creation](./config-creation.md) | Create agent configurations |
| [Custom](./custom.md) | Load custom toolset implementations |

## Common Configuration

All toolsets share these base options:

```yaml
toolsets:
  - type: <toolset_type>
    namespace: optional_prefix  # Prefix for tool names
```

The `namespace` field helps prevent name collisions when using multiple toolsets.
