---
title: Tool Management Toolset
description: Enable and disable tools at runtime
icon: material/wrench-cog
---

# Tool Management Toolset

Allows agents to dynamically enable or disable tools during execution.

## Basic Usage

```yaml
agents:
  my_agent:
    toolsets:
      - type: tool_management
```

## Use Cases

- Temporarily disable destructive tools
- Enable tools only when needed
- Implement tool-based permissions

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.toolsets.ToolManagementToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///
