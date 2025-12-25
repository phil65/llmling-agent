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

## Available Tools

```python exec="true"
from agentpool_toolsets.builtin.tool_management import ToolManagementTools
from agentpool.docs.utils import generate_tool_docs

toolset = ToolManagementTools()
print(generate_tool_docs(toolset))
```

## Use Cases

- Temporarily disable destructive tools
- Enable tools only when needed
- Implement tool-based permissions

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.ToolManagementToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets", header_level=3) }}
///
