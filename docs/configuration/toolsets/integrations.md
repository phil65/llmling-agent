---
title: Integration Toolset
description: Integration utilities
icon: material/connection
---

# Integration Toolset

Utilities for integrating with external systems.

## Basic Usage

```yaml
agents:
  my_agent:
    toolsets:
      - type: integration
```

## Available Tools

```python exec="true"
from agentpool_toolsets.builtin.integration import IntegrationTools
from agentpool.docs.utils import generate_tool_docs

toolset = IntegrationTools()
print(generate_tool_docs(toolset))
```

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.IntegrationToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets", header_level=3) }}
///
