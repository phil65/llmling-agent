---
title: History Toolset
description: Access conversation history
icon: material/history
---

# History Toolset

Access and search conversation history.

## Basic Usage

```yaml
agents:
  my_agent:
    toolsets:
      - type: history
```

## Available Tools

```python exec="true"
from agentpool_toolsets.builtin.history import HistoryTools
from agentpool.docs.utils import generate_tool_docs

toolset = HistoryTools()
print(generate_tool_docs(toolset))
```

## Use Cases

- Recall previous conversations
- Search for specific topics discussed
- Build context from past interactions

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.HistoryToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets", header_level=3) }}
///
