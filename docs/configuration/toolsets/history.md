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

## Use Cases

- Recall previous conversations
- Search for specific topics discussed
- Build context from past interactions

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.HistoryToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///
