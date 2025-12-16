---
title: Workers Toolset
description: Manage worker agents
icon: material/account-hard-hat
---

# Workers Toolset

The Workers toolset provides tools for managing worker agents.

## Basic Usage

```yaml
agents:
  manager:
    toolsets:
      - type: workers
```

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.toolsets.WorkersToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///
