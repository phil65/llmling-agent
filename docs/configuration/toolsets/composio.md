---
title: Composio Toolset
description: Integration with Composio tool platform
icon: material/puzzle
---

# Composio Toolset

Integration with the [Composio](https://composio.dev) platform for accessing pre-built tool integrations.

## Basic Usage

```yaml
agents:
  my_agent:
    toolsets:
      - type: composio
        apps:
          - github
          - slack
```

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.ComposioToolSetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///

## Available Apps

Composio provides integrations with many services. Check the [Composio documentation](https://docs.composio.dev) for the full list.
