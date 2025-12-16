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

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.toolsets.IntegrationToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///
