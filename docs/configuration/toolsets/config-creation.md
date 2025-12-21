---
title: Config Creation Toolset
description: Create agent configurations
icon: material/cog-play
---

# Config Creation Toolset

Tools for creating and managing agent configurations programmatically.

## Basic Usage

```yaml
agents:
  admin:
    toolsets:
      - type: config_creation
```

## Use Cases

- Generate agent configs from templates
- Create configurations dynamically
- Manage configuration files

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.ConfigCreationToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///
