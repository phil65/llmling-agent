---
title: Skills Toolset
description: Load and execute skills
icon: material/lightning-bolt
---

# Skills Toolset

Load and execute skills - reusable prompt-based capabilities.

## Basic Usage

```yaml
agents:
  my_agent:
    toolsets:
      - type: skills
```

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.SkillsToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///
