---
title: Code Toolset
description: Code analysis and manipulation
icon: material/code-braces
---

# Code Toolset

Tools for code analysis and manipulation.

## Basic Usage

```yaml
agents:
  coder:
    toolsets:
      - type: code
```

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.toolsets.CodeToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///
