---
title: Entry Points Toolset
description: Load tools from Python entry points
icon: material/import
---

# Entry Points Toolset

Load tools registered through Python entry points, enabling plugin-style tool discovery.

## Basic Usage

```yaml
agents:
  my_agent:
    toolsets:
      - type: entry_points
        group: my_package.tools
```

## How It Works

Python packages can register tools via entry points in their `pyproject.toml`:

```toml
[project.entry-points."my_package.tools"]
my_tool = "my_package.tools:my_tool_function"
```

The toolset then discovers and loads these tools automatically.

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.toolsets.EntryPointToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///
