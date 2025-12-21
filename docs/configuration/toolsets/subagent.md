---
title: Subagent Toolset
description: Delegate tasks to other agents
icon: material/account-group
---

# Subagent Toolset

The Subagent toolset enables agents to delegate tasks to other agents in the pool.

## Basic Usage

```yaml
agents:
  coordinator:
    toolsets:
      - type: subagent
```

## Available Tools

| Tool | Description |
|------|-------------|
| `list_available_nodes` | List agents available for delegation |
| `delegate_to` | Delegate a task to another agent |
| `ask_agent` | Ask another agent a question |

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.SubagentToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///

## Example Workflow

```yaml
agents:
  coordinator:
    toolsets:
      - type: subagent
    system_prompts:
      - You coordinate work between specialized agents.
  
  researcher:
    model: openai:gpt-4o
    system_prompts:
      - You research topics thoroughly.
  
  writer:
    model: openai:gpt-4o
    system_prompts:
      - You write clear documentation.
```

The coordinator can then delegate research to the researcher and writing to the writer.
