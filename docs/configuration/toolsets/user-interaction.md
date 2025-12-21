---
title: User Interaction Toolset
description: Interact with users
icon: material/account-voice
---

# User Interaction Toolset

Tools for agents to interact with users during execution.

## Basic Usage

```yaml
agents:
  my_agent:
    toolsets:
      - type: user_interaction
```

## Available Tools

| Tool | Description |
|------|-------------|
| `ask_user` | Ask the user a question |
| `show_message` | Display a message to the user |
| `confirm` | Request user confirmation |

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.UserInteractionToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///
