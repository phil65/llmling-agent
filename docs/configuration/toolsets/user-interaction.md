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

```python exec="true"
from agentpool_toolsets.builtin import UserInteractionTools
from agentpool.docs.utils import generate_tool_docs

toolset = UserInteractionTools()
print(generate_tool_docs(toolset))
```

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.UserInteractionToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets", header_level=3) }}
///
