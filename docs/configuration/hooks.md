---
title: Hooks
description: Lifecycle hooks for intercepting agent behavior
icon: material/hook
---

Hooks allow you to intercept and customize agent behavior at key lifecycle points. They can add context, block operations, modify inputs, or trigger side effects.

## Overview

| Event | Trigger | Can Block | Can Modify |
|-------|---------|-----------|------------|
| `pre_run` | Before `agent.run()` processes a prompt | Yes | No |
| `post_run` | After `agent.run()` completes | No | No |
| `pre_tool_use` | Before a tool is called | Yes | Yes (tool input) |
| `post_tool_use` | After a tool completes | No | No |

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.hooks.HooksConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Hook Types

/// mknodes
{{ "llmling_agent_config.hooks.HookConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Command Hook

Executes a shell command. Receives JSON via stdin, returns JSON via stdout.

```yaml
agents:
  my_agent:
    model: openai:gpt-4o
    hooks:
      pre_tool_use:
        - type: command
          command: python /path/to/validate.py
          matcher: "Bash|Write"
          timeout: 30.0
          env:
            LOG_LEVEL: debug
```

**Exit codes:**

- `0` = allow (stdout parsed as JSON)
- `2` = deny (stderr used as reason)
- Other = non-blocking error, logged but continues

**Example script:**

```python
#!/usr/bin/env python3
import json
import sys

data = json.load(sys.stdin)
tool_input = data.get("tool_input", {})

if "rm -rf" in tool_input.get("command", ""):
    print("Dangerous command blocked", file=sys.stderr)
    sys.exit(2)  # Deny

print(json.dumps({"decision": "allow"}))
sys.exit(0)
```

## Callable Hook

Executes a Python function by import path.

```yaml
agents:
  my_agent:
    model: openai:gpt-4o
    hooks:
      post_tool_use:
        - type: callable
          import_path: myproject.hooks.log_tool_usage
          arguments:
            log_file: /var/log/tools.log
```

**Function signature:**

```python
# myproject/hooks.py
from typing import Any

def log_tool_usage(
    tool_name: str,
    tool_output: Any,
    duration_ms: float,
    log_file: str,
    **kwargs
) -> dict | None:
    with open(log_file, "a") as f:
        f.write(f"{tool_name}: {duration_ms:.1f}ms\n")
    return {"decision": "allow"}
```

## Prompt Hook

Uses an LLM to evaluate the action with structured output.

```yaml
agents:
  my_agent:
    model: openai:gpt-4o
    hooks:
      pre_tool_use:
        - type: prompt
          prompt: |
            Should this command be allowed?
            Tool: $TOOL_NAME
            Input: $TOOL_INPUT
          model: openai:gpt-4o-mini
          matcher: "Bash"
```

**Placeholders:**

| Placeholder | Description |
|-------------|-------------|
| `$INPUT` | Full hook input as JSON |
| `$TOOL_NAME` | Name of the tool |
| `$TOOL_INPUT` | Tool input arguments as JSON |
| `$TOOL_OUTPUT` | Tool output (post_tool_use only) |
| `$AGENT_NAME` | Name of the agent |
| `$PROMPT` | The prompt being processed (pre_run/post_run) |
| `$EVENT` | The hook event name |

## Matcher Patterns

The `matcher` field uses regex to filter which tools trigger the hook:

```yaml
matcher: "Bash"           # Only Bash tool
matcher: "Write|Edit"     # Write or Edit tools
matcher: "mcp__.*"        # All MCP tools
matcher: ".*"             # All tools (same as null)
```

## Full Example

```yaml
agents:
  secure_agent:
    model: openai:gpt-4o
    hooks:
      pre_run:
        - type: callable
          import_path: myproject.auth.check_allowed

      pre_tool_use:
        - type: prompt
          prompt: "Is this Bash command safe? $TOOL_INPUT"
          matcher: "Bash"
          model: openai:gpt-4o-mini

        - type: command
          command: python -m myproject.hooks.validate_paths
          matcher: "Write|Edit|Read"

      post_tool_use:
        - type: callable
          import_path: myproject.metrics.record_usage

      post_run:
        - type: command
          command: /scripts/notify-complete.sh
```

## Programmatic Usage

```python
from llmling_agent import Agent
from llmling_agent.hooks import AgentHooks, CallableHook

def my_pre_tool_hook(tool_name: str, tool_input: dict, **kwargs):
    if tool_name == "Bash" and "rm" in tool_input.get("command", ""):
        return {"decision": "deny", "reason": "rm commands not allowed"}
    return {"decision": "allow"}

hooks = AgentHooks(
    pre_tool_use=[
        CallableHook(
            event="pre_tool_use",
            fn=my_pre_tool_hook,
            matcher="Bash",
        )
    ]
)

agent = Agent(model="openai:gpt-4o", hooks=hooks)
```
