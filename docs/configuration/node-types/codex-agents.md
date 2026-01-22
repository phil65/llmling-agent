---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/agentpool/models/codex_agents.py
title: Codex Agents
description: Codex app-server integration
icon: material/robot
---

Codex agents provide AgentPool's integration with the Codex app-server, enabling advanced code editing, terminal access, and tool execution through Codex's JSON-RPC protocol.

## Overview

AgentPool integrates directly with Codex app-server via its JSON-RPC protocol, providing:

- **Native Codex features**: Full access to Codex's reasoning, tools, and model capabilities
- **Low latency**: Direct process communication via stdin/stdout
- **Toolset bridging**: Expose AgentPool's internal toolsets to Codex via MCP
- **Streaming events**: Real-time event propagation from Codex to AgentPool
- **Structured outputs**: Support for typed response schemas via JSON Schema

Codex agents are ideal for:

- Complex coding tasks with extended reasoning
- Terminal operations and shell command execution
- Multi-file refactoring and code generation
- Tasks benefiting from Codex's reasoning effort levels

## Configuration Reference

/// mknodes
{{ "agentpool.models.codex_agents.CodexAgentConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", as_listitem=False) }}
///

## Examples

### Basic Configuration

A minimal Codex agent for coding tasks:

```yaml
agents:
  codex_coder:
    type: codex
    cwd: /path/to/project
    model: gpt-5.1-codex-max
    reasoning_effort: medium
```

### Reasoning Effort Levels

Control how deeply Codex reasons about tasks:

```yaml
agents:
  quick_helper:
    type: codex
    model: gpt-5.1-codex-max
    reasoning_effort: low  # Fast responses

  deep_thinker:
    type: codex
    model: gpt-5.1-codex-max
    reasoning_effort: high  # Complex problem solving
```

### Tool Approval Policies

Configure how Codex handles tool execution approval:

```yaml
agents:
  auto_executor:
    type: codex
    approval_policy: never  # Execute tools without approval (default)

  safe_agent:
    type: codex
    approval_policy: auto  # Auto-approve safe tools, ask for risky ones

  interactive:
    type: codex
    approval_policy: always  # Always request approval
```

### Toolset Integration

Expose AgentPool's internal toolsets to Codex via MCP:

```yaml
agents:
  codex_coordinator:
    type: codex
    cwd: /path/to/project
    tools:
      # Delegate to specialized agents
      - type: subagent
        agents:
          - researcher
          - writer

      # File operations
      - type: file_access

      # Bash commands
      - type: bash

      # Custom tools
      - my_module:my_tool
```

This allows Codex to use AgentPool's toolsets alongside its native capabilities.

### Structured Output

Define typed responses using JSON Schema:

```yaml
responses:
  bug_report:
    type: object
    properties:
      title: { type: string }
      severity: { type: string, enum: [critical, high, medium, low] }
      steps_to_reproduce:
        type: array
        items: { type: string }
      affected_files:
        type: array
        items: { type: string }
      suggested_fix: { type: string }

agents:
  bug_finder:
    type: codex
    output_type: bug_report
```

Then use structured output programmatically:

```python
from agentpool import AgentPool
from pydantic import BaseModel

class BugReport(BaseModel):
    title: str
    severity: str
    steps_to_reproduce: list[str]
    affected_files: list[str]
    suggested_fix: str

async with AgentPool("config.yml") as pool:
    agent = pool.get_agents()["bug_finder"]

    # Configure for structured output
    agent.to_structured(BugReport)

    result = await agent.run("Analyze the authentication code for bugs")
    # result is now a BugReport instance
    print(f"Found {result.title} with severity {result.severity}")
```

### External MCP Servers

Connect Codex to external MCP servers:

```yaml
mcp_servers:
  filesystem:
    transport: stdio
    command: mcp-server-filesystem
    args: ["/path/to/workspace"]

agents:
  codex_with_fs:
    type: codex
    mcp_servers:
      - filesystem
```

## Reasoning Effort Modes

Codex supports three reasoning effort levels that can be changed at runtime:

| Mode | Description | Use Case |
|------|-------------|----------|
| `low` | Fast, minimal reasoning | Quick edits, simple tasks |
| `medium` | Balanced reasoning | General purpose (default) |
| `high` | Deep reasoning | Complex problem solving, architecture |

Example runtime mode switching:

```python
from agentpool import AgentPool

async with AgentPool("config.yml") as pool:
    agent = pool.get_agents()["my_codex"]

    # Switch to high reasoning for complex task
    await agent.set_mode("high", category_id="reasoning_effort")
    result = await agent.run("Design a scalable authentication system")

    # Switch back to medium for normal tasks
    await agent.set_mode("medium", category_id="reasoning_effort")
```

## Approval Policy Modes

Control when Codex requests permission to execute tools:

| Policy | Description |
|--------|-------------|
| `never` | Execute tools without approval (default for programmatic use) |
| `auto` | Auto-approve low-risk tools, request approval for high-risk |
| `always` | Always request approval before executing any tool |

Change approval policy at runtime:

```python
await agent.set_mode("always", category_id="approval_policy")
```

## Model Selection

Codex provides access to various models. The agent can dynamically fetch available models:

```python
from agentpool import AgentPool

async with AgentPool("config.yml") as pool:
    agent = pool.get_agents()["my_codex"]

    # Get available models
    models = await agent.get_available_models()
    for model in models:
        print(f"{model.name}: {model.description}")

    # Switch model
    await agent.set_model("gpt-5.1-codex-mini")
```

Common Codex models:

- `gpt-5.1-codex-max` - Most capable, best for complex tasks
- `gpt-5.1-codex-mini` - Faster, cost-effective
- `gpt-5-codex` - Previous generation
- `claude-opus-4` - Anthropic Claude integration

## Thread Management

Codex uses threads to maintain conversation context. Some settings (model, reasoning_effort) require creating a new thread:

```python
# These require thread restart (archive old, create new):
await agent.set_model("new-model")
await agent.set_mode("high", category_id="reasoning_effort")

# This doesn't require restart:
await agent.set_mode("always", category_id="approval_policy")
```

## Comparison with Claude Code Agents

| Feature | Codex Agent | Claude Code Agent |
|---------|-------------|-------------------|
| Protocol | Codex JSON-RPC | Claude Agent SDK |
| Reasoning control | 3 effort levels | Extended thinking tokens |
| Tool approval | 3 policies (never/auto/always) | 4 permission modes |
| Models | Codex models + Claude | Claude models only |
| Thread management | Archive + restart | Session reconnect |
| Structured output | Via JSON Schema | Via result tools |

Choose Codex agents for:

- Access to Codex-specific models and reasoning
- Projects already using Codex
- Tasks benefiting from Codex's reasoning effort system

Choose Claude Code agents for:

- Pure Claude model access
- Extended thinking capabilities
- Tighter Claude ecosystem integration

## Related Configuration

- [Claude Code Agents](claude-code-agents.md) - Claude Agent SDK integration
- [ACP Agents](acp-agents.md) - ACP protocol integration
- [Toolsets](../toolsets/index.md) - Tool collections for agents
- [Responses](responses.md) - Structured output types
