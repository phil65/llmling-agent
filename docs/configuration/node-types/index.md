---
title: Node Types
description: Agent and team node configuration
icon: material/family-tree
order: 0
---

# Node Types

Nodes are the core building blocks of AgentPool. This section covers the different types of nodes you can configure: agents (individual AI workers) and teams (coordinated groups).

## Agents

### Standard Agents

[Standard agents](agent.md) are the primary node type in AgentPool. They are fully configured within your manifest and support:

- Model selection and configuration
- System prompts and knowledge sources
- Tools and toolsets
- Workers for task delegation
- Event handlers and connections

### ACP Agents

[ACP (Agent Communication Protocol) agents](acp-agents.md) integrate external coding agents like Claude Code, Gemini CLI, Codex, and many others. These run as separate processes and communicate via the ACP protocol.

### AG-UI Agents

[AG-UI agents](agui-agents.md) connect to remote HTTP endpoints implementing the AG-UI protocol, enabling integration with any AG-UI compatible server.

## Teams

[Teams](./team.md) coordinate multiple agents to work together on complex tasks:

- **Parallel teams**: Members execute simultaneously
- **Sequential teams**: Members execute in sequence, passing results along
- **Nested teams**: Teams can contain other teams for complex workflows

## Quick Comparison

| Feature | Standard Agent | ACP Agent | AG-UI Agent | Team |
|---------|---------------|-----------|-------------|------|
| Runs in-process | Yes | No | No | Yes |
| Full configuration | Yes | Limited | Limited | Yes |
| External tools | Via toolsets | Built-in | Via server | Via members |
| Streaming | Yes | Yes | Yes | Yes |
| Use case | Primary agents | Coding tasks | Remote services | Coordination |
