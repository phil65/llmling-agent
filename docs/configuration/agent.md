---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/nodes.py
title: Agent Configuration
description: Agent configuration options
icon: material/robot
---

# Agent Configuration

Individual agent configurations define the behavior, capabilities, and settings for each agent in your manifest. Each agent entry in the `agents` dictionary represents a complete agent setup.

## Overview

Agent configuration includes:

- **Model settings**: LLM provider and model selection
- **System prompts**: Define agent behavior and personality
- **Tools and toolsets**: Capabilities available to the agent
- **Knowledge sources**: Context and information access
- **Output types**: Structured response definitions
- **Workers**: Sub-agents for delegation
- **Connections**: Message routing to other nodes
- **MCP servers**: Model Context Protocol integrations
- **Triggers**: Event-based activation

## Configuration Reference

/// mknodes
{{ "llmling_agent.models.agents.AgentConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", as_listitem=False) }}
///

## Configuration Inheritance

Agents can inherit configuration from other agents or base configurations:

```yaml
agents:
  base_agent:
    model: "openai:gpt-4o"
    retries: 2
    toolsets:
      - type: "resource_access"
  
  specialized_agent:
    inherits: "base_agent"
    description: "Specialized version"
    system_prompts:
      - "You are a specialized agent..."
```

## Related Configuration

- [Model Configuration](./model.md) - Configure LLM providers
- [System Prompts](./prompts.md) - Define agent behavior
- [Tools](./tools.md) - Individual tool configuration
- [Toolsets](./toolsets.md) - Tool collections
- [Workers](./worker.md) - Sub-agent delegation
- [Connections](./connections.md) - Message routing
- [MCP Servers](./mcp.md) - MCP integration
- [ACP Agents](acp-agents.md) - External ACP agent integration
- [AG-UI Agents](agui-agents.md) - AG-UI protocol agents
