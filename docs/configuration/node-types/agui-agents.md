---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent/models/agui_agents.py
title: AG-UI Agents
description: AG-UI protocol agent integration
icon: material/api
---

AG-UI (Agent User Interface) agents connect to remote HTTP endpoints that implement the AG-UI protocol, enabling integration of any AG-UI compatible server into the LLMling-Agent pool.

## Overview

AG-UI is a protocol for building agent interfaces that provides:

- **HTTP-based communication**: Simple REST endpoints for agent interaction
- **Streaming support**: Real-time response streaming
- **Standardized interface**: Consistent API across different agent implementations

AG-UI agents are useful for:

- Integrating existing AG-UI compatible services
- Building distributed agent architectures
- Connecting to remote agent deployments
- Testing with locally spawned servers

## Configuration Reference

/// mknodes
{{ "llmling_agent.models.agui_agents.AGUIAgentConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Basic Usage

```yaml
agui_agents:
  remote_assistant:
    endpoint: http://localhost:8000/agent/run
    timeout: 30.0
    headers:
      X-API-Key: ${API_KEY}

  managed_agent:
    endpoint: http://localhost:8765/agent/run
    startup_command: "uv run ag-ui-server config.yml"
    startup_delay: 3.0
```

## Configuration Notes

- The `endpoint` field specifies the HTTP URL for the AG-UI agent server
- Use `headers` for authentication tokens or custom routing headers
- Environment variables can be used in header values: `${VAR_NAME}`
- When `startup_command` is provided, LLMling-Agent will spawn the server process
- The server process is automatically terminated when the agent pool closes
- Use `startup_delay` to give the server time to initialize before connecting
- The `timeout` setting controls the maximum time for agent responses
