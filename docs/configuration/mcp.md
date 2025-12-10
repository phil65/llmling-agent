---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/mcp_server.py
title: MCP Server Setup
description: MCP server configuration and integration
icon: material/server
---

# MCP Server Configuration

MCP (Model Control Protocol) servers allow agents to use external tools through a standardized protocol. They can be configured at both agent and manifest levels.

## Basic Configuration

MCP servers can be defined at the agent/team level for restricted access or at the manifest level for global access.

```yaml
agents:
  assistant:
    mcp_servers:
      - "python -m mcp_server"           # Simple command
      - "node mcp_server.js --debug"     # With arguments
teams:
  my_team:
    # other settings..
    mcp_servers:
      - "python -m mcp_server"
```

## Detailed Configuration

Full server configuration with all options:

```yaml
agents:
  assistant:
    mcp_servers:
      # Stdio-based server
      - type: "stdio"
        command: "python"
        args: ["-m", "mcp_server", "--debug"]
        enabled: true
        env:
          PYTHONPATH: "src"
          DEBUG: "1"
        timeout: 30.0
        name: "python-server"  # Optional identifier

      # SSE-based server
      - type: "sse"
        url: "http://localhost:3001"
        enabled: true
        timeout: 30.0

      # StreamableHTTP-based server
      - type: "streamable-http"
        url: "http://localhost:3002"
        enabled: true
```

## Manifest Level Configuration

Shared servers available to all agents:

```yaml
# Root level configuration
mcp_servers:
  - type: "stdio"
    command: "python"
    args: ["-m", "shared_mcp_server"]

  - type: "sse"
    url: "http://shared-server:3001"

agents:
  assistant:
    # Agent can still have its own servers
    mcp_servers:
      - "python -m agent_specific_server"
```

## Server Types

### Stdio Server

Uses standard input/output for communication:

```yaml
mcp_servers:
  - type: "stdio"
    command: "python"
    args: ["-m", "mcp_server"]
    env:
      PYTHONPATH: "src"
```

### SSE Server

Uses Server-Sent Events over HTTP:

```yaml
mcp_servers:
  - type: "sse"
    url: "http://localhost:3001"
```

### StreamableHTTP Server

Uses StreamableHTTP for communication:

```yaml
mcp_servers:
  - type: "streamable-http"
    url: "http://localhost:3002"
```

## Shorthand Syntax

Simple commands can use string shorthand:

```yaml
mcp_servers:
  - "python -m mcp_server"     # Converted to stdio config
  - "node server.js --debug"   # With arguments
```

## Pool Server Configuration

Configure how the agent pool exposes nodes and prompts through MCP:

```yaml
# At manifest level
pool_server:
  enabled: true

  # Resource exposure control
  serve_nodes: ["agent1", "agent2"]  # or true for all nodes
  serve_prompts: true                # expose all prompts

  # Transport configuration
  transport: "sse"                   # "stdio", "sse", or "streamable-http"
  host: "localhost"                  # for HTTP-based transports
  port: 3001                         # for HTTP-based transports
  cors_origins: ["*"]                # CORS settings

  # Editor integration
  zed_mode: false                    # Enable Zed editor compatibility
```

## Setting Environment Variables

Environment variables can be passed to MCP servers:

```yaml
mcp_servers:
  - type: "stdio"
    command: "python"
    args: ["-m", "mcp_server"]
    env:
      MODEL_API_KEY: "${API_KEY}"  # From environment
      DEBUG: "true"
      LOG_LEVEL: "debug"
```
