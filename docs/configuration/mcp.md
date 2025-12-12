---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/mcp_server.py
title: MCP Server Setup
description: MCP server configuration and integration
icon: material/server
---

MCP (Model Control Protocol) servers allow agents to use external tools through a standardized protocol. They can be configured at both agent and manifest levels.

## Overview

MCP servers provide a standardized way to expose tools and resources to agents. LLMling-Agent supports multiple server types:

- **Stdio**: Command-line servers using standard input/output
- **SSE**: Server-Sent Events based servers
- **StreamableHTTP**: HTTP streaming servers

## Scope Levels

MCP servers can be configured at different levels:

- **Agent level**: Servers available only to specific agents
- **Team level**: Servers shared within a team
- **Manifest level**: Global servers available to all agents

Servers can use simple string syntax (e.g., `"python -m mcp_server"`) or detailed configuration for advanced use cases.

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.mcp_server.MCPServerConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Configuration Notes

- Stdio servers execute as subprocesses with the specified command and arguments
- SSE and StreamableHTTP servers connect to remote URLs
- All server types support timeout configuration
- Environment variables can be passed to stdio servers
- Servers can be enabled/disabled without removing configuration
- Command strings like `"python -m mcp_server"` are automatically parsed into command and args
- Use manifest-level servers for shared tools, agent-level for specialized tools
