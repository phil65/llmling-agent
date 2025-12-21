---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/agentpool_config/builtin_tools.py
title: Built-in Tools
description: PydanticAI built-in tool configuration
icon: material/wrench
---

Built-in tools are native tools provided by LLM providers that can be used to enhance your agent's capabilities. They are executed directly by the model provider.

## Overview

AgentPool supports the following PydanticAI built-in tools:

- **Web Search**: Search the web using various providers
- **Code Execution**: Execute code in a sandboxed environment
- **URL Context**: Fetch and process content from URLs
- **Image Generation**: Generate images using AI models
- **Memory**: Persistent memory storage for agents
- **MCP Server**: Connect to MCP server tools

These tools integrate seamlessly with the agent's capability system and can be configured with provider-specific options.

## Configuration Reference

/// mknodes
{{ "agentpool_config.builtin_tools.BuiltinToolConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///
