---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/builtin_tools.py
title: Built-in Tools
description: PydanticAI built-in tool configuration
icon: material/wrench
---

Built-in tools are pre-configured tools provided by PydanticAI that offer common functionality like web search, code execution, and image generation.

## Overview

LLMling-Agent supports the following PydanticAI built-in tools:

- **Web Search**: Search the web using various providers
- **Code Execution**: Execute code in a sandboxed environment
- **URL Context**: Fetch and process content from URLs
- **Image Generation**: Generate images using AI models
- **Memory**: Persistent memory storage for agents
- **MCP Server**: Connect to MCP server tools

These tools integrate seamlessly with the agent's capability system and can be configured with provider-specific options.

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.builtin_tools.BuiltinToolConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Usage Notes

- Built-in tools require specific API keys or credentials depending on the provider
- Tools can be restricted using the agent's capability system
- Some tools (like code execution) may require additional security considerations
- Memory tools persist data across agent runs
- MCP Server tools allow integration with external MCP servers
