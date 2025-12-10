---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/toolsets.py
title: Toolset Configuration
description: Toolset setup and management
icon: material/package
---

# Toolset Configuration

Toolsets are collections of tools that can be dynamically loaded from various sources. They provide a way to organize and manage groups of related tools.

## Overview

LLMling-Agent supports multiple toolset types:

- **OpenAPI**: Create tools from OpenAPI/Swagger specifications
- **Entry Points**: Load tools registered through Python entry points
- **Composio**: Integration with Composio tool platform
- **Upsonic**: Integration with Upsonic tool platform
- **Built-in Toolsets**: Agent management, execution environment, tool management, user interaction, history, and skills

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.toolsets.ToolsetConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Configuration Notes

- The `type` field serves as discriminator for toolset types
- Namespaces help prevent tool name collisions when combining multiple toolsets
- Toolsets are loaded when the agent initializes
- OpenAPI specs can be local files or URLs
- Entry points use standard Python entry point format
- Tools from toolsets can be filtered through agent capabilities
- API keys can be provided directly or via environment variables