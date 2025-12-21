---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/agentpool_config/loaders.py
title: Resources
description: Resource loader configuration
icon: material/file-document
---

Resources provide dynamic content that can be accessed by agents during execution. They allow agents to fetch data from various sources like files, CLI commands, source code, and external systems.

## Overview

AgentPool supports multiple resource types:

- **Path**: Load content from file paths with pattern matching
- **Text**: Static text content with optional templating
- **CLI**: Execute command-line tools and capture output
- **Source**: Extract source code from Python modules and classes
- **LangChain**: Integration with LangChain document loaders
- **Callable**: Custom Python functions that return content

Resources are loaded on-demand when agents request them, supporting parameterization for dynamic content generation.

## Configuration Reference

/// mknodes
{{ "agentpool_config.loaders.Resource" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Key Features

- **On-demand loading**: Resources are loaded only when requested
- **Parameterization**: Pass parameters to resources for dynamic content
- **Caching**: Optional caching to improve performance
- **Pattern matching**: Use glob patterns to load multiple files
- **Content processing**: Transform and filter content before delivery
- **Access control**: Restrict resource access through capabilities

## Use Cases

- **Documentation**: Provide agents with access to project documentation
- **Code analysis**: Give agents access to source code for review or modification
- **Data access**: Load configuration files, datasets, or API responses
- **Dynamic content**: Generate content based on current state or parameters
- **External integration**: Fetch data from external systems and tools

## Configuration Notes

- Resources can be defined at manifest level (global) or agent level (local)
- Path resources support glob patterns for batch loading
- CLI resources execute in the system shell with security considerations
- Source resources automatically extract docstrings and type hints
- LangChain resources leverage the extensive LangChain loader ecosystem
- Callable resources provide maximum flexibility for custom logic
