---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/storage.py
title: Storage Configuration
description: Database and storage setup
icon: material/database
---

The storage configuration defines how agent interactions, messages, and tool usage are logged. It's defined at the root level of the manifest.

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.storage.StorageProviderConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Overview

Storage providers define how agent interactions, messages, and tool usage are persisted. The system supports multiple providers including SQL databases, file storage, text logs, and in-memory storage.

Key features:

- **Multiple Providers**: Use multiple storage backends simultaneously
- **Agent Filtering**: Control which agents are logged per provider
- **Flexible Logging**: Configure what gets logged (messages, conversations, commands, context)
- **Provider Selection**: Automatic or explicit provider selection for queries

## Usage Example

```yaml title="agents.yml"
--8<-- "docs/configuration/storage_example.yml"
```

## Configuration Notes

- Multiple providers can be used simultaneously
- Agent filtering works at both global and provider levels
- Provider flags are combined with global flags using AND logic
- SQL provider is recommended for production use
- Memory provider is useful for testing
- Text logs support custom Jinja2 templates for flexible formatting
