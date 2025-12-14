---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/event_handlers.py
title: Event Handlers
description: Event handler setup and configuration
icon: material/bell-ring
---

Event handlers allow you to respond to various agent events such as messages, tool calls, and state changes.

/// mknodes
{{ "llmling_agent_config.event_handlers.EventHandlerConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///
