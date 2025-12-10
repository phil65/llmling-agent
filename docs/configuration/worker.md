---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/workers.py
title: Worker Configuration
description: Worker agent configuration
icon: material/account-hard-hat
---

# Worker Configuration

Workers are specialized agents that can be registered with other agents to provide specific capabilities.

/// mknodes
{{ "llmling_agent_config.workers.WorkerConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///