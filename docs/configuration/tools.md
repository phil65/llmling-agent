---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/tools.py
title: Tool Configuration
description: Tool registration and configuration
icon: material/tools
---

# Tool Configuration

Tools provide agents with specific capabilities. Configure tools using import-based tools, CrewAI tools, or LangChain tools.

/// mknodes
{{ "llmling_agent_config.tools.ToolConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///