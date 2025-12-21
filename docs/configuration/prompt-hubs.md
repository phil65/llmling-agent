---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/agentpool_config/prompt_hubs.py
title: Prompt Hubs
description: External prompt hub integration
icon: material/hub
---

Prompt hubs are external platforms for managing, versioning, and sharing prompts. AgentPool integrates with popular prompt management platforms to leverage curated prompt libraries and collaborative prompt development.

## Overview

AgentPool supports integration with leading prompt hub platforms:

- **PromptLayer**: Comprehensive prompt management with versioning and analytics
- **Langfuse**: Open-source LLM engineering platform with prompt management
- **Fabric**: Community-driven prompt patterns and templates
- **Braintrust**: Enterprise prompt management with evaluation and testing

These integrations allow you to fetch prompts from these services by identifiers.

## Configuration Reference

/// mknodes
{{ "agentpool_config.prompt_hubs.PromptHubConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Configuration Notes

- API keys should be stored in environment variables
- Prompts from hubs can be referenced in agent system_prompts
- Hub integration provides automatic prompt versioning
- Some platforms offer additional features like prompt analytics and testing
- Prompts can be managed both locally and in external hubs
