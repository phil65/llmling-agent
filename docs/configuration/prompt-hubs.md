---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/prompt_hubs.py
title: Prompt Hub Configuration
description: External prompt hub integration
icon: material/hub
---

Prompt hubs are external platforms for managing, versioning, and sharing prompts. LLMling-Agent integrates with popular prompt management platforms to leverage curated prompt libraries and collaborative prompt development.

## Overview

LLMling-Agent supports integration with leading prompt hub platforms:

- **PromptLayer**: Comprehensive prompt management with versioning and analytics
- **Langfuse**: Open-source LLM engineering platform with prompt management
- **Fabric**: Community-driven prompt patterns and templates
- **Braintrust**: Enterprise prompt management with evaluation and testing

These integrations allow you to:

- Access curated prompt libraries
- Version and manage prompts centrally
- A/B test different prompt variations
- Collaborate on prompt development
- Track prompt performance and effectiveness

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.prompt_hubs.PromptHubConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Benefits

- **Centralized management**: Store and manage prompts in a single location
- **Version control**: Track changes and rollback to previous versions
- **Collaboration**: Share prompts across teams and projects
- **Testing**: A/B test prompts and measure effectiveness
- **Analytics**: Track prompt usage and performance metrics
- **Reusability**: Access community-curated prompt libraries

## Use Cases

- **Prompt engineering**: Iterate on prompts with version control
- **Team collaboration**: Share and review prompts with colleagues
- **Production stability**: Deploy tested, versioned prompts
- **Performance optimization**: Track and optimize prompt effectiveness
- **Best practices**: Leverage community-tested prompt patterns

## Configuration Notes

- API keys should be stored in environment variables
- Prompts from hubs can be referenced in agent system_prompts
- Hub integration provides automatic prompt versioning
- Some platforms offer additional features like prompt analytics and testing
- Prompts can be managed both locally and in external hubs
