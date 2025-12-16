---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent/models/acp_agents.py
title: ACP Agents
description: External ACP agent integration
icon: material/connection
---

ACP (Agent Communication Protocol) agents allow integration of external coding agents and AI assistants into the LLMling-Agent pool. These agents run as separate processes and communicate via the ACP protocol.

## Overview

LLMling-Agent supports integration with many popular coding agents:

- **Claude Code**: Anthropic's Claude-based coding agent
- **Gemini CLI**: Google's Gemini-based coding agent
- **Codex**: OpenAI Codex-based agent
- **OpenCode**: Open-source coding agent
- **Goose**: Block's AI coding assistant
- **OpenHands**: Open-source AI coding agent
- **FastAgent**: Fast and lightweight coding agent
- **Amp**: Sourcegraph's AI coding assistant
- **Auggie**: AI pair programming assistant
- **Cagent**: AI coding agent
- **Kimi**: Moonshot's AI coding agent
- **Stakpak**: AI infrastructure assistant
- **VTCode**: Visual Studio Code AI assistant
- **Mistral**: Mistral AI's coding agent

These agents can be configured to work alongside LLMling agents, enabling hybrid workflows that leverage specialized external tools.

## Configuration Reference

/// mknodes
{{ "llmling_agent.models.acp_agents.ACPAgentConfigTypes" | union_to_markdown(display_mode="yaml", header_style="pymdownx", as_listitem=False, wrapped_in="agentname") }}
///

## Configuration Notes

- ACP agents run as separate processes managed by LLMling-Agent
- Each agent type has specific configuration options for its underlying tool
- Model selection depends on what the external agent supports
- MCP servers can be attached to capable agents for extended functionality
- Environment variables can be used for API keys and secrets
- Agents are started on-demand and terminated when the pool closes
