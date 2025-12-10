---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/observability.py
title: Observability Configuration
description: Monitoring and tracing configuration
icon: material/monitor-eye
---

# Observability Configuration

Observability providers enable monitoring, tracing, and logging of agent operations. This helps you understand agent behavior, track performance, and debug issues in production.

## Overview

LLMling-Agent integrates with leading observability platforms:

- **Logfire**: Pydantic's observability platform with native support for structured logging
- **Langsmith**: LangChain's tracing and evaluation platform
- **AgentOps**: Specialized platform for agent monitoring and analytics
- **Arize Phoenix**: Open-source observability for LLM applications
- **Custom**: Integrate your own observability solution

These integrations provide automatic instrumentation of agent operations, capturing traces, spans, and metrics without code changes.

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.observability.ObservabilityProviderConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Key Features

- **Automatic tracing**: All agent operations are automatically traced
- **Structured logging**: Rich context and metadata for every operation
- **Performance metrics**: Token usage, latency, and cost tracking
- **Error tracking**: Detailed error capture with stack traces and context
- **Multi-provider support**: Use multiple observability platforms simultaneously

## Use Cases

- **Development**: Debug agent behavior and optimize prompts
- **Production monitoring**: Track performance and reliability metrics
- **Cost optimization**: Analyze token usage and identify expensive operations
- **Quality assurance**: Evaluate agent responses and identify issues
- **Compliance**: Maintain audit logs of agent interactions

## Configuration Notes

- Providers can be configured at the manifest level for global observability
- API keys should be stored in environment variables for security
- Multiple providers can be active simultaneously
- Each provider captures different metrics and views of agent behavior
- Some providers offer additional features like prompt optimization and A/B testing