---
title: Model Configuration
description: Language model setup and configuration
icon: material/cpu-64-bit
---

# Models & Providers

## Overview

LLMling-Agent supports a wide range of model types through [LLMling-models](https://github.com/phil65/LLMling-models):

- **Standard LLM Providers**: OpenAI, Anthropic, Google, Groq, Mistral, and more
- **Human-Interaction Models**: Console input, remote input, user selection
- **Multi-Models**: Fallback chains, cost-optimization, token-optimization, delegation
- **Wrapper Models**: Custom implementations, monitoring, and specialized behaviors

Models can be specified simply as strings (e.g., `"openai:gpt-4"`), or with detailed configuration for advanced use cases.

## Configuration Reference

/// mknodes
{{ "llmling_models.models.AnyModelConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///
