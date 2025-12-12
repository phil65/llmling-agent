---
title: Models & Providers
description: Advanced model and provider features
icon: material/cpu-64-bit
---

# Provider models

In addition to the regular pydantic-ai models,
LLMling-agent supports all model types from [llmling-models](https://github.com/phil65/llmling-models) through YAML configuration. Each model is identified by its `type` field.
These models often are some kind of "meta-models", allowing model selection patterns as well
as human-in-the-loop interactions.

## Basic Configuration

```yaml
agents:
  my_agent:
    model:
      type: string           # Basic string model identifier
      identifier: gpt-5      # Model name
```

## Available Model Types

See Config section to see the available types.

## Model Settings

You can set common model settings to fine-tune the LLM behavior:

```yaml title="agents.yml"
--8<-- "docs/advanced/models_example.yml"
```

All settings are optional and providers will use their defaults if not specified.

## Setting pydantic-ai models by identifier

LLMling-agent also extends pydantic-ai functionality by allowing to define more models via simple
string identifiers. These providers are

- OpenRouter (`openrouter:provider/model-name`, requires `OPENROUTER_API_KEY` env var)
- Grok (X) (`grok:grok-2-1212`, requires `X_AI_API_KEY` env var)
- DeepSeek (`deepsek:deepsek-chat`, requires `DEEPSEEK_API_KEY` env var)

For detailed model documentation and features, see the [llmling-models repository](https://github.com/phil65/llmling-models).
