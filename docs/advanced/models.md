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

```yaml
agents:
  tuned_agent:
    model_settings:
      max_tokens: 2000          # Maximum tokens to generate
      temperature: 0.7          # Randomness (0.0 - 2.0)
      top_p: 0.9               # Nucleus sampling threshold
      timeout: 30.0            # Request timeout in seconds
      parallel_tool_calls: true # Allow parallel tool execution
      seed: 42                 # Random seed for reproducibility
      presence_penalty: 0.5     # (-2.0 to 2.0) Penalize token reuse
      frequency_penalty: 0.3    # (-2.0 to 2.0) Penalize token frequency
      logit_bias:              # Modify token likelihood
        "1234": 100  # Increase likelihood
        "5678": -100 # Decrease likelihood

### Example with Provider and Model Settings

  advanced_agent:
    provider:
      type: pydantic_ai
      name: "Advanced GPT-5"
      model: openai:gpt-5
      end_strategy: early
      validation_enabled: true
      allow_text_fallback: true
      model_settings:
        temperature: 0.8
        max_tokens: 1000
        presence_penalty: 0.2
        timeout: 60.0

  cautious_agent:
    provider:
      type: pydantic_ai
      name: "Careful Claude"
      model: anthropic:claude-sonnet-4-0
      retries: 3
      model_settings:
        temperature: 0.3  # More deterministic
        max_tokens: 2000
        timeout: 120.0    # Longer timeout
```


All settings are optional and providers will use their defaults if not specified.


## Setting pydantic-ai models by identifier

LLMling-agent also extends pydantic-ai functionality by allowing to define more models via simple
string identifiers. These providers are

- OpenRouter (`openrouter:provider/model-name`, requires `OPENROUTER_API_KEY` env var)
- Grok (X) (`grok:grok-2-1212`, requires `X_AI_API_KEY` env var)
- DeepSeek (`deepsek:deepsek-chat`, requires `DEEPSEEK_API_KEY` env var)

For detailed model documentation and features, see the [llmling-models repository](https://github.com/phil65/llmling-models).
