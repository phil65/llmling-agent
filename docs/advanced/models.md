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

### Fallback Model

Tries models in sequence until one succeeds. Perfect for handling rate limits.

```yaml
agents:
  resilient_agent:
    model:
      type: fallback
      models:
        - openai:gpt-5          # First choice
        - openai:gpt-5-mini  # Fallback option
        - anthropic:claude-sonnet-4-0    # Last resort
```

### Cost-Optimized Model

Selects models based on budget constraints:

```yaml
agents:
  budget_agent:
    model:
      type: cost-optimized
      models:
        - openai:gpt-5
        - openai:gpt-5-mini
      max_input_cost: 0.1     # Maximum USD per request
      strategy: cheapest_possible  # Or best_within_budget
```

### Token-Optimized Model

Selects models based on context length:

```yaml
agents:
  context_aware_agent:
    model:
      type: token-optimized
      models:
        - openai:gpt-4-32k     # 32k context
        - openai:gpt-4         # 8k context
        - openai:gpt-3.5-turbo # 4k context
      strategy: efficient  # Or maximum_context
```

### Delegation Model

Uses a selector model to choose appropriate models for tasks:

```yaml
agents:
  smart_delegator:
    model:
      type: delegation
      selector_model: openai:gpt-5-mini
      models:
        - openai:gpt-5
        - openai:gpt-5-mini
      selection_prompt: |
        Pick gpt-5 for complex tasks,
        gpt-5-mini for simple queries.
```

### Input Model

For human-in-the-loop operations:

```yaml
agents:
  human_assisted:
    model:
      type: input
      prompt_template: "🤖 Question: {prompt}"
      show_system: true
      input_prompt: "Your answer: "
```

### Remote Input Model

Connects to a remote human operator:

```yaml
agents:
  remote_operator:
    model:
      type: remote-input
      url: ws://operator:8000/v1/chat/stream
      protocol: websocket  # or rest
      api_key: your-api-key
```

### LLM Library Integration

Use models from the LLM library:

```yaml
agents:
  llm_based:
    model:
      type: llm
      model_name: gpt-5-mini
```

### AISuite Integration

Use models from AISuite:

```yaml
agents:
  aisuite_agent:
    model:
      type: aisuite
      model: anthropic:claude-3-opus-20240229
      config:
        anthropic:
          api_key: your-api-key
```

### Augmented Model

Enhance prompts with pre/post processing:

```yaml
agents:
  enhanced_agent:
    model:
      type: augmented
      main_model: openai:gpt-5
      pre_prompt:
        text: "Expand this question: {input}"
        model: openai:gpt-5-mini
      post_prompt:
        text: "Summarize response: {output}"
        model: openai:gpt-5-mini
```

## Common Settings

Most models support these common settings:

```yaml
agents:
  example_agent:
    model:
      type: any_model_type
      # Common optional settings:
      name: optional_name          # Custom name for the model
      description: "Description"   # Model description
```

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
```

### Example with Provider and Model Settings

```yaml
agents:
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
