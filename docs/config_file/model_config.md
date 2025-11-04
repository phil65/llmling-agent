# Models & Providers


In addition to the regular models, LLMling-Agent supports some special kinds of models
provided by [LLMling-models](https://github.com/phil65/LLMling-models).


## Human-Interaction Models
Models that facilitate human interaction and input:

### Input Model
Basic console-based human input for testing and debugging:
```yaml
agents:
  reviewer:
    model:
      type: "input"
      prompt_template: "👤 Please respond to: {prompt}"
      show_system: true
      input_prompt: "Your response: "
      handler: "llmling_models:DefaultInputHandler"
```

### Remote Input Model
Connect to a remote human operator via WebSocket:
```yaml
agents:
  remote_reviewer:
    model:
      type: "remote-input"
      url: "ws://operator:8000/v1/chat/stream"
      api_key: "your-api-key"
```

### User Select Model
Let users interactively choose which model to use:
```yaml
agents:
  interactive:
    model:
      type: "user-select"
      models: ["openai:gpt-5", "openai:gpt-5-mini"]
      prompt_template: "🤖 Choose model for: {prompt}"
      input_prompt: "Enter model number (0-{max}): "
      handler: "llmling_models:DefaultInputHandler"
```

## Multi-Models

### Fallback Model
Try models in sequence until one succeeds:
```yaml
agents:
  resilient:
    model:
      type: "fallback"
      models:
        - "openai:gpt-5"         # Try first
        - "openai:5-mini"        # Fallback
        - "anthropic:claude-sonnet-4-0"   # Last resort
```

### Cost-Optimized Model
Select models based on budget constraints:
```yaml
agents:
  budget_aware:
    model:
      type: "cost-optimized"
      models: ["openai:gpt-5", "openai:5-mini"]
      max_input_cost: 0.1  # USD per request
      strategy: "best_within_budget"  # or "cheapest_possible"
```

### Token-Optimized Model
Select models based on context window requirements:
```yaml
agents:
  context_aware:
    model:
      type: "token-optimized"
      models:
        - "openai:gpt-4-32k"     # 32k context
        - "openai:gpt-4"         # 8k context
      strategy: "efficient"  # or "maximum_context"
```

### Delegation Model
Use a model to choose the most appropriate model:
```yaml
agents:
  smart_router:
    model:
      type: "delegation"
      selector_model: "openai:gpt-5-nano"
      models: ["openai:gpt-5", "openai:gpt-5-mini"]
      selection_prompt: "Pick gpt-5 for complex tasks, gpt-5-mini for simple queries."
```

## Wrapper Models


### Import Model
Import and use custom model implementations:
```yaml
agents:
  custom:
    model:
      type: "import"
      model: "myapp.models:CustomModel"
      kw_args:
        param1: "value1"
        param2: "value2"
```
