# Provider Configuration

Providers determine how an agent processes messages and generates responses. The provider configuration is set in the agent's `provider` field.

## AI Provider (PydanticAI)
The default provider, using pydantic-ai for language model integration.

```yaml
agents:
  my-agent:
    provider:
      type: "pydantic_ai"  # provider discriminator
      name: "gpt4-agent"  # optional provider instance name
      end_strategy: "early"  # "early" | "complete" | "confirm"
      output_retries: 3  # max retries for result validation
      model: "openai:gpt-5"  # optional model override
      model_settings:  # additional settings passed to pydantic-ai
        temperature: 0.7
        max_output_tokens: 1000
      validation_enabled: true  # whether to validate outputs against schemas
      allow_text_fallback: true  # accept plain text when structure fails
```


## Model Settings

Available model settings that can be configured:

```yaml
model_settings:
  max_output_tokens: 1000  # Maximum tokens to generate
  temperature: 0.7             # Randomness (0.0-2.0)
  top_p: 0.9                   # Nucleus sampling (0.0-1.0)
  timeout: 60                  # Request timeout in seconds
  parallel_tool_calls: true    # Allow parallel tool calls
  seed: 42                     # Random seed for reproducible outputs
  presence_penalty: 0.0        # Penalty for token presence (-2.0-2.0)
  frequency_penalty: 0.0       # Penalty for token frequency (-2.0-2.0)
  logit_bias: {464: 100}       # Token biases
```
