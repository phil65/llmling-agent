# Quickstart Guide

## CLI Usage

Start an interactive session:
```bash
# Quick chat with GPT-5
uvx llmling-agent[default] quickstart openai:gpt-5-mini

# Enable streaming mode
uvx llmling-agent[default] quickstart openai:gpt-5-mini --stream
```

Initialize and manage configurations:
```bash
# Create starter configuration
llmling-agent init agents.yml

# Add to your configurations
llmling-agent add agents.yml

# Start chatting
llmling-agent chat assistant
```

Launch the web interface:
```bash
# Install UI dependencies
pip install "llmling-agent[ui]"

# Launch the web interface
llmling-agent launch
```

!!! tip
    Set `OPENAI_API_KEY` in your environment before running:
    ```bash
    export OPENAI_API_KEY='your-api-key-here'
    ```

## Configured Agents

Create an agent configuration:

```yaml
# agents.yml
agents:
  assistant:
    name: "Technical Assistant"
    model: openai:gpt-5-mini
    system_prompts:
      - You are a helpful technical assistant.
    toolsets:
      - type: file_access
```

Use it in code:

```python
from llmling_agent import AgentPool

async def main():
    async with AgentPool("agents.yml") as pool:
        agent = pool.get_agent("assistant")
        response = await agent.run("What is Python?")
        print(response.data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

!!! note "Web Interface Features"
    The web interface provides:
    - Interactive chat with your agents
    - Tool management
    - Command execution
    - Message history
    - Cost tracking

    Visit [http://localhost:7860](http://localhost:7860) after launching.

## Functional Interface

Quick model interactions without configuration:

```python
from llmling_agent import run_with_model, run_with_model_sync

# Async usage
async def main():
    # Simple completion
    result = await run_with_model(
        "Analyze this text",
        model="openai:gpt-5-mini"
    )
    print(result)

    # With structured output
    from pydantic import BaseModel

    class Analysis(BaseModel):
        summary: str
        key_points: list[str]

    result = await run_with_model(
        "Analyze the sentiment",
        model="openai:gpt-5-mini",
        result_type=Analysis
    )
    print(f"Summary: {result.summary}")
    print(f"Key points: {result.key_points}")

# Sync usage (convenience wrapper)
result = run_with_model_sync(
    "Quick question",
    model="openai:gpt-5-mini"
)
```

## Next Steps

- Learn about [Key Concepts](../key_concepts.md)
- Explore [Agent Configuration](../agent_config.md)
- Try the [Web Interface](../webui.md)
- See [Running Agents](../running_agents.md) for more usage patterns
- Check the [Command Reference](../commands.md) for CLI options

!!! note
    For details about environment configuration (tools, resources, etc.),
    see the [LLMling documentation](https://github.com/phil65/llmling).
