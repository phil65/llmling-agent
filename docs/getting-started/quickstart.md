---
title: Quickstart
description: Quick introduction to using AgentPool
icon: material/rocket-launch
---

# Quickstart Guide

## ACP Integration (Recommended)

The fastest way to get started is through the **Agent Client Protocol (ACP)**, which integrates agentpool directly into your IDE.

### One-Line Setup

No installation needed - run directly with uvx:

```bash
uvx --python 3.13 agentpool[default]@latest serve-acp agents.yml
```

### IDE Configuration (Zed)

Add to your Zed `settings.json`:

```json
{
  "agent_servers": {
    "LLMling": {
      "command": "uvx",
      "args": [
        "--python", "3.13",
        "agentpool[default]@latest",
        "serve-acp",
        "path/to/your/agents.yml",
        "--model-provider", "openai"
      ],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Your agents now appear as modes in Zed's agent panel - switch between them mid-conversation.

### Wrap External Agents

Integrate existing ACP-compatible agents (Claude Code, Goose, Codex, fast-agent) into your pool:

```yaml
agents:
  claude:
    type: acp
    provider: claude
    cwd: /path/to/project
  goose:
    type: goose
```

See [ACP Integration](../advanced/acp-integration/) for full details.

## CLI Usage

Initialize and manage configurations:

```bash
# Create starter configuration
agentpool init agents.yml

# Add to your configurations
agentpool add agents.yml

# Start chatting
agentpool chat assistant
```

## Configured Agents

Create an agent configuration:

```yaml
# agents.yml
agents:
  assistant:
    name: "Technical Assistant"
    model: openai:gpt-4
    system_prompts:
      - You are a helpful technical assistant.
    toolsets:
      - type: file_access
```

Use it in code:

```python
from agentpool import AgentPool

async def main():
    async with AgentPool("agents.yml") as pool:
        agent = pool.get_agent("assistant")
        response = await agent.run("What is Python?")
        print(response.data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Functional Interface

Quick model interactions without configuration:

```python
from agentpool import run_with_model, run_with_model_sync

# Async usage
async def main():
    result = await run_with_model(
        "Analyze this text",
        model="openai:gpt-4"
    )
    print(result)

    # With structured output
    from pydantic import BaseModel

    class Analysis(BaseModel):
        summary: str
        key_points: list[str]

    result = await run_with_model(
        "Analyze the sentiment",
        model="openai:gpt-4",
        output_type=Analysis
    )
    print(f"Summary: {result.summary}")

# Sync usage (convenience wrapper)
result = run_with_model_sync("Quick question", model="openai:gpt-4")
```
