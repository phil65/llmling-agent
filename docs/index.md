---
title: LLMling-Agent
description: A brand new AI framework. Fully async. Excellently typed. MCP & ACP Integration. Human in the loop. Unique messaging features.
icon: material/robot-outline
---

# LLMling-Agent

**A brand new AI framework. Fully async. Excellently typed. MCP & ACP Integration. Human in the loop. Unique messaging features.**

## Key Features

### 🔌 ACP Integration
First-class support for the Agent Client Protocol (ACP):

- Integrate directly into IDEs like Zed, VS Code, and others
- Wrap external agents (Claude Code, Goose, Codex, fast-agent) as nodes
- Unified node abstraction - ACP agents work like native agents
- Compose ACP agents into teams with native agents

### 📝 Easy Agent Configuration
LLMling-agent excels at static YAML-based agent configuration:

- Define agents with unprecedented detail in pure YAML (Pydantic-backed)
- Expansive JSON schema for IDE autocompletion and validation
- Agent "connection" setup via YAML enabling workflows without step-based code
- Configuration inheritance and reuse

### ⚡ True Async Framework
An async-first Agent framework:

- Proper async context management
- Non-blocking operations
- Streaming responses
- Automatic resource management

### 🧩 Unified Node Architecture
Everything is a MessageNode - enabling seamless composition:

- Native LLM agents
- ACP-wrapped external agents
- Teams (parallel and sequential)
- Human-in-the-loop nodes
- All nodes share the same interface

### 🏊 Pool-Based Architecture
Central coordination point for multi-agent systems:

- Type-safe dependency injection
- Shared resource management
- Dynamic agent/team creation and cloning
- Central monitoring and statistics

### 🔒 Type-Safety on Pydantic-Level
- Excellently typed user APIs
- Type-safe message passing
- Type-safe agent-team forming
- Structured response handling

## Quick Start

### One-Line ACP Setup

No installation needed - run directly with uvx:

```bash
uvx --python 3.13 llmling-agent[default]@latest serve-acp agents.yml
```

### Basic Agent Configuration

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

### Python Usage

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

## Installation

```bash
uv tool install llmling-agent[default]
```

## Available Extras

/// mknodes
{{ "extras"| MkDependencyGroups }}
///

## Dependencies

/// mknodes
{{ "llmling_agent"| MkDependencyTable }}
///

## License

MIT License - see [LICENSE](https://github.com/phil65/llmling-agent/blob/main/LICENSE) for details.